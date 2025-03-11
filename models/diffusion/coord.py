import torch
import os
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from typing import Dict, Optional, List, Union, Any, Tuple
from diffusers import DDPMScheduler
import torch.nn.functional as F

# Import our refactored modules
from .core import DiffusionConfig, DiffusionModelLoader, DiffusionProcessor, DiffusionVisualizer
from models.coord_conv.unet import (
    CoordConvUNet2DModel, 
    convert_to_coordconv, 
    SymmetricCoordConvUNet2DModel,
    convert_to_symmetric_coordconv
)


class CoordDiffusionTrainer:
    """Extends DiffusionTrainer to use CoordConv for fine-tuning."""
    
    def __init__(
        self, 
        model_loader: DiffusionModelLoader,
        config: DiffusionConfig,
        output_dir: str = "checkpoints/coord_fine_tuned_model",
        device: Optional[torch.device] = None,
        with_r: bool = False,
        normalize_coords: bool = True
    ):
        """Initialize the CoordConv trainer.
        
        Args:
            model_loader: Model loader containing the model to fine-tune
            config: Configuration for the diffusion process
            output_dir: Directory to save model checkpoints
            device: Device to use for training (default: from config)
            with_r: Whether to include a radius channel
            normalize_coords: Whether to normalize coordinate values to [-1, 1]
        """
        self.model_loader = model_loader
        self.config = config
        self.output_dir = output_dir
        self.device = device if device is not None else config.device
        
        # Convert the UNet model to a CoordConv version
        original_unet = self.model_loader.model
        
        # Unwrap from DataParallel if necessary
        if hasattr(original_unet, "module"):
            # Model is wrapped in DataParallel, need to unwrap first
            unwrapped_unet = original_unet.module
            self.model = convert_to_coordconv(
                unwrapped_unet, 
                with_r=with_r, 
                normalize=normalize_coords
            )
            # Re-wrap in DataParallel if needed
            if config.use_data_parallel:
                self.model = torch.nn.DataParallel(self.model, device_ids=config.device_ids)
        else:
            # Model is not wrapped
            self.model = convert_to_coordconv(
                original_unet,
                with_r=with_r,
                normalize=normalize_coords
            )
        
        # Move to device
        self.model.to(self.device)
        
        # Initialize optimizer, learning rate scheduler, etc.
        self.optimizer = None
        self.lr_scheduler = None
        self.train_dataloader = None
        self.val_dataloader = None
        
    def train(
        self,
        train_dataloader: DataLoader,
        num_epochs: int = 10,
        val_dataloader: Optional[DataLoader] = None,
        learning_rate: float = 5e-6,
        save_interval: int = 5
    ):
        """Train the CoordConv model."""
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        
        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Train for specified number of epochs
        for epoch in range(num_epochs):
            # Train for one epoch
            self._train_epoch(epoch)
            
            # Save checkpoint at intervals
            if (epoch + 1) % save_interval == 0 or epoch == num_epochs - 1:
                checkpoint_dir = os.path.join(self.output_dir, f"checkpoint-{epoch+1}")
                self.save_model(checkpoint_dir)
                print(f"Saved checkpoint at epoch {epoch+1} to {checkpoint_dir}")
                
    def _train_epoch(self, epoch: int):
        """Train for one epoch."""
        # Set model to training mode
        self.model.train()
        
        total_loss = 0.0
        num_batches = len(self.train_dataloader)
        
        for i, batch in enumerate(self.train_dataloader):
            # Process batch and calculate loss
            images = batch['pixel_values'].to(self.device)
            loss = self._training_step(images)
            
            # Update total loss
            total_loss += loss.item()
            
            # Update model weights
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Log progress
            if (i + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{len(self.train_dataloader)}, Batch {i+1}/{num_batches}, Loss: {loss.item():.4f}")
        
        # Log average loss for the epoch
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1} completed, Average Loss: {avg_loss:.4f}")
        
    def _training_step(self, images):
        """Single training step."""
        # Training step implementation would go here
        # This is a placeholder and should be implemented for your specific task
        raise NotImplementedError("Training step not implemented")
        
    def save_model(self, output_dir: str):
        """Save the fine-tuned CoordConv model."""
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Get the model (unwrap if it's in DataParallel)
        model_to_save = self.model.module if hasattr(self.model, "module") else self.model
        
        # Check that it's a CoordConvUNet2DModel
        if not isinstance(model_to_save, CoordConvUNet2DModel):
            raise ValueError("Model is not a CoordConvUNet2DModel, cannot save")
        
        # Save using the model's save_pretrained method
        model_to_save.save_pretrained(output_dir)
        
        print(f"Saved CoordConv model to {output_dir}")

    def _training_step(self, images):
        """Single training step for diffusion model training."""
        # Get the noise scheduler
        noise_scheduler = DDPMScheduler.from_pretrained("google/ddpm-celebahq-256")
        
        # Add noise to the clean images 
        noise = torch.randn_like(images)
        batch_size = images.shape[0]
        
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, 
            noise_scheduler.config.num_train_timesteps, 
            (batch_size,), 
            device=self.device
        )
        
        # Add noise to the clean images according to the noise magnitude at each timestep
        noisy_images = noise_scheduler.add_noise(images, noise, timesteps)
        
        # Get the model prediction
        noise_pred = self.model(noisy_images, timesteps).sample
        
        # Calculate the loss
        loss = F.mse_loss(noise_pred, noise)
        
        return loss


class SymmetricCoordDiffusionTrainer(CoordDiffusionTrainer):
    """Trainer for symmetric CoordConv diffusion models."""
    
    def __init__(
        self,
        model_loader: DiffusionModelLoader,
        config: DiffusionConfig,
        symmetry_type: str = 'vertical',  # 'vertical', 'horizontal', 'both'
        output_dir: str = "checkpoints/symmetric_coord_model",
        with_r: bool = True,
        normalize_coords: bool = True
    ):
        super().__init__(
            model_loader=model_loader,
            config=config,
            output_dir=output_dir,
            with_r=with_r,
            normalize_coords=normalize_coords
        )
        self.symmetry_type = symmetry_type
        
        # Validate symmetry type
        if symmetry_type not in ['vertical', 'horizontal', 'both']:
            raise ValueError(f"Invalid symmetry type: {symmetry_type}. Must be one of: vertical, horizontal, both")
    
    def save_model(self, output_dir: str):
        """Save the fine-tuned Symmetric CoordConv model."""
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Get the model (unwrap if it's in DataParallel)
        model_to_save = self.model.module if hasattr(self.model, "module") else self.model
        
        # Store symmetry information in config
        model_to_save.config.symmetry_type = self.symmetry_type
        
        # Save using the model's save_pretrained method
        model_to_save.save_pretrained(output_dir)
        
        # Save additional metadata
        metadata = {
            "symmetry_type": self.symmetry_type,
            "with_r": self.with_r,
            "normalize_coords": self.normalize_coords
        }
        
        # Save metadata as JSON
        import json
        with open(os.path.join(output_dir, "symmetric_metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Saved Symmetric CoordConv model to {output_dir}")
    
    def _training_step(self, images):
        """Single training step with symmetry preservation."""
        # Get the noise scheduler
        noise_scheduler = DDPMScheduler.from_pretrained("google/ddpm-celebahq-256")
        
        # Add noise to the clean images 
        noise = torch.randn_like(images)
        batch_size = images.shape[0]
        
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, 
            noise_scheduler.config.num_train_timesteps, 
            (batch_size,), 
            device=self.device
        )
        
        # Add noise to the clean images according to the noise magnitude at each timestep
        noisy_images = noise_scheduler.add_noise(images, noise, timesteps)
        
        # Apply symmetry to the noise based on the chosen symmetry type
        if self.symmetry_type == 'vertical':
            # Make the noise symmetric across the vertical axis
            h, w = noise.shape[2], noise.shape[3]
            # Flip horizontally and average
            flipped_noise = torch.flip(noise, [3])  # Flip along width dimension
            symmetric_noise = (noise + flipped_noise) / 2  # Average to create symmetric noise
            
        elif self.symmetry_type == 'horizontal':
            # Make the noise symmetric across the horizontal axis
            flipped_noise = torch.flip(noise, [2])  # Flip along height dimension
            symmetric_noise = (noise + flipped_noise) / 2
            
        elif self.symmetry_type == 'both':
            # Make the noise symmetric across both axes
            h_flipped = torch.flip(noise, [2])  # Flip along height
            v_flipped = torch.flip(noise, [3])  # Flip along width
            both_flipped = torch.flip(noise, [2, 3])  # Flip along both dimensions
            symmetric_noise = (noise + h_flipped + v_flipped + both_flipped) / 4
        
        # Get the model prediction
        noise_pred = self.model(noisy_images, timesteps).sample
        
        # Calculate the loss against the symmetric noise
        loss = F.mse_loss(noise_pred, symmetric_noise)
        
        return loss


def convert_and_finetune_model(
    model_path: str,
    dataset_path: str,
    output_dir: str,
    batch_size: int = 8,
    num_epochs: int = 15,
    learning_rate: float = 5e-6,
    save_interval: int = 5,
    visible_gpus: str = "2,3",
    with_r: bool = True,
    normalize_coords: bool = True,
    num_workers: int = 4
):
    """
    Convert a pre-trained UNet model to CoordConv and fine-tune it on a dataset.
    
    Args:
        model_path: Path to the pre-trained model
        dataset_path: Path to the dataset for fine-tuning
        output_dir: Directory to save fine-tuned model
        batch_size: Batch size for training
        num_epochs: Number of epochs to train
        learning_rate: Learning rate for optimizer
        save_interval: Interval (in epochs) to save model checkpoints
        visible_gpus: Comma-separated list of GPU IDs to use
        with_r: Whether to include a radius channel
        normalize_coords: Whether to normalize coordinate values
        num_workers: Number of data loader workers
    """
    # Import dataset manager
    from data.datasets import DatasetManager
    
    # Create configuration
    config = DiffusionConfig(
        model_id=model_path,
        visible_gpus=visible_gpus,
        batch_size=batch_size
    )
    
    # Create model loader and load model
    model_loader = DiffusionModelLoader(config)
    model_loader.load_model()
    
    # Create CoordConv trainer
    trainer = CoordDiffusionTrainer(
        model_loader=model_loader,
        config=config,
        output_dir=output_dir,
        with_r=with_r,
        normalize_coords=normalize_coords
    )
    
    # Load the dataset
    print(f"Loading dataset from {dataset_path}...")
    dataset = DatasetManager.get_dataset(
        dataset_name_or_path=dataset_path,
        image_size=model_loader.image_size,
        train=True
    )
    
    # Create data loaders
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"Dataset loaded. Training set: {train_size} images, Validation set: {val_size} images")
    
    # Train the model
    print(f"Starting training for {num_epochs} epochs...")
    trainer.train(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        save_interval=save_interval
    )
    
    # Save the final model
    final_output_dir = os.path.join(output_dir, "final")
    trainer.save_model(final_output_dir)
    print(f"Training complete! Final model saved to {final_output_dir}")


def convert_and_finetune_symmetric_model(
    model_path: str,
    dataset_path: str,
    output_dir: str,
    symmetry_type: str = 'vertical',  # 'vertical', 'horizontal', 'both'
    batch_size: int = 8,
    num_epochs: int = 15,
    learning_rate: float = 5e-6,
    save_interval: int = 5,
    visible_gpus: str = "2,3",
    with_r: bool = True,
    normalize_coords: bool = True,
    num_workers: int = 4
):
    """
    Convert a pre-trained UNet model to Symmetric CoordConv and fine-tune it on a dataset.
    
    Args:
        model_path: Path to the pre-trained model
        dataset_path: Path to the dataset for fine-tuning
        output_dir: Directory to save fine-tuned model
        symmetry_type: Type of symmetry to enforce ('vertical', 'horizontal', 'both')
        batch_size: Batch size for training
        num_epochs: Number of epochs to train
        learning_rate: Learning rate for optimizer
        save_interval: Interval (in epochs) to save model checkpoints
        visible_gpus: Comma-separated list of GPU IDs to use
        with_r: Whether to include a radius channel
        normalize_coords: Whether to normalize coordinate values
        num_workers: Number of data loader workers
    """
    # Import dataset manager
    from data.datasets import DatasetManager
    
    # Create configuration
    config = DiffusionConfig(
        model_id=model_path,
        visible_gpus=visible_gpus,
        batch_size=batch_size
    )
    
    # Create model loader and load model
    model_loader = DiffusionModelLoader(config)
    model_loader.load_model()
    
    # Create Symmetric CoordConv trainer
    trainer = SymmetricCoordDiffusionTrainer(
        model_loader=model_loader,
        config=config,
        symmetry_type=symmetry_type,
        output_dir=output_dir,
        with_r=with_r,
        normalize_coords=normalize_coords
    )
    
    # Load the dataset
    print(f"Loading dataset from {dataset_path}...")
    dataset = DatasetManager.get_dataset(
        dataset_name_or_path=dataset_path,
        image_size=model_loader.image_size,
        train=True
    )
    
    # Create data loaders
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"Dataset loaded. Training set: {train_size} images, Validation set: {val_size} images")
    
    # Train the model
    print(f"Starting symmetric training with {symmetry_type} symmetry for {num_epochs} epochs...")
    trainer.train(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        save_interval=save_interval
    )
    
    # Save the final model
    final_output_dir = os.path.join(output_dir, "final")
    trainer.save_model(final_output_dir)
    print(f"Symmetric training complete! Final model saved to {final_output_dir}")


def sample_from_coord_model(
    model_path: str, 
    num_samples: int = 4, 
    with_r: bool = True,
    output_dir: str = "outputs/coord_samples",
    num_inference_steps: int = 200  # Increased for better quality
):
    """
    Generate samples from a trained CoordConv diffusion model.
    
    Args:
        model_path: Path to the trained CoordConv model
        num_samples: Number of samples to generate
        with_r: Whether the model includes a radius channel
        output_dir: Directory to save generated samples
        num_inference_steps: Number of inference steps for the diffusion process
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create configuration
    config = DiffusionConfig(
        model_id=model_path,  # Not used for loading, but kept for compatibility
        batch_size=num_samples,
        num_inference_steps=num_inference_steps
    )
    
    print(f"Generating samples with {num_inference_steps} inference steps")
    
    # Create model loader
    model_loader = DiffusionModelLoader(config)
    
    # Check if pipeline exists at the model path
    if os.path.exists(os.path.join(model_path, "unet")):
        print("Found UNet model directory, loading custom CoordConv model")
        # Load the CoordConv model
        try:
            print(f"Loading CoordConv model from {model_path} with with_r={with_r}")
            model_loader.model = CoordConvUNet2DModel.from_pretrained(
                model_path,
                with_r=with_r,
                normalize=True
            )
            
            # Extract necessary parameters from model
            model_loader.image_size = model_loader.model.unet.config.sample_size
            model_loader.in_channels = model_loader.model.unet.config.in_channels
            
            # Load the scheduler
            model_loader.scheduler = DDPMScheduler.from_pretrained(model_path)
            
            # Move model to device
            model_loader.model.to(config.device)
            
            # No pipeline available in this case
            model_loader.pipeline = None
            
            print(f"Successfully loaded CoordConv model with parameters:")
            print(f"  - Image size: {model_loader.image_size}")
            print(f"  - Input channels: {model_loader.in_channels}")
            print(f"  - With radius: {with_r}")
        except Exception as e:
            print(f"Error loading CoordConv model: {e}")
            print("Falling back to standard model")
            model_loader.load_model()
    else:
        # Try loading normally, which will try the pipeline first
        print("Loading model using standard loader")
        model_loader.load_model()
    
    print(f"Model info: {model_loader.get_model_info()}")
    
    # Create processor and visualizer
    processor = DiffusionProcessor(model_loader, config)
    visualizer = DiffusionVisualizer(model_loader)
    
    # Generate samples
    print(f"Generating {num_samples} samples...")
    samples, intermediates = processor.generate_sample(save_intermediates=True)
    
    # Visualize samples
    for i in range(min(num_samples, samples.shape[0])):
        sample = samples[i:i+1]  # Extract a single sample with batch dimension
        visualizer.visualize_final_image(
            sample,
            save_path=os.path.join(output_dir, f"sample_{i+1}.png")
        )
    
    # Save more intermediate steps for debugging
    num_intermediates = len(intermediates)
    print(f"Saving {num_intermediates} intermediate steps")
    for i, intermediate in enumerate(intermediates):
        if i % max(1, num_intermediates // 10) == 0 or i == num_intermediates - 1:
            step_num = i * num_inference_steps // num_intermediates
            visualizer.visualize_final_image(
                intermediate,
                save_path=os.path.join(output_dir, f"intermediate_step_{step_num}.png")
            )
    
    # Visualize progression for first sample
    visualizer.visualize_progression(
        intermediates,
        save_path=os.path.join(output_dir, "diffusion_progression.png")
    )
    
    print(f"Generated {num_samples} samples in {output_dir}")
    return samples 


def sample_direct_from_coordconv(
    model_path: str,
    num_samples: int = 4,
    with_r: bool = True,
    output_dir: str = "outputs/coordconv_direct",
    num_inference_steps: int = 200
):
    """
    Generate samples from a trained CoordConv model using direct coordinate handling.
    This is a more explicit implementation that handles the coordinate channels manually.
    
    Args:
        model_path: Path to the trained CoordConv model
        num_samples: Number of samples to generate
        with_r: Whether to include radius channel
        output_dir: Directory to save samples
        num_inference_steps: Number of inference steps
    """
    import os
    import torch
    import matplotlib.pyplot as plt
    from diffusers import UNet2DModel, DDPMScheduler
    from diffusers.utils.torch_utils import randn_tensor
    from models.coord_conv.layers import AddCoordinateChannels
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    print(f"Generating samples with direct CoordConv handling...")
    
    # Load the base UNet model directly
    try:
        unet = UNet2DModel.from_pretrained(os.path.join(model_path, "unet"))
        print(f"Successfully loaded UNet with config: {unet.config}")
    except Exception as e:
        print(f"Error loading UNet: {e}")
        return None
    
    # Load the channel mapper
    mapper_path = os.path.join(model_path, "channel_mapper.pt")
    if os.path.exists(mapper_path):
        print(f"Loading channel mapper from {mapper_path}")
        channel_mapper = torch.nn.Conv2d(
            unet.config.in_channels + (3 if with_r else 2),  # Original + coords
            unet.config.in_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )
        channel_mapper.load_state_dict(torch.load(mapper_path, map_location="cpu"))
        print("Successfully loaded channel mapper")
    else:
        print(f"Channel mapper not found at {mapper_path}")
        return None
    
    # Create coordinate adder
    coord_adder = AddCoordinateChannels(with_r=with_r, normalize=True)
    
    # Load scheduler
    scheduler = DDPMScheduler.from_pretrained(model_path)
    
    # Move models to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    unet.to(device)
    channel_mapper.to(device)
    
    # Set seed for reproducibility
    generator = torch.Generator(device=device).manual_seed(42)
    
    # Start from random noise
    image_size = unet.config.sample_size
    sample = randn_tensor(
        (num_samples, unet.config.in_channels, image_size, image_size),
        generator=generator,
        device=device
    )
    
    # Store intermediate samples
    intermediates = [sample.cpu().clone()]
    
    # Diffusion process (denoising)
    for t in range(num_inference_steps):
        print(f"Inference step {t+1}/{num_inference_steps}", end="\r")
        
        # Set model to evaluation mode
        unet.eval()
        
        # Get the timestep (schedule)
        timestep = scheduler.timesteps[t]
        
        # Add coordinate channels to input
        with torch.no_grad():
            # Add coordinates
            augmented_sample = coord_adder(sample)
            
            # Map back to original channels
            mapped_sample = channel_mapper(augmented_sample)
            
            # Get prediction from UNet
            noise_pred = unet(mapped_sample, timestep).sample
            
            # Update sample with scheduler
            sample = scheduler.step(noise_pred, timestep, sample).prev_sample
        
        # Save intermediate result
        if t % max(1, num_inference_steps // 10) == 0 or t == num_inference_steps - 1:
            intermediates.append(sample.cpu().clone())
    
    print("\nGeneration complete!")
    
    # Visualize samples
    for i in range(num_samples):
        img = sample[i].cpu()
        
        # Format for visualization (normalize and convert shape)
        img = img.permute(1, 2, 0)  # Change from (C,H,W) to (H,W,C)
        img = (img - img.min()) / (img.max() - img.min())  # Normalize to [0,1]
        
        plt.figure(figsize=(8, 8))
        plt.imshow(img.numpy())
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"sample_{i+1}.png"))
        plt.close()
        print(f"Saved sample {i+1} to {os.path.join(output_dir, f'sample_{i+1}.png')}")
    
    # Visualize progression
    num_steps = len(intermediates)
    fig, axes = plt.subplots(1, num_steps, figsize=(num_steps * 3, 3))
    
    for i, step_sample in enumerate(intermediates):
        img = step_sample[0].permute(1, 2, 0)  # Get first sample, convert shape
        img = (img - img.min()) / (img.max() - img.min())  # Normalize to [0,1]
        
        if num_steps == 1:
            axes.imshow(img.numpy())
            axes.set_title(f"Step {i}")
            axes.axis('off')
        else:
            axes[i].imshow(img.numpy())
            axes[i].set_title(f"Step {i}")
            axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "progression.png"))
    plt.close()
    print(f"Saved progression visualization to {os.path.join(output_dir, 'progression.png')}")
    
    return sample 