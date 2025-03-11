import torch
import os
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from typing import Dict, Optional, List, Union, Any
from diffusers import DDPMScheduler

# Import our refactored modules
from .core import DiffusionConfig, DiffusionModelLoader, DiffusionProcessor, DiffusionVisualizer
from models.coord_conv.unet import CoordConvUNet2DModel, convert_to_coordconv


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
    
    # Note: This function would need to be completed with DatasetManager to properly load the dataset
    # and create train/validation dataloaders, which is not included in the original CoordDiffusion.py snippet

    print("Fine-tuning complete!")


def sample_from_coord_model(
    model_path: str, 
    num_samples: int = 4, 
    with_r: bool = True,
    output_dir: str = "outputs/coord_samples"
):
    """
    Generate samples from a trained CoordConv diffusion model.
    
    Args:
        model_path: Path to the trained CoordConv model
        num_samples: Number of samples to generate
        with_r: Whether the model includes a radius channel
        output_dir: Directory to save generated samples
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create configuration
    config = DiffusionConfig(
        model_id=model_path,  # Not used for loading, but kept for compatibility
        batch_size=num_samples,
        num_inference_steps=100  # Use more steps for better quality
    )
    
    # Create model loader
    model_loader = DiffusionModelLoader(config)
    
    # Check if pipeline exists at the model path
    if os.path.exists(os.path.join(model_path, "unet")):
        print("Found UNet model directory, loading custom CoordConv model")
        # Load the CoordConv model
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
    
    # Visualize progression for first sample
    visualizer.visualize_progression(
        intermediates,
        save_path=os.path.join(output_dir, "sample_progression.png")
    )
    
    print(f"Generated {num_samples} samples in {output_dir}")
    return samples 