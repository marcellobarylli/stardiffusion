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