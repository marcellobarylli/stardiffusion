import torch
import os
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from typing import Dict, Optional, List, Union, Any, Tuple
from diffusers import DDPMScheduler
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import json
from datetime import datetime

# Import our refactored modules
from .core import DiffusionConfig, DiffusionModelLoader, DiffusionProcessor, DiffusionVisualizer
from models.coord_conv.unet import (
    CoordConvUNet2DModel, 
    convert_to_coordconv, 
    SymmetricCoordConvUNet2DModel,
    convert_to_symmetric_coordconv
)
from data.datasets import DatasetManager


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
        self.with_r = with_r
        self.normalize_coords = normalize_coords
        
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
        
        # Load the noise scheduler
        self.noise_scheduler = None
        self.load_scheduler()
        
    def load_scheduler(self):
        """Load the noise scheduler for training."""
        # Use the same scheduler that was loaded with the model
        self.noise_scheduler = self.model_loader.scheduler
        
        # If scheduler wasn't loaded, create a new one
        if self.noise_scheduler is None:
            self.noise_scheduler = DDPMScheduler.from_pretrained(self.config.model_id)
        
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
        self.num_epochs = num_epochs
        
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
                print(f"Epoch {epoch+1}/{self.num_epochs}, Batch {i+1}/{num_batches}, Loss: {loss.item():.4f}")
        
        # Log average loss for the epoch
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1}/{self.num_epochs} completed, Average Loss: {avg_loss:.4f}")
        
    def _training_step(self, images):
        """Single training step for diffusion model training."""
        # Get the noise scheduler
        noise_scheduler = self.noise_scheduler
        
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
        
        # Create and save the scheduler config
        # Load a standard DDPM scheduler for reference
        noise_scheduler = DDPMScheduler.from_pretrained("google/ddpm-celebahq-256")
        noise_scheduler.save_pretrained(output_dir)
        
        # Get current diffusers version
        from diffusers import __version__ as diffusers_version
        
        # Create and save model_index.json to enable pipeline loading
        model_index = {
            "_class_name": "DDPMPipeline",
            "_diffusers_version": diffusers_version,
            "scheduler": {
                "_class_name": "DDPMScheduler",
                "_diffusers_version": diffusers_version,
                "beta_end": noise_scheduler.config.beta_end,
                "beta_schedule": noise_scheduler.config.beta_schedule,
                "beta_start": noise_scheduler.config.beta_start,
                "clip_sample": noise_scheduler.config.clip_sample,
                "num_train_timesteps": noise_scheduler.config.num_train_timesteps,
                "prediction_type": noise_scheduler.config.prediction_type,
                "variance_type": noise_scheduler.config.variance_type
            },
            "unet": {
                "_class_name": "UNet2DModel",
                "_diffusers_version": diffusers_version,
            }
        }
        
        with open(os.path.join(output_dir, "model_index.json"), "w") as f:
            json.dump(model_index, f, indent=2)
        
        # Save training info
        training_info = {
            "model_id": self.config.model_id,
            "sample_size": model_to_save.unet.config.sample_size,
            "in_channels": model_to_save.unet.config.in_channels,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "saved_as_pipeline": True,
            "diffusers_version": diffusers_version,
            "with_r": hasattr(model_to_save, "coord_adder") and getattr(model_to_save.coord_adder, "with_r", False),
            "normalize_coords": hasattr(model_to_save, "coord_adder") and getattr(model_to_save.coord_adder, "normalize", True)
        }
        
        # Save training info
        with open(os.path.join(output_dir, "training_info.txt"), "w") as f:
            for key, value in training_info.items():
                f.write(f"{key}: {value}\n")
        
        print(f"Saved CoordConv model to {output_dir}")


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
        with open(os.path.join(output_dir, "symmetric_metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
        
        # Create and save the scheduler config
        # Load a standard DDPM scheduler for reference
        noise_scheduler = DDPMScheduler.from_pretrained("google/ddpm-celebahq-256")
        noise_scheduler.save_pretrained(output_dir)
        
        # Get current diffusers version
        from diffusers import __version__ as diffusers_version
        
        # Create and save model_index.json to enable pipeline loading
        model_index = {
            "_class_name": "DDPMPipeline",
            "_diffusers_version": diffusers_version,
            "scheduler": {
                "_class_name": "DDPMScheduler",
                "_diffusers_version": diffusers_version,
                "beta_end": noise_scheduler.config.beta_end,
                "beta_schedule": noise_scheduler.config.beta_schedule,
                "beta_start": noise_scheduler.config.beta_start,
                "clip_sample": noise_scheduler.config.clip_sample,
                "num_train_timesteps": noise_scheduler.config.num_train_timesteps,
                "prediction_type": noise_scheduler.config.prediction_type,
                "variance_type": noise_scheduler.config.variance_type
            },
            "unet": {
                "_class_name": "UNet2DModel",
                "_diffusers_version": diffusers_version,
            }
        }
        
        with open(os.path.join(output_dir, "model_index.json"), "w") as f:
            json.dump(model_index, f, indent=2)
        
        # Save training info
        training_info = {
            "model_id": self.config.model_id,
            "sample_size": model_to_save.unet.config.sample_size,
            "in_channels": model_to_save.unet.config.in_channels,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "saved_as_pipeline": True,
            "diffusers_version": diffusers_version,
            "symmetry_type": self.symmetry_type,
            "with_r": hasattr(model_to_save, "coord_adder") and getattr(model_to_save.coord_adder, "with_r", False),
            "normalize_coords": hasattr(model_to_save, "coord_adder") and getattr(model_to_save.coord_adder, "normalize", True)
        }
        
        # Save training info
        with open(os.path.join(output_dir, "training_info.txt"), "w") as f:
            for key, value in training_info.items():
                f.write(f"{key}: {value}\n")
        
        print(f"Saved Symmetric CoordConv model to {output_dir}")
    
    def _training_step(self, images):
        """Single training step with symmetry preservation."""
        # Get the noise scheduler
        noise_scheduler = self.noise_scheduler
        
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

# Helper functions for coordinate convolution operations

def convert_and_finetune_model(
    model_path: str,
    dataset_path: str,
    output_dir: str,
    batch_size: int = 8,
    num_epochs: int = 10,
    learning_rate: float = 1e-5,
    save_interval: int = 5,
    with_r: bool = False,
    normalize_coords: bool = True
):
    """
    Convert a standard UNet model to CoordConv and fine-tune it on the given dataset.
    
    Args:
        model_path: Path or ID of the pre-trained model to fine-tune (e.g., 'google/ddpm-celebahq-256')
        dataset_path: Path to the dataset for fine-tuning
        output_dir: Directory to save the fine-tuned model
        batch_size: Batch size for training
        num_epochs: Number of epochs for training
        learning_rate: Learning rate for optimizer
        save_interval: Interval (in epochs) to save model checkpoints
        with_r: Whether to include a radius channel in the coordinate information
        normalize_coords: Whether to normalize coordinate values to [-1, 1]
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Converting and fine-tuning model from {model_path}")
    print(f"Dataset: {dataset_path}")
    print(f"Output directory: {output_dir}")
    print(f"Using coordinate radius: {with_r}")
    
    # Create the configuration
    config = DiffusionConfig(
        model_id=model_path,
        batch_size=batch_size
    )
    
    # Load the model
    model_loader = DiffusionModelLoader(config)
    model_loader.load_model()
    
    # Initialize the CoordConv trainer
    trainer = CoordDiffusionTrainer(
        model_loader=model_loader,
        config=config,
        output_dir=output_dir,
        with_r=with_r,
        normalize_coords=normalize_coords
    )
    
    # Create a dataset and dataloader using the trainer's prepare_dataset method
    # This is the same approach used in train_starcraft.py
    from training.trainer import DiffusionTrainer
    temp_trainer = DiffusionTrainer(
        model_loader=model_loader,
        config=config,
        output_dir=output_dir
    )
    
    dataloader = temp_trainer.prepare_dataset(
        dataset_name_or_path=dataset_path,
        batch_size=batch_size,
        image_size=256  # Use the standard image size for StarCraft maps
    )
    
    print(f"Prepared dataset with {len(dataloader)} batches")
    
    # Train the model
    print(f"Starting training for {num_epochs} epochs with learning rate {learning_rate}")
    trainer.train(
        train_dataloader=dataloader,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        save_interval=save_interval
    )
    
    # Save the final model
    final_model_path = os.path.join(output_dir, "final")
    trainer.save_model(final_model_path)
    print(f"Fine-tuning complete! Final model saved to {final_model_path}")
    
    return final_model_path