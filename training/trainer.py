import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from diffusers import UNet2DModel, DDPMScheduler
from typing import Dict, Optional, Union, List, Any
import os
import sys
import subprocess
from tqdm.auto import tqdm
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from datetime import datetime
import types

# Import from our core module
from models.diffusion.core import DiffusionConfig, DiffusionModelLoader
from data.datasets import DatasetManager


class DiffusionTrainer:
    """Handles fine-tuning of diffusion models."""
    
    def __init__(
        self, 
        model_loader: DiffusionModelLoader,
        config: DiffusionConfig,
        output_dir: str = "checkpoints/fine_tuned_model",
        device: Optional[torch.device] = None
    ):
        """Initialize the diffusion trainer.
        
        Args:
            model_loader: Model loader containing the model to fine-tune
            config: Configuration for the diffusion process
            output_dir: Directory to save model checkpoints
            device: Device to use for training (default: from config)
        """
        self.model_loader = model_loader
        self.config = config
        self.output_dir = output_dir
        self.device = device if device is not None else config.device
        
        # Get the model from the loader
        self.model = model_loader.model
        
        # Move model to device (if not already)
        if self.model.device != self.device:
            self.model.to(self.device)
        
        # Get necessary parameters
        self.noise_scheduler = None
        self.optimizer = None
        self.scaler = None  # For mixed precision training
        
        # Load noise scheduler
        self.load_scheduler()
    
    def load_scheduler(self):
        """Load the noise scheduler for training."""
        # Use the same scheduler that was loaded with the model
        self.noise_scheduler = self.model_loader.scheduler
        
        # If scheduler wasn't loaded, create a new one
        if self.noise_scheduler is None:
            self.noise_scheduler = DDPMScheduler.from_pretrained(self.config.model_id)
    
    def prepare_dataset(
        self, 
        dataset_name_or_path: str,
        batch_size: int = 8,
        image_size: int = None,
        num_workers: int = 4
    ) -> DataLoader:
        """
        Prepare a dataset for training.
        
        Args:
            dataset_name_or_path: Name of a standard dataset or path to custom images
            batch_size: Batch size for training
            image_size: Image size to resize to (default: model's sample_size)
            num_workers: Number of dataloader workers
            
        Returns:
            DataLoader for the dataset
        """
        # If image_size is not specified, use the model's sample_size
        if image_size is None:
            # Get from model (handle DataParallel case)
            if hasattr(self.model, "module"):
                image_size = self.model.module.config.sample_size
            else:
                image_size = self.model.config.sample_size
                
        # Get the dataset using DatasetManager
        dataset = DatasetManager.get_dataset(
            dataset_name_or_path=dataset_name_or_path,
            image_size=image_size,
            train=True,
            download=True
        )
        
        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
        
        return dataloader
    
    def train(
        self,
        dataloader: DataLoader,
        num_epochs: int = 10,
        learning_rate: float = 1e-5,
        save_interval: int = 1,
        gradient_accumulation_steps: int = 1,
        mixed_precision: bool = True
    ):
        """
        Train the diffusion model.
        
        Args:
            dataloader: DataLoader for training data
            num_epochs: Number of epochs to train
            learning_rate: Learning rate for optimizer
            save_interval: Interval (in epochs) to save model checkpoints
            gradient_accumulation_steps: Number of steps to accumulate gradients
            mixed_precision: Whether to use mixed precision training
        """
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set up optimizer
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        
        # Set up mixed precision training if requested
        if mixed_precision and torch.cuda.is_available():
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None
            
        # Track total steps and losses
        global_step = 0
        losses = []
        
        # Training loop
        for epoch in range(num_epochs):
            progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch+1}/{num_epochs}")
            
            # Set model to training mode
            self.model.train()
            
            epoch_losses = []
            
            # Process batches
            for step, batch in progress_bar:
                # Get images from batch
                clean_images = batch["pixel_values"].to(self.device)
                batch_size = clean_images.shape[0]
                
                # Mixed precision context if requested
                with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                    # Sample random noise
                    noise = torch.randn(clean_images.shape).to(self.device)
                    
                    # Sample a random timestep for each image
                    timesteps = torch.randint(
                        0, 
                        self.noise_scheduler.config.num_train_timesteps, 
                        (batch_size,), 
                        device=self.device
                    ).long()
                    
                    # Add noise to the clean images according to the noise magnitude at each timestep
                    noisy_images = self.noise_scheduler.add_noise(clean_images, noise, timesteps)
                    
                    # Get model prediction
                    if isinstance(self.model, torch.nn.DataParallel):
                        # For DataParallel models
                        noise_pred = self.model.module(noisy_images, timesteps).sample
                    else:
                        # For regular models
                        noise_pred = self.model(noisy_images, timesteps).sample
                    
                    # Calculate loss
                    loss = nn.functional.mse_loss(noise_pred, noise)
                    
                    # Divide loss by accumulation steps if using gradient accumulation
                    if gradient_accumulation_steps > 1:
                        loss = loss / gradient_accumulation_steps
                
                # Backward pass with mixed precision if requested
                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                # Track loss
                epoch_losses.append(loss.item() * (gradient_accumulation_steps if gradient_accumulation_steps > 1 else 1))
                
                # Update weights if we've accumulated enough gradients
                if (step + 1) % gradient_accumulation_steps == 0 or step == len(dataloader) - 1:
                    if self.scaler is not None:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    
                    self.optimizer.zero_grad()
                    global_step += 1
                
                # Update progress bar
                progress_bar.set_postfix(loss=epoch_losses[-1])
            
            # Calculate average loss for the epoch
            avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
            losses.append(avg_epoch_loss)
            
            print(f"Epoch {epoch+1}/{num_epochs} - Average Loss: {avg_epoch_loss:.4f}")
            
            # Save checkpoint
            if (epoch + 1) % save_interval == 0 or epoch == num_epochs - 1:
                checkpoint_dir = os.path.join(self.output_dir, f"checkpoint-{epoch+1}")
                self.save_model(checkpoint_dir)
                print(f"Saved checkpoint at epoch {epoch+1} to {checkpoint_dir}")
        
        # Save final model
        final_model_dir = os.path.join(self.output_dir, "final")
        self.save_model(final_model_dir)
        print(f"Training complete. Final model saved to {final_model_dir}")
        
        return losses
    
    def save_model(self, output_dir: str):
        """
        Save the fine-tuned model.
        
        Args:
            output_dir: Directory to save the model
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Get the model (unwrap if it's in DataParallel)
        model_to_save = self.model.module if hasattr(self.model, "module") else self.model
        
        # Save model
        model_to_save.save_pretrained(output_dir)
        
        # Save scheduler config
        self.noise_scheduler.save_pretrained(output_dir)
        
        # Save training info
        training_info = {
            "model_id": self.config.model_id,
            "sample_size": model_to_save.config.sample_size,
            "in_channels": model_to_save.config.in_channels,
            "trained_epochs": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Save training info
        with open(os.path.join(output_dir, "training_info.txt"), "w") as f:
            for key, value in training_info.items():
                f.write(f"{key}: {value}\n")
        
        print(f"Saved model to {output_dir}") 