import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from diffusers import UNet2DModel, DDPMScheduler
from typing import Dict, Optional, Union, List, Any
import os
from tqdm.auto import tqdm
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

# Import from our main module
from UNetFinetune import DiffusionConfig, DiffusionModelLoader

class CustomImageDataset(Dataset):
    """Custom dataset for loading images for diffusion model fine-tuning."""
    
    def __init__(
        self,
        image_dir: str,
        transform=None,
        image_size: int = 256,
        exts: List[str] = ['.jpg', '.jpeg', '.png', '.bmp']
    ):
        """Initialize the dataset.
        
        Args:
            image_dir: Directory containing images
            transform: Optional transforms to apply
            image_size: Target image size
            exts: List of valid image extensions
        """
        self.image_dir = image_dir
        self.image_files = []
        
        # Find all image files
        for root, _, files in os.walk(image_dir):
            for file in files:
                if any(file.lower().endswith(ext) for ext in exts):
                    self.image_files.append(os.path.join(root, file))
        
        # Default transform if none provided
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
        else:
            self.transform = transform
            
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image


class DiffusionTrainer:
    """Training manager for diffusion model fine-tuning."""
    
    def __init__(
        self, 
        model_loader: DiffusionModelLoader,
        config: DiffusionConfig,
        output_dir: str = "fine_tuned_model",
        device: Optional[torch.device] = None
    ):
        """Initialize the trainer.
        
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
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Get model
        self.model = self.model_loader.unet
        
        # Create noise scheduler for training
        self.noise_scheduler = DDPMScheduler.from_pretrained(config.model_id)
    
    def prepare_dataset(
        self, 
        dataset_path: str,
        batch_size: int = 8,
        image_size: int = None,
        num_workers: int = 4
    ) -> DataLoader:
        """Prepare dataset for fine-tuning.
        
        Args:
            dataset_path: Path to the dataset
            batch_size: Batch size for training
            image_size: Target image size (default: from model config)
            num_workers: Number of workers for data loading
            
        Returns:
            DataLoader for the dataset
        """
        if image_size is None:
            image_size = self.model_loader.unet_config.sample_size
        
        print(f"Preparing dataset from {dataset_path} with image size {image_size}...")
        
        # Create dataset
        dataset = CustomImageDataset(
            image_dir=dataset_path,
            image_size=image_size
        )
        
        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            drop_last=True
        )
        
        print(f"Dataset prepared with {len(dataset)} images")
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
        """Fine-tune the model on a custom dataset.
        
        Args:
            dataloader: DataLoader for the dataset
            num_epochs: Number of training epochs
            learning_rate: Learning rate for optimization
            save_interval: Interval for saving checkpoints (in epochs)
            gradient_accumulation_steps: Number of steps to accumulate gradients
            mixed_precision: Whether to use mixed precision training
        """
        # Get model (unwrap from DataParallel if needed)
        model = self.model.module if hasattr(self.model, "module") else self.model
        
        # Setup optimizer
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
        
        # Setup scaler for mixed precision training
        scaler = torch.cuda.amp.GradScaler() if mixed_precision and torch.cuda.is_available() else None
        
        # Training loop
        print(f"Starting fine-tuning for {num_epochs} epochs with lr={learning_rate}...")
        
        global_step = 0
        losses = []
        
        for epoch in range(num_epochs):
            progress_bar = tqdm(total=len(dataloader))
            progress_bar.set_description(f"Epoch {epoch+1}/{num_epochs}")
            
            epoch_losses = []
            
            for step, batch in enumerate(dataloader):
                # Move batch to device
                clean_images = batch.to(self.device)
                batch_size = clean_images.shape[0]
                
                # Sample noise
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
                
                # Predict the noise residual
                with torch.cuda.amp.autocast(enabled=mixed_precision and torch.cuda.is_available()):
                    noise_pred = model(noisy_images, timesteps).sample
                    loss = nn.functional.mse_loss(noise_pred, noise)
                    loss = loss / gradient_accumulation_steps
                
                # Accumulate gradients
                if scaler is not None:
                    scaler.scale(loss).backward()
                    if (step + 1) % gradient_accumulation_steps == 0:
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                else:
                    loss.backward()
                    if (step + 1) % gradient_accumulation_steps == 0:
                        optimizer.step()
                        optimizer.zero_grad()
                
                # Log progress
                progress_bar.update(1)
                logs = {"loss": loss.detach().item() * gradient_accumulation_steps}
                progress_bar.set_postfix(**logs)
                epoch_losses.append(logs["loss"])
                
                global_step += 1
            
            # Compute average loss for the epoch
            avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
            losses.append(avg_epoch_loss)
            print(f"Epoch {epoch+1}/{num_epochs} - Average loss: {avg_epoch_loss:.6f}")
            
            # Save checkpoint
            if (epoch + 1) % save_interval == 0:
                self.save_model(os.path.join(self.output_dir, f"checkpoint-{epoch+1}"))
        
        # Save final model
        self.save_model(os.path.join(self.output_dir, "final_model"))
        
        return losses
    
    def save_model(self, output_dir: str):
        """Save the fine-tuned model.
        
        Args:
            output_dir: Directory to save the model
        """
        # Create directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Unwrap from DataParallel if needed
        model = self.model.module if hasattr(self.model, "module") else self.model
        
        # Save the model
        model.save_pretrained(output_dir)
        
        print(f"Model saved to {output_dir}")


def example_fine_tuning_script():
    """Example script for fine-tuning a diffusion model."""
    # Initialize configuration
    config = DiffusionConfig(
        model_id="google/ddpm-celebahq-256",
        visible_gpus="2,3",
        num_inference_steps=100
    )
    
    # Load the model
    model_loader = DiffusionModelLoader(config)
    model_loader.load_model()
    
    # Create trainer
    trainer = DiffusionTrainer(
        model_loader=model_loader,
        config=config,
        output_dir="fine_tuned_models/my_custom_model"
    )
    
    # Prepare dataset
    dataloader = trainer.prepare_dataset(
        dataset_path="path/to/your/images",
        batch_size=8,
        num_workers=4
    )
    
    # Train the model
    losses = trainer.train(
        dataloader=dataloader,
        num_epochs=20,
        learning_rate=1e-5,
        save_interval=5
    )
    
    print("Fine-tuning complete!")
    
    
if __name__ == "__main__":
    example_fine_tuning_script() 