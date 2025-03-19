import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
import argparse
import logging
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torchvision import transforms

# Add the project root to sys.path
project_root = Path(__file__).absolute().parent.parent
sys.path.append(str(project_root))

from models.coord_conv.learning_symmetry import LearnableSymmetryTransformer
from diffusers import UNet2DModel, DDPMScheduler, DDPMPipeline


# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="End-to-end training of learnable symmetry diffusion")
    
    # Data parameters
    parser.add_argument("--dataset_path", type=str, required=True,
                      help="Path to the dataset directory with images")
    parser.add_argument("--output_dir", type=str, default="checkpoints/end_to_end_learnable",
                      help="Directory to save model checkpoints and training progress")
    
    # Model parameters
    parser.add_argument("--base_model", type=str, default=None,
                      help="Path to a pretrained diffusion model to fine-tune")
    parser.add_argument("--transformer_path", type=str, default=None,
                      help="Path to a pretrained transformer model to use")
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=4,
                      help="Batch size for training")
    parser.add_argument("--image_size", type=int, default=256,
                      help="Size of images for training")
    parser.add_argument("--num_epochs", type=int, default=100,
                      help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                      help="Learning rate for diffusion model")
    parser.add_argument("--transformer_lr", type=float, default=1e-4,
                      help="Learning rate for symmetry transformer")
    parser.add_argument("--save_interval", type=int, default=10,
                      help="Save model checkpoint every N epochs")
    parser.add_argument("--sample_interval", type=int, default=5,
                      help="Generate samples every N epochs")
    
    # Model configuration
    parser.add_argument("--symmetry_type", type=str, default="vertical",
                      choices=["vertical", "horizontal", "both"],
                      help="Type of symmetry to learn")
    parser.add_argument("--with_r", action="store_true",
                      help="Include radius coordinate channel")
    parser.add_argument("--freeze_unet", action="store_true",
                      help="Freeze the UNet parameters")
    parser.add_argument("--unfreeze_epoch", type=int, default=20,
                      help="Epoch at which to unfreeze the UNet (if frozen)")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                      help="Number of gradient accumulation steps")
    parser.add_argument("--symmetry_weight", type=float, default=0.3,
                      help="Weight of symmetry loss in transformer loss")
    
    # Device
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                      help="Device to train on (default: auto-detect)")
    
    return parser.parse_args()


def prepare_dataset(dataset_path, image_size, batch_size):
    """
    Prepare a dataset from a directory of images.
    """
    # Setup image transforms
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    # Create dataset
    class ImageDataset(Dataset):
        def __init__(self, path, transform):
            self.path = path
            self.transform = transform
            self.image_paths = [os.path.join(path, f) for f in os.listdir(path) 
                               if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            logger.info(f"Found {len(self.image_paths)} images in {path}")
            
        def __len__(self):
            return len(self.image_paths)
            
        def __getitem__(self, idx):
            img_path = self.image_paths[idx]
            img = Image.open(img_path).convert('RGB')
            return self.transform(img)
    
    dataset = ImageDataset(dataset_path, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    return dataloader


class EndToEndLearnableSystem:
    """
    End-to-end system for training a learnable symmetry transformer with a diffusion model.
    The system consists of two components:
    1. A symmetry transformer that learns to transform noise into symmetric noise
    2. A diffusion model (UNet) that learns to denoise images starting from the transformed noise
    """
    
    def __init__(
        self,
        transformer=None,
        unet=None,
        scheduler=None,
        symmetry_type="vertical",
        with_r=True,
        learning_rate=1e-5,
        transformer_lr=1e-4,
        device="cuda",
        output_dir="checkpoints/end_to_end_learnable",
        freeze_unet=False,
        unfreeze_epoch=20,
        gradient_accumulation_steps=1,
        symmetry_weight=0.3
    ):
        # Initialize or load models
        self.device = device
        self.output_dir = output_dir
        self.freeze_unet = freeze_unet
        self.unfreeze_epoch = unfreeze_epoch
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.symmetry_weight = symmetry_weight
        self.symmetry_type = symmetry_type
        
        # Create transformer if not provided
        if transformer is None:
            self.transformer = LearnableSymmetryTransformer(
                channels=3,
                hidden_dim=64,
                with_r=with_r,
                normalize=True,
                symmetry_type=symmetry_type,
                symmetry_weight=symmetry_weight
            ).to(device)
        else:
            self.transformer = transformer
            
        # Create UNet if not provided
        if unet is None:
            self.unet = UNet2DModel(
                sample_size=64,  # Default size, will resize input as needed
                in_channels=3,
                out_channels=3,
                layers_per_block=2,
                block_out_channels=(128, 128, 256, 256, 512, 512),
                down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D", 
                                 "DownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
                up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D", 
                               "UpBlock2D", "UpBlock2D", "UpBlock2D")
            ).to(device)
        else:
            self.unet = unet
            
        # Create scheduler if not provided
        if scheduler is None:
            self.scheduler = DDPMScheduler(
                num_train_timesteps=1000,
                beta_start=0.0001,
                beta_end=0.02,
                beta_schedule="linear"
            )
        else:
            self.scheduler = scheduler
            
        # Create optimizers
        self.optimizer_transformer = AdamW(
            self.transformer.parameters(),
            lr=transformer_lr
        )
        
        self.optimizer_unet = AdamW(
            self.unet.parameters(),
            lr=learning_rate
        )
        
        # Freeze UNet if requested
        if freeze_unet:
            logger.info("Freezing UNet parameters")
            for param in self.unet.parameters():
                param.requires_grad = False
                
        # Setup logging directory
        self.setup_logging()
        
    def setup_logging(self):
        """Set up logging and directories."""
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "samples"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "logs"), exist_ok=True)
        
        # Add file handler to logger
        file_handler = logging.FileHandler(os.path.join(self.output_dir, "logs", "training.log"))
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)
        
    def train(self, dataloader, num_epochs, save_interval=10, sample_interval=5):
        """
        End-to-end training of the transformer and diffusion model.
        """
        # Track metrics
        total_losses = []
        diffusion_losses = []
        transformer_losses = []
        
        # Training loop
        for epoch in range(num_epochs):
            epoch_total_loss = 0
            epoch_diffusion_loss = 0
            epoch_transformer_loss = 0
            num_batches = 0
            
            # Progress bar
            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
            
            # Unfreeze UNet if needed
            if self.freeze_unet and epoch >= self.unfreeze_epoch:
                logger.info(f"Unfreezing UNet parameters at epoch {epoch+1}")
                for param in self.unet.parameters():
                    param.requires_grad = True
                self.freeze_unet = False
            
            for batch_idx, images in enumerate(progress_bar):
                batch_loss, diff_loss, trans_loss = self.training_step(
                    images=images.to(self.device),
                    epoch=epoch
                )
                
                # Update metrics
                epoch_total_loss += batch_loss
                epoch_diffusion_loss += diff_loss
                epoch_transformer_loss += trans_loss
                num_batches += 1
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': batch_loss,
                    'diff_loss': diff_loss,
                    'trans_loss': trans_loss
                })
            
            # Average losses for the epoch
            avg_total_loss = epoch_total_loss / num_batches
            avg_diffusion_loss = epoch_diffusion_loss / num_batches
            avg_transformer_loss = epoch_transformer_loss / num_batches
            
            # Record metrics
            total_losses.append(avg_total_loss)
            diffusion_losses.append(avg_diffusion_loss)
            transformer_losses.append(avg_transformer_loss)
            
            # Log progress
            logger.info(f"Epoch {epoch+1}/{num_epochs} - "
                       f"Loss: {avg_total_loss:.6f}, "
                       f"Diffusion Loss: {avg_diffusion_loss:.6f}, "
                       f"Transformer Loss: {avg_transformer_loss:.6f}")
            
            # Generate samples if requested
            if (epoch + 1) % sample_interval == 0 or epoch == num_epochs - 1:
                self.generate_samples(epoch + 1)
                
            # Save checkpoint if requested
            if (epoch + 1) % save_interval == 0 or epoch == num_epochs - 1:
                self.save_checkpoint(epoch + 1, {
                    'total_loss': avg_total_loss,
                    'diffusion_loss': avg_diffusion_loss,
                    'transformer_loss': avg_transformer_loss
                })
                
            # Plot losses
            self.plot_losses(total_losses, diffusion_losses, transformer_losses, epoch)
        
        # Save final model
        logger.info("Training complete, saving final model")
        self.save_final_model()
        
        return total_losses, diffusion_losses, transformer_losses
    
    def training_step(self, images, epoch):
        """
        Perform a single training step (batch).
        """
        batch_size = images.shape[0]
        total_loss = 0
        
        # Reset gradients
        self.optimizer_transformer.zero_grad()
        self.optimizer_unet.zero_grad()
        
        # Generate random noise
        noise = torch.randn(images.shape).to(self.device)
        
        # Transform noise using the transformer
        transformed_noise = self.transformer(noise)
        
        # Calculate the transformer loss (symmetry loss)
        transformer_loss = self.transformer.calculate_symmetry_loss(transformed_noise)
        
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.scheduler.num_train_timesteps, 
            (batch_size,), device=self.device
        ).long()
        
        # Add noise to the images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_images = self.scheduler.add_noise(images, transformed_noise, timesteps)
        
        # Get the model prediction for the noise
        noise_pred = self.unet(noisy_images, timesteps).sample
        
        # Calculate the diffusion loss
        diffusion_loss = F.mse_loss(noise_pred, transformed_noise)
        
        # Combine losses
        loss = diffusion_loss + self.symmetry_weight * transformer_loss
        
        # Backpropagation
        loss.backward()
        
        # Apply gradient clipping to stabilize training
        torch.nn.utils.clip_grad_norm_(self.transformer.parameters(), 1.0)
        if not self.freeze_unet:
            torch.nn.utils.clip_grad_norm_(self.unet.parameters(), 1.0)
        
        # Update weights
        self.optimizer_transformer.step()
        if not self.freeze_unet:
            self.optimizer_unet.step()
        
        return loss.item(), diffusion_loss.item(), transformer_loss.item()
    
    def generate_samples(self, epoch, num_samples=4, image_size=256):
        """
        Generate samples using the current state of the model.
        """
        logger.info(f"Generating samples for epoch {epoch}")
        
        self.transformer.eval()
        self.unet.eval()
        
        # Generate random noise
        noise = torch.randn(num_samples, 3, image_size, image_size, device=self.device)
        
        # Transform noise using the transformer
        with torch.no_grad():
            transformed_noise = self.transformer(noise)
        
        # Initialize pipeline
        pipeline = DDPMPipeline(unet=self.unet, scheduler=self.scheduler)
        
        # Generate samples
        with torch.no_grad():
            # Use transformed noise as starting point
            sample = transformed_noise
            
            # Set up progress bar
            scheduler_timesteps = self.scheduler.timesteps
            progress_bar = tqdm(scheduler_timesteps)
            
            # Denoising loop
            for t in progress_bar:
                t_batch = torch.tensor([t] * num_samples, device=self.device)
                
                # Get model prediction
                model_output = self.unet(sample, t_batch).sample
                
                # Update sample with scheduler
                sample = self.scheduler.step(model_output, t, sample).prev_sample
                
        # Convert samples to images and save
        for i in range(num_samples):
            sample_image = (sample[i].permute(1, 2, 0).cpu().numpy() + 1) / 2
            sample_image = (sample_image * 255).astype(np.uint8)
            
            Image.fromarray(sample_image).save(
                os.path.join(self.output_dir, "samples", f"sample_epoch_{epoch}_{i+1}.png")
            )
        
        # Also save the transformed noise
        for i in range(num_samples):
            noise_image = transformed_noise[i].permute(1, 2, 0).cpu().numpy()
            noise_image = (noise_image - noise_image.min()) / (noise_image.max() - noise_image.min())
            noise_image = (noise_image * 255).astype(np.uint8)
            
            Image.fromarray(noise_image).save(
                os.path.join(self.output_dir, "samples", f"noise_epoch_{epoch}_{i+1}.png")
            )
        
        # Create a grid of all samples
        from torchvision.utils import make_grid
        grid = make_grid(sample.cpu(), nrow=2).permute(1, 2, 0).numpy()
        grid = (grid + 1) / 2 * 255
        grid = grid.astype(np.uint8)
        
        Image.fromarray(grid).save(
            os.path.join(self.output_dir, "samples", f"grid_epoch_{epoch}.png")
        )
        
        self.transformer.train()
        self.unet.train()
        
        logger.info(f"Samples saved to {os.path.join(self.output_dir, 'samples')}")
        
    def plot_losses(self, total_losses, diffusion_losses, transformer_losses, epoch):
        """
        Plot training losses.
        """
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 3, 1)
        plt.plot(total_losses)
        plt.title('Total Loss')
        plt.xlabel('Epoch')
        plt.grid(True)
        
        plt.subplot(1, 3, 2)
        plt.plot(diffusion_losses)
        plt.title('Diffusion Loss')
        plt.xlabel('Epoch')
        plt.grid(True)
        
        plt.subplot(1, 3, 3)
        plt.plot(transformer_losses)
        plt.title('Transformer Loss')
        plt.xlabel('Epoch')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "logs", f"losses_epoch_{epoch+1}.png"))
        plt.close()
    
    def save_checkpoint(self, epoch, metrics):
        """
        Save a checkpoint of the model.
        """
        checkpoint_dir = os.path.join(self.output_dir, "checkpoints", f"checkpoint_epoch_{epoch}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save transformer
        self.transformer.save_pretrained(os.path.join(checkpoint_dir, "transformer"))
        
        # Save UNet and scheduler
        self.unet.save_pretrained(os.path.join(checkpoint_dir, "unet"))
        self.scheduler.save_pretrained(os.path.join(checkpoint_dir, "scheduler"))
        
        # Save metrics
        with open(os.path.join(checkpoint_dir, "metrics.txt"), "w") as f:
            for key, value in metrics.items():
                f.write(f"{key}: {value}\n")
        
        logger.info(f"Checkpoint saved to {checkpoint_dir}")
    
    def save_final_model(self):
        """
        Save the final trained model.
        """
        final_dir = os.path.join(self.output_dir, "final")
        os.makedirs(final_dir, exist_ok=True)
        
        # Save transformer
        self.transformer.save_pretrained(os.path.join(final_dir, "transformer"))
        
        # Save UNet and scheduler in the format expected by the diffusers library
        pipeline = DDPMPipeline(unet=self.unet, scheduler=self.scheduler)
        pipeline.save_pretrained(final_dir)
        
        # Save a configuration file with system details
        config = {
            "symmetry_type": self.symmetry_type,
            "symmetry_weight": self.symmetry_weight,
            "freeze_unet": self.freeze_unet,
            "unfreeze_epoch": self.unfreeze_epoch
        }
        
        import json
        with open(os.path.join(final_dir, "end_to_end_config.json"), "w") as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Final model saved to {final_dir}")


def main():
    # Parse command-line arguments
    args = parse_args()
    
    # Set device
    device = torch.device(args.device)
    logger.info(f"Using device: {device}")
    
    # Prepare dataset
    dataloader = prepare_dataset(args.dataset_path, args.image_size, args.batch_size)
    
    # Initialize models
    transformer = None
    unet = None
    scheduler = None
    
    # Load pretrained transformer if specified
    if args.transformer_path:
        logger.info(f"Loading pretrained transformer from {args.transformer_path}")
        transformer = LearnableSymmetryTransformer.from_pretrained(args.transformer_path).to(device)
    
    # Load pretrained diffusion model if specified
    if args.base_model:
        logger.info(f"Loading pretrained diffusion model from {args.base_model}")
        try:
            # First try loading as pipeline
            pipeline = DDPMPipeline.from_pretrained(args.base_model)
            unet = pipeline.unet.to(device)
            scheduler = pipeline.scheduler
            logger.info("Successfully loaded diffusion model")
        except Exception as e:
            logger.warning(f"Error loading diffusion model: {e}")
            logger.info("Initializing new diffusion model")
    
    # Create system
    system = EndToEndLearnableSystem(
        transformer=transformer,
        unet=unet,
        scheduler=scheduler,
        symmetry_type=args.symmetry_type,
        with_r=args.with_r,
        learning_rate=args.learning_rate,
        transformer_lr=args.transformer_lr,
        device=device,
        output_dir=args.output_dir,
        freeze_unet=args.freeze_unet,
        unfreeze_epoch=args.unfreeze_epoch,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        symmetry_weight=args.symmetry_weight
    )
    
    # Train system
    system.train(
        dataloader=dataloader,
        num_epochs=args.num_epochs,
        save_interval=args.save_interval,
        sample_interval=args.sample_interval
    )
    
    logger.info("End-to-end training complete!")


if __name__ == "__main__":
    main() 