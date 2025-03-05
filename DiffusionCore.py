import torch
from diffusers import UNet2DModel, DDPMScheduler, DDIMScheduler
from diffusers.utils.torch_utils import randn_tensor
import matplotlib.pyplot as plt
import os
import numpy as np
from typing import List, Optional, Tuple, Dict, Any, Union

class DiffusionConfig:
    """Configuration class for diffusion model settings."""
    
    def __init__(
        self,
        model_id: str = "google/ddpm-celebahq-256",
        visible_gpus: str = "2,3",
        num_inference_steps: int = 30,
        batch_size: int = 1,
        seed: int = 52
    ):
        self.model_id = model_id
        self.visible_gpus = visible_gpus
        self.num_inference_steps = num_inference_steps
        self.batch_size = batch_size
        self.seed = seed
        
        # Set visible GPUs
        os.environ["CUDA_VISIBLE_DEVICES"] = self.visible_gpus
        
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_data_parallel = torch.cuda.device_count() > 1
        
        # Print device info
        print(f"Using device: {self.device}")
        print(f"Available GPUs: {torch.cuda.device_count()}")


class DiffusionModelLoader:
    """Handles loading and setup of diffusion models."""
    
    def __init__(self, config: DiffusionConfig):
        self.config = config
        self.unet = None
        self.scheduler = None
        self.unet_config = None
        
    def load_model(self) -> None:
        """Load the UNet model and scheduler."""
        # Load UNet model
        self.unet = UNet2DModel.from_pretrained(self.config.model_id).to(self.config.device)
        
        # Store config before wrapping with DataParallel
        self.unet_config = self.unet.config
        
        # Create scheduler
        self.scheduler = DDIMScheduler.from_pretrained(self.config.model_id)
        
        # Wrap model with DataParallel if multiple GPUs are available
        if self.config.use_data_parallel:
            print(f"Using {torch.cuda.device_count()} GPUs!")
            self.unet = torch.nn.DataParallel(self.unet)
        
        print(f"Model loaded: {self.config.model_id}")
        print(f"Model parameters: {sum(p.numel() for p in self.unet.parameters())}")
        
    def get_model_info(self) -> Dict[str, Any]:
        """Return model configuration and details."""
        if self.unet_config is None:
            raise ValueError("Model has not been loaded yet.")
        
        return {
            "model_id": self.config.model_id,
            "unet_config": self.unet_config,
            "parameter_count": sum(p.numel() for p in self.unet.parameters())
        }


class DiffusionProcessor:
    """Handles the diffusion process for generation and sampling."""
    
    def __init__(
        self, 
        model_loader: DiffusionModelLoader,
        config: DiffusionConfig
    ):
        self.model_loader = model_loader
        self.config = config
        
    def generate_sample(self, save_intermediates: bool = True) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Run the full denoising process and generate a sample.
        
        Args:
            save_intermediates: Whether to save intermediate steps for visualization
            
        Returns:
            Tuple of (final_image, intermediate_images)
        """
        # Set seed for reproducibility
        torch.manual_seed(self.config.seed)
        
        # Determine sample shape based on model config
        unet_config = self.model_loader.unet_config
        sample_shape = (
            self.config.batch_size, 
            unet_config.in_channels, 
            unet_config.sample_size, 
            unet_config.sample_size
        )
        
        # Start with random noise
        noise = randn_tensor(sample_shape, device=self.config.device)
        image = noise
        
        # Set up scheduler
        scheduler = self.model_loader.scheduler
        scheduler.set_timesteps(self.config.num_inference_steps)
        
        # Store intermediates for visualization
        intermediates = [image.cpu()] if save_intermediates else []
        
        # Run denoising process
        for t in scheduler.timesteps:
            print(f"Denoising step {t}/{scheduler.timesteps[0]}", end="\r")
            
            # Get model prediction
            with torch.no_grad():
                if self.config.use_data_parallel:
                    model_output = self.model_loader.unet.module(
                        image, t.unsqueeze(0).to(self.config.device)
                    ).sample
                else:
                    model_output = self.model_loader.unet(
                        image, t.unsqueeze(0).to(self.config.device)
                    ).sample
            
            # Scheduler step for computing the previous image with less noise
            image = scheduler.step(model_output, t, image).prev_sample
            
            # Save intermediate steps for visualization
            if save_intermediates and (t % 20 == 0 or t == scheduler.timesteps[-1]):
                intermediates.append(image.cpu())
        
        print("\nDenoising complete!")
        return image, intermediates


class DiffusionVisualizer:
    """Handles visualization of diffusion process and results."""
    
    def __init__(self, model_loader: DiffusionModelLoader):
        self.model_loader = model_loader
        
    def visualize_progression(
        self,
        intermediates: List[torch.Tensor],
        save_path: str = "diffusion_progression.png"
    ) -> None:
        """Visualize the progression of the diffusion process.
        
        Args:
            intermediates: List of intermediate tensors from the diffusion process
            save_path: Path to save the visualization
        """
        unet_config = self.model_loader.unet_config
        
        plt.figure(figsize=(15, 8))
        
        # Number of images to display
        num_images = min(8, len(intermediates))
        cols = 4
        rows = (num_images + cols - 1) // cols
        
        # Plot the original noise and intermediate steps
        for i in range(num_images):
            plt.subplot(rows, cols, i+1)
            
            # Show the first channel for grayscale, or convert to RGB for color
            if unet_config.in_channels == 1:
                img_to_show = intermediates[i][0, 0].numpy()
                plt.imshow(img_to_show, cmap='gray')
            else:
                img_to_show = intermediates[i][0].permute(1, 2, 0).numpy()  # CHW -> HWC
                # Normalize to [0, 1] range
                img_to_show = (img_to_show - img_to_show.min()) / (img_to_show.max() - img_to_show.min())
                plt.imshow(img_to_show)
            
            step_num = 0 if i == 0 else self.model_loader.scheduler.timesteps[0] - (i-1)*20
            plt.title(f"Step {step_num}")
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        print(f"Progression visualization saved as '{save_path}'")
    
    def visualize_final_image(
        self,
        image: torch.Tensor,
        save_path: str = "generated_image.png"
    ) -> None:
        """Visualize the final generated image.
        
        Args:
            image: The final generated image tensor
            save_path: Path to save the visualization
        """
        unet_config = self.model_loader.unet_config
        
        if unet_config.in_channels >= 3:  # For RGB models
            # Convert the final result to a proper image
            final_image = image.cpu()[0].permute(1, 2, 0)  # Change from CxHxW to HxWxC
            
            # Normalize to [0, 1] range
            final_image = (final_image - final_image.min()) / (final_image.max() - final_image.min())
            
            plt.figure(figsize=(10, 10))
            plt.imshow(final_image)
            plt.title("Generated Image")
            plt.axis('off')
            plt.savefig(save_path, dpi=300)
            print(f"Generated image saved as '{save_path}'")


def run_simple_generation():
    """Run a simple image generation example."""
    # Initialize configuration
    config = DiffusionConfig(
        model_id="google/ddpm-celebahq-256",
        visible_gpus="2,3",
        num_inference_steps=100
    )
    
    # Load the model
    model_loader = DiffusionModelLoader(config)
    model_loader.load_model()
    
    # Create processor and run diffusion
    processor = DiffusionProcessor(model_loader, config)
    final_image, intermediates = processor.generate_sample()
    
    # Visualize results
    visualizer = DiffusionVisualizer(model_loader)
    visualizer.visualize_progression(intermediates)
    visualizer.visualize_final_image(final_image)
    
    # Print model configuration
    print("\nModel Configuration:")
    for key, value in model_loader.unet_config.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    run_simple_generation() 