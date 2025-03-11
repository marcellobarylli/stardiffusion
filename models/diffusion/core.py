import torch
from diffusers import UNet2DModel, DDPMScheduler, DDIMScheduler, DDPMPipeline
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
        visible_gpus: str = "0",
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
        
        # Parse the visible GPUs and create device list
        if torch.cuda.is_available():
            self.gpu_ids = [int(x) for x in self.visible_gpus.split(",")]
            print(f"Requested GPUs: {self.gpu_ids}")
            # Reset device indices after setting CUDA_VISIBLE_DEVICES
            self.device_ids = list(range(len(self.gpu_ids)))
            self.use_data_parallel = len(self.device_ids) > 1
        else:
            self.gpu_ids = []
            self.device_ids = []
            self.use_data_parallel = False
        
        # Print device info
        print(f"Using device: {self.device}")
        print(f"Available GPU count: {torch.cuda.device_count()}")
        if self.use_data_parallel:
            print(f"Will use DataParallel with device IDs: {self.device_ids}")


class DiffusionModelLoader:
    """Handles loading diffusion models and schedulers."""
    
    def __init__(self, config: DiffusionConfig):
        self.config = config
        self.model = None
        self.scheduler = None
        self.image_size = None
        self.in_channels = None
        self.pipeline = None
        
    def load_model(self) -> None:
        """Load the model and scheduler."""
        try:
            # First try to load the full pipeline (recommended approach)
            self.pipeline = DDPMPipeline.from_pretrained(self.config.model_id)
            self.pipeline.to(self.config.device)
            
            # Set properties from pipeline
            self.model = self.pipeline.unet
            self.scheduler = self.pipeline.scheduler
            self.image_size = self.model.config.sample_size
            self.in_channels = self.model.config.in_channels
        except Exception as e:
            print(f"Unable to load as pipeline: {e}. Falling back to UNet model only.")
            # Fallback to loading just the UNet model
            self.model = UNet2DModel.from_pretrained(self.config.model_id)
            
            # Use DataParallel if multiple GPUs are available
            if self.config.use_data_parallel:
                self.model = torch.nn.DataParallel(self.model, device_ids=self.config.device_ids)
                
            self.model.to(self.config.device)
            
            # Extract image size from model config
            self.image_size = self.model.config.sample_size
            self.in_channels = self.model.config.in_channels
            
            # Load noise scheduler
            self.scheduler = DDPMScheduler.from_pretrained(self.config.model_id)
            
            # No pipeline available
            self.pipeline = None
    
    def get_model_info(self) -> Dict[str, Any]:
        """Return information about the loaded model."""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
            
        return {
            "model_id": self.config.model_id,
            "image_size": self.image_size,
            "in_channels": self.in_channels,
            "model_type": type(self.model).__name__,
            "scheduler_type": type(self.scheduler).__name__
        }


class DiffusionProcessor:
    """Processes diffusion model inference."""
    
    def __init__(
        self, 
        model_loader: DiffusionModelLoader,
        config: DiffusionConfig
    ):
        self.model_loader = model_loader
        self.config = config
    
    def generate_sample(self, save_intermediates: bool = True) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Generate a sample using the diffusion model.
        
        Args:
            save_intermediates: Whether to save intermediate steps for visualization
            
        Returns:
            Tuple of (final sample, list of intermediate samples)
        """
        if self.model_loader.model is None:
            raise ValueError("Model not loaded. Call model_loader.load_model() first.")
        
        # Store intermediate samples
        intermediate_samples = []
        
        # Set generator for reproducibility
        generator = torch.Generator(device=self.config.device).manual_seed(self.config.seed)
        
        # Check if pipeline is available (preferred approach)
        if self.model_loader.pipeline is not None:
            print("Using pipeline for generation...")
            # Generate directly using the pipeline
            output = self.model_loader.pipeline(
                batch_size=self.config.batch_size,
                num_inference_steps=self.config.num_inference_steps,
                generator=generator,
                output_type="pt"  # Return PyTorch tensors
            )
            
            # Get the images from the pipeline output
            if hasattr(output, "images"):
                # Most modern pipelines return an object with images attribute
                sample = output.images
            else:
                # Older pipelines might return a tuple where first element is images
                sample = output[0]
            
            # Convert to proper tensor format if needed
            if not isinstance(sample, torch.Tensor):
                sample = torch.tensor(sample, device=self.config.device)
            
            # Ensure we have a batch dimension
            if sample.dim() == 3:
                sample = sample.unsqueeze(0)
            
            # Make sure the format is [B, C, H, W] (PyTorch's image format)
            if sample.shape[-1] == 3 and sample.shape[1] != 3:  # If format is [B, H, W, C]
                sample = sample.permute(0, 3, 1, 2)
            
            # For intermediate visualization, we can just use the final image
            # since we don't have access to intermediates from the pipeline
            if save_intermediates:
                intermediate_samples = [sample.cpu().clone()]
            
            return sample, intermediate_samples
            
        # Fallback to manual approach if pipeline is not available
        print("Pipeline not available, using manual diffusion process...")
        # Get model and scheduler
        model = self.model_loader.model
        scheduler = self.model_loader.scheduler
        image_size = self.model_loader.image_size
        
        # Start from random noise
        sample = randn_tensor(
            (self.config.batch_size, self.model_loader.in_channels, image_size, image_size),
            generator=generator,
            device=self.config.device
        )
        
        # Add initial noise sample to intermediates
        if save_intermediates:
            intermediate_samples.append(sample.cpu().clone())
        
        # Diffusion process (denoising)
        for t in range(self.config.num_inference_steps):
            # Set model to evaluation mode
            model.eval()
            
            # Get the timestep (schedule)
            timestep = scheduler.timesteps[t]
            
            # Prepare inputs for the model
            model_input = sample
            
            # Get model prediction
            with torch.no_grad():
                if isinstance(model, torch.nn.DataParallel):
                    noise_pred = model.module(model_input, timestep).sample
                else:
                    noise_pred = model(model_input, timestep).sample
            
            # Update sample with scheduler
            sample = scheduler.step(noise_pred, timestep, sample).prev_sample
            
            # Save intermediate result if requested
            if save_intermediates and (t % max(1, self.config.num_inference_steps // 10) == 0 or t == self.config.num_inference_steps - 1):
                intermediate_samples.append(sample.cpu().clone())
        
        return sample, intermediate_samples


class DiffusionVisualizer:
    """Visualizes diffusion model outputs."""
    
    def __init__(self, model_loader: DiffusionModelLoader):
        self.model_loader = model_loader
    
    def visualize_progression(
        self,
        intermediates: List[torch.Tensor],
        save_path: str = "diffusion_progression.png"
    ) -> None:
        """
        Visualize the progression of the diffusion process.
        
        Args:
            intermediates: List of tensors from different timesteps
            save_path: Path to save the visualization
        """
        if not intermediates:
            raise ValueError("No intermediate samples provided for visualization")
        
        # Number of samples and steps to visualize
        num_images = len(intermediates)
        
        # Create the figure
        plt.figure(figsize=(2 * num_images, 4))
        
        # For each intermediate step
        for i, img_tensor in enumerate(intermediates):
            # Get the first image if there are multiple in the batch
            img = img_tensor[0]
            
            # Move channel dimension to the end for plotting
            img = img.permute(1, 2, 0)
            
            # Normalize image to [0, 1]
            min_val = img.min()
            max_val = img.max()
            img = (img - min_val) / (max_val - min_val)
            
            # If single channel, repeat for RGB
            if img.shape[2] == 1:
                img = img.repeat(1, 1, 3)
            
            # Handle images with too many channels
            elif img.shape[2] > 3:
                img = img[:, :, :3]  # Just use first 3 channels
            
            # Convert to numpy for matplotlib
            img = img.numpy()
            
            # Plot the image
            plt.subplot(1, num_images, i + 1)
            plt.imshow(img)
            plt.title(f"Step {i}")
            plt.axis('off')
        
        # Save the visualization
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        print(f"Saved diffusion progression visualization to {save_path}")
    
    def visualize_final_image(
        self,
        image: torch.Tensor,
        save_path: str = "generated_image.png"
    ) -> None:
        """
        Visualize the final generated image.
        
        Args:
            image: Tensor containing the generated image (B, C, H, W)
            save_path: Path to save the visualization
        """
        if image.dim() != 4:
            raise ValueError(f"Expected 4D tensor, got {image.dim()}D")
        
        # Get the first image if there are multiple in the batch
        img = image[0].cpu()
        
        # For pipeline outputs, the values are often already in [0, 1]
        # But let's check and normalize if needed
        if img.min() < -0.1 or img.max() > 1.1:  # Allow some small margin for floating point errors
            print("Normalizing image from range [{:.2f}, {:.2f}] to [0, 1]".format(img.min().item(), img.max().item()))
            img = (img - img.min()) / (img.max() - img.min())
        
        # Move channel dimension to the end for plotting if it's not already
        if img.shape[0] == 3 or img.shape[0] == 1:  # If channels-first format
            img = img.permute(1, 2, 0)
            
        # If single channel, repeat for RGB
        if img.shape[2] == 1:
            img = img.repeat(1, 1, 3)
        
        # Handle images with too many channels
        elif img.shape[2] > 3:
            print(f"Warning: Image has {img.shape[2]} channels, using only first 3")
            img = img[:, :, :3]  # Just use first 3 channels
        
        # Clamp values to valid range [0, 1] for display
        img = torch.clamp(img, 0, 1)
            
        # Convert to numpy for matplotlib
        img = img.numpy()
        
        # Print image shape and values info
        print(f"Image shape: {img.shape}, Range: [{img.min():.4f}, {img.max():.4f}]")
        
        # Plot the image
        plt.figure(figsize=(10, 10))
        plt.imshow(img)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        print(f"Saved generated image to {save_path}")


def run_simple_generation(output_dir: str = "outputs", model_id: str = "google/ddpm-celebahq-256"):
    """
    Run a simple image generation example.
    
    Args:
        output_dir: Directory to save outputs
        model_id: Hugging Face model ID to use
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up config
    config = DiffusionConfig(
        model_id=model_id,
        num_inference_steps=100  # Increased from 50 for better quality
    )
    
    # Create model loader
    loader = DiffusionModelLoader(config)
    loader.load_model()
    
    # Print model info
    model_info = loader.get_model_info()
    print(f"Loaded model: {model_info}")
    
    # Create processor and visualizer
    processor = DiffusionProcessor(loader, config)
    visualizer = DiffusionVisualizer(loader)
    
    # Generate image
    final_image, intermediates = processor.generate_sample(save_intermediates=True)
    
    # Visualize results
    visualizer.visualize_progression(
        intermediates, 
        save_path=os.path.join(output_dir, "diffusion_progression.png")
    )
    visualizer.visualize_final_image(
        final_image, 
        save_path=os.path.join(output_dir, "generated_image.png")
    )
    
    return final_image 