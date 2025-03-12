import os
import sys
import torch
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

# Add the project root to the Python path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from diffusers import UNet2DModel, DDPMScheduler
from diffusers.utils.torch_utils import randn_tensor

def parse_args():
    parser = argparse.ArgumentParser(description="Generate samples from fine-tuned StarCraft diffusion model using components")
    parser.add_argument(
        "--model_path",
        type=str,
        default="checkpoints/starcraft_fine_tuned_fixed/final",
        help="Path to the fine-tuned model"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/starcraft_samples",
        help="Directory to save generated samples"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=3,
        help="Number of samples to generate"
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=100,
        help="Number of diffusion steps"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for generation"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for generation (cuda or cpu)"
    )
    return parser.parse_args()

def generate_sample(unet, scheduler, num_steps, seed, device):
    """Generate a sample using individual UNet and scheduler components."""
    # Convert device string to torch.device
    device = torch.device(device)
    
    # Set the seed for reproducibility
    generator = torch.Generator(device=device).manual_seed(seed)
    
    # Get image dimensions from the UNet config
    sample_size = unet.config.sample_size
    in_channels = unet.config.in_channels
    
    # Start from random noise
    sample = randn_tensor(
        (1, in_channels, sample_size, sample_size),
        generator=generator,
        device=device
    )
    
    # Store intermediate steps
    intermediates = [sample.cpu().clone()]
    
    # Set the scheduler timesteps
    scheduler.set_timesteps(num_steps)
    
    # Diffusion process (denoising)
    for t in tqdm(scheduler.timesteps):
        # Set model to evaluation mode
        unet.eval()
        
        # Prepare inputs for the model (add time dimension)
        with torch.no_grad():
            noise_pred = unet(sample, t).sample
        
        # Update sample with scheduler
        sample = scheduler.step(noise_pred, t, sample).prev_sample
        
        # Save intermediate step (every 10 steps to save memory)
        if t % max(1, len(scheduler.timesteps) // 10) == 0:
            intermediates.append(sample.cpu().clone())
    
    # Add final result if not already added
    if len(intermediates) > 0 and not torch.allclose(intermediates[-1], sample.cpu()):
        intermediates.append(sample.cpu().clone())
    
    return sample, intermediates

def normalize_image(image):
    """Normalize image tensor to [0, 1] range for visualization."""
    # Move to CPU if on GPU
    if image.device.type != 'cpu':
        image = image.cpu()
        
    image = image.squeeze().permute(1, 2, 0)  # [C, H, W] -> [H, W, C]
    image = (image - image.min()) / (image.max() - image.min())
    
    # Handle single channel or convert as needed
    if image.shape[2] == 1:
        image = image.repeat(1, 1, 3)
    elif image.shape[2] > 3:
        image = image[:, :, :3]
    
    # Clamp to valid range
    return torch.clamp(image, 0, 1).numpy()

def save_image(image_tensor, save_path):
    """Save normalized image tensor as PNG."""
    image_array = normalize_image(image_tensor)
    
    # Convert to uint8 for saving
    image_array = (image_array * 255).astype(np.uint8)
    
    # Save using PIL
    Image.fromarray(image_array).save(save_path)
    print(f"Saved image to {save_path}")

def visualize_progression(intermediates, save_path):
    """Create visualization of diffusion process progression."""
    num_images = len(intermediates)
    cols = min(num_images, 10)
    rows = (num_images + cols - 1) // cols
    
    # Create figure
    plt.figure(figsize=(cols * 2, rows * 2))
    
    # Plot each intermediate
    for i, img in enumerate(intermediates):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(normalize_image(img))
        step_num = i * max(1, len(intermediates) // 10) if i < len(intermediates) - 1 else "final"
        plt.title(f"Step {step_num}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved progression visualization to {save_path}")

def main():
    # Parse arguments
    args = parse_args()
    
    print("=== StarCraft Maps Generation ===")
    print(f"Model path: {args.model_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Number of samples: {args.num_samples}")
    print(f"Number of steps: {args.num_steps}")
    print(f"Seed: {args.seed}")
    print(f"Device: {args.device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Convert device string to torch.device
    device = torch.device(args.device)
    
    # Load UNet model
    print(f"Loading UNet from {args.model_path}...")
    unet = UNet2DModel.from_pretrained(args.model_path, subfolder="")
    unet.to(device)
    
    # Load scheduler
    print(f"Loading scheduler from {args.model_path}...")
    scheduler = DDPMScheduler.from_pretrained(args.model_path, subfolder="")
    
    # Generate samples
    for i in range(args.num_samples):
        current_seed = args.seed + i
        print(f"\nGenerating sample {i+1}/{args.num_samples} with seed {current_seed}...")
        
        sample, intermediates = generate_sample(
            unet=unet,
            scheduler=scheduler,
            num_steps=args.num_steps,
            seed=current_seed,
            device=device
        )
        
        # Save final image
        save_path = os.path.join(args.output_dir, f"sample_{i+1}.png")
        save_image(sample, save_path)
        
        # Save progression visualization (only for first few samples)
        if i < min(3, args.num_samples):
            progression_path = os.path.join(args.output_dir, f"progression_{i+1}.png")
            visualize_progression(intermediates, progression_path)
    
    print(f"\nSuccessfully generated {args.num_samples} samples in {args.output_dir}")

if __name__ == "__main__":
    main() 