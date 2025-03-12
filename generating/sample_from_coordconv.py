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

from models.coord_conv.unet import CoordConvUNet2DModel
from diffusers import DDPMScheduler
from diffusers.utils.torch_utils import randn_tensor

def parse_args():
    parser = argparse.ArgumentParser(description="Generate samples from fine-tuned StarCraft CoordConv diffusion model")
    parser.add_argument(
        "--model_path",
        type=str,
        default="checkpoints/coord_fine_tuned_models/starcraft_maps_coordconv_30_epochs/final",
        help="Path to the fine-tuned CoordConv model"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/coordconv_samples",
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
        "--with_r",
        action="store_true",
        help="Whether the model uses the radius coordinate channel"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for generation (cuda or cpu)"
    )
    return parser.parse_args()

def generate_sample(model, scheduler, num_steps, seed, device, with_r=False):
    """Generate a sample using CoordConv UNet and scheduler components."""
    # Convert device string to torch.device
    device = torch.device(device)
    
    # Set the seed for reproducibility
    generator = torch.Generator(device=device).manual_seed(seed)
    
    # Get image dimensions from the model config - fixed this
    sample_size = model.unet.config.sample_size
    in_channels = model.unet.config.in_channels
    
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
        model.eval()
        
        # Prepare inputs for the model (add time dimension)
        with torch.no_grad():
            noise_pred = model(sample, t).sample
        
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
    
    print("=== StarCraft Maps CoordConv Generation ===")
    print(f"Model path: {args.model_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Number of samples: {args.num_samples}")
    print(f"Number of steps: {args.num_steps}")
    print(f"Seed: {args.seed}")
    print(f"With radius channel: {args.with_r}")
    print(f"Device: {args.device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Convert device string to torch.device
    device = torch.device(args.device)
    
    # Determine the actual model path
    model_path = args.model_path
    
    # Check if a final or checkpoint directory was directly specified
    if not (os.path.basename(model_path) == "final" or os.path.basename(model_path).startswith("checkpoint-")):
        # Check if there's a final directory
        final_dir = os.path.join(model_path, "final")
        if os.path.exists(final_dir) and os.path.isdir(final_dir) and os.path.exists(os.path.join(final_dir, "config.json")):
            model_path = final_dir
            print(f"Using final model at {model_path}")
        else:
            # Try to find the latest checkpoint directory
            checkpoint_dirs = [d for d in os.listdir(model_path) 
                             if d.startswith("checkpoint-") and os.path.isdir(os.path.join(model_path, d))]
            
            if checkpoint_dirs:
                # Sort by checkpoint number
                checkpoint_dirs.sort(key=lambda x: int(x.split("-")[1]), reverse=True)
                latest_checkpoint = os.path.join(model_path, checkpoint_dirs[0])
                
                if os.path.exists(os.path.join(latest_checkpoint, "config.json")):
                    model_path = latest_checkpoint
                    print(f"Using latest checkpoint at {model_path}")
                else:
                    print(f"Warning: Latest checkpoint at {latest_checkpoint} doesn't have config.json")
            else:
                print(f"Warning: No final model or checkpoints found in {model_path}")
    
    # Load CoordConv UNet model
    print(f"Loading CoordConv UNet from {model_path}...")
    try:
        model = CoordConvUNet2DModel.from_pretrained(model_path, with_r=args.with_r)
        model.to(device)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
    
    # Load scheduler
    print(f"Loading scheduler...")
    try:
        scheduler = DDPMScheduler.from_pretrained(model_path)
    except Exception as e:
        print(f"Error loading scheduler from model path, falling back to default scheduler: {e}")
        scheduler = DDPMScheduler.from_pretrained("google/ddpm-celebahq-256")
    
    # Generate samples
    for i in range(args.num_samples):
        current_seed = args.seed + i
        print(f"\nGenerating sample {i+1}/{args.num_samples} with seed {current_seed}...")
        
        sample, intermediates = generate_sample(
            model=model,
            scheduler=scheduler,
            num_steps=args.num_steps,
            seed=current_seed,
            device=device,
            with_r=args.with_r
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