import torch
import argparse
import os
import sys
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from diffusers import DDPMPipeline, DDIMPipeline

# Add the project root to sys.path
project_root = Path(__file__).absolute().parent.parent
sys.path.append(str(project_root))

from models.coord_conv.learning_symmetry import LearnableSymmetryTransformer, sample_with_model


def parse_args():
    parser = argparse.ArgumentParser(description="Generate samples with learned symmetry")
    
    # Model paths
    parser.add_argument("--transformer_path", type=str, required=True,
                      help="Path to the trained symmetry transformer model")
    parser.add_argument("--diffusion_model_path", type=str, required=True,
                      help="Path to the pretrained diffusion model")
    
    # Generation parameters
    parser.add_argument("--output_dir", type=str, default="outputs/learned_symmetry_samples",
                      help="Directory to save generated samples")
    parser.add_argument("--num_samples", type=int, default=4,
                      help="Number of samples to generate")
    parser.add_argument("--batch_size", type=int, default=1,
                      help="Batch size for generation")
    parser.add_argument("--seed", type=int, default=None,
                      help="Random seed for reproducibility")
    parser.add_argument("--guidance_scale", type=float, default=7.5,
                      help="Guidance scale for guided diffusion (not used with DDPM/DDIM)")
    parser.add_argument("--num_inference_steps", type=int, default=50,
                      help="Number of denoising steps")
    parser.add_argument("--image_size", type=int, default=256,
                      help="Image size for generation")
    parser.add_argument("--save_progressive", action="store_true",
                      help="Save intermediate denoising steps")
    parser.add_argument("--show_transformed_noise", action="store_true",
                      help="Save the transformed noise before denoising")
    
    # Device
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                      help="Device to generate on (default: auto-detect)")
    
    return parser.parse_args()


def save_image_grid(images, path, nrow=4, padding=2):
    """Save a grid of images."""
    from torchvision.utils import make_grid
    grid = make_grid(images, nrow=nrow, padding=padding).permute(1, 2, 0).cpu().numpy()
    grid = (grid * 255).astype(np.uint8)
    Image.fromarray(grid).save(path)
    return grid


def main():
    # Parse command-line arguments
    args = parse_args()
    
    # Set device
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Set random seed for reproducibility
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load the learned symmetry transformer
    print(f"Loading symmetry transformer from {args.transformer_path}")
    transformer = LearnableSymmetryTransformer.from_pretrained(args.transformer_path).to(device)
    transformer.eval()
    
    # Print model config
    print(f"Transformer config:")
    print(f"  Channels: {transformer.channels}")
    print(f"  Hidden dimension: {transformer.hidden_dim}")
    print(f"  With radius: {transformer.with_r}")
    print(f"  Symmetry type: {transformer.symmetry_type}")
    
    # Load the diffusion model
    print(f"Loading diffusion model from {args.diffusion_model_path}")
    try:
        # Try loading with DDPM pipeline first
        pipeline = DDPMPipeline.from_pretrained(args.diffusion_model_path).to(device)
        print("Loaded DDPM pipeline")
    except:
        try:
            # Try DDIM if DDPM fails
            pipeline = DDIMPipeline.from_pretrained(args.diffusion_model_path).to(device)
            print("Loaded DDIM pipeline")
        except Exception as e:
            print(f"Error loading diffusion model: {e}")
            print("Using standard DDPM pipeline")
            from diffusers import UNet2DModel
            unet = UNet2DModel.from_pretrained(args.diffusion_model_path, subfolder="unet").to(device)
            from diffusers import DDPMScheduler
            scheduler = DDPMScheduler.from_pretrained(args.diffusion_model_path, subfolder="scheduler")
            pipeline = DDPMPipeline(unet=unet, scheduler=scheduler)
            pipeline.to(device)
    
    # Generate samples
    print(f"Generating {args.num_samples} samples...")
    
    all_images = []
    all_transformed_noise = []
    
    for i in range(0, args.num_samples, args.batch_size):
        batch_size = min(args.batch_size, args.num_samples - i)
        
        # Generate samples with transformed noise
        if args.save_progressive:
            samples, intermediates = sample_with_model(
                transformer=transformer,
                diffusion_pipeline=pipeline,
                num_samples=batch_size,
                height=args.image_size,
                width=args.image_size,
                device=device,
                num_inference_steps=args.num_inference_steps,
                return_all_steps=True
            )
            
            # Save progressive denoising steps for first sample in batch
            if i == 0:
                step_interval = max(1, len(intermediates) // 10)  # Save ~10 steps
                for step_idx in range(0, len(intermediates), step_interval):
                    step_sample = intermediates[step_idx][0]
                    step_image = (step_sample.permute(1, 2, 0).cpu().numpy() + 1) / 2
                    step_image = (step_image * 255).astype(np.uint8)
                    Image.fromarray(step_image).save(
                        os.path.join(args.output_dir, f"progressive_step_{step_idx}.png")
                    )
        else:
            samples = sample_with_model(
                transformer=transformer,
                diffusion_pipeline=pipeline,
                num_samples=batch_size,
                height=args.image_size,
                width=args.image_size,
                device=device,
                num_inference_steps=args.num_inference_steps,
                return_all_steps=False
            )
        
        # Save individual samples
        for j in range(batch_size):
            sample_idx = i + j
            sample = samples[j]
            
            # Convert to image
            sample_image = (sample.permute(1, 2, 0).cpu().numpy() + 1) / 2
            sample_image = (sample_image * 255).astype(np.uint8)
            
            # Save image
            Image.fromarray(sample_image).save(
                os.path.join(args.output_dir, f"sample_{sample_idx+1}.png")
            )
            
            all_images.append(sample.cpu())
    
    # Generate a grid of all samples
    print("Generating sample grid...")
    save_image_grid(
        torch.stack(all_images),
        os.path.join(args.output_dir, "all_samples_grid.png"),
        nrow=min(4, args.num_samples)
    )
    
    # If requested, also save the transformed noise
    if args.show_transformed_noise:
        print("Generating transformed noise...")
        
        # Generate random noise
        noise_batch = torch.randn(args.num_samples, transformer.channels, 
                                 args.image_size, args.image_size, device=device)
        
        # Transform noise
        with torch.no_grad():
            transformed_noise = transformer(noise_batch)
        
        # Calculate and print symmetry scores
        symmetry_scores = [
            transformer.calculate_symmetry_loss(transformed_noise[i:i+1]).item()
            for i in range(args.num_samples)
        ]
        print(f"Transformed noise symmetry scores: {symmetry_scores}")
        print(f"Average symmetry score: {sum(symmetry_scores) / len(symmetry_scores):.6f}")
        
        # Save individual transformed noise images
        for i in range(args.num_samples):
            # Convert to image
            tn_image = transformed_noise[i].permute(1, 2, 0).cpu().numpy()
            tn_image = (tn_image - tn_image.min()) / (tn_image.max() - tn_image.min())
            tn_image = (tn_image * 255).astype(np.uint8)
            
            # Save image
            Image.fromarray(tn_image).save(
                os.path.join(args.output_dir, f"transformed_noise_{i+1}.png")
            )
        
        # Create a demonstration comparing original and transformed noise
        transformer.demo(
            output_path=os.path.join(args.output_dir, "noise_comparison"),
            num_samples=min(4, args.num_samples),
            height=args.image_size,
            width=args.image_size,
            device=device,
            show_symmetry_scores=True
        )
    
    print(f"All samples saved to {args.output_dir}")


if __name__ == "__main__":
    main() 