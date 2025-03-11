#!/usr/bin/env python3
"""Generate StarCraft-style maps using a fine-tuned diffusion model."""
import os
import argparse
from models.diffusion.core import DiffusionConfig, DiffusionModelLoader, DiffusionProcessor, DiffusionVisualizer
from PIL import Image

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate StarCraft-style maps")
    
    parser.add_argument(
        "--model_path", 
        type=str, 
        default="checkpoints/fine_tuned_models/starcraft_maps/final",
        help="Path to the fine-tuned model"
    )
    parser.add_argument(
        "--num_images", 
        type=int, 
        default=32,
        help="Number of images to generate"
    )
    parser.add_argument(
        "--steps", 
        type=int, 
        default=100,
        help="Number of denoising steps"
    )
    parser.add_argument(
        "--gpu", 
        type=str, 
        default="0",
        help="GPU ID to use"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="outputs/generated_starcraft_maps",
        help="Directory to save generated images"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--skip_progression", 
        action="store_true",
        help="Skip saving the diffusion progression visualization"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=4,
        help="Batch size for generation"
    )
    
    return parser.parse_args()

def main():
    """Generate StarCraft-style maps."""
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Generating {args.num_images} StarCraft-style maps...")
    print(f"Using model: {args.model_path}")
    print(f"Output directory: {args.output_dir}")
    
    # Create configuration
    config = DiffusionConfig(
        model_id=args.model_path,
        visible_gpus=args.gpu,
        num_inference_steps=args.steps,
        batch_size=args.batch_size,
        seed=args.seed
    )
    
    # Create model loader
    model_loader = DiffusionModelLoader(config)
    model_loader.load_model()
    
    # Create processor and visualizer
    processor = DiffusionProcessor(model_loader, config)
    visualizer = DiffusionVisualizer(model_loader)
    
    # Number of batches
    num_batches = (args.num_images + args.batch_size - 1) // args.batch_size
    
    # Generate images in batches
    total_generated = 0
    for i in range(num_batches):
        # Calculate how many images to generate in this batch
        batch_count = min(args.batch_size, args.num_images - total_generated)
        if batch_count <= 0:
            break
            
        # Adjust batch size for the last batch if needed
        if batch_count < args.batch_size:
            config.batch_size = batch_count
            
        print(f"Generating batch {i+1}/{num_batches} ({batch_count} images)...")
        
        # Generate the images
        images, intermediates = processor.generate_sample(
            save_intermediates=not args.skip_progression
        )
        
        # Save the images
        for j in range(batch_count):
            idx = total_generated + j + 1
            image_tensor = images[j:j+1]  # Get a single image with batch dimension
            
            # Save individual image
            visualizer.visualize_final_image(
                image_tensor,
                save_path=os.path.join(args.output_dir, f"starcraft_map_{idx:04d}.png")
            )
        
        # Save progression for the first batch
        if i == 0 and not args.skip_progression:
            visualizer.visualize_progression(
                intermediates,
                save_path=os.path.join(args.output_dir, "diffusion_progression.png")
            )
        
        # Update counter
        total_generated += batch_count
    
    print(f"Successfully generated {total_generated} images in {args.output_dir}")

if __name__ == "__main__":
    main() 