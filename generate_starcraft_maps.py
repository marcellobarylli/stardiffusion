#!/usr/bin/env python3
"""Generate StarCraft-style maps using a fine-tuned diffusion model."""
import os
import argparse
from DiffusionCore import DiffusionConfig, DiffusionModelLoader, DiffusionProcessor, DiffusionVisualizer
from PIL import Image

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate StarCraft-style maps")
    
    parser.add_argument(
        "--model_path", 
        type=str, 
        default="fine_tuned_models/starcraft_maps/final",
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
        default="2",
        help="GPU ID to use"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="generated_starcraft_maps",
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
        "--save_individual", 
        action="store_true",
        help="Save each generated image individually"
    )
    
    return parser.parse_args()

def main():
    """Generate StarCraft-style maps."""
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Generating {args.num_images} StarCraft-style maps...")
    print(f"Using model: {args.model_path}")
    print(f"Output directory: {args.output_dir}")
    
    # Initialize configuration
    config = DiffusionConfig(
        model_id=args.model_path,
        visible_gpus=args.gpu,
        num_inference_steps=args.steps,
        batch_size=args.num_images,
        seed=args.seed
    )
    
    # Load the model
    model_loader = DiffusionModelLoader(config)
    model_loader.load_model()
    
    # Create processor and generate images
    processor = DiffusionProcessor(model_loader, config)
    final_images, intermediates = processor.generate_sample()
    
    # Visualize and save results
    visualizer = DiffusionVisualizer(model_loader)
    
    # Save the final image grid
    final_image_path = os.path.join(args.output_dir, f"starcraft_maps_seed{args.seed}.png")
    visualizer.visualize_final_image(
        final_images,
        save_path=final_image_path
    )
    
    # Save individual images if requested
    if args.save_individual:
        individual_dir = os.path.join(args.output_dir, f"individual_seed{args.seed}")
        os.makedirs(individual_dir, exist_ok=True)
        print(f"Saving individual images to {individual_dir}")
        
        for i, img_tensor in enumerate(final_images):
            # Convert tensor to PIL Image and save
            # Assuming the tensor is in the format [C, H, W] and normalized
            img_tensor = (img_tensor + 1) / 2  # Denormalize from [-1, 1] to [0, 1]
            img_tensor = img_tensor.clamp(0, 1)
            img_tensor = img_tensor.cpu().permute(1, 2, 0).numpy() * 255
            img = Image.fromarray(img_tensor.astype('uint8'))
            img_path = os.path.join(individual_dir, f"map_{i+1}.png")
            img.save(img_path)
        
        print(f"Saved {len(final_images)} individual images")
    
    # Optionally save the diffusion progression for the first image
    if not args.skip_progression:
        progression_path = os.path.join(args.output_dir, f"progression_seed{args.seed}.png")
        visualizer.visualize_progression(
            intermediates,
            save_path=progression_path
        )
        print(f"Diffusion progression: {progression_path}")
    
    print(f"Generated images saved to {args.output_dir}")
    print(f"Final image grid: {final_image_path}")

if __name__ == "__main__":
    main() 