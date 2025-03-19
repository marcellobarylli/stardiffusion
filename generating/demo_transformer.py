import torch
import argparse
import os
import sys
from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from diffusers import DDPMPipeline, DDPMScheduler, UNet2DModel
from tqdm import tqdm
from torchvision import transforms

# Add the project root to the Python path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import our modules
from models.coord_conv.learning_symmetry import LearnableSymmetryTransformer, sample_with_model
from data.datasets import DatasetManager, CustomImageDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Demo of symmetric noise transformer for diffusion")
    parser.add_argument(
        "--transformer_path",
        type=str,
        default=None,
        help="Path to a pretrained transformer (optional)"
    )
    parser.add_argument(
        "--diffusion_model_path",
        type=str,
        required=True,
        help="Path to the diffusion model"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/transformer_demo",
        help="Directory to save demo outputs"
    )
    parser.add_argument(
        "--symmetry_type",
        type=str,
        default="vertical",
        choices=["vertical", "horizontal", "both"],
        help="Type of symmetry to enforce if no transformer provided"
    )
    parser.add_argument(
        "--with_r",
        action="store_true",
        help="Use radius channel in transformer if creating a new one"
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=20,
        help="Number of denoising steps to visualize"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--create_video",
        action="store_true",
        help="Create a video of the denoising process"
    )
    # Add new arguments for dataset transformation
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="data/StarCraft_Map_Dataset",
        help="Path to the dataset for transformation demo"
    )
    parser.add_argument(
        "--transform_dataset",
        action="store_true",
        help="Transform real images from the dataset instead of noise"
    )
    parser.add_argument(
        "--num_dataset_samples",
        type=int,
        default=8,
        help="Number of dataset samples to transform"
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=256,
        help="Size to resize images to"
    )
    return parser.parse_args()


def visualize_denoising(samples, output_path, create_video=False):
    """Visualize the denoising process from noise to image."""
    os.makedirs(output_path, exist_ok=True)
    
    num_steps = len(samples)
    num_samples = samples[0].shape[0]
    
    # Save each intermediate step for each sample
    for i in range(num_samples):
        step_images = []
        
        for step, step_samples in enumerate(samples):
            # Convert tensor to image
            img = step_samples[i].cpu().permute(1, 2, 0).numpy()
            img = (img + 1.0) / 2.0  # Convert from [-1, 1] to [0, 1]
            img = (img * 255).astype(np.uint8)
            
            # Save individual step
            img_path = os.path.join(output_path, f"sample_{i+1}_step_{step}.png")
            Image.fromarray(img).save(img_path)
            
            step_images.append(img)
        
        # Create a grid of all steps
        rows = 4
        cols = (num_steps + rows - 1) // rows  # Ceiling division
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
        
        # Flatten axes for easy iteration
        if rows == 1 and cols == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for j, (ax, img) in enumerate(zip(axes, step_images)):
            ax.imshow(img)
            ax.set_title(f"Step {j}")
            ax.axis('off')
            
        # Hide any unused subplots
        for j in range(len(step_images), len(axes)):
            axes[j].axis('off')
            
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, f"denoising_grid_sample_{i+1}.png"))
        plt.close()
        
        # Optionally create a video
        if create_video:
            try:
                import imageio
                
                frames = [np.array(Image.fromarray(img).resize((512, 512))) for img in step_images]
                
                # Create video with 4 fps (slow enough to see changes)
                video_path = os.path.join(output_path, f"denoising_video_sample_{i+1}.mp4")
                imageio.mimsave(video_path, frames, fps=4)
                print(f"Created denoising video: {video_path}")
            except ImportError:
                print("Could not create video: imageio package not installed")
                print("Install with: pip install imageio imageio-ffmpeg")


def transform_dataset_images(transformer, dataset_path, output_path, num_samples=8, image_size=256, device=None):
    """
    Apply the transformer to real images from a dataset.
    
    Args:
        transformer: The LearnableSymmetryTransformer model
        dataset_path: Path to the dataset directory
        output_path: Directory to save the transformed images
        num_samples: Number of samples to transform
        image_size: Size to resize images to
        device: Device to use for processing
    """
    os.makedirs(output_path, exist_ok=True)
    
    if device is None:
        device = next(transformer.parameters()).device
    
    # Load dataset
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # Normalize to [-1, 1]
    ])
    
    dataset = CustomImageDataset(dataset_path, transform, image_size)
    
    # Ensure we don't try to access more samples than we have
    num_samples = min(num_samples, len(dataset))
    
    # Generate indices to sample from the dataset
    indices = torch.randperm(len(dataset))[:num_samples]
    
    # Process each sample
    transformer.eval()
    all_images = []
    all_transformed = []
    symmetry_scores = []
    
    print(f"\n=== Transforming {num_samples} images from dataset ===")
    print(f"Dataset: {dataset_path}")
    print(f"Image size: {image_size}x{image_size}")
    
    for idx in indices:
        # Get image from dataset
        sample = dataset[idx.item()]
        image = sample["pixel_values"].unsqueeze(0).to(device)  # Add batch dimension
        
        # Transform the image
        with torch.no_grad():
            transformed_image = transformer(image)
            symmetry_score = transformer.calculate_symmetry_loss(transformed_image).item()
            symmetry_scores.append(symmetry_score)
        
        # Add to collections
        all_images.append(image)
        all_transformed.append(transformed_image)
    
    # Create visualization grid
    fig, axes = plt.subplots(num_samples, 2, figsize=(10, 5 * num_samples))
    
    for i in range(num_samples):
        # Original image
        original = all_images[i][0].cpu().permute(1, 2, 0).numpy()
        original = (original + 1.0) / 2.0  # Convert from [-1, 1] to [0, 1]
        
        if num_samples > 1:
            ax_orig = axes[i, 0]
        else:
            ax_orig = axes[0]
            
        ax_orig.imshow(original)
        ax_orig.set_title(f"Original Image {i+1}")
        ax_orig.axis('off')
        
        # Transformed image
        transformed = all_transformed[i][0].cpu().permute(1, 2, 0).numpy()
        transformed = (transformed + 1.0) / 2.0  # Convert from [-1, 1] to [0, 1]
        
        if num_samples > 1:
            ax_trans = axes[i, 1]
        else:
            ax_trans = axes[1]
            
        title = f"Transformed Image {i+1} ({transformer.symmetry_type})"
        title += f"\nSymmetry Score: {symmetry_scores[i]:.4f}"
            
        ax_trans.imshow(transformed)
        ax_trans.set_title(title)
        ax_trans.axis('off')
        
        # Also save individual images
        orig_img = (original * 255).astype(np.uint8)
        trans_img = (transformed * 255).astype(np.uint8)
        
        Image.fromarray(orig_img).save(os.path.join(output_path, f"original_{i+1}.png"))
        Image.fromarray(trans_img).save(os.path.join(output_path, f"transformed_{i+1}.png"))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f"dataset_transformation_{transformer.symmetry_type}.png"))
    plt.close()
    
    print(f"Transformed dataset images saved to {output_path}")
    print(f"Symmetry scores (lower is better): {symmetry_scores}")
    print(f"Average symmetry score: {sum(symmetry_scores)/len(symmetry_scores):.6f}")
    
    return all_images, all_transformed, symmetry_scores


def main():
    # Parse arguments
    args = parse_args()
    
    # Set random seed if provided
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Saving outputs to {args.output_dir}")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load or create transformer
    if args.transformer_path:
        print(f"Loading transformer from {args.transformer_path}")
        transformer = LearnableSymmetryTransformer.from_pretrained(args.transformer_path).to(device)
    else:
        print(f"Creating new transformer with symmetry_type={args.symmetry_type}, with_r={args.with_r}")
        transformer = LearnableSymmetryTransformer(
            channels=3,  # Standard RGB
            hidden_dim=64,
            with_r=args.with_r,
            normalize=True,
            symmetry_type=args.symmetry_type
        ).to(device)
    
    # Load diffusion model
    print(f"Loading diffusion model from {args.diffusion_model_path}")
    try:
        # Try to load as full pipeline
        diffusion_pipeline = DDPMPipeline.from_pretrained(args.diffusion_model_path).to(device)
    except Exception as e:
        print(f"Error loading pipeline: {e}")
        print("Trying to load components separately...")
        
        # Load UNet and scheduler separately
        unet = UNet2DModel.from_pretrained(args.diffusion_model_path, subfolder="unet").to(device)
        scheduler = DDPMScheduler.from_pretrained(args.diffusion_model_path, subfolder="scheduler")
        
        # Create a minimal pipeline
        diffusion_pipeline = DDPMPipeline(unet=unet, scheduler=scheduler)
        diffusion_pipeline.to(device)
    
    # If transforming dataset images is requested
    if args.transform_dataset:
        print("\n=== Demonstrating Transformer on Dataset Images ===")
        transform_dataset_images(
            transformer,
            args.dataset_path,
            os.path.join(args.output_dir, "dataset_transformation"),
            num_samples=args.num_dataset_samples,
            image_size=args.image_size,
            device=device
        )
    
    # First, let the transformer demonstrate its capabilities
    print("\n=== Demonstrating Noise Transformer ===")
    transformer.demo(
        output_path=os.path.join(args.output_dir, "noise_demo"),
        num_samples=4,
        height=args.image_size,
        width=args.image_size,
        device=device
    )
    
    # Generate with intermediate steps for visualization
    print("\n=== Generating with Intermediate Steps ===")
    _, all_steps = sample_with_model(
        transformer,
        diffusion_pipeline,
        num_samples=2,
        height=args.image_size,
        width=args.image_size,
        device=device,
        num_inference_steps=args.num_inference_steps,
        return_all_steps=True
    )
    
    # Visualize the denoising process
    print("\n=== Visualizing Denoising Process ===")
    visualize_denoising(
        all_steps,
        output_path=os.path.join(args.output_dir, "denoising_steps"),
        create_video=args.create_video
    )
    
    print("\nDemo complete! Generated samples and visualizations")
    print(f"All outputs saved to {args.output_dir}")


if __name__ == "__main__":
    main() 