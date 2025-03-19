import torch
import argparse
import os
import sys
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms

# Add the project root to sys.path
project_root = Path(__file__).absolute().parent.parent
sys.path.append(str(project_root))

from models.coord_conv.paired_symmetrizer import PairedSymmetrizer
from models.coord_conv.symmetrize import SymmetryPairedDataset, calculate_symmetry_loss


def parse_args():
    parser = argparse.ArgumentParser(description="Demo a trained paired symmetrizer model")
    
    # Input parameters
    parser.add_argument("--model_path", type=str, required=True,
                      help="Path to the trained model")
    parser.add_argument("--dataset_path", type=str, required=True,
                      help="Path to the dataset to symmetrize")
    parser.add_argument("--output_dir", type=str, default="outputs/paired_symmetrizer_demo",
                      help="Directory to save generated images")
    
    # Generation parameters
    parser.add_argument("--num_samples", type=int, default=8,
                      help="Number of samples to generate")
    parser.add_argument("--image_size", type=int, default=128,
                      help="Size of images for generation")
    parser.add_argument("--batch_size", type=int, default=8,
                      help="Batch size for processing")
    
    # Device
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                      help="Device to use (default: auto-detect)")
    
    return parser.parse_args()


def load_images(dataset_path, image_size, max_images=None):
    """
    Load images from a directory.
    
    Args:
        dataset_path: Path to the dataset directory
        image_size: Size to resize images to
        max_images: Maximum number of images to load (None for all)
    
    Returns:
        List of image tensors
    """
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # Normalize to [-1, 1]
    ])
    
    # Find image files
    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(list(Path(dataset_path).glob(f'**/*{ext}')))
    
    print(f"Found {len(image_files)} images in {dataset_path}")
    
    # Limit number of images if specified
    if max_images is not None:
        np.random.shuffle(image_files)
        image_files = image_files[:max_images]
    
    # Load and transform images
    images = []
    for img_path in image_files:
        img = Image.open(img_path).convert('RGB')
        img_tensor = transform(img)
        images.append((img_tensor, str(img_path)))
    
    return images


def process_images(model, images, device, batch_size=8):
    """
    Process a list of image tensors with the model.
    
    Args:
        model: The paired symmetrizer model
        images: List of (image_tensor, image_path) tuples
        device: Device to use
        batch_size: Batch size for processing
    
    Returns:
        List of (original_tensor, symmetrized_tensor, symmetry_score, image_path) tuples
    """
    model.eval()
    results = []
    
    # Process in batches
    for i in range(0, len(images), batch_size):
        batch_images = [img for img, _ in images[i:i+batch_size]]
        batch_paths = [path for _, path in images[i:i+batch_size]]
        
        # Stack images into a batch
        batch_tensor = torch.stack(batch_images).to(device)
        
        # Process batch
        with torch.no_grad():
            output_tensor = model(batch_tensor)
            
            # Calculate symmetry scores
            symmetry_scores = [
                model.calculate_symmetry_loss(output_tensor[j:j+1]).item()
                for j in range(len(output_tensor))
            ]
        
        # Add results to list
        for j in range(len(batch_tensor)):
            results.append((
                batch_tensor[j].cpu(),
                output_tensor[j].cpu(),
                symmetry_scores[j],
                batch_paths[j]
            ))
    
    return results


def save_results(results, output_dir, create_comparison=True):
    """
    Save processed images and optionally create comparison visualizations.
    
    Args:
        results: List of (original_tensor, symmetrized_tensor, symmetry_score, image_path) tuples
        output_dir: Directory to save results
        create_comparison: Whether to create comparison visualizations
    """
    # Create output directories
    os.makedirs(os.path.join(output_dir, "original"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "symmetrized"), exist_ok=True)
    if create_comparison:
        os.makedirs(os.path.join(output_dir, "comparison"), exist_ok=True)
    
    # Save individual images
    for i, (orig, sym, score, path) in enumerate(results):
        # Convert tensors to numpy arrays
        orig_img = (orig.permute(1, 2, 0).numpy() + 1.0) / 2.0
        orig_img = (orig_img * 255).astype(np.uint8)
        
        sym_img = (sym.permute(1, 2, 0).numpy() + 1.0) / 2.0
        sym_img = (sym_img * 255).astype(np.uint8)
        
        # Get filename from path
        filename = Path(path).stem
        
        # Save images
        Image.fromarray(orig_img).save(os.path.join(output_dir, "original", f"{filename}.png"))
        Image.fromarray(sym_img).save(os.path.join(output_dir, "symmetrized", f"{filename}.png"))
        
        # Create comparison visualization
        if create_comparison:
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            
            axes[0].imshow(orig_img)
            axes[0].set_title("Original")
            axes[0].axis('off')
            
            axes[1].imshow(sym_img)
            axes[1].set_title(f"Symmetrized\nSymmetry Score: {score:.4f}")
            axes[1].axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "comparison", f"{filename}_comparison.png"))
            plt.close()
    
    # Create a grid of all results
    if len(results) > 1 and create_comparison:
        # Determine grid dimensions
        grid_size = min(len(results), 16)  # Show at most 16 images
        cols = min(4, grid_size)
        rows = (grid_size + cols - 1) // cols
        
        # Create the figure
        plt.figure(figsize=(cols * 5, rows * 5))
        
        for i in range(grid_size):
            if i >= len(results):
                break
                
            orig, sym, score, _ = results[i]
            
            # Convert to numpy
            orig_img = (orig.permute(1, 2, 0).numpy() + 1.0) / 2.0
            sym_img = (sym.permute(1, 2, 0).numpy() + 1.0) / 2.0
            
            # Original
            plt.subplot(rows, cols * 2, i * 2 + 1)
            plt.imshow(orig_img)
            plt.title(f"Original {i+1}")
            plt.axis('off')
            
            # Symmetrized
            plt.subplot(rows, cols * 2, i * 2 + 2)
            plt.imshow(sym_img)
            plt.title(f"Symmetrized {i+1}\nScore: {score:.4f}")
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "all_comparisons.png"))
        plt.close()
    
    print(f"Saved {len(results)} processed images to {output_dir}")


def main():
    # Parse command-line arguments
    args = parse_args()
    
    # Set device
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Load the model
    print(f"Loading model from {args.model_path}")
    model = PairedSymmetrizer.from_pretrained(args.model_path).to(device)
    model.eval()
    
    # Print model config
    print(f"Model config:")
    print(f"  Hidden dimension: {model.hidden_dim}")
    print(f"  With radius: {model.with_r}")
    print(f"  Symmetry type: {model.symmetry_type}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load images
    images = load_images(
        dataset_path=args.dataset_path,
        image_size=args.image_size,
        max_images=args.num_samples
    )
    
    print(f"Processing {len(images)} images...")
    
    # Process images
    results = process_images(
        model=model,
        images=images,
        device=device,
        batch_size=args.batch_size
    )
    
    # Calculate average symmetry score
    avg_score = sum(score for _, _, score, _ in results) / len(results)
    print(f"Average symmetry score: {avg_score:.6f}")
    
    # Save results
    save_results(
        results=results,
        output_dir=args.output_dir,
        create_comparison=True
    )
    
    print(f"Demo complete. Results saved to {args.output_dir}")


if __name__ == "__main__":
    main() 