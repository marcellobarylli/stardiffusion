import torch
import argparse
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Add the project root to sys.path
project_root = Path(__file__).absolute().parent.parent
sys.path.append(str(project_root))

from models.coord_conv.learning_symmetry import LearnableSymmetryTransformer, train_symmetry_transformer


def parse_args():
    parser = argparse.ArgumentParser(description="Train a learnable symmetry transformer model")
    
    # Training parameters
    parser.add_argument("--output_dir", type=str, default="checkpoints/learnable_symmetry",
                      help="Directory to save model checkpoints and training progress")
    parser.add_argument("--batch_size", type=int, default=32,
                      help="Batch size for training")
    parser.add_argument("--image_size", type=int, default=256,
                      help="Size of generated noise images")
    parser.add_argument("--num_epochs", type=int, default=1000,
                      help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                      help="Learning rate for optimizer")
    parser.add_argument("--save_interval", type=int, default=100,
                      help="Save model checkpoint every N epochs")
    parser.add_argument("--visualize_interval", type=int, default=25,
                      help="Visualize progress every N epochs")
    
    # Model parameters
    parser.add_argument("--hidden_dim", type=int, default=64,
                      help="Hidden dimension size")
    parser.add_argument("--with_r", action="store_true",
                      help="Include radius coordinate channel")
    parser.add_argument("--symmetry_type", type=str, default="vertical", 
                      choices=["vertical", "horizontal", "both"],
                      help="Type of symmetry to learn")
    parser.add_argument("--channels", type=int, default=3,
                      help="Number of image channels")
    parser.add_argument("--symmetry_weight", type=float, default=0.7,
                      help="Weight of symmetry loss in total loss")
    
    # Device
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                      help="Device to train on (default: auto-detect)")
    
    return parser.parse_args()


def evaluate_symmetry(model, num_samples=10, image_size=256, device="cuda"):
    """
    Evaluate the model's ability to generate symmetric noise.
    
    Args:
        model: The transformer model to evaluate
        num_samples: Number of samples to evaluate
        image_size: Size of noise images
        device: Device to generate on
        
    Returns:
        Average symmetry score (lower is better)
    """
    model.eval()
    symmetry_scores = []
    
    with torch.no_grad():
        for _ in range(num_samples):
            # Generate random noise
            noise = torch.randn(1, model.channels, image_size, image_size, device=device)
            
            # Transform noise
            transformed = model(noise)
            
            # Calculate symmetry score
            score = model.calculate_symmetry_loss(transformed).item()
            symmetry_scores.append(score)
    
    return sum(symmetry_scores) / len(symmetry_scores)


def main():
    # Parse command-line arguments
    args = parse_args()
    
    # Set device
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Create model
    model = LearnableSymmetryTransformer(
        channels=args.channels,
        hidden_dim=args.hidden_dim,
        with_r=args.with_r,
        normalize=True,  # Always normalize coordinates
        symmetry_type=args.symmetry_type,
        symmetry_weight=args.symmetry_weight
    ).to(device)
    
    print(f"Created LearnableSymmetryTransformer with config:")
    print(f"  Channels: {args.channels}")
    print(f"  Hidden dimension: {args.hidden_dim}")
    print(f"  With radius: {args.with_r}")
    print(f"  Symmetry type: {args.symmetry_type}")
    print(f"  Symmetry weight: {args.symmetry_weight}")
    
    # Train model
    model = train_symmetry_transformer(
        transformer=model,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        image_size=args.image_size,
        learning_rate=args.learning_rate,
        device=device,
        save_dir=args.output_dir,
        save_interval=args.save_interval,
        visualize_interval=args.visualize_interval
    )
    
    # Final evaluation
    final_symmetry_score = evaluate_symmetry(
        model=model,
        num_samples=20,
        image_size=args.image_size,
        device=device
    )
    
    print(f"Training complete!")
    print(f"Final average symmetry score: {final_symmetry_score:.6f} (lower is better)")
    print(f"Model saved to: {os.path.join(args.output_dir, 'final')}")
    
    # Generate a final demo
    model.demo(
        output_path=os.path.join(args.output_dir, "final_demo"),
        num_samples=8,
        height=args.image_size,
        width=args.image_size,
        device=device,
        show_symmetry_scores=True
    )


if __name__ == "__main__":
    main() 