import sys
import os
import argparse
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from models.diffusion.coord import (
    convert_and_finetune_model,
)

def parse_args():
    parser = argparse.ArgumentParser(description="Train a diffusion model with CoordConv for StarCraft maps")
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="data/StarCraft_Map_Dataset",
        help="Path to the StarCraft maps dataset"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoints/coord_fine_tuned_models/starcraft_maps_coordconv",
        help="Directory to save the fine-tuned model"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for training"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=10,
        help="Number of epochs to train"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Learning rate for training"
    )
    parser.add_argument(
        "--save_interval",
        type=int,
        default=2,
        help="Save model every N epochs"
    )
    parser.add_argument(
        "--with_r",
        action="store_true",
        help="Use radius channel in CoordConv"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    print(f"Training CoordConv UNet model for StarCraft maps...")
    
    # Run the fine-tuning process
    convert_and_finetune_model(
        model_path="google/ddpm-celebahq-256",  # Base model to convert
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        save_interval=args.save_interval,
        with_r=args.with_r,
        normalize_coords=True
    )
    
    print(f"Training complete!")

if __name__ == "__main__":
    main() 