import sys
import os
import argparse
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from models.diffusion.coord import (
    convert_and_finetune_model,
    sample_direct_from_coordconv,
    debug_coordconv_sampling,
    multi_seed_test
)

def parse_args():
    parser = argparse.ArgumentParser(description="Train and sample from coordconv UNet models")
    parser.add_argument(
        "--mode",
        type=str, 
        default="train", 
        choices=["train", "sample", "debug", "multi_seed"],
        help="Whether to train the model, generate samples, run in debug mode, or test multiple seeds"
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="data/starcraft_maps_256",
        help="Path to the StarCraft maps dataset"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="checkpoints/coord_fine_tuned_models/starcraft_maps_coordconv/final",
        help="Path to the model to use for sampling"
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
        default=8,
        help="Number of samples to generate"
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=50,
        help="Number of diffusion steps for sampling"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
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
        default=1e-4,
        help="Learning rate for training"
    )
    parser.add_argument(
        "--save_interval",
        type=int,
        default=5,
        help="Save model every N epochs"
    )
    parser.add_argument(
        "--with_r",
        action="store_true",
        help="Use radius channel in CoordConv"
    )
    parser.add_argument(
        "--num_seeds",
        type=int,
        default=5,
        help="Number of seeds to test (for multi_seed mode)"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    if args.mode == "train":
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
        
    elif args.mode == "sample":
        print(f"Generating samples from {args.model_path}...")
        
        # Generate samples
        sample_direct_from_coordconv(
            model_path=args.model_path,
            output_dir=args.output_dir,
            num_samples=args.num_samples,
            num_steps=args.num_steps,
            with_r=args.with_r
        )
        
        print(f"Samples generated and saved to {args.output_dir}")
        
    elif args.mode == "debug":
        print(f"Running debug mode to diagnose sampling issues...")
        
        # Debug sampling
        debug_coordconv_sampling(
            model_path=args.model_path,
            output_dir="outputs/starcraft_debug",
            num_steps=args.num_steps,
            with_r=args.with_r
        )
        
        print(f"Debug information saved to outputs/starcraft_debug")
        print(f"Please check the debug_log.txt file and visualizations to diagnose the issue.")
    elif args.mode == "multi_seed":
        print(f"Testing with {args.num_seeds} different seeds...")
        
        # Run multi-seed testing
        multi_seed_test(
            model_path=args.model_path,
            output_dir=args.output_dir,
            num_seeds=args.num_seeds,
            num_steps=args.num_steps,
            with_r=args.with_r
        )
    else:
        print(f"Unknown mode: {args.mode}")
        sys.exit(1)

if __name__ == "__main__":
    main() 