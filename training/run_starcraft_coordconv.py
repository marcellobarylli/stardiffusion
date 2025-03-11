import os
import argparse
from models.diffusion.coord import convert_and_finetune_model, sample_from_coord_model

def parse_args():
    parser = argparse.ArgumentParser(description="StarCraft CoordConv Diffusion Model Finetuning")
    parser.add_argument(
        "--mode", 
        type=str, 
        default="train", 
        choices=["train", "sample"],
        help="Whether to train the model or generate samples"
    )
    parser.add_argument(
        "--model_path", 
        type=str, 
        default="checkpoints/coord_fine_tuned_models/starcraft_maps_coordconv/final",
        help="Path to trained CoordConv model (for sampling)"
    )
    parser.add_argument(
        "--num_samples", 
        type=int, 
        default=4,
        help="Number of samples to generate"
    )
    parser.add_argument(
        "--with_r", 
        action="store_true",
        help="Include radius channel in coordinate information"
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="data/StarCraft_Map_Dataset",
        help="Path to StarCraft dataset for training"
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
        default=15,
        help="Number of epochs for training"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-6,
        help="Learning rate for training"
    )
    parser.add_argument(
        "--save_interval",
        type=int,
        default=5,
        help="Interval for saving checkpoints during training"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoints/coord_fine_tuned_models/starcraft_maps_coordconv",
        help="Directory to save the fine-tuned model"
    )
    return parser.parse_args()

def main():
    """Run StarCraft CoordConv model training or sampling."""
    args = parse_args()
    
    # Create output directory if it doesn't exist
    if args.mode == "train":
        os.makedirs(args.output_dir, exist_ok=True)
    
    if args.mode == "train":
        print(f"Starting CoordConv fine-tuning of StarCraft model...")
        print(f"Dataset path: {args.dataset_path}")
        print(f"Output directory: {args.output_dir}")
        print(f"Using radius channel: {args.with_r}")
        
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
        
        print(f"Fine-tuning complete! Model saved to {args.output_dir}")
        
    elif args.mode == "sample":
        print(f"Generating {args.num_samples} samples from CoordConv model...")
        print(f"Model path: {args.model_path}")
        print(f"Using radius channel: {args.with_r}")
        
        # Generate samples
        sample_from_coord_model(
            model_path=args.model_path,
            num_samples=args.num_samples,
            with_r=args.with_r,
            output_dir="outputs/starcraft_samples"
        )
        
        print(f"Samples generated and saved to outputs/starcraft_samples")

if __name__ == "__main__":
    main() 