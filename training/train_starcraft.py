import torch
import os
import sys
import argparse
from pathlib import Path

# Add the project root to the Python path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import from our modules
from models.diffusion.core import DiffusionConfig, DiffusionModelLoader
from training.trainer import DiffusionTrainer

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune diffusion model on StarCraft maps")
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="data/StarCraft_Map_Dataset",
        help="Path to the StarCraft maps dataset"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoints/starcraft_fine_tuned",
        help="Directory to save model checkpoints"
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="google/ddpm-celebahq-256",
        help="Base model to fine-tune"
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=256,
        help="Image size to use for training"
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
        default=20,
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
        default=5,
        help="Interval (in epochs) to save checkpoints"
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default="0",
        help="Comma-separated list of GPU IDs to use"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of steps to accumulate gradients"
    )
    parser.add_argument(
        "--mixed_precision",
        action="store_true",
        help="Use mixed precision training"
    )
    return parser.parse_args()

def main():
    # Parse arguments
    args = parse_args()
    
    print("=== StarCraft Maps Diffusion Model Fine-tuning ===")
    print(f"Dataset: {args.dataset_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Base model: {args.base_model}")
    print(f"Image size: {args.image_size}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.num_epochs}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"GPUs: {args.gpus}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set up diffusion config
    config = DiffusionConfig(
        model_id=args.base_model,
        visible_gpus=args.gpus,
        batch_size=args.batch_size
    )
    
    # Load the base model
    model_loader = DiffusionModelLoader(config)
    model_loader.load_model()
    
    print(f"Loaded base model: {model_loader.get_model_info()}")
    
    # Initialize trainer
    trainer = DiffusionTrainer(
        model_loader=model_loader,
        config=config,
        output_dir=args.output_dir
    )
    
    # Prepare the dataset
    dataloader = trainer.prepare_dataset(
        dataset_name_or_path=args.dataset_path,
        batch_size=args.batch_size,
        image_size=args.image_size
    )
    
    print(f"Prepared dataset with {len(dataloader)} batches")
    
    # Train the model
    losses = trainer.train(
        dataloader=dataloader,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        save_interval=args.save_interval,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision
    )
    
    print("Fine-tuning complete!")
    print(f"Final model saved to: {os.path.join(args.output_dir, 'final')}")

if __name__ == "__main__":
    main() 