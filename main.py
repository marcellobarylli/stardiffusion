#!/usr/bin/env python3
import argparse
import torch
import os
import matplotlib.pyplot as plt
from typing import Optional, List, Dict, Any

# Import our modules
from UNetFinetune import (
    DiffusionConfig,
    DiffusionModelLoader,
    DiffusionProcessor,
    DiffusionVisualizer,
    main as run_generation
)
from DiffusionFineTuner import DiffusionTrainer, example_fine_tuning_script


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Diffusion model generation and fine-tuning")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Generate command
    generate_parser = subparsers.add_parser("generate", help="Generate images with a diffusion model")
    generate_parser.add_argument(
        "--model_id", type=str, default="google/ddpm-celebahq-256",
        help="Hugging Face model ID of the diffusion model"
    )
    generate_parser.add_argument(
        "--num_steps", type=int, default=100,
        help="Number of denoising steps"
    )
    generate_parser.add_argument(
        "--batch_size", type=int, default=1,
        help="Batch size for generation"
    )
    generate_parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility"
    )
    generate_parser.add_argument(
        "--output_dir", type=str, default="output",
        help="Directory to save generated images"
    )
    generate_parser.add_argument(
        "--gpus", type=str, default="0,1",
        help="Comma-separated list of GPU IDs to use"
    )
    
    # Fine-tune command
    finetune_parser = subparsers.add_parser("finetune", help="Fine-tune a diffusion model")
    finetune_parser.add_argument(
        "--model_id", type=str, default="google/ddpm-celebahq-256",
        help="Hugging Face model ID of the diffusion model to fine-tune"
    )
    finetune_parser.add_argument(
        "--dataset_path", type=str, required=True,
        help="Path to the dataset for fine-tuning"
    )
    finetune_parser.add_argument(
        "--output_dir", type=str, default="fine_tuned_model",
        help="Directory to save the fine-tuned model"
    )
    finetune_parser.add_argument(
        "--num_epochs", type=int, default=10,
        help="Number of training epochs"
    )
    finetune_parser.add_argument(
        "--batch_size", type=int, default=8,
        help="Batch size for training"
    )
    finetune_parser.add_argument(
        "--learning_rate", type=float, default=1e-5,
        help="Learning rate for optimization"
    )
    finetune_parser.add_argument(
        "--save_interval", type=int, default=5,
        help="Interval for saving checkpoints (in epochs)"
    )
    finetune_parser.add_argument(
        "--gpus", type=str, default="0,1",
        help="Comma-separated list of GPU IDs to use"
    )
    finetune_parser.add_argument(
        "--mixed_precision", action="store_true",
        help="Use mixed precision training"
    )
    finetune_parser.add_argument(
        "--image_size", type=int, default=None,
        help="Target image size (default: from model config)"
    )
    finetune_parser.add_argument(
        "--num_workers", type=int, default=4,
        help="Number of workers for data loading"
    )
    
    # Generate with fine-tuned model command
    generate_ft_parser = subparsers.add_parser(
        "generate_ft", 
        help="Generate images with a fine-tuned diffusion model"
    )
    generate_ft_parser.add_argument(
        "--model_path", type=str, required=True,
        help="Path to the fine-tuned model"
    )
    generate_ft_parser.add_argument(
        "--num_steps", type=int, default=100,
        help="Number of denoising steps"
    )
    generate_ft_parser.add_argument(
        "--batch_size", type=int, default=1,
        help="Batch size for generation"
    )
    generate_ft_parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility"
    )
    generate_ft_parser.add_argument(
        "--output_dir", type=str, default="output",
        help="Directory to save generated images"
    )
    generate_ft_parser.add_argument(
        "--gpus", type=str, default="0,1",
        help="Comma-separated list of GPU IDs to use"
    )
    
    return parser.parse_args()


def generate(args):
    """Generate images with a diffusion model."""
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize configuration
    config = DiffusionConfig(
        model_id=args.model_id,
        visible_gpus=args.gpus,
        num_inference_steps=args.num_steps,
        batch_size=args.batch_size,
        seed=args.seed
    )
    
    # Load the model
    model_loader = DiffusionModelLoader(config)
    model_loader.load_model()
    
    # Create processor and run diffusion
    processor = DiffusionProcessor(model_loader, config)
    final_image, intermediates = processor.generate_sample()
    
    # Visualize results
    visualizer = DiffusionVisualizer(model_loader)
    visualizer.visualize_progression(
        intermediates,
        save_path=os.path.join(args.output_dir, "diffusion_progression.png")
    )
    visualizer.visualize_final_image(
        final_image,
        save_path=os.path.join(args.output_dir, "generated_image.png")
    )
    
    print(f"\nGeneration complete! Results saved to {args.output_dir}")


def generate_from_finetuned(args):
    """Generate images with a fine-tuned diffusion model."""
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load the model from local path
    print(f"Loading fine-tuned model from {args.model_path}")
    
    # Initialize configuration
    config = DiffusionConfig(
        model_id=args.model_path,  # Use local path instead of HF model ID
        visible_gpus=args.gpus,
        num_inference_steps=args.num_steps,
        batch_size=args.batch_size,
        seed=args.seed
    )
    
    # Load the model
    model_loader = DiffusionModelLoader(config)
    model_loader.load_model()
    
    # Create processor and run diffusion
    processor = DiffusionProcessor(model_loader, config)
    final_image, intermediates = processor.generate_sample()
    
    # Visualize results
    visualizer = DiffusionVisualizer(model_loader)
    visualizer.visualize_progression(
        intermediates,
        save_path=os.path.join(args.output_dir, "ft_diffusion_progression.png")
    )
    visualizer.visualize_final_image(
        final_image,
        save_path=os.path.join(args.output_dir, "ft_generated_image.png")
    )
    
    print(f"\nGeneration with fine-tuned model complete! Results saved to {args.output_dir}")


def finetune(args):
    """Fine-tune a diffusion model on a custom dataset."""
    # Initialize configuration
    config = DiffusionConfig(
        model_id=args.model_id,
        visible_gpus=args.gpus,
        num_inference_steps=100,  # Not used for fine-tuning
        batch_size=args.batch_size,
        seed=42  # Not critical for fine-tuning
    )
    
    # Load the model
    print(f"Loading base model {args.model_id} for fine-tuning")
    model_loader = DiffusionModelLoader(config)
    model_loader.load_model()
    
    # Create trainer
    trainer = DiffusionTrainer(
        model_loader=model_loader,
        config=config,
        output_dir=args.output_dir
    )
    
    # Prepare dataset
    dataloader = trainer.prepare_dataset(
        dataset_path=args.dataset_path,
        batch_size=args.batch_size,
        image_size=args.image_size,
        num_workers=args.num_workers
    )
    
    # Train the model
    losses = trainer.train(
        dataloader=dataloader,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        save_interval=args.save_interval,
        mixed_precision=args.mixed_precision
    )
    
    # Plot and save training loss
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig(os.path.join(args.output_dir, "training_loss.png"))
    
    print(f"Fine-tuning complete! Model saved to {args.output_dir}")
    print(f"Final loss: {losses[-1]:.6f}")


def main():
    """Main entry point."""
    args = parse_args()
    
    if args.command == "generate":
        generate(args)
    elif args.command == "finetune":
        finetune(args)
    elif args.command == "generate_ft":
        generate_from_finetuned(args)
    else:
        print("Please specify a command. Use --help for more information.")
        # If no command is specified, run the simple generation example
        run_generation()


if __name__ == "__main__":
    main() 