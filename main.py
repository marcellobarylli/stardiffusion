#!/usr/bin/env python3
import argparse
import torch
import os
import matplotlib.pyplot as plt
from typing import Optional, List, Dict, Any

# Import our modules
from models.diffusion.core import (
    DiffusionConfig,
    DiffusionModelLoader,
    DiffusionProcessor,
    DiffusionVisualizer,
    run_simple_generation
)
from training.trainer import DiffusionTrainer
from data.datasets import DatasetManager
from diffusers import DDPMPipeline


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
        "--output_dir", type=str, default="outputs",
        help="Directory to save generated images"
    )
    generate_parser.add_argument(
        "--gpus", type=str, default="2,3",
        help="Comma-separated list of GPU IDs to use"
    )
    
    # Generate from fine-tuned model
    generate_finetuned_parser = subparsers.add_parser(
        "generate-finetuned", 
        help="Generate images from a fine-tuned diffusion model"
    )
    generate_finetuned_parser.add_argument(
        "--model_path", type=str, required=True,
        help="Path to the fine-tuned model"
    )
    generate_finetuned_parser.add_argument(
        "--num_steps", type=int, default=100,
        help="Number of denoising steps"
    )
    generate_finetuned_parser.add_argument(
        "--batch_size", type=int, default=1,
        help="Batch size for generation"
    )
    generate_finetuned_parser.add_argument(
        "--num_images", type=int, default=4,
        help="Number of images to generate"
    )
    generate_finetuned_parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility"
    )
    generate_finetuned_parser.add_argument(
        "--output_dir", type=str, default="outputs/finetuned",
        help="Directory to save generated images"
    )
    generate_finetuned_parser.add_argument(
        "--gpus", type=str, default="2,3",
        help="Comma-separated list of GPU IDs to use"
    )
    
    # Fine-tune command
    finetune_parser = subparsers.add_parser("finetune", help="Fine-tune a diffusion model")
    finetune_parser.add_argument(
        "--model_id", type=str, default="google/ddpm-celebahq-256",
        help="Hugging Face model ID of the diffusion model to fine-tune"
    )
    finetune_parser.add_argument(
        "--dataset", type=str, required=True,
        help="Dataset to fine-tune on (name or path)"
    )
    finetune_parser.add_argument(
        "--batch_size", type=int, default=4,
        help="Batch size for training"
    )
    finetune_parser.add_argument(
        "--num_epochs", type=int, default=10,
        help="Number of epochs to train"
    )
    finetune_parser.add_argument(
        "--learning_rate", type=float, default=5e-6,
        help="Learning rate for the optimizer"
    )
    finetune_parser.add_argument(
        "--output_dir", type=str, default="checkpoints/fine_tuned_models",
        help="Directory to save the fine-tuned model"
    )
    finetune_parser.add_argument(
        "--save_interval", type=int, default=1,
        help="Interval (in epochs) to save checkpoints"
    )
    finetune_parser.add_argument(
        "--gpus", type=str, default="2,3",
        help="Comma-separated list of GPU IDs to use"
    )
    finetune_parser.add_argument(
        "--gradient_accumulation_steps", type=int, default=1,
        help="Number of steps to accumulate gradients"
    )
    finetune_parser.add_argument(
        "--mixed_precision", action="store_true",
        help="Use mixed precision training"
    )
    
    # List datasets command
    list_datasets_parser = subparsers.add_parser(
        "list-datasets", 
        help="List available standard datasets"
    )
    
    # CoordConv subcommand for future implementation
    coordconv_parser = subparsers.add_parser(
        "coordconv", 
        help="CoordConv diffusion operations"
    )
    coordconv_parser.add_argument(
        "--operation", choices=["convert", "finetune", "generate"],
        help="CoordConv operation to perform"
    )
    
    return parser.parse_args()


def generate(args):
    """Generate images with a pre-trained diffusion model."""
    print(f"Generating images with model {args.model_id}...")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set up configuration
    config = DiffusionConfig(
        model_id=args.model_id,
        visible_gpus=args.gpus,
        num_inference_steps=args.num_steps,
        batch_size=args.batch_size,
        seed=args.seed
    )
    
    # Create model loader and load model
    model_loader = DiffusionModelLoader(config)
    model_loader.load_model()
    
    # Create processor and visualizer
    processor = DiffusionProcessor(model_loader, config)
    visualizer = DiffusionVisualizer(model_loader)
    
    # Generate sample
    final_image, intermediates = processor.generate_sample(save_intermediates=True)
    
    # Visualize and save results
    visualizer.visualize_progression(
        intermediates, 
        save_path=os.path.join(args.output_dir, "diffusion_progression.png")
    )
    visualizer.visualize_final_image(
        final_image, 
        save_path=os.path.join(args.output_dir, "generated_image.png")
    )
    
    print(f"Generation complete! Results saved to {args.output_dir}")


def generate_from_finetuned(args):
    """Generate images from a fine-tuned diffusion model."""
    print(f"Generating images from fine-tuned model at {args.model_path}...")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set up configuration
    config = DiffusionConfig(
        model_id=args.model_path,  # Use the path as model_id
        visible_gpus=args.gpus,
        num_inference_steps=args.num_steps,
        batch_size=args.batch_size,
        seed=args.seed
    )
    
    # Create model loader and load model
    model_loader = DiffusionModelLoader(config)
    model_loader.load_model()
    
    # Create processor and visualizer
    processor = DiffusionProcessor(model_loader, config)
    visualizer = DiffusionVisualizer(model_loader)
    
    # Generate images
    for i in range(args.num_images):
        # Generate sample
        final_image, intermediates = processor.generate_sample(save_intermediates=(i==0))
        
        # Save the image
        visualizer.visualize_final_image(
            final_image, 
            save_path=os.path.join(args.output_dir, f"generated_image_{i+1}.png")
        )
        
        # Save progression for the first image only
        if i == 0:
            visualizer.visualize_progression(
                intermediates, 
                save_path=os.path.join(args.output_dir, "diffusion_progression.png")
            )
    
    print(f"Generation complete! {args.num_images} images saved to {args.output_dir}")


def finetune(args):
    """Fine-tune a diffusion model on a dataset."""
    print(f"Fine-tuning model {args.model_id} on dataset {args.dataset}...")
    
    # Create output directory path
    model_name = args.model_id.split("/")[-1]
    dataset_name = os.path.basename(args.dataset)
    output_dir = os.path.join(args.output_dir, f"{model_name}_{dataset_name}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up configuration
    config = DiffusionConfig(
        model_id=args.model_id,
        visible_gpus=args.gpus,
        batch_size=args.batch_size
    )
    
    # Create model loader and load model
    model_loader = DiffusionModelLoader(config)
    model_loader.load_model()
    
    # Create trainer
    trainer = DiffusionTrainer(
        model_loader=model_loader,
        config=config,
        output_dir=output_dir
    )
    
    # Prepare dataset
    dataloader = trainer.prepare_dataset(
        dataset_name_or_path=args.dataset,
        batch_size=args.batch_size
    )
    
    # Fine-tune the model
    trainer.train(
        dataloader=dataloader,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        save_interval=args.save_interval,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision
    )
    
    print(f"Fine-tuning complete! Model saved to {output_dir}")


def list_datasets():
    """List available standard datasets."""
    datasets = DatasetManager.list_available_datasets()
    
    print("Available standard datasets:")
    for dataset in datasets:
        print(f"  - {dataset}")
    
    print("\nYou can also provide a path to a directory containing images for custom datasets.")


def main():
    """Run the main program based on command-line arguments."""
    args = parse_args()
    
    if args.command == "generate":
        generate(args)
    elif args.command == "generate-finetuned":
        generate_from_finetuned(args)
    elif args.command == "finetune":
        finetune(args)
    elif args.command == "list-datasets":
        list_datasets()
    elif args.command == "coordconv":
        print("CoordConv operations are still under implementation.")
        print("Please use the specific scripts in training/ and inference/ for CoordConv operations.")
    elif args.command is None:
        # Default action if no command is provided
        print("No command specified. Running simple generation example.")
        run_simple_generation()
    else:
        print(f"Unknown command: {args.command}")
        

if __name__ == "__main__":
    main() 