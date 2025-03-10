import torch
import os
import sys
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from typing import Dict, Optional, List, Union, Any

# Import our custom modules
from DiffusionCore import DiffusionConfig, DiffusionModelLoader, DiffusionProcessor, DiffusionVisualizer
from DiffusionFineTuner import DiffusionTrainer, DatasetManager
from CoordConv import CoordConvUNet2DModel, convert_to_coordconv


class CoordDiffusionTrainer(DiffusionTrainer):
    """Extends DiffusionTrainer to use CoordConv for fine-tuning."""
    
    def __init__(
        self, 
        model_loader: DiffusionModelLoader,
        config: DiffusionConfig,
        output_dir: str = "coord_fine_tuned_model",
        device: Optional[torch.device] = None,
        with_r: bool = False,
        normalize_coords: bool = True
    ):
        """Initialize the CoordConv trainer.
        
        Args:
            model_loader: Model loader containing the model to fine-tune
            config: Configuration for the diffusion process
            output_dir: Directory to save model checkpoints
            device: Device to use for training (default: from config)
            with_r: Whether to include a radius channel
            normalize_coords: Whether to normalize coordinate values to [-1, 1]
        """
        # Initialize the base trainer
        super().__init__(model_loader, config, output_dir, device)
        
        # Convert the UNet model to a CoordConv version
        original_unet = self.model
        
        # Unwrap from DataParallel if necessary
        if hasattr(original_unet, "module"):
            # Model is wrapped in DataParallel, need to unwrap first
            unwrapped_unet = original_unet.module
            self.model = convert_to_coordconv(
                unwrapped_unet, 
                with_r=with_r, 
                normalize=normalize_coords
            )
            # Re-wrap in DataParallel
            if self.config.use_data_parallel:
                self.model = torch.nn.DataParallel(
                    self.model, 
                    device_ids=self.config.device_ids
                )
        else:
            # Model is not wrapped, convert directly
            self.model = convert_to_coordconv(
                original_unet, 
                with_r=with_r, 
                normalize=normalize_coords
            )
            
        # Store the CoordConv settings
        self.with_r = with_r
        self.normalize_coords = normalize_coords
        
        # Print model information
        print(f"Converted to CoordConv UNet model")
        print(f"With radius channel: {self.with_r}")
        print(f"Normalize coordinates: {self.normalize_coords}")
        print(f"Additional channels: {2 + (1 if self.with_r else 0)}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters())}")
    
    def save_model(self, output_dir: str):
        """Save the fine-tuned CoordConv model.
        
        Args:
            output_dir: Directory to save the model
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Check if model is wrapped in DataParallel and unwrap if needed
        model_to_save = self.model.module if hasattr(self.model, "module") else self.model
        
        # Save model using the custom save method
        print(f"Saving CoordConv UNet weights...")
        model_to_save.save_pretrained(output_dir)
        
        # Save scheduler configuration
        print(f"Saving scheduler configuration...")
        self.noise_scheduler.save_pretrained(output_dir)
        
        # Save a README with training details
        with open(os.path.join(output_dir, "training_info.txt"), "w") as f:
            f.write(f"Fine-tuned from: {self.config.model_id}\n")
            f.write(f"Training device: {self.device}\n")
            f.write(f"With CoordConv - radius channel: {self.with_r}\n")
            f.write(f"With CoordConv - normalized coordinates: {self.normalize_coords}\n")
            if hasattr(self.config, 'use_data_parallel') and self.config.use_data_parallel:
                f.write(f"Trained on GPUs: {self.config.visible_gpus}\n")
            
        print(f"Model successfully saved to {output_dir}")


# Function to convert and fine-tune a model using CoordConv
def convert_and_finetune_model(
    model_path: str,
    dataset_path: str,
    output_dir: str,
    batch_size: int = 8,
    num_epochs: int = 15,
    learning_rate: float = 5e-6,
    save_interval: int = 5,
    visible_gpus: str = "2",
    with_r: bool = True,
    normalize_coords: bool = True,
    num_workers: int = 4
):
    """Convert a model to use CoordConv and fine-tune it.
    
    Args:
        model_path: Path to the model to convert
        dataset_path: Path to the dataset for fine-tuning
        output_dir: Directory to save the fine-tuned model
        batch_size: Batch size for training
        num_epochs: Number of epochs for fine-tuning
        learning_rate: Learning rate for fine-tuning
        save_interval: Interval for saving checkpoints
        visible_gpus: GPU indices to use
        with_r: Whether to include radius channel
        normalize_coords: Whether to normalize coordinates
        num_workers: Number of workers for data loading
    
    Returns:
        Fine-tuned trainer
    """
    # Initialize configuration
    config = DiffusionConfig(
        model_id=model_path,
        visible_gpus=visible_gpus,
        num_inference_steps=100
    )
    
    # Load the model
    model_loader = DiffusionModelLoader(config)
    model_loader.load_model()
    
    # Create CoordConv trainer
    trainer = CoordDiffusionTrainer(
        model_loader=model_loader,
        config=config,
        output_dir=output_dir,
        with_r=with_r,
        normalize_coords=normalize_coords
    )
    
    # Prepare dataset
    dataloader = trainer.prepare_dataset(
        dataset_name_or_path=dataset_path,
        batch_size=batch_size,
        num_workers=num_workers
    )
    
    # Train the model
    losses = trainer.train(
        dataloader=dataloader,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        save_interval=save_interval
    )
    
    # Plot training losses
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title('CoordConv Fine-Tuning Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "training_loss.png"))
    
    return trainer


# Example of how to use the CoordConv implementation with the starcraft model
def convert_and_finetune_starcraft_model(
    output_dir: str = "coord_fine_tuned_models/starcraft_maps_coordconv",
    with_r: bool = True,
    normalize_coords: bool = True,
    num_epochs: int = 15,
    batch_size: int = 8,
    learning_rate: float = 5e-6,
    save_interval: int = 5
):
    """Example of how to convert and further fine-tune the starcraft model using CoordConv."""
    return convert_and_finetune_model(
        model_path="fine_tuned_models/starcraft_maps/final",
        dataset_path="data/StarCraft_Map_Dataset",
        output_dir=output_dir,
        batch_size=batch_size,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        save_interval=save_interval,
        with_r=with_r,
        normalize_coords=normalize_coords
    )


# Function to sample from the converted model
def sample_from_coord_model(model_path: str, num_samples: int = 4, with_r: bool = True):
    """Generate samples from a CoordConv model.
    
    Args:
        model_path: Path to the CoordConv model
        num_samples: Number of samples to generate
        with_r: Whether the model includes a radius channel
    """
    # Initialize config
    config = DiffusionConfig(
        model_id=model_path,  # This will be overridden when we load the CoordConv model
        visible_gpus="2",
        num_inference_steps=100,
        batch_size=num_samples
    )
    
    # Create a basic model loader (we'll replace this model with our CoordConv model)
    model_loader = DiffusionModelLoader(config)
    model_loader.load_model()
    
    # Replace with CoordConv model
    original_unet = model_loader.unet
    # Unwrap from DataParallel if necessary
    if hasattr(original_unet, "module"):
        original_unet.module = CoordConvUNet2DModel.from_pretrained(
            model_path, with_r=with_r
        )
    else:
        model_loader.unet = CoordConvUNet2DModel.from_pretrained(
            model_path, with_r=with_r
        )
    
    # Create processor and generate samples
    processor = DiffusionProcessor(model_loader, config)
    final_images, intermediates = processor.generate_sample()
    
    # Visualize results
    visualizer = DiffusionVisualizer(model_loader)
    visualizer.visualize_progression(
        intermediates, 
        save_path=f"{model_path}/sample_progression.png"
    )
    
    # Save individual final samples
    for i, img in enumerate(final_images):
        plt.figure(figsize=(8, 8))
        if img.shape[0] == 1:  # Grayscale
            plt.imshow(img[0].cpu(), cmap='gray')
        else:  # RGB
            plt.imshow(img.permute(1, 2, 0).cpu())
        plt.axis('off')
        plt.savefig(f"{model_path}/sample_{i}.png", bbox_inches='tight')
        plt.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="CoordConv Diffusion Model Training and Sampling")
    parser.add_argument("--mode", type=str, choices=["train", "sample"], default="train",
                        help="Mode: train or sample")
    
    # Training arguments
    parser.add_argument("--model_path", type=str, default="fine_tuned_models/starcraft_maps/final",
                        help="Path to model to convert/fine-tune or sample from")
    parser.add_argument("--dataset_path", type=str, default="data/StarCraft_Map_Dataset",
                        help="Path to dataset for fine-tuning")
    parser.add_argument("--output_dir", type=str, default="coord_fine_tuned_models/starcraft_maps_coordconv",
                        help="Directory to save fine-tuned model")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=15,
                        help="Number of epochs for training")
    parser.add_argument("--learning_rate", type=float, default=5e-6,
                        help="Learning rate for training")
    parser.add_argument("--save_interval", type=int, default=5,
                        help="Interval for saving checkpoints")
    parser.add_argument("--visible_gpus", type=str, default="2",
                        help="Comma-separated list of GPU indices to use")
    
    # CoordConv options
    parser.add_argument("--with_r", action="store_true",
                        help="Include radius channel")
    parser.add_argument("--normalize_coords", action="store_true", default=True,
                        help="Normalize coordinate values to [-1, 1]")
    
    # Sampling arguments
    parser.add_argument("--num_samples", type=int, default=4,
                        help="Number of samples to generate")
    
    args = parser.parse_args()
    
    if args.mode == "train":
        print(f"Converting and fine-tuning model with CoordConv...")
        trainer = convert_and_finetune_model(
            model_path=args.model_path,
            dataset_path=args.dataset_path,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            save_interval=args.save_interval,
            visible_gpus=args.visible_gpus,
            with_r=args.with_r,
            normalize_coords=args.normalize_coords
        )
        print(f"Fine-tuning complete! Model saved to {args.output_dir}")
        
    elif args.mode == "sample":
        print(f"Generating {args.num_samples} samples from CoordConv model...")
        sample_from_coord_model(
            model_path=args.model_path,
            num_samples=args.num_samples,
            with_r=args.with_r
        )
        print(f"Samples generated and saved to {args.model_path}") 