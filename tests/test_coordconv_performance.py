import os
import torch
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
from pathlib import Path
from torch.utils.data import DataLoader

# Import from our modules
from models.diffusion.core import DiffusionConfig, DiffusionModelLoader, DiffusionProcessor, DiffusionVisualizer
from models.coord_conv.unet import convert_to_coordconv
from models.diffusion.coord import sample_from_coord_model
from training.trainer import DiffusionTrainer
from data.datasets import DatasetManager

class CoordConvTester:
    """Class to test CoordConv implementation and compare it with standard model."""
    
    def __init__(
        self,
        dataset_path: str,
        output_dir: str = "outputs/coordconv_test",
        model_id: str = "google/ddpm-celebahq-256",
        batch_size: int = 8,
        num_epochs: int = 10,
        learning_rate: float = 1e-5,
        with_r: bool = True,
        normalize_coords: bool = True,
        image_size: int = 64,
    ):
        """Initialize the CoordConv tester.
        
        Args:
            dataset_path: Path to the dataset for training
            output_dir: Directory to save testing results and models
            model_id: HuggingFace model ID to use as base model
            batch_size: Batch size for training
            num_epochs: Number of epochs to train
            learning_rate: Learning rate for training
            with_r: Whether to include radius channel in CoordConv
            normalize_coords: Whether to normalize coordinates
            image_size: Size of images to use for training
        """
        self.dataset_path = dataset_path
        self.output_dir = output_dir
        self.model_id = model_id
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.with_r = with_r
        self.normalize_coords = normalize_coords
        self.image_size = image_size
        
        # Create output directories
        self.standard_model_dir = os.path.join(output_dir, "standard_model")
        self.coordconv_model_dir = os.path.join(output_dir, "coordconv_model")
        self.standard_samples_dir = os.path.join(output_dir, "standard_samples")
        self.coordconv_samples_dir = os.path.join(output_dir, "coordconv_samples")
        
        os.makedirs(self.standard_model_dir, exist_ok=True)
        os.makedirs(self.coordconv_model_dir, exist_ok=True)
        os.makedirs(self.standard_samples_dir, exist_ok=True)
        os.makedirs(self.coordconv_samples_dir, exist_ok=True)
        
        # Create device configuration - DiffusionConfig will set the device automatically
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create diffusion configuration with only supported parameters
        self.config = DiffusionConfig(
            model_id=model_id,
            batch_size=batch_size,
            seed=42  # Use a fixed seed for reproducibility
        )
        
    def prepare_dataset(self):
        """Prepare dataset for training."""
        print(f"Preparing dataset from {self.dataset_path}")
        
        # Get the dataset using DatasetManager's get_dataset method
        dataset = DatasetManager.get_dataset(
            dataset_name_or_path=self.dataset_path,
            image_size=self.image_size,
            train=True,
            download=True
        )
        
        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        return dataloader
    
    def train_standard_model(self):
        """Train a standard UNet diffusion model."""
        print("\n=== Training Standard UNet Model ===")
        
        # Load model
        model_loader = DiffusionModelLoader(self.config)
        model_loader.load_model()
        
        # Create trainer
        trainer = DiffusionTrainer(
            model_loader=model_loader,
            config=self.config,
            output_dir=self.standard_model_dir
        )
        
        # Prepare dataset
        dataloader = self.prepare_dataset()
        
        # Train model
        start_time = time.time()
        trainer.train(
            dataloader=dataloader,
            num_epochs=self.num_epochs,
            learning_rate=self.learning_rate
        )
        training_time = time.time() - start_time
        
        # Save training metrics
        with open(os.path.join(self.standard_model_dir, "training_metrics.txt"), "w") as f:
            f.write(f"Training time: {training_time:.2f} seconds\n")
        
        print(f"Standard model training completed in {training_time:.2f} seconds")
        return model_loader, training_time
    
    def train_coordconv_model(self):
        """Train a CoordConv UNet diffusion model."""
        print("\n=== Training CoordConv UNet Model ===")
        
        # Load model
        model_loader = DiffusionModelLoader(self.config)
        model_loader.load_model()
        
        # Convert to CoordConv
        original_model = model_loader.model
        model_loader.model = convert_to_coordconv(
            original_model,
            with_r=self.with_r,
            normalize=self.normalize_coords
        )
        
        # Move the CoordConv model to the correct device manually
        # since CoordConvUNet2DModel doesn't have a device attribute
        model_loader.model.to(self.config.device)
        
        # Create a custom trainer class to handle the CoordConv model
        class CoordTrainer(DiffusionTrainer):
            def __init__(self, model_loader, config, output_dir):
                self.model_loader = model_loader
                self.config = config
                self.output_dir = output_dir
                self.device = config.device
                
                # Get the model from the loader
                self.model = model_loader.model
                
                # Skip the device check that would cause an error
                
                # Get necessary parameters
                self.noise_scheduler = None
                self.optimizer = None
                self.scaler = None  # For mixed precision training
                
                # Load noise scheduler
                self.load_scheduler()
                
            def save_model(self, output_dir):
                """
                Override the save_model method to handle CoordConvUNet2DModel.
                
                Args:
                    output_dir: Directory to save the model
                """
                # Create output directory if it doesn't exist
                os.makedirs(output_dir, exist_ok=True)
                
                # Get the model (unwrap if it's in DataParallel)
                model_to_save = self.model.module if hasattr(self.model, "module") else self.model
                
                # Check if it's a CoordConvUNet2DModel and use its save method
                if hasattr(model_to_save, 'save_pretrained'):
                    model_to_save.save_pretrained(output_dir)
                else:
                    # Fallback for other model types
                    torch.save(model_to_save.state_dict(), os.path.join(output_dir, "model.pt"))
                
                # Save scheduler config
                self.noise_scheduler.save_pretrained(output_dir)
                
                # Save training info
                training_info = {
                    "model_id": self.config.model_id,
                    "trained_epochs": "CoordConv Model"
                }
                
                # Save training info
                with open(os.path.join(output_dir, "training_info.txt"), "w") as f:
                    for key, value in training_info.items():
                        f.write(f"{key}: {value}\n")
                
                print(f"Saved model to {output_dir}")
        
        # Create trainer with our custom class
        trainer = CoordTrainer(
            model_loader=model_loader,
            config=self.config,
            output_dir=self.coordconv_model_dir
        )
        
        # Prepare dataset
        dataloader = self.prepare_dataset()
        
        # Train model
        start_time = time.time()
        trainer.train(
            dataloader=dataloader,
            num_epochs=self.num_epochs,
            learning_rate=self.learning_rate
        )
        training_time = time.time() - start_time
        
        # Save training metrics
        with open(os.path.join(self.coordconv_model_dir, "training_metrics.txt"), "w") as f:
            f.write(f"Training time: {training_time:.2f} seconds\n")
        
        print(f"CoordConv model training completed in {training_time:.2f} seconds")
        return model_loader, training_time
    
    def generate_samples(self, num_samples=10):
        """Generate samples from both models for comparison."""
        print("\n=== Generating Samples for Comparison ===")
        
        # Check if standard model exists or if we're skipping training
        if not os.path.exists(os.path.join(self.standard_model_dir, "final")):
            print(f"Standard model directory {os.path.join(self.standard_model_dir, 'final')} not found.")
            print("Using base model for standard samples generation instead.")
            
            # Generate samples directly with the base model
            config = DiffusionConfig(
                model_id=self.model_id,
                batch_size=1,
                num_inference_steps=100,
                seed=42
            )
            
            # Create directories
            os.makedirs(self.standard_samples_dir, exist_ok=True)
            
            # Load model
            model_loader = DiffusionModelLoader(config)
            model_loader.load_model()
            
            # Create processor and visualizer
            processor = DiffusionProcessor(model_loader, config)
            visualizer = DiffusionVisualizer(model_loader)
            
            # Generate samples
            for i in range(num_samples):
                print(f"Generating standard sample {i+1}/{num_samples}...")
                # Use a different seed for each sample
                config.seed = 42 + i
                samples, _ = processor.generate_sample(save_intermediates=False)
                
                # Save sample
                visualizer.visualize_final_image(
                    samples,
                    save_path=os.path.join(self.standard_samples_dir, f"sample_{i+1}.png")
                )
        else:
            # Generate samples from standard model
            print("Generating samples from standard model...")
            sample_from_coord_model(
                model_path=os.path.join(self.standard_model_dir, "final"),
                num_samples=num_samples,
                with_r=False,
                output_dir=self.standard_samples_dir
            )
        
        # Check if CoordConv model exists or if we're skipping training
        if not os.path.exists(os.path.join(self.coordconv_model_dir, "final")):
            print(f"CoordConv model directory {os.path.join(self.coordconv_model_dir, 'final')} not found.")
            print("Using base model with CoordConv conversion for CoordConv samples generation instead.")
            
            # Generate samples with a converted base model
            config = DiffusionConfig(
                model_id=self.model_id,
                batch_size=1,
                num_inference_steps=100,
                seed=42
            )
            
            # Create directories
            os.makedirs(self.coordconv_samples_dir, exist_ok=True)
            
            # Load model and convert to CoordConv
            model_loader = DiffusionModelLoader(config)
            model_loader.load_model()
            
            # Convert model to CoordConv
            original_model = model_loader.model
            model_loader.model = convert_to_coordconv(
                original_model,
                with_r=self.with_r,
                normalize=self.normalize_coords
            )
            
            # Create processor and visualizer
            processor = DiffusionProcessor(model_loader, config)
            visualizer = DiffusionVisualizer(model_loader)
            
            # Generate samples
            for i in range(num_samples):
                print(f"Generating CoordConv sample {i+1}/{num_samples}...")
                # Use a different seed for each sample
                config.seed = 42 + i
                samples, _ = processor.generate_sample(save_intermediates=False)
                
                # Save sample
                visualizer.visualize_final_image(
                    samples,
                    save_path=os.path.join(self.coordconv_samples_dir, f"sample_{i+1}.png")
                )
        else:
            # Generate samples from CoordConv model
            print("Generating samples from CoordConv model...")
            sample_from_coord_model(
                model_path=os.path.join(self.coordconv_model_dir, "final"),
                num_samples=num_samples,
                with_r=self.with_r,
                output_dir=self.coordconv_samples_dir
            )
    
    def compare_samples(self):
        """Compare generated samples and create visual comparison."""
        print("\n=== Creating Visual Comparison ===")
        
        # Get samples
        standard_samples = sorted(Path(self.standard_samples_dir).glob("*.png"))
        coordconv_samples = sorted(Path(self.coordconv_samples_dir).glob("*.png"))
        
        if not standard_samples or not coordconv_samples:
            print("No samples found for comparison")
            return
        
        # Create comparison grid
        num_samples = min(len(standard_samples), len(coordconv_samples))
        fig, axes = plt.subplots(2, num_samples, figsize=(num_samples * 3, 6))
        
        for i in range(num_samples):
            # Load images
            std_img = Image.open(standard_samples[i])
            coord_img = Image.open(coordconv_samples[i])
            
            # Plot images
            axes[0, i].imshow(np.array(std_img))
            axes[0, i].set_title(f"Standard {i+1}")
            axes[0, i].axis("off")
            
            axes[1, i].imshow(np.array(coord_img))
            axes[1, i].set_title(f"CoordConv {i+1}")
            axes[1, i].axis("off")
        
        # Add super titles for rows
        fig.text(0.5, 0.95, "Standard UNet vs CoordConv UNet", ha="center", fontsize=16)
        fig.text(0.5, 0.05, "CoordConv Parameters: with_r=" + str(self.with_r) + ", normalize=" + str(self.normalize_coords), ha="center")
        
        # Save comparison
        comparison_path = os.path.join(self.output_dir, "comparison.png")
        plt.tight_layout()
        plt.savefig(comparison_path)
        plt.close()
        
        print(f"Visual comparison saved to {comparison_path}")
    
    def run_test(self, skip_training=False):
        """Run the full test suite."""
        # Create test output directories
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.standard_model_dir, exist_ok=True)
        os.makedirs(self.coordconv_model_dir, exist_ok=True)
        os.makedirs(self.standard_samples_dir, exist_ok=True)
        os.makedirs(self.coordconv_samples_dir, exist_ok=True)
        
        if not skip_training:
            # Train standard model
            standard_model_loader, standard_time = self.train_standard_model()
            
            # Train CoordConv model
            coordconv_model_loader, coordconv_time = self.train_coordconv_model()
            
            # Print training time comparison
            print("\n=== Training Time Comparison ===")
            print(f"Standard model: {standard_time:.2f} seconds")
            print(f"CoordConv model: {coordconv_time:.2f} seconds")
            print(f"Difference: {coordconv_time - standard_time:.2f} seconds")
            print(f"CoordConv training time is {(coordconv_time / standard_time) * 100:.2f}% of standard model time")
        else:
            print("\n=== Skipping Training Phase ===")
        
        # Generate samples
        self.generate_samples()
        
        # Compare samples
        self.compare_samples()
        
        print("\n=== Test Completed ===")
        print(f"Results saved to {self.output_dir}")
        
        return {
            "standard_model_dir": self.standard_model_dir,
            "coordconv_model_dir": self.coordconv_model_dir,
            "standard_samples_dir": self.standard_samples_dir,
            "coordconv_samples_dir": self.coordconv_samples_dir
        }


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Test CoordConv in diffusion models")
    parser.add_argument(
        "--dataset_path", type=str, required=True,
        help="Path to the dataset for training"
    )
    parser.add_argument(
        "--output_dir", type=str, default="outputs/coordconv_test",
        help="Directory to save testing results"
    )
    parser.add_argument(
        "--model_id", type=str, default="google/ddpm-celebahq-256",
        help="HuggingFace model ID to use as base model"
    )
    parser.add_argument(
        "--batch_size", type=int, default=8,
        help="Batch size for training"
    )
    parser.add_argument(
        "--num_epochs", type=int, default=10,
        help="Number of epochs to train"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-5,
        help="Learning rate for training"
    )
    parser.add_argument(
        "--with_r", action="store_true",
        help="Whether to include radius channel in CoordConv"
    )
    parser.add_argument(
        "--skip_training", action="store_true",
        help="Skip training and just generate samples from existing models"
    )
    parser.add_argument(
        "--image_size", type=int, default=64,
        help="Size of images to use for training"
    )
    return parser.parse_args()


def main():
    """Run the CoordConv test."""
    args = parse_args()
    
    tester = CoordConvTester(
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        model_id=args.model_id,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        with_r=args.with_r,
        image_size=args.image_size
    )
    
    tester.run_test(skip_training=args.skip_training)


if __name__ == "__main__":
    main() 