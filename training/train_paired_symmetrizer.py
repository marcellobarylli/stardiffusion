import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
import argparse
import logging
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import Adam

# Add the project root to sys.path
project_root = Path(__file__).absolute().parent.parent
sys.path.append(str(project_root))

from models.coord_conv.paired_symmetrizer import PairedSymmetrizer
from models.coord_conv.symmetrize import SymmetryPairedDataset, symmetrize_dataset, visualize_pairs

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a paired symmetrizer model using pairs of original and symmetrized images")
    
    # Data parameters
    parser.add_argument("--dataset_path", type=str, required=True,
                      help="Path to the dataset directory with images")
    parser.add_argument("--output_dir", type=str, default="checkpoints/paired_symmetrizer",
                      help="Directory to save model checkpoints and training progress")
    parser.add_argument("--symmetrized_dataset_path", type=str, default=None,
                      help="Path to pre-created symmetrized dataset (if None, will create on the fly)")
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=8,
                      help="Batch size for training")
    parser.add_argument("--image_size", type=int, default=128,
                      help="Size of images for training")
    parser.add_argument("--num_epochs", type=int, default=50,
                      help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=2e-4,
                      help="Learning rate for optimizer")
    parser.add_argument("--save_interval", type=int, default=5,
                      help="Save model checkpoint every N epochs")
    parser.add_argument("--visualize_interval", type=int, default=2,
                      help="Visualize progress every N epochs")
    
    # Model parameters
    parser.add_argument("--hidden_dim", type=int, default=64,
                      help="Hidden dimension size")
    parser.add_argument("--with_r", action="store_true",
                      help="Include radius coordinate channel")
    parser.add_argument("--symmetry_type", type=str, default="vertical", 
                      choices=["vertical", "horizontal", "both"],
                      help="Type of symmetry to learn")
    parser.add_argument("--symmetry_weight", type=float, default=0.3,
                      help="Weight of symmetry loss in total loss")
    parser.add_argument("--recon_weight", type=float, default=0.7,
                      help="Weight of reconstruction loss in total loss")
    
    # Dataset creation 
    parser.add_argument("--create_dataset", action="store_true",
                      help="Create a symmetrized dataset before training")
    parser.add_argument("--create_dataset_only", action="store_true",
                      help="Only create the dataset, don't train")
    
    # Device
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                      help="Device to train on (default: auto-detect)")
    
    return parser.parse_args()


def train_paired_symmetrizer(
    model,
    dataloader,
    num_epochs,
    learning_rate,
    device,
    save_dir,
    save_interval,
    visualize_interval,
    symmetry_weight,
    recon_weight
):
    """
    Train a paired symmetrizer model.
    
    Args:
        model: The model to train
        dataloader: DataLoader with training data
        num_epochs: Number of training epochs
        learning_rate: Learning rate for the optimizer
        device: Device to train on
        save_dir: Directory to save checkpoints
        save_interval: Save model every N epochs
        visualize_interval: Visualize progress every N epochs
        symmetry_weight: Weight of symmetry loss in total loss
        recon_weight: Weight of reconstruction loss in total loss
    
    Returns:
        Trained model
    """
    # Create optimizer
    optimizer = Adam(model.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    
    # Create directories
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, "progress"), exist_ok=True)
    
    # Training history
    history = {
        "total_loss": [],
        "recon_loss": [],
        "symmetry_loss": []
    }
    
    logger.info(f"Starting training for {num_epochs} epochs")
    logger.info(f"Training on {device}")
    logger.info(f"Dataset size: {len(dataloader.dataset)} images")
    logger.info(f"Batch size: {dataloader.batch_size}")
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        epoch_total_loss = 0
        epoch_recon_loss = 0
        epoch_symmetry_loss = 0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch in progress_bar:
            inputs = batch["input"].to(device)
            targets = batch["target"].to(device)
            
            # Forward pass
            outputs = model(inputs)
            
            # Calculate losses
            # Ensure targets have the right shape (could be [B, 1, C, H, W] instead of [B, C, H, W])
            if len(targets.shape) == 5:
                targets = targets.squeeze(1)
            recon_loss = F.mse_loss(outputs, targets)
            symmetry_loss = model.calculate_symmetry_loss(outputs)
            
            # Weighted sum of losses
            total_loss = recon_weight * recon_loss + symmetry_weight * symmetry_loss
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            
            # Update weights
            optimizer.step()
            
            # Track losses
            epoch_total_loss += total_loss.item()
            epoch_recon_loss += recon_loss.item()
            epoch_symmetry_loss += symmetry_loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                "total_loss": total_loss.item(),
                "recon_loss": recon_loss.item(),
                "sym_loss": symmetry_loss.item()
            })
        
        # Calculate average losses for epoch
        avg_total_loss = epoch_total_loss / len(dataloader)
        avg_recon_loss = epoch_recon_loss / len(dataloader)
        avg_symmetry_loss = epoch_symmetry_loss / len(dataloader)
        
        # Record history
        history["total_loss"].append(avg_total_loss)
        history["recon_loss"].append(avg_recon_loss)
        history["symmetry_loss"].append(avg_symmetry_loss)
        
        # Log progress
        logger.info(f"Epoch {epoch+1}/{num_epochs}: "
                   f"Total Loss = {avg_total_loss:.6f}, "
                   f"Recon Loss = {avg_recon_loss:.6f}, "
                   f"Symmetry Loss = {avg_symmetry_loss:.6f}")
        
        # Visualize progress
        if (epoch + 1) % visualize_interval == 0 or epoch == 0 or epoch == num_epochs - 1:
            model.eval()
            
            # Get a few samples from the dataset
            samples = []
            for i, batch in enumerate(dataloader):
                samples.append(batch)
                if i >= 0:  # Just get the first batch
                    break
            
            sample_batch = samples[0]
            inputs = sample_batch["input"].to(device)
            targets = sample_batch["target"].to(device)
            
            # Get outputs
            with torch.no_grad():
                outputs = model(inputs)
            
            # Visualize
            vis_samples = min(4, inputs.shape[0])
            model.visualize_results(
                inputs=inputs[:vis_samples],
                targets=targets[:vis_samples],
                outputs=outputs[:vis_samples],
                num_samples=vis_samples,
                show=False,
                save_path=os.path.join(save_dir, "progress", f"samples_epoch_{epoch+1}.png")
            )
            
            # Plot loss curves
            plt.figure(figsize=(12, 4))
            
            plt.subplot(1, 3, 1)
            plt.plot(history["total_loss"])
            plt.title("Total Loss")
            plt.xlabel("Epoch")
            plt.grid(True)
            
            plt.subplot(1, 3, 2)
            plt.plot(history["recon_loss"])
            plt.title("Reconstruction Loss")
            plt.xlabel("Epoch")
            plt.grid(True)
            
            plt.subplot(1, 3, 3)
            plt.plot(history["symmetry_loss"])
            plt.title("Symmetry Loss")
            plt.xlabel("Epoch")
            plt.grid(True)
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, "progress", f"losses_epoch_{epoch+1}.png"))
            plt.close()
        
        # Save checkpoint
        if (epoch + 1) % save_interval == 0 or epoch == num_epochs - 1:
            checkpoint_dir = os.path.join(save_dir, f"checkpoint_{epoch+1}")
            model.save_pretrained(checkpoint_dir)
            logger.info(f"Saved checkpoint at epoch {epoch+1} to {checkpoint_dir}")
    
    # Save final model
    model.save_pretrained(os.path.join(save_dir, "final"))
    logger.info(f"Training complete! Final model saved to {os.path.join(save_dir, 'final')}")
    
    return model


def main():
    # Parse command-line arguments
    args = parse_args()
    
    # Set device
    device = torch.device(args.device)
    
    # Dataset paths
    dataset_path = args.dataset_path
    symmetrized_dataset_path = args.symmetrized_dataset_path or os.path.join(args.output_dir, "symmetrized_dataset")
    
    # Create symmetrized dataset if requested
    if args.create_dataset or args.symmetrized_dataset_path is None:
        logger.info(f"Creating symmetrized dataset at {symmetrized_dataset_path}")
        symmetrize_dataset(
            dataset_path=dataset_path,
            output_path=symmetrized_dataset_path,
            symmetry_type=args.symmetry_type,
            size=args.image_size
        )
        
        # Visualize some samples from the created dataset
        dataset = SymmetryPairedDataset(
            dataset_path=symmetrized_dataset_path,
            image_size=args.image_size,
            create_pairs=False
        )
        visualize_pairs(
            dataset=dataset,
            num_samples=4,
            output_path=os.path.join(symmetrized_dataset_path, "sample_pairs.png")
        )
        
        if args.create_dataset_only:
            logger.info("Dataset creation complete. Exiting as --create_dataset_only was specified.")
            return
    
    # Create model
    model = PairedSymmetrizer(
        in_channels=3,
        hidden_dim=args.hidden_dim,
        with_r=args.with_r,
        normalize=True,
        symmetry_type=args.symmetry_type,
        symmetry_weight=args.symmetry_weight
    ).to(device)
    
    logger.info(f"Created PairedSymmetrizer with config:")
    logger.info(f"  Hidden dimension: {args.hidden_dim}")
    logger.info(f"  With radius: {args.with_r}")
    logger.info(f"  Symmetry type: {args.symmetry_type}")
    logger.info(f"  Symmetry weight: {args.symmetry_weight}")
    logger.info(f"  Reconstruction weight: {args.recon_weight}")
    
    # Create dataset
    if args.symmetrized_dataset_path is None and not args.create_dataset:
        # Create pairs on the fly
        logger.info(f"Creating pairs on the fly from dataset: {dataset_path}")
        dataset = SymmetryPairedDataset(
            dataset_path=dataset_path,
            image_size=args.image_size,
            symmetry_type=args.symmetry_type,
            create_pairs=True
        )
    else:
        # Use pre-created pairs
        logger.info(f"Using pre-created pairs from: {symmetrized_dataset_path}")
        dataset = SymmetryPairedDataset(
            dataset_path=symmetrized_dataset_path,
            image_size=args.image_size,
            create_pairs=False
        )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    # Train model
    trained_model = train_paired_symmetrizer(
        model=model,
        dataloader=dataloader,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        device=device,
        save_dir=args.output_dir,
        save_interval=args.save_interval,
        visualize_interval=args.visualize_interval,
        symmetry_weight=args.symmetry_weight,
        recon_weight=args.recon_weight
    )
    
    # Final evaluation on a few test samples
    model.eval()
    test_samples = next(iter(DataLoader(dataset, batch_size=8, shuffle=True)))
    inputs = test_samples["input"].to(device)
    targets = test_samples["target"].to(device)
    
    with torch.no_grad():
        outputs = model(inputs)
        symmetry_scores = [model.calculate_symmetry_loss(outputs[i:i+1]).item() for i in range(len(outputs))]
    
    logger.info(f"Final evaluation on {len(inputs)} samples:")
    logger.info(f"Average symmetry score: {sum(symmetry_scores) / len(symmetry_scores):.6f}")
    
    # Save final visualization
    model.visualize_results(
        inputs=inputs,
        targets=targets,
        outputs=outputs,
        num_samples=min(8, len(inputs)),
        show=False,
        save_path=os.path.join(args.output_dir, "final_results.png")
    )


if __name__ == "__main__":
    main() 