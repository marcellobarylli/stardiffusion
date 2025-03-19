import subprocess
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Run the symmetrizer on CelebA dataset")
    parser.add_argument("--image_size", type=int, default=256, help="Size of images for training")
    parser.add_argument("--symmetry_type", type=str, default="vertical", 
                      choices=["vertical", "horizontal", "both"], help="Type of symmetry to learn")
    parser.add_argument("--symmetry_weight", type=float, default=0.8, help="Weight of symmetry loss")
    parser.add_argument("--recon_weight", type=float, default=0.2, help="Weight of reconstruction loss")
    parser.add_argument("--hidden_dim", type=int, default=64, help="Hidden dimension size")
    parser.add_argument("--num_epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--processed_dir", type=str, default="data/celeba_processed", 
                      help="Directory with processed CelebA images")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Check if processed directory exists
    if not os.path.exists(args.processed_dir):
        print(f"Processed directory {args.processed_dir} not found.")
        print("Running preprocessing script first...")
        subprocess.run(["python", "preprocess_celeba.py"])
    
    # Run the training script
    output_dir = f"checkpoints/celeba_symmetrizer_{args.symmetry_type}"
    
    command = [
        "python", "training/train_paired_symmetrizer.py",
        "--dataset_path", args.processed_dir,
        "--output_dir", output_dir,
        "--create_dataset",
        "--image_size", str(args.image_size),
        "--symmetry_type", args.symmetry_type,
        "--symmetry_weight", str(args.symmetry_weight),
        "--recon_weight", str(args.recon_weight),
        "--hidden_dim", str(args.hidden_dim),
        "--num_epochs", str(args.num_epochs),
        "--batch_size", str(args.batch_size)
    ]
    
    print("Running symmetrizer with the following command:")
    print(" ".join(command))
    
    subprocess.run(command)

if __name__ == "__main__":
    main() 