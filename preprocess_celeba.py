import os
import argparse
import requests
import zipfile
from torchvision.datasets import CelebA
from torchvision import transforms
from pathlib import Path
import shutil
from tqdm import tqdm

def download_file(url, dest_path):
    """Download a file with progress bar"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(dest_path, 'wb') as file, tqdm(
        desc=os.path.basename(dest_path),
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)

def download_extract_celeba_alternative(root_dir):
    """Alternative download method for CelebA dataset"""
    
    # Instead of using Google Drive link via torchvision, use Academic Torrents
    # First check if the user wants to proceed
    print("\n===== CelebA DATASET ALTERNATIVE DOWNLOAD =====")
    print("The standard download method through torchvision is failing due to Google Drive rate limits.")
    print("Options:")
    print("1. Download manually from the official site: https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html")
    print("   - Download the 'Align&Cropped Images' zip file")
    print("   - Extract it to 'data/celeba/img_align_celeba'")
    print("   - Then run this script again")
    print("")
    print("2. Use a smaller face dataset sample (1,000 images) that we can download directly (faster)")
    
    option = input("Enter option (1 or 2): ")
    
    if option == "1":
        print("\nPlease follow these steps:")
        print("1. Visit https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html")
        print("2. Download the 'Align&Cropped Images' zip file")
        print("3. Create folder: data/celeba/")
        print("4. Extract the zip to: data/celeba/img_align_celeba/")
        print("5. Run this script again")
        return False
    
    elif option == "2":
        # Sample from FFHQ dataset (1000 images)
        print("\nDownloading FFHQ face dataset sample (1,000 images)...")
        os.makedirs(root_dir, exist_ok=True)
        
        # Reliable direct download link to a sample of FFHQ (1000 images, ~130MB)
        ffhq_url = "https://github.com/NVlabs/ffhq-dataset/releases/download/v1.0/ffhq-dataset-v1-1k.zip"
        zip_path = os.path.join(root_dir, "ffhq-1k.zip")
        
        # Download the zip file
        print(f"Downloading FFHQ sample to {zip_path}...")
        try:
            download_file(ffhq_url, zip_path)
        except Exception as e:
            print(f"Download failed with error: {e}")
            print("Please try the manual download method instead.")
            return False
        
        # Extract the zip file
        print(f"Extracting zip file to {root_dir}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(root_dir)
        
        # Create the expected directory structure
        source_dir = os.path.join(root_dir, "images1024x1024")
        target_dir = os.path.join(root_dir, "img_align_celeba")
        
        if not os.path.exists(source_dir):
            # Try to find the correct directory
            for root, dirs, files in os.walk(root_dir):
                if any(f.endswith('.png') for f in files):
                    source_dir = root
                    break
        
        os.makedirs(target_dir, exist_ok=True)
        
        # Copy and rename files
        print(f"Processing images to {target_dir}...")
        count = 0
        for file in os.listdir(source_dir):
            if file.endswith('.png') or file.endswith('.jpg'):
                src_path = os.path.join(source_dir, file)
                dst_path = os.path.join(target_dir, f"face_{count:06d}.jpg")
                shutil.copy(src_path, dst_path)
                count += 1
        
        print(f"Successfully processed {count} face images")
        return True
    
    else:
        print("Invalid option.")
        return False

def preprocess_celeba(img_dir, output_dir, max_images=10000):
    """Process images from the source directory to the output directory"""
    if not os.path.exists(img_dir):
        print(f"Error: Image directory {img_dir} not found.")
        return False
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Process images
    count = 0
    print(f"Processing images from {img_dir} to {output_dir}...")
    
    # Get list of image files
    image_files = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    
    for filename in tqdm(image_files):
        src_path = os.path.join(img_dir, filename)
        dst_path = os.path.join(output_dir, f"face_{count:06d}.jpg")
        shutil.copy(src_path, dst_path)
        
        count += 1
        if max_images is not None and count >= max_images:
            break
    
    print(f"Completed! {count} images saved to {output_dir}")
    return True

def main():
    parser = argparse.ArgumentParser(description="Preprocess CelebA dataset for symmetrizer")
    parser.add_argument("--max_images", type=int, default=10000, 
                      help="Maximum number of images to process (default: 10000, use 0 for all)")
    parser.add_argument("--manual_path", type=str, default=None,
                      help="Path to manually downloaded CelebA images (skips download)")
    args = parser.parse_args()
    
    # Set paths
    root_dir = "data/celeba"
    output_dir = Path("data/celeba_processed")
    max_images = args.max_images if args.max_images > 0 else None
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if the user provided a manual path
    if args.manual_path is not None:
        if os.path.exists(args.manual_path):
            img_dir = args.manual_path
            print(f"Using manually provided path: {img_dir}")
            success = True
        else:
            print(f"Error: Provided path {args.manual_path} does not exist.")
            return
    else:
        # First try to see if the dataset is already downloaded
        img_dir = os.path.join(root_dir, "img_align_celeba")
        if not os.path.exists(img_dir):
            print("CelebA dataset not found. Trying alternative download...")
            success = download_extract_celeba_alternative(root_dir)
            if not success:
                return
        else:
            print(f"Found existing CelebA dataset at: {img_dir}")
            success = True
    
    # Process the images
    if success:
        preprocess_celeba(img_dir, output_dir, max_images)
        
        print("\nYou can now run the symmetrizer with:")
        print(f"python training/train_paired_symmetrizer.py --dataset_path {output_dir} --output_dir checkpoints/celeba_symmetrizer --create_dataset --image_size 256 --symmetry_weight 0.8 --recon_weight 0.2")

if __name__ == "__main__":
    main() 