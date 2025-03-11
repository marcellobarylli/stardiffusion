import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from diffusers import DDPMScheduler, UNet2DModel
from models.coord_conv.unet import CoordConvUNet2DModel
from models.coord_conv.layers import AddCoordinateChannels

def sample_from_model(
    model_path="./my_new_model/final",
    num_samples=4,
    with_r=True,
    output_dir="outputs/samples"
):
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading model from {model_path}...")
    
    # Load the model
    model = CoordConvUNet2DModel.from_pretrained(
        model_path,
        with_r=with_r,
        normalize=True
    )
    
    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    # Create a scheduler (use default config since we don't have a saved one)
    scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="linear"
    )
    
    # Get sample size from model config
    sample_size = model.unet.config.sample_size
    
    # Generate random noise
    noise = torch.randn(
        (num_samples, 3, sample_size, sample_size),
        device=device
    )
    
    # Sampling parameters
    num_inference_steps = 100
    
    # Set timesteps
    scheduler.set_timesteps(num_inference_steps)
    
    # Denoising loop
    image = noise
    for t in scheduler.timesteps:
        print(f"Step {t}")
        
        # Add noise to image according to timestep
        with torch.no_grad():
            noisy_residual = model(image, t).sample
            
        # Update sample with scheduler
        image = scheduler.step(noisy_residual, t, image).prev_sample
    
    # Convert to numpy and prepare for visualization
    with torch.no_grad():
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
    
    # Save samples
    for i, img in enumerate(image):
        plt.figure(figsize=(8, 8))
        plt.imshow(img)
        plt.axis("off")
        plt.savefig(os.path.join(output_dir, f"sample_{i}.png"))
        plt.close()
    
    print(f"Generated {num_samples} samples and saved to {output_dir}")
    return image

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Sample from CoordConv model")
    parser.add_argument("--model_path", type=str, default="./my_new_model/final",
                        help="Path to trained model")
    parser.add_argument("--num_samples", type=int, default=4,
                        help="Number of samples to generate")
    parser.add_argument("--with_r", action="store_true",
                        help="Use radius channel")
    parser.add_argument("--output_dir", type=str, default="outputs/my_new_model_samples",
                        help="Output directory for samples")
    
    args = parser.parse_args()
    
    sample_from_model(
        model_path=args.model_path,
        num_samples=args.num_samples,
        with_r=args.with_r,
        output_dir=args.output_dir
    ) 