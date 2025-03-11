import os
import torch
from models.diffusion.coord import sample_from_symmetric_coord_model
from models.diffusion.core import DiffusionConfig, DiffusionModelLoader
from models.coord_conv.unet import convert_to_symmetric_coordconv

def convert_model_example():
    """
    Example of converting a model to a symmetric CoordConv model and saving it.
    """
    # Create output directory
    output_dir = "outputs/symmetric_example"
    os.makedirs(output_dir, exist_ok=True)
    
    # Configuration
    config = DiffusionConfig(
        model_id="google/ddpm-celebahq-256",  # Example base model
        batch_size=1
    )
    
    # Load model
    model_loader = DiffusionModelLoader(config)
    model_loader.load_model()
    
    # Convert to symmetric CoordConv
    symmetric_model = convert_to_symmetric_coordconv(
        model_loader.model,
        symmetry_type="vertical",  # Options: 'vertical', 'horizontal', 'both'
        with_r=True,
        normalize=True
    )
    
    # Save the converted model
    symmetric_model.save_pretrained(os.path.join(output_dir, "symmetric_model"))
    print(f"Saved symmetric model to {os.path.join(output_dir, 'symmetric_model')}")

def generate_symmetric_samples():
    """
    Example of generating samples using a symmetric CoordConv model.
    """
    # Generate with vertical symmetry
    vertical_output = sample_from_symmetric_coord_model(
        model_path="google/ddpm-celebahq-256",  # Can be any model path, will be converted
        num_samples=2,
        symmetry_type="vertical",
        with_r=True,
        output_dir="outputs/symmetric_examples/vertical"
    )
    print(f"Generated vertical symmetric samples in {vertical_output}")
    
    # Generate with horizontal symmetry
    horizontal_output = sample_from_symmetric_coord_model(
        model_path="google/ddpm-celebahq-256",  # Can be any model path, will be converted
        num_samples=2,
        symmetry_type="horizontal",
        with_r=True,
        output_dir="outputs/symmetric_examples/horizontal"
    )
    print(f"Generated horizontal symmetric samples in {horizontal_output}")
    
    # Generate with both symmetries
    both_output = sample_from_symmetric_coord_model(
        model_path="google/ddpm-celebahq-256",  # Can be any model path, will be converted
        num_samples=2,
        symmetry_type="both",
        with_r=True,
        output_dir="outputs/symmetric_examples/both"
    )
    print(f"Generated samples with both symmetries in {both_output}")

def compare_symmetry_types():
    """
    Compare different symmetry types by generating samples with each.
    """
    # Set up common parameters
    model_path = "google/ddpm-celebahq-256"
    seed = 42
    num_samples = 4
    
    # Generate samples with each symmetry type using the same seed
    for symmetry_type in ["vertical", "horizontal", "both"]:
        output_dir = f"outputs/symmetric_comparison/{symmetry_type}"
        
        # Use the same seed for all runs to compare the effect of symmetry type
        torch.manual_seed(seed)
        
        # Generate samples
        sample_from_symmetric_coord_model(
            model_path=model_path,
            num_samples=num_samples,
            symmetry_type=symmetry_type,
            with_r=True,
            output_dir=output_dir
        )
        print(f"Generated {symmetry_type} symmetric samples in {output_dir}")

if __name__ == "__main__":
    # Uncomment the function you want to run
    # convert_model_example()
    # generate_symmetric_samples()
    compare_symmetry_types() 