import torch
import torch.nn as nn
from typing import Union
from diffusers import UNet2DModel
from .layers import AddCoordinateChannels, SymmetricCoordinateChannels
import os
import json


class CoordConvUNet2DModel(nn.Module):
    """
    Wrapper for UNet2DModel that adds coordinate channels to the input
    """
    
    def __init__(self, 
                 unet_model: UNet2DModel,
                 with_r: bool = False,
                 normalize: bool = True):
        """
        Initialize the CoordConv UNet wrapper.
        
        Args:
            unet_model: The UNet2DModel to wrap
            with_r: Whether to include a radius channel
            normalize: Whether to normalize coordinate values to [-1, 1]
        """
        super().__init__()
        self.unet = unet_model
        self.coord_adder = AddCoordinateChannels(with_r=with_r, normalize=normalize)
        
        # Calculate number of additional channels
        self.additional_channels = 2 + (1 if with_r else 0)
        
        # Get original in_channels from UNet config
        self.original_in_channels = self.unet.config.in_channels
        
        # Create a new conv layer to handle the additional coordinate channels
        # This maps from original channels + coordinate channels back to original channels
        self.channel_mapper = nn.Conv2d(
            self.original_in_channels + self.additional_channels,
            self.original_in_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )
        
        # Ensure all parameters have the same data type
        # This is the key fix for the mixed precision issue
        model_dtype = next(unet_model.parameters()).dtype
        self.channel_mapper.to(dtype=model_dtype)
        
    def forward(self, 
                sample: torch.FloatTensor,
                timestep: Union[torch.Tensor, float, int],
                **kwargs):
        """
        Forward pass adding coordinate channels.
        
        Args:
            sample: Input tensor
            timestep: The timestep for the diffusion model
            
        Returns:
            Output of the UNet
        """
        # Add coordinate channels to input
        augmented_sample = self.coord_adder(sample)
        
        # Ensure augmented_sample has the same dtype as sample
        if augmented_sample.dtype != sample.dtype:
            augmented_sample = augmented_sample.to(sample.dtype)
        
        # Map back to original number of channels
        mapped_sample = self.channel_mapper(augmented_sample)
        
        # Pass to original UNet
        return self.unet(mapped_sample, timestep, **kwargs)
    
    def save_pretrained(self, save_directory):
        """
        Save the model to a directory.
        
        Args:
            save_directory: Directory to save the model to
        """
        # Create directory if it doesn't exist
        os.makedirs(save_directory, exist_ok=True)
        print(f"Saving CoordConvUNet2DModel to {save_directory}")
        
        # Save the UNet model
        self.unet.save_pretrained(save_directory)
        print(f"Saved UNet component")
        
        # Save the channel mapper
        mapper_path = os.path.join(save_directory, "channel_mapper.pt")
        torch.save(self.channel_mapper.state_dict(), mapper_path)
        print(f"Saved channel mapper to {mapper_path}")
        
        # Save additional info
        info = {
            "with_r": hasattr(self, "coord_adder") and getattr(self.coord_adder, "with_r", False),
            "normalize": hasattr(self, "coord_adder") and getattr(self.coord_adder, "normalize", True),
            "additional_channels": self.additional_channels,
            "original_in_channels": self.original_in_channels
        }
        
        # Save as JSON
        with open(os.path.join(save_directory, "coordconv_config.json"), "w") as f:
            json.dump(info, f, indent=2)
        print(f"Saved CoordConv configuration")
        
        # Return the directory path
        return save_directory
    
    @classmethod
    def from_pretrained(cls, 
                        pretrained_model_path: str,
                        with_r: bool = False,
                        normalize: bool = True):
        """
        Load a pretrained CoordConvUNet2DModel.
        
        Args:
            pretrained_model_path: Path to the pretrained model
            with_r: Whether to include a radius channel
            normalize: Whether to normalize coordinate values
            
        Returns:
            Loaded CoordConvUNet2DModel
        """
        print(f"Loading UNet model from {pretrained_model_path}")
        
        # Load UNet
        try:
            unet = UNet2DModel.from_pretrained(pretrained_model_path)
            print(f"Successfully loaded UNet with config: {unet.config}")
        except Exception as e:
            print(f"Error loading UNet: {e}")
            raise
        
        # Create instance
        model = cls(unet, with_r=with_r, normalize=normalize)
        print(f"Created CoordConvUNet2DModel wrapper with with_r={with_r}, normalize={normalize}")
        
        # Try to load channel mapper if it exists
        mapper_path = os.path.join(pretrained_model_path, "channel_mapper.pt")
        try:
            if os.path.exists(mapper_path):
                print(f"Loading channel mapper from {mapper_path}")
                
                channel_mapper_state = torch.load(mapper_path, map_location="cpu")
                model.channel_mapper.load_state_dict(channel_mapper_state)
                
                print(f"Successfully loaded channel mapper weights")
            else:
                print(f"Channel mapper weights not found at {mapper_path}, using initialized weights")
        except Exception as e:
            print(f"Error loading channel mapper: {e}")
            print("Using initialized weights instead")
        
        return model


class SymmetricCoordConvUNet2DModel(CoordConvUNet2DModel):
    """
    CoordConv UNet model with symmetric coordinate channels
    """
    
    def __init__(self, 
                 unet_model: UNet2DModel,
                 symmetry_type: str = 'vertical',
                 with_r: bool = False,
                 normalize: bool = True):
        """
        Initialize the symmetric CoordConv UNet wrapper.
        
        Args:
            unet_model: The UNet2DModel to wrap
            symmetry_type: Type of symmetry to enforce ('vertical', 'horizontal', 'both')
            with_r: Whether to include a radius channel
            normalize: Whether to normalize coordinate values to [-1, 1]
        """
        # Call parent constructor but we'll replace the coord_adder
        super().__init__(unet_model, with_r=with_r, normalize=normalize)
        
        # Replace standard coordinate adder with symmetric version
        self.coord_adder = SymmetricCoordinateChannels(
            symmetry_type=symmetry_type,
            with_r=with_r, 
            normalize=normalize
        )
        
        # Store symmetry type for config
        self.symmetry_type = symmetry_type
        
    def save_pretrained(self, save_directory):
        """
        Save both the wrapped model and the coordinate channel mapper.
        
        Args:
            save_directory: Directory to save the model
        """
        # Use parent method to save most things
        super().save_pretrained(save_directory)
        
        # Save additional config with symmetry type
        with open(f"{save_directory}/coord_config.py", "a") as f:
            f.write(f"SYMMETRY_TYPE = '{self.symmetry_type}'\n")
    
    @classmethod
    def from_pretrained(cls, 
                        pretrained_model_path: str,
                        symmetry_type: str = 'vertical',
                        with_r: bool = False,
                        normalize: bool = True):
        """
        Load a pretrained SymmetricCoordConvUNet2DModel.
        
        Args:
            pretrained_model_path: Path to the pretrained model
            symmetry_type: Type of symmetry to enforce
            with_r: Whether to include a radius channel
            normalize: Whether to normalize coordinate values
            
        Returns:
            Loaded SymmetricCoordConvUNet2DModel
        """
        # Load config if exists
        config_path = f"{pretrained_model_path}/coord_config.py"
        loaded_symmetry_type = symmetry_type
        
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                for line in f:
                    if line.startswith("SYMMETRY_TYPE"):
                        # Extract symmetry type from config
                        loaded_symmetry_type = line.split("=")[1].strip().strip("'").strip('"')
                        print(f"Loaded symmetry type: {loaded_symmetry_type}")
                        break
            
            if loaded_symmetry_type != symmetry_type:
                print(f"Warning: Using loaded symmetry type '{loaded_symmetry_type}' instead of requested '{symmetry_type}'")
        
        # Load UNet
        unet = UNet2DModel.from_pretrained(pretrained_model_path)
        
        # Create instance with loaded or specified symmetry type
        model = cls(unet, symmetry_type=loaded_symmetry_type, with_r=with_r, normalize=normalize)
        
        # Try to load channel mapper if it exists
        mapper_path = f"{pretrained_model_path}/channel_mapper.pt"
        try:
            model.channel_mapper.load_state_dict(torch.load(mapper_path))
        except FileNotFoundError:
            print(f"Channel mapper weights not found at {mapper_path}, using initialized weights")
        
        return model


# Function to convert a standard model to symmetric CoordConv model
def convert_to_symmetric_coordconv(
    model: UNet2DModel, 
    symmetry_type: str = 'vertical',
    with_r: bool = False, 
    normalize: bool = True
) -> SymmetricCoordConvUNet2DModel:
    """
    Convert a standard UNet2DModel to a SymmetricCoordConvUNet2DModel.
    
    Args:
        model: The UNet2DModel to convert
        symmetry_type: Type of symmetry to enforce ('vertical', 'horizontal', 'both')
        with_r: Whether to include a radius channel
        normalize: Whether to normalize coordinate values
        
    Returns:
        SymmetricCoordConvUNet2DModel
    """
    return SymmetricCoordConvUNet2DModel(
        model, 
        symmetry_type=symmetry_type, 
        with_r=with_r, 
        normalize=normalize
    )


# Utility function to convert a standard model to CoordConv model
def convert_to_coordconv(model: UNet2DModel, 
                         with_r: bool = False, 
                         normalize: bool = True) -> CoordConvUNet2DModel:
    """
    Convert a standard UNet2DModel to a CoordConvUNet2DModel.
    
    Args:
        model: The UNet2DModel to convert
        with_r: Whether to include a radius channel
        normalize: Whether to normalize coordinate values
        
    Returns:
        CoordConvUNet2DModel
    """
    return CoordConvUNet2DModel(model, with_r=with_r, normalize=normalize) 