import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Union, List
from diffusers import UNet2DModel


class AddCoordinateChannels(nn.Module):
    """
    Transform that adds coordinate channels to images based on the CoordConv paper
    (https://arxiv.org/abs/1807.03247)
    """
    
    def __init__(self, 
                 with_r: bool = False, 
                 normalize: bool = True,
                 coord_size: Optional[Tuple[int, int]] = None):
        """
        Initialize the coordinate channel adder.
        
        Args:
            with_r: Whether to include a radius channel
            normalize: Whether to normalize coordinate values to [-1, 1]
            coord_size: Size of the coordinate grid (height, width), if None will use input size
        """
        super().__init__()
        self.with_r = with_r
        self.normalize = normalize
        self.coord_size = coord_size
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add coordinate channels to the input tensor.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Tensor with added coordinate channels (B, C+2(+1), H, W)
        """
        batch_size, _, height, width = x.size()
        
        # Use specified size or input size
        h, w = self.coord_size if self.coord_size else (height, width)
        
        # Create coordinate grid
        y_coords = torch.linspace(-1 if self.normalize else 0, 
                                 1 if self.normalize else h-1, 
                                 h)
        x_coords = torch.linspace(-1 if self.normalize else 0, 
                                 1 if self.normalize else w-1, 
                                 w)
        
        # Meshgrid to create 2D coordinate grid
        y_coords, x_coords = torch.meshgrid(y_coords, x_coords, indexing='ij')
        
        # Add batch dimension and move to same device as input
        y_coords = y_coords.unsqueeze(0).repeat(batch_size, 1, 1).unsqueeze(1).to(x.device)
        x_coords = x_coords.unsqueeze(0).repeat(batch_size, 1, 1).unsqueeze(1).to(x.device)
        
        # Resize to match input if needed
        if (h, w) != (height, width):
            y_coords = F.interpolate(y_coords, size=(height, width), mode='bilinear', align_corners=True)
            x_coords = F.interpolate(x_coords, size=(height, width), mode='bilinear', align_corners=True)
        
        # Concatenate coordinate channels with input
        out = torch.cat([x, y_coords, x_coords], dim=1)
        
        # Add radius channel if requested
        if self.with_r:
            r = torch.sqrt(y_coords**2 + x_coords**2)  # Euclidean distance from center
            r = r / torch.max(r) if self.normalize else r  # Normalize if requested
            out = torch.cat([out, r], dim=1)
            
        return out


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
        Save both the wrapped model and the coordinate channel mapper.
        
        Args:
            save_directory: Directory to save the model
        """
        # Save UNet
        self.unet.save_pretrained(save_directory)
        
        # Save the channel mapper separately
        mapper_path = f"{save_directory}/channel_mapper.pt"
        torch.save(self.channel_mapper.state_dict(), mapper_path)
        
        # Save additional config
        with open(f"{save_directory}/coord_config.py", "w") as f:
            f.write(f"ADDITIONAL_CHANNELS = {self.additional_channels}\n")
            f.write(f"WITH_R = {self.coord_adder.with_r}\n")
            f.write(f"NORMALIZE = {self.coord_adder.normalize}\n")
    
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
        # Load UNet
        unet = UNet2DModel.from_pretrained(pretrained_model_path)
        
        # Create instance
        model = cls(unet, with_r=with_r, normalize=normalize)
        
        # Try to load channel mapper if it exists
        mapper_path = f"{pretrained_model_path}/channel_mapper.pt"
        try:
            model.channel_mapper.load_state_dict(torch.load(mapper_path))
        except FileNotFoundError:
            print(f"Channel mapper weights not found at {mapper_path}, using initialized weights")
        
        return model


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