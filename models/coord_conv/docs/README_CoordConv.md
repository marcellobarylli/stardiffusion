# CoordConv for Diffusion Models

This implementation adds coordinate channels to diffusion models based on the CoordConv paper ([Liu et al., 2018](https://arxiv.org/abs/1807.03247)). By adding explicit coordinate information as additional input channels, the model can better understand spatial relationships, which can be particularly useful for tasks like image generation where position awareness is important.

## Overview

This CoordConv implementation for diffusion models:

1. Adds x,y coordinate channels to the input images of the UNet model
2. Optionally adds a radius channel (distance from center)
3. Allows for continued fine-tuning of existing diffusion models with CoordConv functionality

## Implementation Details

The implementation consists of three main components:

1. `models/coord_conv/layers.py` and `models/coord_conv/unet.py` - Core implementation of the CoordConv layers and model conversion
2. `models/diffusion/coord.py` - Integration with the diffusion model training pipeline
3. `training/run_starcraft_coordconv.py` - Example script to run training and sampling

### Coordinate Channels

Two coordinate channels are added to the input:
- x-coordinate channel: values range from -1 to 1 (or 0 to width-1 if not normalized)
- y-coordinate channel: values range from -1 to 1 (or 0 to height-1 if not normalized)

An optional third channel can be added:
- radius channel: Euclidean distance from the center of the image

## Usage

**Note:** All scripts need to be run with the Python path set to the project root using `PYTHONPATH=.`

### Converting and Fine-tuning a Model

```bash
PYTHONPATH=. python training/run_starcraft_coordconv.py --mode train \
    --dataset_path data/StarCraft_Map_Dataset \
    --output_dir checkpoints/coord_fine_tuned_models/starcraft_maps_coordconv \
    --batch_size 8 \
    --num_epochs 15 \
    --learning_rate 5e-6 \
    --with_r
```

### Generating Samples from a CoordConv Model

```bash
PYTHONPATH=. python training/run_starcraft_coordconv.py --mode sample \
    --model_path checkpoints/coord_fine_tuned_models/starcraft_maps_coordconv/final \
    --num_samples 4 \
    --with_r
```

### Using the StarCraft Example Script

```bash
# For training
PYTHONPATH=. python training/run_starcraft_coordconv.py --mode train --with_r

# For sampling
PYTHONPATH=. python training/run_starcraft_coordconv.py --mode sample --model_path checkpoints/coord_fine_tuned_models/starcraft_maps_coordconv/final --num_samples 4 --with_r
```

## Command Line Arguments

### Training Arguments

- `--mode`: "train" or "sample"
- `--model_path`: Path to the model to convert/fine-tune or sample from
- `--dataset_path`: Path to the dataset for fine-tuning
- `--output_dir`: Directory to save the fine-tuned model
- `--batch_size`: Batch size for training
- `--num_epochs`: Number of epochs for training
- `--learning_rate`: Learning rate for fine-tuning
- `--save_interval`: Interval for saving checkpoints
- `--visible_gpus`: GPU indices to use

### CoordConv Options

- `--with_r`: Include radius channel (flag)
- `--normalize_coords`: Normalize coordinate values to [-1, 1] (flag)

### Sampling Arguments

- `--num_samples`: Number of samples to generate

## How CoordConv Works

CoordConv adds explicit coordinate information to convolutional neural networks. In traditional CNNs, position information must be implicitly learned, which can be difficult for the network. By providing explicit coordinates as additional input channels, the model can more easily learn location-dependent features.

This is particularly useful for diffusion models where the UNet architecture needs to understand spatial relationships in the image during the denoising process.

## Benefits for StarCraft Map Generation

For StarCraft map generation, using CoordConv can improve:

1. Position awareness - generating appropriate terrain and features based on location
2. Structural coherence - better understanding of distance relationships between map elements
3. Edge handling - better awareness of image boundaries
4. Global pattern understanding - improved ability to create coherent, playable map layouts

## References

Liu, R., Lehman, J., Molino, P., Petroski Such, F., Frank, E., Sergeev, A., & Yosinski, J. (2018). [An intriguing failing of convolutional neural networks and the CoordConv solution](https://arxiv.org/abs/1807.03247). Advances in Neural Information Processing Systems (NeurIPS). 