# StarDiffusion

A modular implementation of diffusion models with CoordConv enhancement for spatial awareness.

## Overview

StarDiffusion is a project that enhances diffusion models with spatial awareness through CoordConv layers. It allows for the generation of images with better spatial relationships, particularly useful for structured content like game maps, layouts, and structured images.

## Project Structure

```
stardiffusion-new/
├── models/                  # Model implementations
│   ├── coord_conv/          # CoordConv implementation
│   │   ├── layers.py        # CoordConv layer implementation
│   │   └── unet.py          # UNet with CoordConv
│   └── diffusion/           # Diffusion model implementations
│       ├── core.py          # Core diffusion functionality
│       └── coord.py         # CoordConv diffusion extensions
├── training/                # Training modules
│   ├── trainer.py           # Diffusion model trainer
│   ├── finetune_starcraft.py
│   └── run_starcraft_coordconv.py
├── inference/               # Inference modules
│   └── generate_starcraft_maps.py
├── data/                    # Data handling
│   ├── datasets/            # Dataset implementations
│   └── preprocessing/       # Data preprocessing
├── utils/                   # Utility functions
├── configs/                 # Configuration files
├── checkpoints/             # Model checkpoints
├── outputs/                 # Generated outputs
├── main.py                  # Main entry point
└── requirements.txt         # Dependencies
```

## Key Features

- **CoordConv Integration**: Adds spatial awareness to diffusion models based on the [CoordConv paper](https://arxiv.org/abs/1807.03247)
- **Modular Design**: Separates models, training, and inference for better code organization
- **Fine-tuning Support**: Built-in support for fine-tuning on custom datasets
- **Visualization Tools**: Utilities to visualize diffusion progression

## Getting Started

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/stardiffusion-new.git
cd stardiffusion-new

# Install dependencies
pip install -r requirements.txt
```

### Running Scripts

All scripts need to be run with the Python path set to the project root:

```bash
# When running the main script
PYTHONPATH=. python main.py [arguments]

# When running module scripts
PYTHONPATH=. python training/run_starcraft_coordconv.py [arguments]
PYTHONPATH=. python inference/generate_starcraft_maps.py [arguments]
```

### Basic Usage

#### Generate Images with a Pretrained Model

```bash
PYTHONPATH=. python main.py generate --model_id google/ddpm-celebahq-256 --output_dir outputs/celebahq
```

#### Fine-tune a Model on a Custom Dataset

```bash
PYTHONPATH=. python main.py finetune --model_id google/ddpm-celebahq-256 --dataset path/to/your/images --output_dir checkpoints/fine_tuned
```

#### Generate Images with a Fine-tuned Model

```bash
PYTHONPATH=. python main.py generate-finetuned --model_path checkpoints/fine_tuned/final --output_dir outputs/fine_tuned_samples
```

### CoordConv Features

#### Generate StarCraft Maps

```bash
PYTHONPATH=. python inference/generate_starcraft_maps.py --model_path checkpoints/fine_tuned_models/starcraft_maps/final --num_images 10
```

#### Fine-tune with CoordConv

```bash
PYTHONPATH=. python training/run_starcraft_coordconv.py --mode train --dataset_path data/StarCraft_Map_Dataset --with_r
```

## Requirements

- Python 3.8+
- PyTorch 1.9+
- Diffusers library
- See `requirements.txt` for a complete list

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The CoordConv paper: [An Intriguing Failing of Convolutional Neural Networks and the CoordConv Solution](https://arxiv.org/abs/1807.03247)
- Hugging Face Diffusers library 