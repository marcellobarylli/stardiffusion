# StarCraft Map Diffusion Project

This project fine-tunes diffusion models to generate StarCraft-style maps.

## Project Structure

- `training/`: Scripts for training and fine-tuning models
  - `train_starcraft.py`: Fine-tunes a diffusion model on the StarCraft map dataset
  - `save_model_as_pipeline.py`: Utility to convert models to pipeline format
  
- `generating/`: Scripts for generating samples from trained models
  - `sample_components.py`: Main script for generating samples (component-based approach)
  - `sample_starcraft.py`: Legacy sample generation script (pipeline-based approach)
  - `generate_starcraft_maps.py`: Legacy batch generation script

- `models/`: Model implementations and utilities
  - `diffusion/`: Core diffusion model code
  - `coord_conv/`: CoordConv implementation

- `data/`: Datasets and data processing utilities
  - `StarCraft_Map_Dataset/`: Dataset of StarCraft maps

## Getting Started

### Fine-tuning a Model

To fine-tune a diffusion model on StarCraft maps:

```bash
python training/train_starcraft.py --num_epochs 20 \
  --batch_size 8 \
  --save_interval 5 \
  --output_dir checkpoints/starcraft_fine_tuned \
  --learning_rate 1e-5 \
  --mixed_precision
```

### Generating Samples

To generate samples from a fine-tuned model:

```bash
python generating/sample_components.py \
  --model_path checkpoints/starcraft_fine_tuned/final \
  --output_dir outputs/starcraft_samples \
  --num_samples 5 \
  --num_steps 100
```