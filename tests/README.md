# CoordConv Tests for StarDiffusion

This directory contains tests to evaluate the impact of CoordConv layers on diffusion models for image generation.

## Test Overview

The test suite includes two main components:

1. **Performance Test**: Trains both a standard UNet and a CoordConv UNet with identical hyperparameters on the same dataset, then compares their performance in generating images.

2. **Spatial Awareness Test**: Analyzes the generated images to determine if CoordConv enhances the model's ability to capture spatial relationships, which is a key benefit claimed in the original CoordConv paper.

## Running Tests

All commands need to be run from the project root directory with PYTHONPATH set:

```bash
PYTHONPATH=. python -m tests.run_coordconv_tests <command> [options]
```

### Available Commands

- `all`: Run both performance and spatial awareness tests in sequence
- `performance`: Run only the performance comparison test
- `spatial`: Run only the spatial awareness analysis

### Example Usage

#### Running All Tests

```bash
PYTHONPATH=. python -m tests.run_coordconv_tests all \
  --dataset_path data/StarCraft_Map_Dataset \
  --output_dir outputs/coordconv_tests \
  --num_epochs 10 \
  --with_r
```

#### Running Performance Test Only

```bash
PYTHONPATH=. python -m tests.run_coordconv_tests performance \
  --dataset_path data/StarCraft_Map_Dataset \
  --output_dir outputs/coordconv_performance \
  --num_epochs 10 \
  --with_r
```

#### Running Spatial Awareness Test Only

```bash
PYTHONPATH=. python -m tests.run_coordconv_tests spatial \
  --standard_samples_dir outputs/coordconv_performance/standard_samples \
  --coordconv_samples_dir outputs/coordconv_performance/coordconv_samples \
  --output_dir outputs/spatial_analysis \
  --with_r
```

### Important Parameters

- `--dataset_path`: Path to the dataset for training models
- `--output_dir`: Directory to save test results
- `--model_id`: HuggingFace diffusion model ID (default: "google/ddpm-celebahq-256")
- `--batch_size`: Batch size for training
- `--num_epochs`: Number of epochs for training
- `--with_r`: Include radius channel in CoordConv (flag, no value needed)
- `--skip_training`: Skip training and just generate samples/analysis

## Test Outputs

The tests generate various outputs to help understand the impact of CoordConv:

### Performance Test Outputs

- Trained model checkpoints for both standard and CoordConv UNet
- Generated image samples from both models
- Visual comparison of generated samples
- Training time metrics

### Spatial Awareness Test Outputs

- Edge detection visualizations showing structural differences
- Quadrant analysis showing content distribution
- Metrics comparing spatial properties:
  - Edge density (measuring detail level)
  - Edge distance from center (measuring centrality)
  - Quadrant variance (measuring evenness of distribution)
- Comprehensive analysis report

## Requirements

These tests require additional packages beyond the main requirements:
- scikit-image
- opencv-python
- matplotlib
- numpy
- pillow

You can install them with:

```bash
pip install scikit-image opencv-python matplotlib
``` 