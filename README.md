# StarCraft Map Diffusion Project

This project fine-tunes diffusion models to generate StarCraft-style maps.

## Project Structure

- `training/`: Scripts for training and fine-tuning models
  - `train_starcraft.py`: Fine-tunes a diffusion model on the StarCraft map dataset
  - `train_starcraft_coordconv.py`: Fine-tunes a diffusion model with CoordConv for better spatial awareness
  - `train_noise_transformer.py`: Trains a transformer that creates symmetric noise 
  - `train_learnable_symmetry.py`: Trains a transformer that learns to create symmetric noise without explicit enforcement
  - `train_end_to_end.py`: End-to-end training of transformer + diffusion model
  - `train_end_to_end_learnable.py`: End-to-end training of learnable symmetry transformer + diffusion model
  - `save_model_as_pipeline.py`: Utility to convert models to pipeline format
  
- `generating/`: Scripts for generating samples from trained models
  - `sample_components.py`: Main script for generating samples (component-based approach)
  - `sample_with_transformer.py`: Generates symmetric samples using the transformer approach
  - `sample_with_learned_symmetry.py`: Generates symmetric samples using the LearnableSymmetryTransformer
  - `demo_transformer.py`: Interactive demo showcasing the symmetric noise transformer
  - `sample_starcraft.py`: Legacy sample generation script (pipeline-based approach)
  - `generate_starcraft_maps.py`: Legacy batch generation script

- `models/`: Model implementations and utilities
  - `diffusion/`: Core diffusion model code
  - `coord_conv/`: CoordConv implementation and symmetric noise transformer
      - `layers.py`: Core CoordConv layer implementations
      - `unet.py`: CoordConv UNet models
      - `transformer.py`: Explicit symmetry enforcement transformer
      - `learning_symmetry.py`: Learnable symmetry transformer that uses loss functions to guide learning

- `data/`: Datasets and data processing utilities
  - `StarCraft_Map_Dataset/`: Dataset of StarCraft maps

## Approaches

This project explores several approaches to generate symmetric StarCraft-style maps:

### 1. Basic Diffusion

A standard diffusion model fine-tuned on StarCraft maps with no symmetry enforcing.

### 2. CoordConv Approach

Integrates coordinate channels directly into the UNet architecture to provide spatial awareness, which can help the model learn symmetries naturally.

### 3. Symmetric Noise Transformer

A two-stage process:
1. A small transformer network that converts random noise into symmetrical noise
2. A standard diffusion model that denoises the symmetric noise

This approach provides good results by enforcing symmetry at the noise level, allowing the diffusion model to focus on generating high-quality features.

### 4. Learnable Symmetry Transformer (Recommended)

Our latest approach improves on the Symmetric Noise Transformer by:
1. Using coordinate channels to provide positional hints, but letting the model learn symmetry naturally
2. Using a symmetry loss function to guide learning instead of explicitly enforcing symmetry
3. Maintaining a learnable skip connection to preserve noise characteristics

This approach better aligns with the CoordConv philosophy by letting the model learn symmetry rather than enforcing it mathematically.

## Getting Started

### Fine-tuning a Basic Model

To fine-tune a diffusion model on StarCraft maps:

```bash
python training/train_starcraft.py --num_epochs 20 \
  --batch_size 8 \
  --save_interval 5 \
  --output_dir checkpoints/starcraft_fine_tuned \
  --learning_rate 1e-5 \
  --mixed_precision
```

### Training with CoordConv

To train with CoordConv for better spatial awareness:

```bash
python training/train_starcraft_coordconv.py \
  --dataset_path data/StarCraft_Map_Dataset \
  --output_dir checkpoints/coordconv_starcraft \
  --batch_size 8 \
  --num_epochs 20 \
  --with_r
```

### Training a Symmetric Noise Transformer

Step a: Train a transformer with explicit symmetry enforcement:

```bash
python training/train_noise_transformer.py \
  --output_dir checkpoints/noise_transformer \
  --batch_size 32 \
  --image_size 256 \
  --num_epochs 500 \
  --learning_rate 1e-4 \
  --symmetry_type vertical \
  --with_r
```

### Training a Learnable Symmetry Transformer

Step 1: Train a transformer that learns symmetry without explicit enforcement:

```bash
python training/train_learnable_symmetry.py \
  --output_dir checkpoints/learnable_symmetry \
  --batch_size 32 \
  --image_size 256 \
  --num_epochs 1000 \
  --learning_rate 1e-4 \
  --symmetry_type vertical \
  --with_r \
  --symmetry_weight 0.7
```

Step 2 (Optional): End-to-end training of learnable transformer and diffusion model:

```bash
python training/train_end_to_end_learnable.py \
  --dataset_path data/StarCraft_Map_Dataset \
  --output_dir checkpoints/e2e_learnable_diffusion \
  --base_model google/ddpm-celebahq-256 \
  --transformer_path checkpoints/learnable_symmetry/final \
  --batch_size 8 \
  --num_epochs 20 \
  --symmetry_type vertical \
  --freeze_unet \
  --unfreeze_epoch 3 \
  --symmetry_weight 0.3
```

### Generating Samples

#### Basic Diffusion Samples

```bash
python generating/sample_components.py \
  --model_path checkpoints/starcraft_fine_tuned/final \
  --output_dir outputs/starcraft_samples \
  --num_samples 5 \
  --num_steps 100
```

#### Transformer-based Symmetric Samples

```bash
python generating/sample_with_transformer.py \
  --transformer_path checkpoints/noise_transformer/final \
  --diffusion_model_path checkpoints/starcraft_fine_tuned/final \
  --output_dir outputs/symmetric_samples \
  --num_samples 5 \
  --show_transformed_noise
```

#### Learnable Symmetry Samples (Recommended)

```bash
python generating/sample_with_learned_symmetry.py \
  --transformer_path checkpoints/learnable_symmetry/final \
  --diffusion_model_path checkpoints/starcraft_fine_tuned/final \
  --output_dir outputs/learned_symmetry_samples \
  --num_samples 5 \
  --show_transformed_noise
```

### Interactive Demonstration

To visualize the symmetric noise transformation and denoising process:

```bash
python generating/demo_transformer.py \
  --transformer_path checkpoints/noise_transformer/final \
  --diffusion_model_path checkpoints/starcraft_fine_tuned/final \
  --output_dir outputs/transformer_demo \
  --num_inference_steps 20 \
  --create_video
```

This creates:
- Visualizations of original vs. symmetric noise
- Step-by-step denoising process images
- Grid visualization of all steps
- Optional video of the denoising process

You can also test without a pre-trained transformer:

```bash
python generating/demo_transformer.py \
  --diffusion_model_path checkpoints/starcraft_fine_tuned/final \
  --symmetry_type vertical \
  --with_r
```

## Troubleshooting Pipeline Loading

If you encounter issues loading the model as a pipeline, try using the component-based approach, which loads the UNet and scheduler components separately.

## Converting Models

To convert a trained model to a standard pipeline format:

```bash
python training/save_model_as_pipeline.py \
  --model_path checkpoints/starcraft_fine_tuned/final \
  --output_dir checkpoints/starcraft_fine_tuned/pipeline
```

## Understanding the Different Symmetry Approaches

### Explicit Symmetry (Original Approach)
In the original `transformer.py` implementation, we mathematically enforce symmetry through direct operations like flipping and averaging. This guarantees symmetry but doesn't let the model learn it naturally.

### Learnable Symmetry (New Approach)
The new `learning_symmetry.py` implementation provides the model with symmetry hints through coordinate channels, but uses loss functions to encourage symmetry learning rather than enforcing it. This approach:

1. Better aligns with the CoordConv philosophy of letting the model learn spatial relationships
2. May lead to more natural-looking outputs by avoiding explicit symmetry constraints
3. Allows for varying degrees of symmetry by adjusting the symmetry weight in the loss function

The key philosophical difference is that we're using coordinate channels to provide context for learning, not to enforce an outcome.