# stardiffusion

A modular framework for generating images with diffusion models and fine-tuning them on custom datasets.

## Features

- Image generation using pre-trained diffusion models from Hugging Face
- Fine-tuning capabilities for customizing models on your own datasets
- Support for standard datasets (CIFAR-10, MNIST, etc.) and custom image directories
- Multi-GPU support via PyTorch DataParallel
- Command-line interface for easy use
- Modular design for extensibility

## Project Structure

- `DiffusionCore.py`: Core classes for diffusion model loading, processing, and visualization
- `DiffusionFineTuner.py`: Classes and utilities for fine-tuning diffusion models
- `main.py`: Command-line interface and entry point for the application

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/huggingdiffusion.git
cd huggingdiffusion
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install torch torchvision torchaudio
pip install diffusers transformers accelerate
pip install matplotlib pillow tqdm
```

## Usage

### Image Generation

To generate images using a pre-trained diffusion model:

```bash
python main.py generate --model_id="google/ddpm-celebahq-256" --gpus="0,1" --output_dir="output"
```

Options:
- `--model_id`: Hugging Face model ID (default: "google/ddpm-celebahq-256")
- `--num_steps`: Number of denoising steps (default: 100)
- `--batch_size`: Batch size for generation (default: 1)
- `--seed`: Random seed for reproducibility (default: 42)
- `--output_dir`: Directory to save generated images (default: "output")
- `--gpus`: Comma-separated list of GPU IDs to use (default: "0,1")

### Fine-tuning

To fine-tune a diffusion model on a dataset:

```bash
python main.py finetune --model_id="google/ddpm-celebahq-256" --dataset="cifar10" --output_dir="fine_tuned_model" --num_epochs=20
```

To fine-tune on your own custom images:

```bash
python main.py finetune --model_id="google/ddpm-celebahq-256" --dataset="path/to/images" --output_dir="fine_tuned_model" --num_epochs=20
```

To see the list of available standard datasets:

```bash
python main.py list_datasets
```

Options:
- `--model_id`: Hugging Face model ID to fine-tune (default: "google/ddpm-celebahq-256")
- `--dataset`: Dataset to use for fine-tuning (can be a standard dataset name like 'cifar10' or a path to custom images)
- `--output_dir`: Directory to save the fine-tuned model (default: "fine_tuned_model")
- `--num_epochs`: Number of training epochs (default: 10)
- `--batch_size`: Batch size for training (default: 8)
- `--learning_rate`: Learning rate for optimization (default: 1e-5)
- `--save_interval`: Interval for saving checkpoints in epochs (default: 5)
- `--gpus`: Comma-separated list of GPU IDs to use (default: "0,1")
- `--mixed_precision`: Use mixed precision training (flag, default: False)
- `--image_size`: Target image size (default: from model config)
- `--num_workers`: Number of workers for data loading (default: 4)

### Generating with a Fine-tuned Model

To generate images using your fine-tuned model:

```bash
python main.py generate_ft --model_path="fine_tuned_model/final_model" --output_dir="ft_output"
```

Options:
- `--model_path`: Path to the fine-tuned model (required)
- `--num_steps`: Number of denoising steps (default: 100)
- `--batch_size`: Batch size for generation (default: 1)
- `--seed`: Random seed for reproducibility (default: 42)
- `--output_dir`: Directory to save generated images (default: "output")
- `--gpus`: Comma-separated list of GPU IDs to use (default: "0,1")

## Supported Datasets

### Standard Datasets
The framework supports the following standard datasets:
- CIFAR-10 (`cifar10`)
- CIFAR-100 (`cifar100`)
- MNIST (`mnist`) 
- Fashion-MNIST (`fashion_mnist`)
- SVHN (`svhn`)
- STL10 (`stl10`)
- LSUN (`lsun-bedroom`, `lsun-church`, etc.)

### Custom Datasets
For fine-tuning on your own images, organize them in a directory structure. The framework will recursively find all images with the following extensions: `.jpg`, `.jpeg`, `.png`, `.bmp`.

Example dataset structure:
```
dataset/
  ├── image1.jpg
  ├── image2.png
  ├── category1/
  │   ├── image3.jpg
  │   └── image4.png
  └── category2/
      ├── image5.jpg
      └── image6.png
```

## Examples

### Basic Generation
```python
from DiffusionCore import DiffusionConfig, DiffusionModelLoader, DiffusionProcessor, DiffusionVisualizer

# Initialize configuration
config = DiffusionConfig(model_id="google/ddpm-celebahq-256", visible_gpus="0,1")

# Load the model
model_loader = DiffusionModelLoader(config)
model_loader.load_model()

# Create processor and run diffusion
processor = DiffusionProcessor(model_loader, config)
final_image, intermediates = processor.generate_sample()

# Visualize results
visualizer = DiffusionVisualizer(model_loader)
visualizer.visualize_progression(intermediates)
visualizer.visualize_final_image(final_image)
```

### Fine-tuning with CIFAR-10
```python
from DiffusionCore import DiffusionConfig, DiffusionModelLoader
from DiffusionFineTuner import DiffusionTrainer

# Initialize configuration
config = DiffusionConfig(model_id="google/ddpm-celebahq-256", visible_gpus="0,1")

# Load the model
model_loader = DiffusionModelLoader(config)
model_loader.load_model()

# Create trainer
trainer = DiffusionTrainer(model_loader=model_loader, config=config, output_dir="cifar10_model")

# Prepare CIFAR-10 dataset
dataloader = trainer.prepare_dataset(dataset_name_or_path="cifar10", batch_size=8)

# Train the model
losses = trainer.train(dataloader=dataloader, num_epochs=20, learning_rate=1e-5)
```

### Fine-tuning with Custom Images
```python
from DiffusionCore import DiffusionConfig, DiffusionModelLoader
from DiffusionFineTuner import DiffusionTrainer

# Initialize configuration
config = DiffusionConfig(model_id="google/ddpm-celebahq-256", visible_gpus="0,1")

# Load the model
model_loader = DiffusionModelLoader(config)
model_loader.load_model()

# Create trainer
trainer = DiffusionTrainer(model_loader=model_loader, config=config, output_dir="custom_model")

# Prepare custom dataset
dataloader = trainer.prepare_dataset(dataset_name_or_path="path/to/images", batch_size=8)

# Train the model
losses = trainer.train(dataloader=dataloader, num_epochs=20, learning_rate=1e-5)
```

## License

This project is licensed under the MIT License - see the LICENSE file for details. 
