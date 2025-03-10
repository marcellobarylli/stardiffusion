#!/usr/bin/env python3
"""Script for fine-tuning a diffusion model on StarCraft maps."""
from DiffusionCore import DiffusionConfig, DiffusionModelLoader
from DiffusionFineTuner import DiffusionTrainer

def finetune_starcraft_maps():
    """Fine-tune a diffusion model on StarCraft map images."""
    # Initialize configuration
    # Changed the model to church images as they might transfer better to map-like visuals
    config = DiffusionConfig(
        model_id="google/ddpm-ema-church-256",  # Better starting point for map-like images
        visible_gpus="2",  # Using GPU ID 2
        num_inference_steps=100
    )
    
    print(f"Starting fine-tuning with model: {config.model_id}")
    print(f"Using GPU(s): {config.visible_gpus}")
    
    # Load the model
    model_loader = DiffusionModelLoader(config)
    model_loader.load_model()
    
    # Create trainer
    output_dir = "fine_tuned_models/starcraft_maps"
    trainer = DiffusionTrainer(
        model_loader=model_loader,
        config=config,
        output_dir=output_dir
    )
    
    # Use our custom StarCraft map dataset
    print("Preparing StarCraft map dataset...")
    dataloader = trainer.prepare_dataset(
        dataset_name_or_path="data/StarCraft_Map_Dataset",
        batch_size=8,
        num_workers=4
    )
    
    # Train the model
    print(f"Starting training for 30 epochs...")
    losses = trainer.train(
        dataloader=dataloader,
        num_epochs=30,  # More epochs for better results
        learning_rate=1e-5,
        save_interval=5
    )
    
    print(f"Fine-tuning complete! Model saved to {output_dir}")
    if losses:
        print(f"Final loss: {losses[-1]:.4f}")

if __name__ == "__main__":
    finetune_starcraft_maps() 