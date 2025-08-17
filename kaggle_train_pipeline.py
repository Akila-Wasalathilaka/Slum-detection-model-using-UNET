#!/usr/bin/env python3
"""
Kaggle Training Pipeline for Slum Detection
==========================================
Minimal training pipeline for Kaggle environment
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Setup paths
sys.path.append('/kaggle/working')

def setup_kaggle_environment():
    """Setup Kaggle environment"""
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    return device

def main():
    """Main training pipeline"""
    print("🚀 Kaggle Training Pipeline")
    print("=" * 40)
    
    # Setup environment
    device = setup_kaggle_environment()
    
    # Import modules after path setup
    from models import create_model, create_loss
    from models.metrics import create_metrics, MetricsTracker
    from config import get_model_config, get_training_config, get_data_config
    from utils.dataset import SlumDataset, create_data_loaders
    from utils.transforms import get_train_transforms, get_val_transforms
    from utils.checkpoint import CheckpointManager
    
    # Load configs
    model_config = get_model_config('balanced')
    training_config = get_training_config('development')
    data_config = get_data_config('standard')
    
    # Create datasets
    paths = data_config.get_paths()
    train_transforms = get_train_transforms(data_config)
    val_transforms = get_val_transforms(data_config)
    
    train_dataset = SlumDataset(
        images_dir=paths['train_images'],
        masks_dir=paths['train_masks'],
        transform=train_transforms,
        slum_rgb=data_config.slum_rgb,
        image_size=data_config.image_size
    )
    
    val_dataset = SlumDataset(
        images_dir=paths['val_images'],
        masks_dir=paths['val_masks'],
        transform=val_transforms,
        slum_rgb=data_config.slum_rgb,
        image_size=data_config.image_size
    )
    
    # Create data loaders
    data_loaders = create_data_loaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=training_config.batch_size,
        num_workers=2
    )
    
    # Create model
    model = create_model(
        architecture=model_config.architecture,
        encoder=model_config.encoder,
        pretrained=model_config.pretrained,
        num_classes=model_config.num_classes
    ).to(device)
    
    # Create loss and optimizer
    criterion = create_loss(training_config.loss_type)
    optimizer = torch.optim.Adam(model.parameters(), lr=training_config.learning_rate)
    
    # Training loop
    print("🏋️ Training...")
    model.train()
    
    for epoch in range(training_config.epochs):
        epoch_loss = 0
        for batch_idx, (images, masks) in enumerate(data_loaders['train']):
            images, masks = images.to(device), masks.to(device)
            
            if len(masks.shape) == 3:
                masks = masks.unsqueeze(1)
            masks = masks.float()
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}/{training_config.epochs} [{batch_idx}/{len(data_loaders['train'])}] Loss: {loss.item():.4f}")
        
        print(f"Epoch {epoch+1} Average Loss: {epoch_loss/len(data_loaders['train']):.4f}")
    
    # Save model
    torch.save(model.state_dict(), '/kaggle/working/slum_model.pth')
    print("✅ Model saved to /kaggle/working/slum_model.pth")

if __name__ == "__main__":
    main()