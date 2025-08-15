#!/usr/bin/env python3
"""
Minimal Colab Training Script for Global Slum Detection
======================================================
Clone repo, setup environment, and train model
"""

# Setup and clone repository
import os
import subprocess
import sys

def setup_colab():
    """Setup Colab environment"""
    print("🚀 Setting up Colab environment...")
    
    # Clone repository
    if not os.path.exists('Slum-detection-model-using-UNET'):
        subprocess.run(['git', 'clone', 'https://github.com/Akila-Wasalathilaka/Slum-detection-model-using-UNET.git'], check=True)
    
    # Change to repo directory
    os.chdir('Slum-detection-model-using-UNET')
    
    # Install requirements
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'], check=True)
    
    print("✅ Setup complete!")

# Run setup
setup_colab()

# Import required modules
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from tqdm import tqdm

from models.enhanced_unet import create_enhanced_model
from models.global_losses import ComboLossV2
from utils.global_transforms import get_global_train_transforms, get_global_val_transforms
from utils.dataset import SlumDataset
from torch.utils.data import DataLoader

class MinimalTrainer:
    """Minimal trainer for Colab"""
    def __init__(self, data_root='data', batch_size=8, lr=1e-4):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.data_root = Path(data_root)
        self.batch_size = batch_size
        self.lr = lr
        
        print(f"🔥 Device: {self.device}")
        
        # Create model
        self.model = create_enhanced_model(encoder="resnet34", in_channels=6)
        self.model.to(self.device)
        
        # Loss and optimizer
        self.criterion = ComboLossV2()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        
        # Setup data
        self._setup_data()
    
    def _setup_data(self):
        """Setup datasets and loaders"""
        train_transform = get_global_train_transforms()
        val_transform = get_global_val_transforms()
        
        # Create datasets
        self.train_dataset = SlumDataset(
            images_dir=self.data_root / 'train' / 'images',
            masks_dir=self.data_root / 'train' / 'masks',
            transform=train_transform
        )
        
        self.val_dataset = SlumDataset(
            images_dir=self.data_root / 'val' / 'images',
            masks_dir=self.data_root / 'val' / 'masks',
            transform=val_transform
        )
        
        # Create loaders
        self.train_loader = DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=2
        )
        
        self.val_loader = DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=2
        )
        
        print(f"📊 Train: {len(self.train_dataset)}, Val: {len(self.val_dataset)}")
    
    def train_epoch(self):
        """Train one epoch"""
        self.model.train()
        total_loss = 0
        
        for images, masks in tqdm(self.train_loader, desc="Training"):
            images, masks = images.to(self.device), masks.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            
            if isinstance(outputs, dict):
                main_loss, _ = self.criterion(outputs['main'], masks)
                boundary_loss = nn.BCEWithLogitsLoss()(outputs['boundary'], masks)
                loss = main_loss + 0.3 * boundary_loss
            else:
                loss, _ = self.criterion(outputs, masks)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(self.train_loader)
    
    def validate(self):
        """Validate model"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for images, masks in tqdm(self.val_loader, desc="Validation"):
                images, masks = images.to(self.device), masks.to(self.device)
                
                outputs = self.model(images)
                if isinstance(outputs, dict):
                    outputs = outputs['main']
                
                loss, _ = self.criterion(outputs, masks)
                total_loss += loss.item()
        
        return total_loss / len(self.val_loader)
    
    def train(self, epochs=50):
        """Train model"""
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            
            train_loss = self.train_epoch()
            val_loss = self.validate()
            
            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'epoch': epoch,
                    'val_loss': val_loss
                }, 'best_global_model.pth')
                print(f"✅ Best model saved! Val Loss: {val_loss:.4f}")
        
        print("🎉 Training completed!")

# Main training execution
if __name__ == "__main__":
    # Check if data exists
    if not os.path.exists('data/train'):
        print("❌ Data folder not found! Please ensure data/ folder exists with train/val splits")
        sys.exit(1)
    
    # Create trainer and start training
    trainer = MinimalTrainer(
        data_root='data',
        batch_size=8,  # Small batch for Colab
        lr=1e-4
    )
    
    # Train model
    trainer.train(epochs=50)
    
    print("🌍 Global slum detection model training complete!")
    print("📁 Model saved as: best_global_model.pth")