#!/usr/bin/env python3
"""
Minimal Colab Training Script for Global Slum Detection
======================================================
Idempotent setup: detects current working directory to avoid double clone or nested paths.
"""

# Setup and clone repository
import os
import subprocess
import sys

def setup_colab():
    """Setup Colab environment without cloning twice or nesting paths."""
    print("🚀 Setting up Colab environment...")

    repo_name = 'Slum-detection-model-using-UNET'
    cwd = os.getcwd()

    # Heuristic: if we're already inside the repo (has requirements and scripts/), don't clone or chdir
    already_in_repo = (
        os.path.isfile(os.path.join(cwd, 'requirements.txt')) and
        os.path.isdir(os.path.join(cwd, 'scripts')) and
        os.path.isdir(os.path.join(cwd, 'utils'))
    )

    if already_in_repo:
        repo_dir = cwd
    else:
        # Prefer /content in Colab if available
        base_dir = '/content' if os.path.isdir('/content') else cwd
        repo_dir = os.path.join(base_dir, repo_name)

        if not os.path.exists(repo_dir):
            # Clone into base_dir
            subprocess.run(['git', '-C', base_dir, 'clone', 'https://github.com/Akila-Wasalathilaka/Slum-detection-model-using-UNET.git'], check=True)

    # Ensure we're in the repo root
    os.chdir(repo_dir)

    # Note: avoid resetting the repo while this script is running to prevent overwriting it.

    # Install requirements (safe to re-run)
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'], check=True)

    print(f"✅ Setup complete in: {os.getcwd()}")

# Run setup
setup_colab()

# Import required modules
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from tqdm import tqdm

import segmentation_models_pytorch as smp
from models.global_losses import ComboLossV2
from utils.global_transforms import get_global_train_transforms, get_global_val_transforms
from utils.dataset import SlumDataset
from torch.utils.data import DataLoader


def _build_smp_unet(encoder_name: str = "resnet34", in_channels: int = 6, classes: int = 1, pretrained: bool = True) -> nn.Module:
    """Create an SMP Unet and safely adapt first conv to arbitrary input channels.

    - Loads ImageNet weights with 3 input channels when pretrained=True.
    - Replaces the first conv with a new one matching `in_channels`, copying RGB weights and
      initializing extra channels from the RGB mean to preserve pretrained features.
    """
    base_in = 3 if (pretrained and in_channels != 3) else in_channels
    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=("imagenet" if pretrained else None),
        in_channels=base_in,
        classes=classes,
    )

    # If needed, adapt first conv
    if base_in != in_channels:
        first_conv = model.encoder.conv1
        new_conv = nn.Conv2d(
            in_channels,
            first_conv.out_channels,
            kernel_size=first_conv.kernel_size,
            stride=first_conv.stride,
            padding=first_conv.padding,
            bias=first_conv.bias is not None,
        )
        with torch.no_grad():
            old_w = first_conv.weight
            old_in = old_w.shape[1]

            # Copy existing channels up to min
            copy_c = min(old_in, in_channels)
            new_conv.weight[:, :copy_c].copy_(old_w[:, :copy_c])

            # Initialize any extra channels from RGB (or available) mean
            if in_channels > old_in:
                mean_src = old_w[:, :min(3, old_in)].mean(dim=1, keepdim=True)
                repeat = in_channels - old_in
                new_conv.weight[:, old_in:].copy_(mean_src.expand(-1, repeat, -1, -1))

            if first_conv.bias is not None and new_conv.bias is not None:
                new_conv.bias.copy_(first_conv.bias)

        model.encoder.conv1 = new_conv

    return model

class MinimalTrainer:
    """Minimal trainer for Colab"""
    def __init__(self, data_root='data', batch_size=8, lr=1e-4):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.data_root = Path(data_root)
        self.batch_size = batch_size
        self.lr = lr

        print(f"🔥 Device: {self.device}")

    # Create model (robust to 6-channel input with pretrained weights)
    self.model = _build_smp_unet(encoder_name="resnet34", in_channels=6, classes=1, pretrained=True)
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
            images_dir=str(self.data_root / 'train' / 'images'),
            masks_dir=str(self.data_root / 'train' / 'masks'),
            transform=train_transform
        )

        self.val_dataset = SlumDataset(
            images_dir=str(self.data_root / 'val' / 'images'),
            masks_dir=str(self.data_root / 'val' / 'masks'),
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