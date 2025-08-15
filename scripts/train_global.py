#!/usr/bin/env python3
"""
Global Training Script with Domain Generalization
================================================
Train slum detection model for worldwide deployment
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import sys
sys.path.insert(0, '.')
sys.path.insert(0, '/content/Slum-detection-model-using-UNET')

from models.enhanced_unet import create_enhanced_model
from models.global_losses import ComboLossV2, OHEMLoss
from utils.global_transforms import get_global_train_transforms, get_global_val_transforms
from utils.dataset import SlumDataset
from torch.utils.data import DataLoader

class EMAModel:
    """Exponential Moving Average for stable training"""
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = self.decay * self.shadow[name] + (1 - self.decay) * param.data
    
    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name]
    
    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]

def train_epoch(model, loader, criterion, optimizer, device, ema=None):
    """Train one epoch with curriculum"""
    model.train()
    total_loss = 0
    
    pbar = tqdm(loader, desc="Training")
    for batch_idx, (images, masks) in enumerate(pbar):
        images, masks = images.to(device), masks.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        
        if isinstance(outputs, dict):
            # Multi-output (main + boundary)
            main_loss, _ = criterion(outputs['main'], masks)
            boundary_loss = nn.BCEWithLogitsLoss()(outputs['boundary'], masks)
            loss = main_loss + 0.3 * boundary_loss
        else:
            loss, _ = criterion(outputs, masks)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        
        # EMA update
        if ema:
            ema.update()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})
    
    return total_loss / len(loader)

def validate_epoch(model, loader, criterion, device):
    """Validate one epoch"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for images, masks in tqdm(loader, desc="Validation"):
            images, masks = images.to(device), masks.to(device)
            
            outputs = model(images)
            if isinstance(outputs, dict):
                outputs = outputs['main']
            
            loss, _ = criterion(outputs, masks)
            total_loss += loss.item()
    
    return total_loss / len(loader)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='data', help='Data root directory')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--device', default='auto', help='Device to use')
    args = parser.parse_args()
    
    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"🔥 Training Global Slum Detection Model")
    print(f"Device: {device}")
    
    # Create datasets with global transforms
    train_transform = get_global_train_transforms()
    val_transform = get_global_val_transforms()
    
    train_dataset = SlumDataset(
        images_dir=Path(args.data_root) / 'train' / 'images',
        masks_dir=Path(args.data_root) / 'train' / 'masks',
        transform=train_transform
    )
    
    val_dataset = SlumDataset(
        images_dir=Path(args.data_root) / 'val' / 'images',
        masks_dir=Path(args.data_root) / 'val' / 'masks',
        transform=val_transform
    )
    
    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    print(f"📊 Train samples: {len(train_dataset)}")
    print(f"📊 Val samples: {len(val_dataset)}")
    
    # Create model
    model = create_enhanced_model(encoder="resnet34", in_channels=6)
    model.to(device)
    
    # Loss function
    criterion = ComboLossV2()
    
    # Optimizer with different LRs for encoder/decoder
    encoder_params = []
    decoder_params = []
    
    for name, param in model.named_parameters():
        if 'encoder' in name:
            encoder_params.append(param)
        else:
            decoder_params.append(param)
    
    optimizer = torch.optim.AdamW([
        {'params': encoder_params, 'lr': args.lr * 0.1},  # Lower LR for encoder
        {'params': decoder_params, 'lr': args.lr}
    ], weight_decay=1e-4)
    
    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # EMA
    ema = EMAModel(model)
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        print(f"\\nEpoch {epoch+1}/{args.epochs}")
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, ema)
        
        # Validate with EMA model
        ema.apply_shadow()
        val_loss = validate_epoch(model, val_loader, criterion, device)
        ema.restore()
        
        # Scheduler step
        scheduler.step()
        
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        print(f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            
            # Save EMA model
            ema.apply_shadow()
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'val_loss': val_loss
            }, 'best_global_model.pth')
            ema.restore()
            
            print(f"✅ New best model saved! Val Loss: {val_loss:.4f}")
    
    print("🎉 Training completed!")

if __name__ == "__main__":
    main()