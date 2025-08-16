#!/usr/bin/env python3
"""
Kaggle-Ready Slum Detection Pipeline
===================================
Minimal, clean implementation for multi-class slum detection using U-Net.
Based on comprehensive dataset analysis showing 7 classes.
"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
from tqdm import tqdm
import glob
from pathlib import Path

# Set device and seeds
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(42)
np.random.seed(42)

print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    torch.cuda.empty_cache()

class SlumDataset(Dataset):
    """Dataset for multi-class slum detection"""
    
    def __init__(self, images_dir, masks_dir, transform=None, max_samples=None):
        self.image_paths = sorted(glob.glob(f"{images_dir}/*.tif"))
        self.mask_paths = sorted(glob.glob(f"{masks_dir}/*.png"))
        
        if max_samples:
            self.image_paths = self.image_paths[:max_samples]
            self.mask_paths = self.mask_paths[:max_samples]
        
        self.transform = transform
        print(f"Dataset: {len(self.image_paths)} samples")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load mask
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
        
        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        
        return image, mask.long()

class UNet(nn.Module):
    """Simple U-Net for multi-class segmentation"""
    
    def __init__(self, in_channels=3, num_classes=7):
        super(UNet, self).__init__()
        
        # Encoder
        self.enc1 = self.conv_block(in_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)
        
        # Bottleneck
        self.bottleneck = self.conv_block(512, 1024)
        
        # Decoder
        self.upconv4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = self.conv_block(1024, 512)
        
        self.upconv3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = self.conv_block(512, 256)
        
        self.upconv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = self.conv_block(256, 128)
        
        self.upconv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = self.conv_block(128, 64)
        
        # Output
        self.final = nn.Conv2d(64, num_classes, 1)
        
        self.pool = nn.MaxPool2d(2)
    
    def conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        
        # Bottleneck
        b = self.bottleneck(self.pool(e4))
        
        # Decoder
        d4 = self.upconv4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)
        
        d3 = self.upconv3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.upconv2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.upconv1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        
        return self.final(d1)

def get_transforms(is_train=True, img_size=128):
    """Get data transforms"""
    if is_train:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

def calculate_class_weights(dataset):
    """Calculate class weights for balanced training"""
    class_counts = torch.zeros(7)
    
    print("Calculating class weights...")
    for i in tqdm(range(min(len(dataset), 1000))):  # Sample subset for speed
        _, mask = dataset[i]
        for class_id in range(7):
            class_counts[class_id] += (mask == class_id).sum()
    
    # Convert to weights (inverse frequency)
    total_pixels = class_counts.sum()
    class_weights = total_pixels / (7 * class_counts + 1e-8)
    
    print(f"Class weights: {class_weights}")
    return class_weights

def train_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct_pixels = 0
    total_pixels = 0
    
    for images, masks in tqdm(loader, desc="Training"):
        images, masks = images.to(device), masks.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Calculate accuracy
        pred = outputs.argmax(dim=1)
        correct_pixels += (pred == masks).sum().item()
        total_pixels += masks.numel()
    
    avg_loss = total_loss / len(loader)
    accuracy = correct_pixels / total_pixels
    
    return avg_loss, accuracy

def validate_epoch(model, loader, criterion, device):
    """Validate for one epoch"""
    model.eval()
    total_loss = 0
    correct_pixels = 0
    total_pixels = 0
    
    with torch.no_grad():
        for images, masks in tqdm(loader, desc="Validation"):
            images, masks = images.to(device), masks.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            total_loss += loss.item()
            
            # Calculate accuracy
            pred = outputs.argmax(dim=1)
            correct_pixels += (pred == masks).sum().item()
            total_pixels += masks.numel()
    
    avg_loss = total_loss / len(loader)
    accuracy = correct_pixels / total_pixels
    
    return avg_loss, accuracy

def visualize_predictions(model, dataset, device, num_samples=4):
    """Visualize model predictions"""
    model.eval()
    
    fig, axes = plt.subplots(3, num_samples, figsize=(16, 8))
    
    with torch.no_grad():
        for i in range(num_samples):
            # Get sample
            image, mask = dataset[i * 100]  # Sample every 100th image
            
            # Predict
            image_tensor = image.unsqueeze(0).to(device)
            pred = model(image_tensor).argmax(dim=1).squeeze().cpu()
            
            # Convert image for display
            img_display = image.permute(1, 2, 0).numpy()
            img_display = (img_display * [0.229, 0.224, 0.225]) + [0.485, 0.456, 0.406]
            img_display = np.clip(img_display, 0, 1)
            
            # Plot
            axes[0, i].imshow(img_display)
            axes[0, i].set_title(f'Image {i+1}')
            axes[0, i].axis('off')
            
            axes[1, i].imshow(mask.numpy(), cmap='tab10', vmin=0, vmax=6)
            axes[1, i].set_title('Ground Truth')
            axes[1, i].axis('off')
            
            axes[2, i].imshow(pred.numpy(), cmap='tab10', vmin=0, vmax=6)
            axes[2, i].set_title('Prediction')
            axes[2, i].axis('off')
    
    plt.tight_layout()
    plt.savefig('predictions.png', dpi=150, bbox_inches='tight')
    plt.show()

def main():
    """Main training function"""
    
    # Configuration
    IMG_SIZE = 128
    BATCH_SIZE = 16
    EPOCHS = 20
    LEARNING_RATE = 1e-3
    
    print("SLUM DETECTION TRAINING")
    print("=" * 40)
    print(f"Image size: {IMG_SIZE}x{IMG_SIZE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}")
    
    # Create datasets
    train_transform = get_transforms(is_train=True, img_size=IMG_SIZE)
    val_transform = get_transforms(is_train=False, img_size=IMG_SIZE)
    
    train_dataset = SlumDataset('data/train/images', 'data/train/masks', 
                               transform=train_transform, max_samples=2000)  # Limit for speed
    val_dataset = SlumDataset('data/val/images', 'data/val/masks', 
                             transform=val_transform, max_samples=500)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    # Create model
    model = UNet(in_channels=3, num_classes=7).to(device)
    
    # Calculate class weights
    class_weights = calculate_class_weights(train_dataset).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    
    # Training loop
    best_val_acc = 0
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    
    for epoch in range(EPOCHS):
        print(f"\\nEpoch {epoch+1}/{EPOCHS}")
        print("-" * 30)
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Save metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_slum_model.pth')
            print(f"New best model saved! Val Acc: {val_acc:.4f}")
    
    # Plot training history
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(val_losses, label='Val Loss')
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    ax2.plot(train_accs, label='Train Acc')
    ax2.plot(val_accs, label='Val Acc')
    ax2.set_title('Training Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Load best model and visualize
    model.load_state_dict(torch.load('best_slum_model.pth'))
    visualize_predictions(model, val_dataset, device)
    
    print(f"\\nTraining complete! Best validation accuracy: {best_val_acc:.4f}")
    print("Model saved as 'best_slum_model.pth'")

if __name__ == "__main__":
    main()