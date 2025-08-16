#!/usr/bin/env python3
"""
🏘️ QUICK SLUM DETECTION MODEL
============================
Simplified version for quick testing and evaluation
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
import os
import glob
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥️ Using device: {device}")

class QuickSlumDataset(Dataset):
    def __init__(self, images_dir, masks_dir):
        self.image_paths = sorted(glob.glob(f"{images_dir}/*.tif"))[:100]  # Limit for quick testing
        self.mask_paths = sorted(glob.glob(f"{masks_dir}/*.png"))[:100]
        print(f"📊 Quick dataset: {len(self.image_paths)} images")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load and resize image
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (256, 256))
        image = image.astype(np.float32) / 255.0
        
        # Load and process mask
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (256, 256))
        
        # Convert to binary slum detection (any non-zero value = slum)
        mask = (mask > 0).astype(np.float32)
        
        # Convert to tensors
        image = torch.FloatTensor(image.transpose(2, 0, 1))
        mask = torch.FloatTensor(mask).unsqueeze(0)
        
        return image, mask

class QuickUNet(nn.Module):
    def __init__(self):
        super(QuickUNet, self).__init__()
        
        # Encoder
        self.enc1 = self.conv_block(3, 64)
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
        
        self.final = nn.Conv2d(64, 1, 1)
        
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
        # Encoder path
        e1 = self.enc1(x)
        e2 = self.enc2(nn.MaxPool2d(2)(e1))
        e3 = self.enc3(nn.MaxPool2d(2)(e2))
        e4 = self.enc4(nn.MaxPool2d(2)(e3))
        
        # Bottleneck
        b = self.bottleneck(nn.MaxPool2d(2)(e4))
        
        # Decoder path
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
        
        return torch.sigmoid(self.final(d1))

def quick_train():
    """Quick training function"""
    print("🏋️ Quick UNET Training for Slum Detection")
    print("=" * 45)
    
    # Create datasets
    train_dataset = QuickSlumDataset('data/train/images', 'data/train/masks')
    val_dataset = QuickSlumDataset('data/val/images', 'data/val/masks')
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    
    # Initialize model
    model = QuickUNet().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print(f"🧠 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training loop
    num_epochs = 15
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss, train_acc = 0, 0
        
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for images, masks in train_bar:
            images, masks = images.to(device), masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # Calculate accuracy
            predicted = (outputs > 0.5).float()
            accuracy = (predicted == masks).float().mean()
            train_acc += accuracy.item()
            
            train_bar.set_postfix({'loss': f"{loss.item():.4f}", 'acc': f"{accuracy.item():.3f}"})
        
        # Validation
        model.eval()
        val_loss, val_acc = 0, 0
        
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                
                val_loss += loss.item()
                predicted = (outputs > 0.5).float()
                accuracy = (predicted == masks).float().mean()
                val_acc += accuracy.item()
        
        # Store metrics
        train_losses.append(train_loss / len(train_loader))
        val_losses.append(val_loss / len(val_loader))
        train_accs.append(train_acc / len(train_loader))
        val_accs.append(val_acc / len(val_loader))
        
        print(f"Epoch {epoch+1}: Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}, "
              f"Train Acc: {train_accs[-1]:.3f}, Val Acc: {val_accs[-1]:.3f}")
    
    # Save model
    torch.save(model.state_dict(), 'quick_slum_model.pth')
    print("💾 Model saved!")
    
    return model, train_losses, val_losses, train_accs, val_accs

def visualize_results(model, train_losses, val_losses, train_accs, val_accs):
    """Visualize training results and predictions"""
    
    # Training curves
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss curves
    epochs = range(1, len(train_losses) + 1)
    ax1.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
    ax1.set_title('Loss Curves', fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy curves
    ax2.plot(epochs, train_accs, 'g-', label='Train Acc', linewidth=2)
    ax2.plot(epochs, val_accs, 'orange', label='Val Acc', linewidth=2)
    ax2.set_title('Accuracy Curves', fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Test predictions
    test_images = []
    test_paths = sorted(glob.glob('data/test/images/*.tif'))[:8]
    
    for path in test_paths:
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (256, 256))
        test_images.append(img / 255.0)
    
    # Generate predictions
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for img in test_images:
            img_tensor = torch.FloatTensor(img.transpose(2, 0, 1)).unsqueeze(0).to(device)
            pred = model(img_tensor).squeeze().cpu().numpy()
            predictions.append(pred)
    
    # Show predictions
    ax3.axis('off')
    ax4.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Detailed predictions visualization
    fig, axes = plt.subplots(2, 8, figsize=(20, 5))
    fig.suptitle('Quick UNET - Slum Detection Results', fontsize=14, fontweight='bold')
    
    for i in range(8):
        # Original image
        axes[0, i].imshow(test_images[i])
        axes[0, i].set_title(f'Test {i+1}')
        axes[0, i].axis('off')
        
        # Prediction with overlay
        axes[1, i].imshow(test_images[i])
        
        # Create red overlay for slum areas
        pred_binary = predictions[i] > 0.5
        if pred_binary.sum() > 0:
            overlay = np.zeros((*pred_binary.shape, 3))
            overlay[pred_binary] = [1, 0, 0]  # Red for slums
            axes[1, i].imshow(overlay, alpha=0.6)
        
        slum_pct = (pred_binary.sum() / pred_binary.size) * 100
        conf = predictions[i].max()
        axes[1, i].set_title(f'Slum: {slum_pct:.1f}%\nConf: {conf:.3f}')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Print final results
    print("\n🎯 FINAL RESULTS:")
    print("=" * 30)
    print(f"📊 Final Training Accuracy: {train_accs[-1]:.3f}")
    print(f"📊 Final Validation Accuracy: {val_accs[-1]:.3f}")
    print(f"📊 Final Training Loss: {train_losses[-1]:.4f}")
    print(f"📊 Final Validation Loss: {val_losses[-1]:.4f}")
    
    # Calculate slum detection statistics
    slum_detected = sum(1 for p in predictions if (p > 0.5).sum() > 0)
    print(f"🏘️ Images with slums detected: {slum_detected}/8")
    print(f"📈 Detection rate: {slum_detected/8*100:.1f}%")

if __name__ == "__main__":
    # Run quick training
    model, train_losses, val_losses, train_accs, val_accs = quick_train()
    
    # Visualize results
    visualize_results(model, train_losses, val_losses, train_accs, val_accs)
    
    print("\n✅ Quick slum detection model complete!")
