#!/usr/bin/env python3
"""
🏘️ ADVANCED SLUM DETECTION MODEL
=====================================
State-of-the-art UNET with latest techniques for slum detection
- EfficientNet encoder
- Attention mechanisms
- Multi-scale features
- Advanced data augmentation
- Class-weighted loss functions
- Comprehensive evaluation metrics
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

import numpy as np
import cv2
import os
import glob
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class AdvancedSlumDataset(Dataset):
    """Advanced dataset class with sophisticated augmentations"""
    
    def __init__(self, images_dir, masks_dir, transform=None, is_train=True):
        self.image_paths = sorted(glob.glob(f"{images_dir}/*.tif"))
        self.mask_paths = sorted(glob.glob(f"{masks_dir}/*.png"))
        self.is_train = is_train
        
        print(f"📊 Found {len(self.image_paths)} images and {len(self.mask_paths)} masks")
        
        # Advanced augmentations for training
        if is_train:
            self.transform = A.Compose([
                A.Resize(512, 512),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.3),
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.1, 
                    scale_limit=0.2, 
                    rotate_limit=15, 
                    p=0.5
                ),
                A.OneOf([
                    A.OpticalDistortion(p=0.3),
                    A.GridDistortion(p=0.3),
                    A.ElasticTransform(p=0.3),
                ], p=0.3),
                A.OneOf([
                    A.GaussNoise(p=0.3),
                    A.GaussianBlur(blur_limit=3, p=0.3),
                    A.MotionBlur(blur_limit=3, p=0.3),
                ], p=0.3),
                A.OneOf([
                    A.RandomBrightnessContrast(p=0.3),
                    A.CLAHE(p=0.3),
                    A.HueSaturationValue(p=0.3),
                    A.RGBShift(p=0.3),
                ], p=0.5),
                A.CoarseDropout(
                    max_holes=8, 
                    max_height=32, 
                    max_width=32, 
                    p=0.3
                ),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2()
            ])
        else:
            self.transform = A.Compose([
                A.Resize(512, 512),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2()
            ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image and mask
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
        
        # Handle multi-class masks - convert to binary slum detection
        # Assuming slums are non-zero values in mask
        mask = (mask > 0).astype(np.uint8)
        
        # Apply transformations
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        
        # Ensure mask is float and in correct range
        mask = mask.float().unsqueeze(0)  # Add channel dimension
        
        return image, mask

class AttentionBlock(nn.Module):
    """Attention mechanism for focusing on important features"""
    
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class AdvancedUNet(nn.Module):
    """Advanced U-Net with EfficientNet encoder and attention mechanisms"""
    
    def __init__(self, encoder_name='efficientnet-b4', encoder_weights='imagenet', 
                 classes=1, activation=None):
        super(AdvancedUNet, self).__init__()
        
        # Use segmentation_models_pytorch for advanced encoder
        self.model = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            classes=classes,
            activation=activation,
            decoder_attention_type='scse'  # Spatial and Channel SE attention
        )
        
    def forward(self, x):
        return self.model(x)

class CombinedLoss(nn.Module):
    """Combined loss function for better training"""
    
    def __init__(self, alpha=0.5, beta=0.5, gamma=2):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha  # Weight for BCE
        self.beta = beta    # Weight for Dice
        self.gamma = gamma  # Focal loss gamma
        
    def forward(self, pred, target):
        # Binary Cross Entropy
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='mean')
        
        # Dice Loss
        pred_sigmoid = torch.sigmoid(pred)
        intersection = (pred_sigmoid * target).sum()
        dice = 1 - (2 * intersection) / (pred_sigmoid.sum() + target.sum() + 1e-8)
        
        # Focal Loss
        pred_sigmoid = torch.sigmoid(pred)
        pt = target * pred_sigmoid + (1 - target) * (1 - pred_sigmoid)
        focal = -((1 - pt) ** self.gamma) * torch.log(pt + 1e-8)
        focal = focal.mean()
        
        return self.alpha * bce + self.beta * dice + (1 - self.alpha - self.beta) * focal

def calculate_metrics(pred, target, threshold=0.5):
    """Calculate comprehensive metrics"""
    pred_binary = (pred > threshold).float()
    target_binary = target.float()
    
    # Flatten for metric calculation
    pred_flat = pred_binary.view(-1).cpu().numpy()
    target_flat = target_binary.view(-1).cpu().numpy()
    pred_prob_flat = pred.view(-1).cpu().numpy()
    
    # Basic metrics
    intersection = (pred_binary * target_binary).sum().item()
    union = pred_binary.sum().item() + target_binary.sum().item() - intersection
    
    iou = intersection / (union + 1e-8)
    dice = 2 * intersection / (pred_binary.sum().item() + target_binary.sum().item() + 1e-8)
    
    # Pixel accuracy
    accuracy = (pred_flat == target_flat).mean()
    
    # Precision, Recall, F1
    tp = intersection
    fp = pred_binary.sum().item() - intersection
    fn = target_binary.sum().item() - intersection
    
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    
    # AUC if possible
    try:
        auc = roc_auc_score(target_flat, pred_prob_flat)
    except:
        auc = 0.5
    
    return {
        'iou': iou,
        'dice': dice,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc
    }

def train_advanced_model():
    """Train the advanced slum detection model"""
    
    print("🏗️ ADVANCED SLUM DETECTION MODEL TRAINING")
    print("=" * 50)
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️ Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"🚀 GPU: {torch.cuda.get_device_name()}")
        print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory // 1024**3} GB")
    
    # Create datasets
    print("📊 Loading datasets...")
    train_dataset = AdvancedSlumDataset(
        'data/train/images', 
        'data/train/masks', 
        is_train=True
    )
    
    val_dataset = AdvancedSlumDataset(
        'data/val/images', 
        'data/val/masks', 
        is_train=False
    )
    
    # Data loaders
    batch_size = 4 if device.type == 'cuda' else 2
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    print(f"📊 Training samples: {len(train_dataset)}")
    print(f"📊 Validation samples: {len(val_dataset)}")
    print(f"📊 Batch size: {batch_size}")
    
    # Initialize model
    print("🧠 Initializing advanced model...")
    model = AdvancedUNet(
        encoder_name='efficientnet-b4',
        encoder_weights='imagenet',
        classes=1,
        activation=None  # We'll use sigmoid in loss
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"🔢 Total parameters: {total_params:,}")
    print(f"🔢 Trainable parameters: {trainable_params:,}")
    
    # Loss and optimizer
    criterion = CombinedLoss(alpha=0.3, beta=0.4, gamma=2)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Training tracking
    train_losses, val_losses = [], []
    train_metrics, val_metrics = [], []
    best_val_iou = 0
    patience_counter = 0
    max_patience = 10
    
    num_epochs = 50
    print(f"🏋️ Starting training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        print(f"\n📅 Epoch {epoch+1}/{num_epochs}")
        print("-" * 40)
        
        # Training phase
        model.train()
        train_loss = 0
        train_epoch_metrics = {
            'iou': 0, 'dice': 0, 'accuracy': 0, 
            'precision': 0, 'recall': 0, 'f1': 0, 'auc': 0
        }
        
        train_bar = tqdm(train_loader, desc="Training")
        for batch_idx, (images, masks) in enumerate(train_bar):
            images, masks = images.to(device), masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            
            # Calculate metrics
            with torch.no_grad():
                batch_metrics = calculate_metrics(torch.sigmoid(outputs), masks)
                for key in train_epoch_metrics:
                    train_epoch_metrics[key] += batch_metrics[key]
            
            train_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'iou': f"{batch_metrics['iou']:.3f}"
            })
        
        # Average training metrics
        train_loss /= len(train_loader)
        for key in train_epoch_metrics:
            train_epoch_metrics[key] /= len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_epoch_metrics = {
            'iou': 0, 'dice': 0, 'accuracy': 0, 
            'precision': 0, 'recall': 0, 'f1': 0, 'auc': 0
        }
        
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc="Validation")
            for images, masks in val_bar:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
                
                # Calculate metrics
                batch_metrics = calculate_metrics(torch.sigmoid(outputs), masks)
                for key in val_epoch_metrics:
                    val_epoch_metrics[key] += batch_metrics[key]
                
                val_bar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'iou': f"{batch_metrics['iou']:.3f}"
                })
        
        # Average validation metrics
        val_loss /= len(val_loader)
        for key in val_epoch_metrics:
            val_epoch_metrics[key] /= len(val_loader)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Store metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_metrics.append(train_epoch_metrics.copy())
        val_metrics.append(val_epoch_metrics.copy())
        
        # Print epoch results
        print(f"📊 Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"📊 Train IoU: {train_epoch_metrics['iou']:.3f} | Val IoU: {val_epoch_metrics['iou']:.3f}")
        print(f"📊 Train F1: {train_epoch_metrics['f1']:.3f} | Val F1: {val_epoch_metrics['f1']:.3f}")
        print(f"📊 Val Accuracy: {val_epoch_metrics['accuracy']:.3f} | Val AUC: {val_epoch_metrics['auc']:.3f}")
        
        # Save best model
        if val_epoch_metrics['iou'] > best_val_iou:
            best_val_iou = val_epoch_metrics['iou']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_iou': best_val_iou,
                'train_metrics': train_metrics,
                'val_metrics': val_metrics
            }, 'advanced_slum_model_best.pth')
            print(f"💾 New best model saved! IoU: {best_val_iou:.3f}")
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= max_patience:
            print(f"⏰ Early stopping triggered after {epoch+1} epochs")
            break
    
    print("\n✅ Training completed!")
    print(f"🏆 Best validation IoU: {best_val_iou:.3f}")
    
    # Load best model for evaluation
    checkpoint = torch.load('advanced_slum_model_best.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model, train_losses, val_losses, train_metrics, val_metrics

def visualize_training_results(train_losses, val_losses, train_metrics, val_metrics):
    """Create comprehensive training visualization"""
    
    print("📊 Creating training visualization...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Advanced Slum Detection Model - Training Results', fontsize=16, fontweight='bold')
    
    epochs = range(1, len(train_losses) + 1)
    
    # Loss curves
    axes[0,0].plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    axes[0,0].plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    axes[0,0].set_title('Loss Curves', fontweight='bold')
    axes[0,0].set_xlabel('Epoch')
    axes[0,0].set_ylabel('Loss')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # IoU curves
    train_ious = [m['iou'] for m in train_metrics]
    val_ious = [m['iou'] for m in val_metrics]
    axes[0,1].plot(epochs, train_ious, 'g-', label='Training IoU', linewidth=2)
    axes[0,1].plot(epochs, val_ious, 'orange', label='Validation IoU', linewidth=2)
    axes[0,1].set_title('IoU (Intersection over Union)', fontweight='bold')
    axes[0,1].set_xlabel('Epoch')
    axes[0,1].set_ylabel('IoU')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # F1 Score curves
    train_f1s = [m['f1'] for m in train_metrics]
    val_f1s = [m['f1'] for m in val_metrics]
    axes[0,2].plot(epochs, train_f1s, 'purple', label='Training F1', linewidth=2)
    axes[0,2].plot(epochs, val_f1s, 'brown', label='Validation F1', linewidth=2)
    axes[0,2].set_title('F1 Score', fontweight='bold')
    axes[0,2].set_xlabel('Epoch')
    axes[0,2].set_ylabel('F1 Score')
    axes[0,2].legend()
    axes[0,2].grid(True, alpha=0.3)
    
    # Precision and Recall
    val_precisions = [m['precision'] for m in val_metrics]
    val_recalls = [m['recall'] for m in val_metrics]
    axes[1,0].plot(epochs, val_precisions, 'cyan', label='Precision', linewidth=2)
    axes[1,0].plot(epochs, val_recalls, 'magenta', label='Recall', linewidth=2)
    axes[1,0].set_title('Precision and Recall', fontweight='bold')
    axes[1,0].set_xlabel('Epoch')
    axes[1,0].set_ylabel('Score')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # Accuracy curves
    train_accs = [m['accuracy'] for m in train_metrics]
    val_accs = [m['accuracy'] for m in val_metrics]
    axes[1,1].plot(epochs, train_accs, 'darkgreen', label='Training Accuracy', linewidth=2)
    axes[1,1].plot(epochs, val_accs, 'darkred', label='Validation Accuracy', linewidth=2)
    axes[1,1].set_title('Accuracy', fontweight='bold')
    axes[1,1].set_xlabel('Epoch')
    axes[1,1].set_ylabel('Accuracy')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    # AUC curves
    val_aucs = [m['auc'] for m in val_metrics]
    axes[1,2].plot(epochs, val_aucs, 'navy', label='Validation AUC', linewidth=2)
    axes[1,2].set_title('AUC Score', fontweight='bold')
    axes[1,2].set_xlabel('Epoch')
    axes[1,2].set_ylabel('AUC')
    axes[1,2].legend()
    axes[1,2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('advanced_training_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print final metrics
    print("\n🎯 FINAL TRAINING METRICS:")
    print("=" * 40)
    final_val_metrics = val_metrics[-1]
    for metric, value in final_val_metrics.items():
        print(f"📊 {metric.upper()}: {value:.4f}")

def run_predictions_on_test_data(model, device, num_samples=20):
    """Run predictions on test data and visualize results"""
    
    print(f"🔍 Running predictions on {num_samples} test samples...")
    
    # Create test dataset
    test_dataset = AdvancedSlumDataset(
        'data/test/images', 
        'data/test/images',  # Use images as dummy masks for test
        is_train=False
    )
    
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
    
    model.eval()
    predictions = []
    images = []
    
    with torch.no_grad():
        for i, (image, _) in enumerate(test_loader):
            if i >= num_samples:
                break
                
            image = image.to(device)
            output = torch.sigmoid(model(image))
            
            # Convert to numpy
            image_np = image.squeeze().cpu().numpy()
            # Denormalize image
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            image_np = image_np.transpose(1, 2, 0)
            image_np = image_np * std + mean
            image_np = np.clip(image_np, 0, 1)
            
            pred_np = output.squeeze().cpu().numpy()
            
            images.append(image_np)
            predictions.append(pred_np)
    
    # Visualize predictions
    fig, axes = plt.subplots(4, 10, figsize=(25, 10))
    fig.suptitle('Advanced UNET - Slum Detection Predictions', fontsize=16, fontweight='bold')
    
    for i in range(min(20, len(images))):
        row = (i // 10) * 2
        col = i % 10
        
        # Original image
        axes[row, col].imshow(images[i])
        axes[row, col].set_title(f'Test {i+1}', fontsize=10)
        axes[row, col].axis('off')
        
        # Prediction overlay
        axes[row + 1, col].imshow(images[i])
        
        # Create red overlay for slum areas
        pred_binary = predictions[i] > 0.5
        if pred_binary.sum() > 0:
            overlay = np.zeros((*pred_binary.shape, 3))
            overlay[pred_binary] = [1, 0, 0]  # Red for slums
            axes[row + 1, col].imshow(overlay, alpha=0.6)
        
        # Calculate statistics
        slum_percentage = (pred_binary.sum() / pred_binary.size) * 100
        max_confidence = predictions[i].max()
        
        axes[row + 1, col].set_title(
            f'Slum: {slum_percentage:.1f}%\nConf: {max_confidence:.3f}', 
            fontsize=9
        )
        axes[row + 1, col].axis('off')
    
    plt.tight_layout()
    plt.savefig('advanced_predictions.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Predictions visualization complete!")
    
    return images, predictions

if __name__ == "__main__":
    try:
        # Install required packages
        import subprocess
        import sys
        
        packages = [
            'segmentation-models-pytorch',
            'albumentations',
            'opencv-python',
            'scikit-learn'
        ]
        
        for package in packages:
            try:
                __import__(package.replace('-', '_'))
            except ImportError:
                print(f"📦 Installing {package}...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        
        # Run training
        model, train_losses, val_losses, train_metrics, val_metrics = train_advanced_model()
        
        # Visualize results
        visualize_training_results(train_losses, val_losses, train_metrics, val_metrics)
        
        # Run predictions
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        images, predictions = run_predictions_on_test_data(model, device)
        
        print("\n🎉 ADVANCED SLUM DETECTION MODEL COMPLETE!")
        print("=" * 50)
        print("✅ Model trained with state-of-the-art techniques")
        print("✅ Comprehensive evaluation metrics calculated")
        print("✅ Prediction visualizations generated")
        print("💾 Best model saved as 'advanced_slum_model_best.pth'")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
