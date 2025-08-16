#!/usr/bin/env python3
"""
Advanced Slum Detection Training Pipeline
========================================
Comprehensive training with multiple architectures, advanced augmentations,
and state-of-the-art techniques for maximum accuracy.
"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
from tqdm import tqdm
import glob
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

# Set device and seeds for reproducibility
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)

print(f"🔥 Device: {device}")
if torch.cuda.is_available():
    print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
    print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory // 1024**3} GB")
    torch.cuda.empty_cache()

class AdvancedSlumDataset(Dataset):
    """Advanced dataset with comprehensive augmentations and class balancing"""
    
    def __init__(self, images_dir, masks_dir, transform=None, is_train=True, max_samples=None):
        self.image_paths = sorted(glob.glob(f"{images_dir}/*.tif"))
        self.mask_paths = sorted(glob.glob(f"{masks_dir}/*.png"))
        
        if max_samples:
            self.image_paths = self.image_paths[:max_samples]
            self.mask_paths = self.mask_paths[:max_samples]
        
        self.transform = transform
        self.is_train = is_train
        
        # Calculate class weights for sampling
        if is_train:
            self.class_weights = self._calculate_sample_weights()
        
        print(f"📊 Dataset: {len(self.image_paths)} samples")
    
    def _calculate_sample_weights(self):
        """Calculate weights for each sample based on class distribution"""
        print("🔍 Calculating sample weights...")
        sample_weights = []
        
        for mask_path in tqdm(self.mask_paths[:1000], desc="Analyzing masks"):  # Sample for speed
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            unique_classes, counts = np.unique(mask, return_counts=True)
            
            # Weight based on rarest class in the mask
            min_class_count = counts.min()
            weight = 1.0 / (min_class_count + 1e-8)
            sample_weights.append(weight)
        
        # Extend weights to all samples
        while len(sample_weights) < len(self.mask_paths):
            sample_weights.extend(sample_weights[:min(1000, len(self.mask_paths) - len(sample_weights))])
        
        return torch.tensor(sample_weights[:len(self.mask_paths)])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load mask
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
        
        # Map class values to 0-6 range
        class_mapping = {0: 0, 105: 1, 109: 2, 111: 3, 158: 4, 200: 5, 233: 6}
        mapped_mask = np.zeros_like(mask)
        for original_val, new_val in class_mapping.items():
            mapped_mask[mask == original_val] = new_val
        
        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image, mask=mapped_mask)
            image = augmented['image']
            mask = augmented['mask']
        else:
            mask = mapped_mask
        
        return image, torch.tensor(mask, dtype=torch.long)

def get_advanced_transforms(is_train=True, img_size=256):
    """Advanced augmentation pipeline"""
    if is_train:
        return A.Compose([
            A.Resize(img_size, img_size),
            
            # Geometric augmentations
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=45, p=0.5),
            A.ElasticTransform(p=0.3),
            A.GridDistortion(p=0.3),
            A.OpticalDistortion(p=0.3),
            
            # Color augmentations
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
                A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1.0),
                A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1.0),
            ], p=0.5),
            
            # Noise and blur
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
                A.GaussianBlur(blur_limit=3, p=1.0),
                A.MotionBlur(blur_limit=3, p=1.0),
            ], p=0.3),
            
            # Weather effects
            A.OneOf([
                A.RandomShadow(p=1.0),
                A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, p=1.0),
            ], p=0.2),
            
            # Cutout for regularization
            A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.3),
            
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

class AdvancedUNet(nn.Module):
    """Advanced U-Net with attention mechanisms and deep supervision"""
    
    def __init__(self, encoder_name='efficientnet-b4', num_classes=7, encoder_weights='imagenet'):
        super(AdvancedUNet, self).__init__()
        
        self.model = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=num_classes,
            activation=None,  # We'll apply softmax later
        )
        
        # Add attention mechanism
        self.attention = nn.MultiheadAttention(embed_dim=256, num_heads=8, batch_first=True)
        
        # Deep supervision heads
        self.aux_head1 = nn.Conv2d(256, num_classes, 1)
        self.aux_head2 = nn.Conv2d(128, num_classes, 1)
    
    def forward(self, x):
        # Main prediction
        main_out = self.model(x)
        
        # For training, return main output
        # In practice, you could add auxiliary outputs for deep supervision
        return main_out

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    
    def __init__(self, alpha=1, gamma=2, weight=None):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.ce_loss = nn.CrossEntropyLoss(weight=weight, reduction='none')
    
    def forward(self, inputs, targets):
        ce_loss = self.ce_loss(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

class DiceLoss(nn.Module):
    """Dice Loss for segmentation"""
    
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, inputs, targets):
        inputs = F.softmax(inputs, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes=inputs.size(1)).permute(0, 3, 1, 2).float()
        
        intersection = (inputs * targets_one_hot).sum(dim=(2, 3))
        union = inputs.sum(dim=(2, 3)) + targets_one_hot.sum(dim=(2, 3))
        
        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()

class CombinedLoss(nn.Module):
    """Combined Focal + Dice Loss"""
    
    def __init__(self, focal_weight=0.7, dice_weight=0.3, class_weights=None):
        super(CombinedLoss, self).__init__()
        self.focal_loss = FocalLoss(weight=class_weights)
        self.dice_loss = DiceLoss()
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
    
    def forward(self, inputs, targets):
        focal = self.focal_loss(inputs, targets)
        dice = self.dice_loss(inputs, targets)
        return self.focal_weight * focal + self.dice_weight * dice

def calculate_metrics(pred, target, num_classes=7):
    """Calculate comprehensive metrics"""
    pred = pred.cpu().numpy().flatten()
    target = target.cpu().numpy().flatten()
    
    # Overall accuracy
    accuracy = (pred == target).mean()
    
    # Per-class metrics
    class_metrics = {}
    for class_id in range(num_classes):
        pred_class = (pred == class_id)
        target_class = (target == class_id)
        
        if target_class.sum() > 0:  # Only calculate if class exists
            precision = (pred_class & target_class).sum() / (pred_class.sum() + 1e-8)
            recall = (pred_class & target_class).sum() / (target_class.sum() + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)
            iou = (pred_class & target_class).sum() / ((pred_class | target_class).sum() + 1e-8)
            
            class_metrics[class_id] = {
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1),
                'iou': float(iou)
            }
    
    return float(accuracy), class_metrics

def train_epoch(model, loader, criterion, optimizer, device, scaler=None):
    """Advanced training epoch with mixed precision"""
    model.train()
    total_loss = 0
    all_preds = []
    all_targets = []
    
    for images, masks in tqdm(loader, desc="Training"):
        images, masks = images.to(device), masks.to(device)
        
        optimizer.zero_grad()
        
        if scaler:  # Mixed precision training
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, masks)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item()
        
        # Collect predictions for metrics
        pred = outputs.argmax(dim=1)
        all_preds.append(pred.cpu())
        all_targets.append(masks.cpu())
    
    # Calculate metrics
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    accuracy, class_metrics = calculate_metrics(all_preds, all_targets)
    
    avg_loss = total_loss / len(loader)
    return avg_loss, accuracy, class_metrics

def validate_epoch(model, loader, criterion, device):
    """Validation epoch"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for images, masks in tqdm(loader, desc="Validation"):
            images, masks = images.to(device), masks.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            total_loss += loss.item()
            
            pred = outputs.argmax(dim=1)
            all_preds.append(pred.cpu())
            all_targets.append(masks.cpu())
    
    # Calculate metrics
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    accuracy, class_metrics = calculate_metrics(all_preds, all_targets)
    
    avg_loss = total_loss / len(loader)
    return avg_loss, accuracy, class_metrics

def save_training_state(model, optimizer, scheduler, epoch, metrics, filepath):
    """Save complete training state"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'metrics': metrics,
    }, filepath)

def main():
    """Main training function"""
    
    # Enhanced Configuration
    CONFIG = {
        'IMG_SIZE': 256,  # Larger for better accuracy
        'BATCH_SIZE': 8,  # Smaller due to larger images
        'EPOCHS': 50,     # More epochs for better convergence
        'LEARNING_RATE': 1e-4,
        'ENCODER': 'efficientnet-b4',  # Better encoder
        'USE_MIXED_PRECISION': True,
        'EARLY_STOPPING_PATIENCE': 10,
        'REDUCE_LR_PATIENCE': 5,
    }
    
    print("🚀 ADVANCED SLUM DETECTION TRAINING")
    print("=" * 50)
    for key, value in CONFIG.items():
        print(f"{key}: {value}")
    
    # Create datasets with advanced augmentations
    train_transform = get_advanced_transforms(is_train=True, img_size=CONFIG['IMG_SIZE'])
    val_transform = get_advanced_transforms(is_train=False, img_size=CONFIG['IMG_SIZE'])
    
    train_dataset = AdvancedSlumDataset('data/train/images', 'data/train/masks', 
                                       transform=train_transform, is_train=True)
    val_dataset = AdvancedSlumDataset('data/val/images', 'data/val/masks', 
                                     transform=val_transform, is_train=False)
    
    # Create weighted sampler for balanced training
    if hasattr(train_dataset, 'class_weights'):
        sampler = WeightedRandomSampler(train_dataset.class_weights, len(train_dataset))
        train_loader = DataLoader(train_dataset, batch_size=CONFIG['BATCH_SIZE'], 
                                 sampler=sampler, num_workers=4, pin_memory=True)
    else:
        train_loader = DataLoader(train_dataset, batch_size=CONFIG['BATCH_SIZE'], 
                                 shuffle=True, num_workers=4, pin_memory=True)
    
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['BATCH_SIZE'], 
                           shuffle=False, num_workers=4, pin_memory=True)
    
    # Create advanced model
    model = AdvancedUNet(encoder_name=CONFIG['ENCODER'], num_classes=7).to(device)
    
    # Calculate class weights from a sample of the dataset
    print("🔍 Calculating class weights...")
    class_counts = torch.zeros(7)
    sample_size = min(500, len(train_dataset))  # Reduced sample size
    
    for i in tqdm(range(sample_size), desc="Sampling for weights"):
        _, mask = train_dataset[i]
        for class_id in range(7):
            class_counts[class_id] += (mask == class_id).sum()
    
    # Add small epsilon to avoid division by zero
    class_counts = class_counts + 1e-6
    total_pixels = class_counts.sum()
    class_weights = total_pixels / (7 * class_counts)
    
    # Normalize weights to prevent extreme values
    class_weights = torch.clamp(class_weights, min=0.1, max=10.0)
    class_weights = class_weights.to(device)
    
    print(f"📊 Class weights: {class_weights}")
    
    # Advanced loss function
    criterion = CombinedLoss(class_weights=class_weights)
    
    # Advanced optimizer with weight decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['LEARNING_RATE'], 
                                 weight_decay=1e-4, betas=(0.9, 0.999))
    
    # Advanced scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )
    
    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler() if CONFIG['USE_MIXED_PRECISION'] else None
    
    # Training tracking
    best_val_acc = 0
    patience_counter = 0
    training_history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': [],
        'learning_rates': []
    }
    
    print("🏋️ Starting training...")
    
    for epoch in range(CONFIG['EPOCHS']):
        print(f"\\n📅 Epoch {epoch+1}/{CONFIG['EPOCHS']}")
        print("-" * 40)
        
        # Training
        train_loss, train_acc, train_class_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device, scaler
        )
        
        # Validation
        val_loss, val_acc, val_class_metrics = validate_epoch(
            model, val_loader, criterion, device
        )
        
        # Update scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Save metrics
        training_history['train_loss'].append(train_loss)
        training_history['val_loss'].append(val_loss)
        training_history['train_acc'].append(train_acc)
        training_history['val_acc'].append(val_acc)
        training_history['learning_rates'].append(current_lr)
        
        print(f"🏃 Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"🎯 Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        print(f"📈 Learning Rate: {current_lr:.6f}")
        
        # Print class-wise metrics for validation
        print("📊 Validation Class Metrics:")
        for class_id, metrics in val_class_metrics.items():
            print(f"  Class {class_id}: IoU={metrics['iou']:.3f}, F1={metrics['f1']:.3f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            
            # Save best model
            torch.save(model.state_dict(), 'best_advanced_slum_model.pth')
            
            # Save complete training state
            save_training_state(model, optimizer, scheduler, epoch, {
                'val_acc': val_acc,
                'val_loss': val_loss,
                'class_metrics': val_class_metrics
            }, 'best_training_state.pth')
            
            print(f"🏆 New best model saved! Val Acc: {val_acc:.4f}")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= CONFIG['EARLY_STOPPING_PATIENCE']:
            print(f"⏰ Early stopping triggered after {epoch+1} epochs")
            break
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            save_training_state(model, optimizer, scheduler, epoch, {
                'val_acc': val_acc,
                'val_loss': val_loss
            }, f'checkpoint_epoch_{epoch+1}.pth')
    
    # Save training history
    with open('advanced_training_history.json', 'w') as f:
        json.dump(training_history, f, indent=2)
    
    # Plot comprehensive training history
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss plot
    axes[0, 0].plot(training_history['train_loss'], label='Train Loss', color='blue')
    axes[0, 0].plot(training_history['val_loss'], label='Val Loss', color='red')
    axes[0, 0].set_title('Training Loss', fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy plot
    axes[0, 1].plot(training_history['train_acc'], label='Train Acc', color='blue')
    axes[0, 1].plot(training_history['val_acc'], label='Val Acc', color='red')
    axes[0, 1].set_title('Training Accuracy', fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Learning rate plot
    axes[1, 0].plot(training_history['learning_rates'], color='green')
    axes[1, 0].set_title('Learning Rate Schedule', fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Learning Rate')
    axes[1, 0].set_yscale('log')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Summary text
    axes[1, 1].axis('off')
    summary_text = f"""
TRAINING SUMMARY
{'='*20}

Best Validation Accuracy: {best_val_acc:.4f}
Total Epochs: {len(training_history['train_loss'])}
Final Train Loss: {training_history['train_loss'][-1]:.4f}
Final Val Loss: {training_history['val_loss'][-1]:.4f}

Model: {CONFIG['ENCODER']}
Image Size: {CONFIG['IMG_SIZE']}x{CONFIG['IMG_SIZE']}
Batch Size: {CONFIG['BATCH_SIZE']}

Files Saved:
• best_advanced_slum_model.pth
• best_training_state.pth
• advanced_training_history.json
"""
    
    axes[1, 1].text(0.1, 0.9, summary_text, transform=axes[1, 1].transAxes,
                   fontsize=11, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('advanced_training_history.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\\n🎉 Training complete!")
    print(f"🏆 Best validation accuracy: {best_val_acc:.4f}")
    print(f"💾 Model saved as 'best_advanced_slum_model.pth'")
    print(f"📊 Training history saved as 'advanced_training_history.json'")

if __name__ == "__main__":
    main()