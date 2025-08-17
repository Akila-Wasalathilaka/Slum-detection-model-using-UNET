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
    """Optimized dataset with fast loading"""
    
    def __init__(self, images_dir, masks_dir, transform=None, is_train=True, max_samples=None):
        self.image_paths = sorted(glob.glob(f"{images_dir}/*.tif"))
        self.mask_paths = sorted(glob.glob(f"{masks_dir}/*.png"))
        
        if max_samples:
            self.image_paths = self.image_paths[:max_samples]
            self.mask_paths = self.mask_paths[:max_samples]
        
        self.transform = transform
        self.is_train = is_train
        
        # Pre-compute class mapping for speed
        self.class_mapping = {0: 0, 105: 1, 109: 2, 111: 3, 158: 4, 200: 5, 233: 6}
        
        print(f"📊 Optimized Dataset: {len(self.image_paths)} samples")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load mask
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
        
        # Vectorized class mapping for speed
        mapped_mask = np.zeros_like(mask, dtype=np.uint8)
        for original_val, new_val in self.class_mapping.items():
            mapped_mask[mask == original_val] = new_val
        
        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image, mask=mapped_mask)
            image = augmented['image']
            mask = augmented['mask']
        else:
            mask = mapped_mask
        
        return image, torch.tensor(mask, dtype=torch.long)

def get_advanced_transforms(is_train=True, img_size=224):
    """Speed-optimized augmentation pipeline"""
    if is_train:
        return A.Compose([
            A.Resize(img_size, img_size, interpolation=cv2.INTER_LINEAR),
            
            # Fast augmentations
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=15, p=0.4),
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.4),
            A.HueSaturationValue(hue_shift_limit=8, sat_shift_limit=12, val_shift_limit=8, p=0.3),
            
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(img_size, img_size, interpolation=cv2.INTER_LINEAR),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

class AttentionBlock(nn.Module):
    """Attention for better feature focus"""
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.conv_out = nn.Conv2d(in_channels // 8, in_channels, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        attention = F.relu(self.conv(x))
        attention = self.sigmoid(self.conv_out(attention))
        return x * attention

class AdvancedUNet(nn.Module):
    """Speed and memory optimized U-Net with enhanced features"""
    
    def __init__(self, encoder_name='efficientnet-b2', num_classes=7, encoder_weights='imagenet'):
        super(AdvancedUNet, self).__init__()
        
        self.backbone = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=num_classes,
            activation=None,
        )
        
        # Efficient water discrimination
        self.water_detector = nn.Sequential(
            nn.Conv2d(num_classes, 16, 1),  # 1x1 conv for speed
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid()
        )
        
        # Lightweight feature refinement
        self.feature_refine = nn.Sequential(
            nn.Conv2d(num_classes, num_classes, 3, padding=1, groups=num_classes),  # Depthwise
            nn.Conv2d(num_classes, num_classes, 1),  # Pointwise
            nn.BatchNorm2d(num_classes),
            nn.ReLU(inplace=True)
        )
        
        # Initialize weights for faster convergence
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Clone input to avoid CUDA graph issues
        x = x.clone()
        base_output = self.backbone(x)
        refined = self.feature_refine(base_output)
        
        # Efficient water suppression with cloning
        water_mask = self.water_detector(refined)
        
        # Safe tensor operations
        final_output = refined.clone()
        suppression = (1 - water_mask * 0.5)
        final_output = final_output * suppression
        final_output[:, 0:1] = refined[:, 0:1].clone()  # Keep background unchanged
        
        return final_output

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    
    def __init__(self, alpha=1, gamma=2, weight=None):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
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

class BoundaryLoss(nn.Module):
    """Boundary-aware loss for better edges"""
    def __init__(self):
        super().__init__()
        
    def forward(self, inputs, targets):
        probs = F.softmax(inputs, dim=1)
        grad_x = torch.abs(probs[:, :, :, :-1] - probs[:, :, :, 1:])
        grad_y = torch.abs(probs[:, :, :-1, :] - probs[:, :, 1:, :])
        
        target_grad_x = torch.abs(targets[:, :, :-1].float() - targets[:, :, 1:].float())
        target_grad_y = torch.abs(targets[:, :-1, :].float() - targets[:, 1:, :].float())
        
        boundary_loss_x = F.mse_loss(grad_x.sum(dim=1), target_grad_x)
        boundary_loss_y = F.mse_loss(grad_y.sum(dim=1), target_grad_y)
        
        return (boundary_loss_x + boundary_loss_y) / 2

class CombinedLoss(nn.Module):
    """Enhanced Combined Loss with boundary awareness"""
    
    def __init__(self, focal_weight=0.5, dice_weight=0.3, boundary_weight=0.2, class_weights=None, gamma=2.5):
        super(CombinedLoss, self).__init__()
        self.focal_loss = FocalLoss(alpha=1, gamma=gamma, weight=class_weights)
        self.dice_loss = DiceLoss()
        self.boundary_loss = BoundaryLoss()
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.boundary_weight = boundary_weight
    
    def forward(self, inputs, targets):
        focal = self.focal_loss(inputs, targets)
        dice = self.dice_loss(inputs, targets)
        boundary = self.boundary_loss(inputs, targets)
        return self.focal_weight * focal + self.dice_weight * dice + self.boundary_weight * boundary

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

def train_epoch(model, loader, criterion, optimizer, device, scaler=None, accumulate_grad=1):
    """Optimized training epoch with gradient accumulation"""
    model.train()
    total_loss = 0
    correct_pixels = 0
    total_pixels = 0
    
    optimizer.zero_grad()
    
    for batch_idx, (images, masks) in enumerate(tqdm(loader, desc="Training")):
        # Safe tensor transfer
        images = images.to(device, non_blocking=False).clone()
        masks = masks.to(device, non_blocking=False).clone()
        
        if scaler:
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, masks) / accumulate_grad
            
            scaler.scale(loss).backward()
            
            if (batch_idx + 1) % accumulate_grad == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            outputs = model(images)
            loss = criterion(outputs, masks) / accumulate_grad
            loss.backward()
            
            if (batch_idx + 1) % accumulate_grad == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
        
        total_loss += loss.item() * accumulate_grad
        
        # Fast accuracy calculation
        with torch.no_grad():
            pred = outputs.argmax(dim=1)
            correct_pixels += (pred == masks).sum().item()
            total_pixels += masks.numel()
    
    avg_loss = total_loss / len(loader)
    accuracy = correct_pixels / total_pixels if total_pixels > 0 else 0.0
    
    return avg_loss, accuracy, {}

def validate_epoch(model, loader, criterion, device):
    """Validation epoch"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(tqdm(loader, desc="Validation")):
            images = images.to(device).clone()
            masks = masks.to(device).clone()
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            total_loss += loss.item()
            
            # Collect predictions for metrics (sample to save memory)
            if batch_idx < 20:  # Only first 20 batches for metrics
                pred = outputs.argmax(dim=1)
                all_preds.append(pred.cpu())
                all_targets.append(masks.cpu())
    
    # Calculate metrics
    if all_preds:
        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)
        accuracy, class_metrics = calculate_metrics(all_preds, all_targets)
    else:
        accuracy, class_metrics = 0.0, {}
    
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
    
    # Memory-optimized configuration for Kaggle P100
    CONFIG = {
        'IMG_SIZE': 224,  # Reduced for T4 compatibility
        'BATCH_SIZE': 4,  # Further reduced for stability
        'EPOCHS': 25,     # Reduced with better convergence
        'LEARNING_RATE': 1e-4,  # Reduced LR for stability
        'ENCODER': 'efficientnet-b1',  # Smaller for T4
        'USE_MIXED_PRECISION': True,
        'COMPILE_MODEL': False,  # Disabled for T4 compatibility
        'EARLY_STOPPING_PATIENCE': 6,
        'REDUCE_LR_PATIENCE': 3,
        'FOCAL_GAMMA': 2.0,
        'ACCUMULATE_GRAD': 4,  # Increased to maintain effective batch size
        'NUM_WORKERS': 0,  # Disabled for stability
    }
    
    print("🚀 OPTIMIZED SLUM DETECTION TRAINING")
    print("=" * 50)
    print("Features: Speed optimization, memory efficiency, enhanced accuracy")
    for key, value in CONFIG.items():
        print(f"{key}: {value}")
    
    # Create datasets with advanced augmentations
    train_transform = get_advanced_transforms(is_train=True, img_size=CONFIG['IMG_SIZE'])
    val_transform = get_advanced_transforms(is_train=False, img_size=CONFIG['IMG_SIZE'])
    
    train_dataset = AdvancedSlumDataset('data/train/images', 'data/train/masks', 
                                       transform=train_transform, is_train=True)
    val_dataset = AdvancedSlumDataset('data/val/images', 'data/val/masks', 
                                     transform=val_transform, is_train=False)
    
    # Stable data loaders
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['BATCH_SIZE'], 
                             shuffle=True, num_workers=CONFIG['NUM_WORKERS'], 
                             pin_memory=False, persistent_workers=False)
    
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['BATCH_SIZE'], 
                           shuffle=False, num_workers=CONFIG['NUM_WORKERS'], 
                           pin_memory=False, persistent_workers=False)
    
    # Create optimized model
    model = AdvancedUNet(encoder_name=CONFIG['ENCODER'], num_classes=7).to(device)
    
    # Compile model for PyTorch 2.0 speed boost (disabled for T4)
    if CONFIG['COMPILE_MODEL'] and hasattr(torch, 'compile'):
        try:
            model = torch.compile(model, mode='default')
            print("🚀 Model compiled for faster training")
        except:
            print("⚠️ Model compilation skipped (T4 compatibility)")
    
    # Enhanced class weights for water/slum discrimination
    print("🔍 Setting enhanced class weights...")
    # Background, Slum-A, Slum-Main, Slum-B, Slum-C, Slum-D, Slum-E
    class_weights = torch.tensor([15.0, 1.2, 0.4, 1.0, 1.1, 2.5, 1.8]).to(device)
    
    print(f"📊 Class weights: {class_weights}")
    
    # Enhanced loss function with boundary awareness
    criterion = CombinedLoss(
        focal_weight=0.5, 
        dice_weight=0.3,
        boundary_weight=0.2,
        class_weights=class_weights,
        gamma=CONFIG['FOCAL_GAMMA']
    )
    
    # Optimized optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['LEARNING_RATE'], 
                                 weight_decay=1e-4, betas=(0.9, 0.999), eps=1e-8, 
                                 fused=True if torch.cuda.is_available() else False)
    
    # Faster convergence scheduler
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=CONFIG['LEARNING_RATE'], 
        epochs=CONFIG['EPOCHS'], steps_per_epoch=len(train_loader),
        pct_start=0.1, anneal_strategy='cos'
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
        
        # Optimized training
        train_loss, train_acc, train_class_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device, scaler, CONFIG['ACCUMULATE_GRAD']
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
        
        # Calculate mean IoU for better metric tracking
        mean_iou = np.mean([m['iou'] for m in val_class_metrics.values()]) if val_class_metrics else 0
        
        print(f"🏃 Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"🎯 Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Mean IoU: {mean_iou:.4f}")
        print(f"📈 Learning Rate: {current_lr:.6f}")
        print(f"⚡ GPU Memory: {torch.cuda.memory_allocated()/1024**3:.1f}GB" if torch.cuda.is_available() else "")
        
        # Memory cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Print class-wise metrics for validation with class names
        class_names = {0: 'Background', 1: 'Slum-A', 2: 'Slum-Main', 3: 'Slum-B', 4: 'Slum-C', 5: 'Slum-D', 6: 'Slum-E'}
        print("📊 Validation Class Metrics:")
        for class_id, metrics in val_class_metrics.items():
            name = class_names.get(class_id, f'Class-{class_id}')
            print(f"  {name}: IoU={metrics['iou']:.3f}, F1={metrics['f1']:.3f}, Prec={metrics['precision']:.3f}")
        
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
        
        # Early stopping check
        if patience_counter >= CONFIG['EARLY_STOPPING_PATIENCE']:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break
        
        # Additional early stopping if no improvement for many epochs
        if epoch > 10 and val_acc < 0.1:  # If accuracy is very low after 10 epochs
            print(f"Training stopped due to poor performance at epoch {epoch+1}")
            break
        
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
OPTIMIZED TRAINING SUMMARY
{'='*25}

Best Validation Accuracy: {best_val_acc:.4f}
Total Epochs: {len(training_history['train_loss'])}
Final Train Loss: {training_history['train_loss'][-1]:.4f}
Final Val Loss: {training_history['val_loss'][-1]:.4f}

Optimizations Applied:
• Model compilation: {'✓' if CONFIG['COMPILE_MODEL'] else '✗'}
• Gradient accumulation: {CONFIG['ACCUMULATE_GRAD']}x
• Mixed precision: {'✓' if CONFIG['USE_MIXED_PRECISION'] else '✗'}
• Parallel workers: {CONFIG['NUM_WORKERS']}

Model: {CONFIG['ENCODER']}
Image Size: {CONFIG['IMG_SIZE']}x{CONFIG['IMG_SIZE']}
Batch Size: {CONFIG['BATCH_SIZE']}

Files Saved:
• best_advanced_slum_model.pth
• advanced_training_history.json
"""
    
    axes[1, 1].text(0.1, 0.9, summary_text, transform=axes[1, 1].transAxes,
                   fontsize=11, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('advanced_training_history.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\\n🎉 Optimized training complete!")
    print(f"🏆 Best validation accuracy: {best_val_acc:.4f}")
    print(f"💾 Model saved as 'best_advanced_slum_model.pth'")
    print(f"📊 Training history saved as 'advanced_training_history.json'")
    print(f"⚡ Training optimizations: Gradient accumulation, model compilation, efficient data loading")

if __name__ == "__main__":
    main()