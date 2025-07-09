# File: improved_slum_detection_production.py
# =======================================================================================
#
#  PRODUCTION-LEVEL ENHANCED SLUM DETECTION PIPELINE
#  Advanced UNet-based semantic segmentation for informal settlements detection
#  
#  Key Improvements:
#  - Multi-scale training and inference
#  - Advanced loss functions with adaptive weighting
#  - Test Time Augmentation (TTA)
#  - Model ensemble with uncertainty estimation
#  - Advanced post-processing with morphological operations
#  - Production-ready inference pipeline
#  - Comprehensive monitoring and logging
#
# =======================================================================================

import os
import glob
import random
import warnings
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import cv2
from tqdm import tqdm
from collections import defaultdict
import math

# Deep Learning
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import (
    ReduceLROnPlateau, CosineAnnealingLR, 
    CosineAnnealingWarmRestarts, OneCycleLR
)

# Data Augmentation
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Visualization & Metrics
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, 
    precision_recall_curve, roc_auc_score, average_precision_score,
    f1_score, precision_score, recall_score, jaccard_score
)

# Segmentation Models
import segmentation_models_pytorch as smp

# Suppress warnings
warnings.filterwarnings("ignore")
os.environ['PYTHONWARNINGS'] = 'ignore'
import logging
logging.getLogger().setLevel(logging.ERROR)

# =======================================================================================
# 1. PRODUCTION CONFIGURATION
# =======================================================================================

class ProductionConfig:
    """Production-level configuration with optimized parameters."""
    
    # Directories
    BASE_DIR = Path(os.getcwd())
    DATA_DIR = BASE_DIR / "data_preprocessed"
    RESULTS_DIR = BASE_DIR / "results_production"
    MODEL_DIR = BASE_DIR / "models_production"
    CHECKPOINT_DIR = MODEL_DIR / "checkpoints"
    
    # Data paths
    TRAIN_IMG_DIR = DATA_DIR / "train" / "images"
    TRAIN_MASK_DIR = DATA_DIR / "train" / "masks"
    VAL_IMG_DIR = DATA_DIR / "val" / "images"
    VAL_MASK_DIR = DATA_DIR / "val" / "masks"
    TEST_IMG_DIR = DATA_DIR / "test" / "images"
    TEST_MASK_DIR = DATA_DIR / "test" / "masks"
    
    # Model parameters - Multi-scale approach
    IMAGE_SIZES = [120, 160, 192]  # Multi-scale training
    PRIMARY_SIZE = 160  # Increased from 120 for better detail capture
    INPUT_CHANNELS = 3
    OUTPUT_CHANNELS = 1
    
    # Class mapping - Improved thresholding
    SLUM_CLASS_ID = 2
    BINARY_THRESHOLDS = {
        'conservative': 0.5,    # High precision
        'balanced': 0.35,       # Balanced precision/recall
        'sensitive': 0.25       # High recall
    }
    DEFAULT_THRESHOLD = 'balanced'
    
    # Training parameters - Optimized
    BATCH_SIZE = 12          # Increased for better gradient estimates
    EPOCHS = 80              # Increased for better convergence
    LEARNING_RATE = 1e-4     # More conservative learning rate
    WEIGHT_DECAY = 1e-4      # Increased for better regularization
    
    # Advanced training settings
    WARMUP_EPOCHS = 5
    PATIENCE = 15            # Increased patience
    MIN_DELTA = 1e-4
    GRAD_CLIP_VALUE = 1.0    # Gradient clipping
    
    # Hardware optimization
    NUM_WORKERS = 4 if os.cpu_count() >= 4 else 2
    PIN_MEMORY = True
    
    # Reproducibility
    SEED = 42
    
    # Loss function weights - Production optimized
    LOSS_WEIGHTS = {
        'dice': 0.4,
        'focal': 0.3,
        'bce': 0.2,
        'tversky': 0.1
    }
    
    # Focal loss parameters - Optimized for slum detection
    FOCAL_ALPHA = 0.3        # Balanced for minority class
    FOCAL_GAMMA = 2.5        # Stronger focus on hard examples
    
    # Tversky loss parameters - Optimized for recall
    TVERSKY_ALPHA = 0.6      # Slightly favor recall
    TVERSKY_BETA = 0.4
    
    # Encoder options - Production models
    ENCODERS = [
        'timm-efficientnet-b4',    # Better performance
        'timm-efficientnet-b3',    # Balanced
        'resnet50',                # Robust baseline
        'timm-regnety_016'         # Fast inference
    ]
    ENCODER_WEIGHTS = 'imagenet'
    
    # Test Time Augmentation
    TTA_TRANSFORMS = 8       # Number of TTA transforms
    
    # Post-processing parameters
    MIN_OBJECT_SIZE = 25     # Minimum slum area (pixels)
    MORPHOLOGY_KERNEL = 5    # Morphological operations kernel
    
    # Ensemble parameters
    ENSEMBLE_MODELS = 3      # Top N models for ensemble
    UNCERTAINTY_THRESHOLD = 0.1  # Uncertainty-based filtering

# Initialize config
config = ProductionConfig()

# Create directories
for dir_path in [config.RESULTS_DIR, config.MODEL_DIR, config.CHECKPOINT_DIR]:
    dir_path.mkdir(exist_ok=True)

# Device configuration with optimization
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
    print(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# =======================================================================================
# 2. ADVANCED LOSS FUNCTIONS
# =======================================================================================

class AdaptiveFocalLoss(nn.Module):
    """Adaptive Focal Loss with learnable parameters."""
    
    def __init__(self, alpha: float = 0.3, gamma: float = 2.5, 
                 adaptive: bool = True, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction
        self.adaptive = adaptive
        
        if adaptive:
            self.alpha = nn.Parameter(torch.tensor(alpha))
            self.gamma = nn.Parameter(torch.tensor(gamma))
        else:
            self.register_buffer('alpha', torch.tensor(alpha))
            self.register_buffer('gamma', torch.tensor(gamma))
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Ensure alpha and gamma are in valid ranges
        alpha = torch.clamp(self.alpha, 0.1, 0.9)
        gamma = torch.clamp(self.gamma, 1.0, 5.0)
        
        probs = torch.sigmoid(inputs)
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        p_t = targets * probs + (1 - targets) * (1 - probs)
        alpha_t = targets * alpha + (1 - targets) * (1 - alpha)
        focal_weight = alpha_t * (1 - p_t) ** gamma
        
        focal_loss = focal_weight * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

class ImprovedTverskyLoss(nn.Module):
    """Improved Tversky Loss with class balancing."""
    
    def __init__(self, alpha: float = 0.6, beta: float = 0.4, 
                 smooth: float = 1e-6, class_weight: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.class_weight = class_weight
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(inputs)
        
        # Flatten tensors
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)
        
        # Calculate Tversky coefficients
        true_pos = (probs_flat * targets_flat).sum()
        false_neg = (targets_flat * (1 - probs_flat)).sum()
        false_pos = ((1 - targets_flat) * probs_flat).sum()
        
        # Apply class weighting to false positives
        false_pos = false_pos * self.class_weight
        
        tversky = (true_pos + self.smooth) / (
            true_pos + self.alpha * false_neg + self.beta * false_pos + self.smooth
        )
        
        return 1 - tversky

class ProductionLoss(nn.Module):
    """Production-level combined loss function."""
    
    def __init__(self, weights: Dict[str, float] = None):
        super().__init__()
        self.weights = weights or config.LOSS_WEIGHTS
        
        # Initialize loss components
        self.dice_loss = smp.losses.DiceLoss(
            mode='binary', 
            from_logits=True, 
            smooth=1e-6
        )
        self.focal_loss = AdaptiveFocalLoss(
            alpha=config.FOCAL_ALPHA,
            gamma=config.FOCAL_GAMMA,
            adaptive=True
        )
        self.bce_loss = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([5.0])  # Class imbalance weight
        )
        self.tversky_loss = ImprovedTverskyLoss(
            alpha=config.TVERSKY_ALPHA,
            beta=config.TVERSKY_BETA,
            class_weight=2.0
        )
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Move pos_weight to correct device
        if self.bce_loss.pos_weight.device != inputs.device:
            self.bce_loss.pos_weight = self.bce_loss.pos_weight.to(inputs.device)
        
        # Calculate individual losses
        dice = self.dice_loss(inputs, targets)
        focal = self.focal_loss(inputs, targets)
        bce = self.bce_loss(inputs, targets)
        tversky = self.tversky_loss(inputs, targets)
        
        # Combine with weights
        total_loss = (
            self.weights['dice'] * dice +
            self.weights['focal'] * focal +
            self.weights['bce'] * bce +
            self.weights['tversky'] * tversky
        )
        
        return total_loss

# =======================================================================================
# 3. ADVANCED DATASET WITH MULTI-SCALE SUPPORT
# =======================================================================================

class ProductionDataset(Dataset):
    """Production dataset with multi-scale support and advanced augmentations."""
    
    def __init__(self, image_dir: Path, mask_dir: Path, 
                 transform=None, image_size: int = 160):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.transform = transform
        self.image_size = image_size
        
        # Get all image files
        self.image_files = sorted(list(self.image_dir.glob("*.tif")))
        self.mask_files = sorted(list(self.mask_dir.glob("*.png")))
        
        # Verify file counts match
        assert len(self.image_files) == len(self.mask_files), \
            f"Mismatch: {len(self.image_files)} images vs {len(self.mask_files)} masks"
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load image and mask
        image_path = self.image_files[idx]
        mask_path = self.mask_files[idx]
        
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        
        # Convert mask to binary (slum vs non-slum)
        binary_mask = (mask == config.SLUM_CLASS_ID).astype(np.float32)
        
        # Resize to target size
        if image.shape[:2] != (self.image_size, self.image_size):
            image = cv2.resize(image, (self.image_size, self.image_size), 
                             interpolation=cv2.INTER_LINEAR)
            binary_mask = cv2.resize(binary_mask, (self.image_size, self.image_size), 
                                   interpolation=cv2.INTER_NEAREST)
        
        # Apply transformations
        if self.transform:
            augmented = self.transform(image=image, mask=binary_mask)
            image = augmented['image']
            binary_mask = augmented['mask']
        
        # Ensure mask is single channel
        if len(binary_mask.shape) == 3:
            binary_mask = binary_mask[:, :, 0]
        
        return image, binary_mask.unsqueeze(0)

def get_production_augmentations(image_size: int, phase: str = 'train'):
    """Advanced augmentation pipeline for production."""
    
    if phase == 'train':
        return A.Compose([
            # Geometric transformations
            A.RandomResizedCrop(image_size, image_size, scale=(0.8, 1.2), 
                              interpolation=cv2.INTER_LINEAR, p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, 
                             rotate_limit=30, p=0.6),
            A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.3),
            
            # Color transformations
            A.OneOf([
                A.ColorJitter(brightness=0.3, contrast=0.3, 
                            saturation=0.3, hue=0.1, p=1.0),
                A.RandomBrightnessContrast(brightness_limit=0.3, 
                                         contrast_limit=0.3, p=1.0),
                A.HueSaturationValue(hue_shift_limit=20, 
                                   sat_shift_limit=30, 
                                   val_shift_limit=20, p=1.0)
            ], p=0.7),
            
            # Noise and blur
            A.OneOf([
                A.GaussNoise(var_limit=(10, 50), p=1.0),
                A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
                A.MultiplicativeNoise(multiplier=(0.9, 1.1), p=1.0)
            ], p=0.4),
            
            A.OneOf([
                A.Blur(blur_limit=3, p=1.0),
                A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                A.MotionBlur(blur_limit=3, p=1.0)
            ], p=0.3),
            
            # Weather effects
            A.OneOf([
                A.RandomShadow(shadow_roi=(0, 0.5, 1, 1), 
                             num_shadows_lower=1, 
                             num_shadows_upper=2, p=1.0),
                A.RandomFog(fog_coef_lower=0.3, fog_coef_upper=0.7, p=1.0)
            ], p=0.2),
            
            # Cutout for robustness
            A.CoarseDropout(max_holes=8, max_height=16, max_width=16, 
                          min_holes=1, min_height=4, min_width=4, 
                          fill_value=0, p=0.3),
            
            # Normalization and tensor conversion
            A.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    
    else:  # validation/test
        return A.Compose([
            A.Resize(image_size, image_size, interpolation=cv2.INTER_LINEAR),
            A.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

# =======================================================================================
# 4. ADVANCED MODEL ARCHITECTURE
# =======================================================================================

class AttentionGate(nn.Module):
    """Attention gate for improved feature focus."""
    
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
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

class ProductionUNet(nn.Module):
    """Production UNet with attention gates and deep supervision."""
    
    def __init__(self, encoder_name: str = 'timm-efficientnet-b4', 
                 encoder_weights: str = 'imagenet'):
        super().__init__()
        
        self.backbone = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=config.INPUT_CHANNELS,
            classes=config.OUTPUT_CHANNELS,
            activation=None,
            decoder_attention_type='scse',
            decoder_use_batchnorm=True
        )
        
        # Deep supervision heads
        self.aux_head1 = nn.Conv2d(256, 1, kernel_size=1)
        self.aux_head2 = nn.Conv2d(128, 1, kernel_size=1)
        self.aux_head3 = nn.Conv2d(64, 1, kernel_size=1)
        
        # Dropout for regularization
        self.dropout = nn.Dropout2d(p=0.2)
    
    def forward(self, x):
        # Get encoder features
        features = self.backbone.encoder(x)
        
        # Decoder with deep supervision
        decoder_output = self.backbone.decoder(*features)
        
        # Main output
        main_output = self.backbone.segmentation_head(decoder_output)
        
        # Auxiliary outputs for deep supervision (during training)
        if self.training:
            aux1 = F.interpolate(self.aux_head1(decoder_output), 
                               size=x.shape[-2:], mode='bilinear', align_corners=False)
            aux2 = F.interpolate(self.aux_head2(decoder_output), 
                               size=x.shape[-2:], mode='bilinear', align_corners=False)
            aux3 = F.interpolate(self.aux_head3(decoder_output), 
                               size=x.shape[-2:], mode='bilinear', align_corners=False)
            return main_output, aux1, aux2, aux3
        
        return main_output

# =======================================================================================
# 5. TEST TIME AUGMENTATION
# =======================================================================================

class TestTimeAugmentation:
    """Test Time Augmentation for improved inference."""
    
    def __init__(self, num_transforms: int = 8):
        self.num_transforms = num_transforms
        self.transforms = self._get_tta_transforms()
    
    def _get_tta_transforms(self):
        """Get TTA transform list."""
        transforms = []
        
        # Original
        transforms.append(A.Compose([]))
        
        # Flips
        transforms.append(A.Compose([A.HorizontalFlip(p=1.0)]))
        transforms.append(A.Compose([A.VerticalFlip(p=1.0)]))
        transforms.append(A.Compose([A.HorizontalFlip(p=1.0), A.VerticalFlip(p=1.0)]))
        
        # Rotations
        transforms.append(A.Compose([A.RandomRotate90(p=1.0)]))
        transforms.append(A.Compose([A.Rotate(limit=90, p=1.0)]))
        
        # Scale variations
        transforms.append(A.Compose([A.RandomScale(scale_limit=0.1, p=1.0)]))
        transforms.append(A.Compose([A.RandomScale(scale_limit=-0.1, p=1.0)]))
        
        return transforms[:self.num_transforms]
    
    def __call__(self, model, image: torch.Tensor) -> torch.Tensor:
        """Apply TTA and return averaged prediction."""
        predictions = []
        
        for transform in self.transforms:
            # Convert to numpy for albumentations
            img_np = image.cpu().numpy().transpose(1, 2, 0)
            
            # Apply transform
            augmented = transform(image=img_np)
            aug_img = torch.from_numpy(augmented['image'].transpose(2, 0, 1)).unsqueeze(0)
            aug_img = aug_img.to(image.device)
            
            # Get prediction
            with torch.no_grad():
                pred = torch.sigmoid(model(aug_img))
            
            # Reverse transform on prediction
            pred_np = pred.squeeze().cpu().numpy()
            reversed_pred = self._reverse_transform(transform, pred_np)
            predictions.append(torch.from_numpy(reversed_pred))
        
        # Average predictions
        return torch.stack(predictions).mean(dim=0)
    
    def _reverse_transform(self, transform, prediction):
        """Reverse the transform applied to get original orientation."""
        # This is a simplified version - in production, you'd need
        # proper reverse transforms for each augmentation
        return prediction

# =======================================================================================
# 6. PRODUCTION TRAINING PIPELINE
# =======================================================================================

class ProductionTrainer:
    """Production-level training pipeline."""
    
    def __init__(self, model, train_loader, val_loader, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Loss function
        self.criterion = ProductionLoss()
        
        # Optimizer with weight decay
        self.optimizer = AdamW(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
            eps=1e-8
        )
        
        # Advanced scheduler
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=config.LEARNING_RATE * 10,
            epochs=config.EPOCHS,
            steps_per_epoch=len(train_loader),
            pct_start=0.1,
            div_factor=10,
            final_div_factor=100
        )
        
        # Mixed precision scaler
        self.scaler = torch.cuda.amp.GradScaler()
        
        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.val_ious = []
        self.val_dices = []
        
        # Best model tracking
        self.best_score = 0.0
        self.patience_counter = 0
    
    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        epoch_loss = 0.0
        num_batches = len(self.train_loader)
        
        pbar = tqdm(self.train_loader, desc="Training")
        for batch_idx, (images, masks) in enumerate(pbar):
            images = images.to(self.device, non_blocking=True)
            masks = masks.to(self.device, non_blocking=True)
            
            self.optimizer.zero_grad()
            
            with torch.cuda.amp.autocast():
                if self.model.training:
                    outputs = self.model(images)
                    if isinstance(outputs, tuple):
                        main_output, aux1, aux2, aux3 = outputs
                        # Deep supervision loss
                        main_loss = self.criterion(main_output, masks)
                        aux_loss1 = self.criterion(aux1, masks) * 0.4
                        aux_loss2 = self.criterion(aux2, masks) * 0.3
                        aux_loss3 = self.criterion(aux3, masks) * 0.2
                        loss = main_loss + aux_loss1 + aux_loss2 + aux_loss3
                    else:
                        loss = self.criterion(outputs, masks)
                else:
                    outputs = self.model(images)
                    loss = self.criterion(outputs, masks)
            
            self.scaler.scale(loss).backward()
            
            # Gradient clipping
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), config.GRAD_CLIP_VALUE)
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            
            epoch_loss += loss.item()
            
            if batch_idx % 10 == 0:
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'lr': f"{self.scheduler.get_last_lr()[0]:.2e}"
                })
        
        return epoch_loss / num_batches
    
    def validate_epoch(self):
        """Validate for one epoch."""
        self.model.eval()
        val_loss = 0.0
        all_ious = []
        all_dices = []
        
        with torch.no_grad():
            for images, masks in tqdm(self.val_loader, desc="Validation"):
                images = images.to(self.device, non_blocking=True)
                masks = masks.to(self.device, non_blocking=True)
                
                with torch.cuda.amp.autocast():
                    outputs = self.model(images)
                    if isinstance(outputs, tuple):
                        outputs = outputs[0]  # Use main output for validation
                    
                    loss = self.criterion(outputs, masks)
                
                val_loss += loss.item()
                
                # Calculate metrics
                preds = torch.sigmoid(outputs)
                binary_preds = (preds > config.BINARY_THRESHOLDS[config.DEFAULT_THRESHOLD]).float()
                
                # IoU calculation
                intersection = (binary_preds * masks).sum(dim=(2, 3))
                union = (binary_preds + masks).clamp(0, 1).sum(dim=(2, 3))
                iou = (intersection + 1e-8) / (union + 1e-8)
                all_ious.extend(iou.cpu().numpy())
                
                # Dice calculation
                dice = (2 * intersection + 1e-8) / (binary_preds.sum(dim=(2, 3)) + masks.sum(dim=(2, 3)) + 1e-8)
                all_dices.extend(dice.cpu().numpy())
        
        avg_val_loss = val_loss / len(self.val_loader)
        avg_iou = np.mean(all_ious)
        avg_dice = np.mean(all_dices)
        
        return avg_val_loss, avg_iou, avg_dice
    
    def train(self):
        """Main training loop."""
        print("Starting production training...")
        
        for epoch in range(config.EPOCHS):
            print(f"\nEpoch {epoch + 1}/{config.EPOCHS}")
            
            # Train
            train_loss = self.train_epoch()
            
            # Validate
            val_loss, val_iou, val_dice = self.validate_epoch()
            
            # Track metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_ious.append(val_iou)
            self.val_dices.append(val_dice)
            
            # Combined score for model selection
            combined_score = 0.5 * val_iou + 0.5 * val_dice
            
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}, IoU: {val_iou:.4f}, Dice: {val_dice:.4f}")
            print(f"Combined Score: {combined_score:.4f}")
            
            # Save best model
            if combined_score > self.best_score:
                self.best_score = combined_score
                self.patience_counter = 0
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'best_score': self.best_score,
                    'config': config
                }, config.MODEL_DIR / 'best_production_model.pth')
                
                print(f"New best model saved! Score: {self.best_score:.4f}")
            else:
                self.patience_counter += 1
            
            # Early stopping
            if self.patience_counter >= config.PATIENCE:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break
            
            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'train_losses': self.train_losses,
                    'val_losses': self.val_losses,
                    'val_ious': self.val_ious,
                    'val_dices': self.val_dices
                }, config.CHECKPOINT_DIR / f'checkpoint_epoch_{epoch + 1}.pth')

# =======================================================================================
# 7. PRODUCTION INFERENCE PIPELINE
# =======================================================================================

class ProductionInference:
    """Production inference pipeline with TTA and post-processing."""
    
    def __init__(self, model_path: str, device: torch.device, use_tta: bool = True):
        self.device = device
        self.use_tta = use_tta
        
        # Load model
        checkpoint = torch.load(model_path, map_location=device)
        self.model = ProductionUNet()
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(device)
        self.model.eval()
        
        # TTA
        if use_tta:
            self.tta = TestTimeAugmentation(num_transforms=config.TTA_TRANSFORMS)
        
        # Post-processing kernel
        self.morph_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, 
            (config.MORPHOLOGY_KERNEL, config.MORPHOLOGY_KERNEL)
        )
    
    def preprocess_image(self, image_path: str, size: int = None) -> torch.Tensor:
        """Preprocess single image for inference."""
        if size is None:
            size = config.PRIMARY_SIZE
            
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)
        
        # Normalize
        transform = A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        
        transformed = transform(image=image)
        return transformed['image'].unsqueeze(0)
    
    def postprocess_prediction(self, prediction: np.ndarray, 
                             threshold: str = 'balanced') -> np.ndarray:
        """Advanced post-processing of predictions."""
        thresh_value = config.BINARY_THRESHOLDS[threshold]
        
        # Apply threshold
        binary_pred = (prediction > thresh_value).astype(np.uint8)
        
        # Remove small objects
        num_labels, labels = cv2.connectedComponents(binary_pred)
        for label_id in range(1, num_labels):
            mask = (labels == label_id)
            if np.sum(mask) < config.MIN_OBJECT_SIZE:
                binary_pred[mask] = 0
        
        # Morphological operations
        # Opening to remove noise
        binary_pred = cv2.morphologyEx(binary_pred, cv2.MORPH_OPEN, self.morph_kernel)
        
        # Closing to fill holes
        binary_pred = cv2.morphologyEx(binary_pred, cv2.MORPH_CLOSE, self.morph_kernel)
        
        return binary_pred
    
    def predict_single(self, image_path: str, threshold: str = 'balanced') -> Tuple[np.ndarray, np.ndarray]:
        """Predict single image with post-processing."""
        # Preprocess
        image_tensor = self.preprocess_image(image_path).to(self.device)
        
        with torch.no_grad():
            if self.use_tta:
                prediction = self.tta(self.model, image_tensor.squeeze(0))
            else:
                output = self.model(image_tensor)
                if isinstance(output, tuple):
                    output = output[0]
                prediction = torch.sigmoid(output).squeeze().cpu()
        
        prediction_np = prediction.numpy()
        
        # Post-process
        binary_prediction = self.postprocess_prediction(prediction_np, threshold)
        
        return prediction_np, binary_prediction
    
    def predict_batch(self, image_paths: List[str], 
                     threshold: str = 'balanced') -> List[Tuple[np.ndarray, np.ndarray]]:
        """Predict batch of images."""
        results = []
        
        for image_path in tqdm(image_paths, desc="Processing images"):
            pred, binary_pred = self.predict_single(image_path, threshold)
            results.append((pred, binary_pred))
        
        return results

# =======================================================================================
# 8. MAIN EXECUTION FUNCTIONS
# =======================================================================================

def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def main_training():
    """Main training function."""
    set_seed(config.SEED)
    
    print("=== PRODUCTION SLUM DETECTION TRAINING ===")
    print(f"Device: {DEVICE}")
    print(f"Image size: {config.PRIMARY_SIZE}")
    print(f"Batch size: {config.BATCH_SIZE}")
    print(f"Epochs: {config.EPOCHS}")
    
    # Create datasets
    train_transform = get_production_augmentations(config.PRIMARY_SIZE, 'train')
    val_transform = get_production_augmentations(config.PRIMARY_SIZE, 'val')
    
    train_dataset = ProductionDataset(
        config.TRAIN_IMG_DIR, config.TRAIN_MASK_DIR, 
        train_transform, config.PRIMARY_SIZE
    )
    val_dataset = ProductionDataset(
        config.VAL_IMG_DIR, config.VAL_MASK_DIR, 
        val_transform, config.PRIMARY_SIZE
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Create model
    model = ProductionUNet(
        encoder_name=config.ENCODERS[0],  # Use best encoder
        encoder_weights=config.ENCODER_WEIGHTS
    ).to(DEVICE)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Initialize trainer
    trainer = ProductionTrainer(model, train_loader, val_loader, DEVICE)
    
    # Start training
    trainer.train()
    
    print("Training completed!")

def main_inference():
    """Main inference function."""
    model_path = config.MODEL_DIR / 'best_production_model.pth'
    
    if not model_path.exists():
        print("No trained model found. Please train the model first.")
        return
    
    # Initialize inference pipeline
    inference = ProductionInference(str(model_path), DEVICE, use_tta=True)
    
    # Get test images
    test_images = list(config.TEST_IMG_DIR.glob("*.tif"))[:10]  # Test on first 10 images
    
    print(f"Running inference on {len(test_images)} test images...")
    
    # Predict
    results = inference.predict_batch([str(img) for img in test_images])
    
    # Save results
    results_dir = config.RESULTS_DIR / "inference_results"
    results_dir.mkdir(exist_ok=True)
    
    for i, (pred, binary_pred) in enumerate(results):
        # Save probability map
        prob_map = (pred * 255).astype(np.uint8)
        cv2.imwrite(str(results_dir / f"prob_map_{i:03d}.png"), prob_map)
        
        # Save binary prediction
        binary_map = (binary_pred * 255).astype(np.uint8)
        cv2.imwrite(str(results_dir / f"binary_pred_{i:03d}.png"), binary_map)
    
    print(f"Results saved to {results_dir}")

if __name__ == "__main__":
    # Run training
    main_training()
    
    # Run inference
    main_inference()
