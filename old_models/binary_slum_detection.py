# File: binary_slum_detection.py
# =======================================================================================
#
#  ENHANCED BINARY SLUM DETECTION PIPELINE
#  UNet-based semantic segmentation for informal settlements detection
#  
#  Features:
#  - Binary classification: Slum (1) vs Non-Slum (0)
#  - Multiple loss functions comparison
#  - Advanced data augmentation
#  - Post-processing with morphological operations
#  - Comprehensive evaluation metrics
#
# =======================================================================================

import os
import glob
import random
import warnings
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import cv2
from tqdm import tqdm
from collections import defaultdict

# Deep Learning
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR

# Data Augmentation
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Visualization & Metrics
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, 
    precision_recall_curve, roc_auc_score, average_precision_score
)

# Segmentation Models
import segmentation_models_pytorch as smp

# Suppress warnings and info messages
warnings.filterwarnings("ignore")
os.environ['PYTHONWARNINGS'] = 'ignore'

# Suppress TensorFlow warnings if present
import logging
logging.getLogger().setLevel(logging.ERROR)

# =======================================================================================
# 1. CONFIGURATION
# =======================================================================================

class Config:
    """Configuration class for the slum detection pipeline."""
    
    # Directories
    BASE_DIR = Path(os.getcwd())
    DATA_DIR = BASE_DIR / "data_preprocessed"  # Use preprocessed data
    RESULTS_DIR = BASE_DIR / "results_binary"
    MODEL_DIR = BASE_DIR / "models_binary"
    
    # Data paths
    TRAIN_IMG_DIR = DATA_DIR / "train" / "images"
    TRAIN_MASK_DIR = DATA_DIR / "train" / "masks"
    VAL_IMG_DIR = DATA_DIR / "val" / "images"
    VAL_MASK_DIR = DATA_DIR / "val" / "masks"
    TEST_IMG_DIR = DATA_DIR / "test" / "images"
    TEST_MASK_DIR = DATA_DIR / "test" / "masks"
    
    # Model parameters
    IMAGE_SIZE = 120
    INPUT_CHANNELS = 3
    OUTPUT_CHANNELS = 1  # Binary segmentation
    
    # Class mapping for binary classification
    SLUM_CLASS_ID = 2  # 'informal settlements' class from original dataset
    BINARY_THRESHOLD = 0.3  # Lower threshold for better recall
    
    # Training parameters
    BATCH_SIZE = 8  # Reduced for better gradient updates
    EPOCHS = 50     # Reduced to prevent overfitting
    LEARNING_RATE = 2e-4  # Slightly higher for better learning
    WEIGHT_DECAY = 1e-5   # Reduced weight decay
    
    # Early stopping
    PATIENCE = 10
    MIN_DELTA = 1e-3
    
    # Hardware
    NUM_WORKERS = 0  # Set to 0 to avoid multiprocessing issues
    PIN_MEMORY = True
    
    # Reproducibility
    SEED = 42
    
    # Loss function weights for combination losses
    DICE_WEIGHT = 0.7  # Increase dice weight for better segmentation
    BCE_WEIGHT = 0.3
    
    # Focal loss parameters - adjusted for class imbalance
    FOCAL_ALPHA = 0.25  # Lower alpha since slums are minority class
    FOCAL_GAMMA = 2.0
    
    # Tversky loss parameters
    TVERSKY_ALPHA = 0.7
    TVERSKY_BETA = 0.3

    # Encoder options
    ENCODERS = ['resnet34', 'efficientnet-b0', 'timm-efficientnet-b3']
    ENCODER_WEIGHTS = 'imagenet'

# Initialize config
config = Config()

# Create directories
config.RESULTS_DIR.mkdir(exist_ok=True)
config.MODEL_DIR.mkdir(exist_ok=True)

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Clear CUDA cache if using GPU
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB total")

# =======================================================================================
# 2. UTILITY FUNCTIONS
# =======================================================================================

# Fix: CUDA memory management utility
def clear_cuda_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def create_binary_mask(mask: np.ndarray, slum_class_id: int) -> np.ndarray:
    """Convert multi-class mask to binary slum vs non-slum mask."""
    binary_mask = (mask == slum_class_id).astype(np.uint8)
    return binary_mask

# --- STRONGER DATA AUGMENTATION ---
def get_training_augmentation(image_size: int) -> A.Compose:
    # Multi-scale, CutMix, MixUp, and strong augmentations
    return A.Compose([
        A.OneOf([
            A.Resize(image_size, image_size, always_apply=True),
            A.RandomResizedCrop(image_size, image_size, scale=(0.7, 1.0), ratio=(0.9, 1.1), always_apply=True),
        ], p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Rotate(limit=30, p=0.5, border_mode=cv2.BORDER_REFLECT),
        A.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.15, rotate_limit=20, p=0.5, border_mode=cv2.BORDER_REFLECT),
        A.OneOf([
            A.ElasticTransform(p=1.0, alpha=120, sigma=6, alpha_affine=3.6),
            A.GridDistortion(p=1.0, num_steps=5, distort_limit=0.3),
            A.OpticalDistortion(p=1.0, distort_limit=0.5, shift_limit=0.5),
        ], p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
        A.RandomGamma(gamma_limit=(80, 120), p=0.3),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.3),
        A.RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=0.3),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        A.GaussianBlur(blur_limit=3, p=0.2),
        A.CoarseDropout(max_holes=8, max_height=8, max_width=8, min_holes=1, min_height=4, min_width=4, fill_value=0, p=0.3),
        A.OneOf([
            A.Cutout(num_holes=8, max_h_size=8, max_w_size=8, fill_value=0, p=1.0),
            A.Cutout(num_holes=4, max_h_size=16, max_w_size=16, fill_value=0, p=1.0),
        ], p=0.2),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

def get_validation_augmentation(image_size: int) -> A.Compose:
    """Get validation augmentation pipeline (minimal processing)."""
    
    val_transform = A.Compose([
        A.Resize(image_size, image_size, always_apply=True),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    
    return val_transform

# =======================================================================================
# 4. DATASET CLASS
# =======================================================================================

class BinarySlumDataset(Dataset):
    """Dataset class for binary slum detection with optional oversampling."""
    
    def __init__(
        self, 
        image_dir: Path, 
        mask_dir: Path, 
        transform: Optional[A.Compose] = None,
        slum_class_id: int = 2,
        oversample_slum: bool = True
    ):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.transform = transform
        self.slum_class_id = slum_class_id
        self.oversample_slum = oversample_slum
        
        # Get image and mask paths
        self.image_paths = sorted(list(self.image_dir.glob("*.tif")))
        self.mask_paths = sorted(list(self.mask_dir.glob("*.png")))
        
        # Verify matching files
        assert len(self.image_paths) == len(self.mask_paths), \
            f"Number of images ({len(self.image_paths)}) != number of masks ({len(self.mask_paths)})"
        
        # Oversample slum images if enabled
        if self.oversample_slum:
            slum_indices = []
            for i, mask_path in enumerate(self.mask_paths):
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if np.sum(mask == self.slum_class_id) > 0:
                    slum_indices.append(i)
            # Duplicate slum images to balance
            n_dup = max(1, int(0.5 * len(self.image_paths) / (len(slum_indices) + 1)))
            self.image_paths += [self.image_paths[i] for i in slum_indices for _ in range(n_dup)]
            self.mask_paths += [self.mask_paths[i] for i in slum_indices for _ in range(n_dup)]
        
        print(f"Dataset initialized with {len(self.image_paths)} samples (oversample_slum={self.oversample_slum})")
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Load image
        image = cv2.imread(str(self.image_paths[idx]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load mask and convert to binary
        mask = cv2.imread(str(self.mask_paths[idx]), cv2.IMREAD_GRAYSCALE)
        binary_mask = create_binary_mask(mask, self.slum_class_id)
        
        # Apply augmentations
        if self.transform:
            augmented = self.transform(image=image, mask=binary_mask)
            image = augmented['image']
            binary_mask = augmented['mask']
        
        # Convert mask to tensor and add channel dimension
        if not isinstance(binary_mask, torch.Tensor):
            binary_mask = torch.from_numpy(binary_mask).float()
        
        if binary_mask.dim() == 2:
            binary_mask = binary_mask.unsqueeze(0)  # Add channel dimension
        
        return image, binary_mask

# =======================================================================================
# 5. LOSS FUNCTIONS
# =======================================================================================

class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance."""
    
    def __init__(self, alpha: float = 0.75, gamma: float = 2.0, reduction: str = 'mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(inputs)
        
        # Calculate BCE loss
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # Calculate focal weight
        p_t = targets * probs + (1 - targets) * (1 - probs)
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        
        # Apply focal weight
        focal_loss = focal_weight * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class TverskyLoss(nn.Module):
    """Tversky Loss for handling class imbalance."""
    
    def __init__(self, alpha: float = 0.7, beta: float = 0.3, smooth: float = 1e-6):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(inputs)
        
        # Flatten tensors
        probs = probs.view(-1)
        targets = targets.view(-1)
        
        # Calculate Tversky coefficients
        true_pos = (probs * targets).sum()
        false_neg = (targets * (1 - probs)).sum()
        false_pos = ((1 - targets) * probs).sum()
        
        # Calculate Tversky index
        tversky = (true_pos + self.smooth) / (
            true_pos + self.alpha * false_neg + self.beta * false_pos + self.smooth
        )
        
        return 1 - tversky

class CombinedLoss(nn.Module):
    """Combined BCE and Dice Loss with class weighting."""
    
    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5, pos_weight: float = 5.0):
        super(CombinedLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))
        self.dice_loss = smp.losses.DiceLoss(mode='binary', from_logits=True, smooth=1e-6)
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Move pos_weight to same device as inputs
        if self.bce_loss.pos_weight.device != inputs.device:
            self.bce_loss.pos_weight = self.bce_loss.pos_weight.to(inputs.device)
        
        bce = self.bce_loss(inputs, targets)
        dice = self.dice_loss(inputs, targets)
        return self.bce_weight * bce + self.dice_weight * dice

# --- ADVANCED LOSS: BCE + DICE + FOCAL ---
class AdvancedComboLoss(nn.Module):
    def __init__(self, bce_weight=0.3, dice_weight=0.4, focal_weight=0.3, pos_weight=8.0):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))
        self.dice = smp.losses.DiceLoss(mode='binary', from_logits=True)
        self.focal = FocalLoss(alpha=0.25, gamma=2.0)
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
    def forward(self, inputs, targets):
        bce = self.bce(inputs, targets)
        dice = self.dice(inputs, targets)
        focal = self.focal(inputs, targets)
        return self.bce_weight * bce + self.dice_weight * dice + self.focal_weight * focal

class AsymmetricFocalTverskyLoss(nn.Module):
    def __init__(self, delta=0.7, gamma=0.75, smooth=1e-6):
        super().__init__()
        self.delta = delta
        self.gamma = gamma
        self.smooth = smooth
    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        targets = targets.float()
        tp = (inputs * targets).sum()
        fn = ((1 - inputs) * targets).sum()
        fp = (inputs * (1 - targets)).sum()
        tversky = (tp + self.smooth) / (tp + self.delta * fn + (1 - self.delta) * fp + self.smooth)
        loss = torch.pow((1 - tversky), self.gamma)
        return loss

class OHEMLoss(nn.Module):
    def __init__(self, base_loss, ratio=0.25):
        super().__init__()
        self.base_loss = base_loss
        self.ratio = ratio
    def forward(self, inputs, targets):
        loss = self.base_loss(inputs, targets)
        if loss.dim() > 1:
            loss = loss.view(-1)
        num_hard = int(self.ratio * loss.numel())
        if num_hard < 1:
            return loss.mean()
        topk_loss, _ = torch.topk(loss, num_hard)
        return topk_loss.mean()

def get_loss_function(loss_name: str) -> nn.Module:
    """Get loss function by name."""
    
    loss_functions = {
        'bce': nn.BCEWithLogitsLoss(pos_weight=torch.tensor([3.0])),  # Weight positive class more
        'bce_dice': CombinedLoss(
            bce_weight=config.BCE_WEIGHT, 
            dice_weight=config.DICE_WEIGHT,
            pos_weight=5.0  # Higher weight for slum class
        ),
        'focal': FocalLoss(
            alpha=config.FOCAL_ALPHA, 
            gamma=config.FOCAL_GAMMA
        ),
        'tversky': TverskyLoss(
            alpha=config.TVERSKY_ALPHA, 
            beta=config.TVERSKY_BETA
        ),
        'advanced_combo': AdvancedComboLoss(),
        'asym_focal_tversky': AsymmetricFocalTverskyLoss(),
        'ohem_bce': OHEMLoss(nn.BCEWithLogitsLoss(pos_weight=torch.tensor([5.0])), ratio=0.25),
    }
    
    if loss_name not in loss_functions:
        raise ValueError(f"Unknown loss function: {loss_name}")
    
    return loss_functions[loss_name]

# =======================================================================================
# 6. METRICS
# =======================================================================================

def calculate_iou(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    """Calculate Intersection over Union (IoU) for binary segmentation."""
    try:
        pred_binary = (torch.sigmoid(pred) > threshold).float()
        target_binary = target.float()
        
        intersection = (pred_binary * target_binary).sum()
        union = pred_binary.sum() + target_binary.sum() - intersection
        
        if union == 0:
            return 1.0 if intersection == 0 else 0.0
        
        iou = (intersection / union).item()
        return max(0.0, min(1.0, iou))  # Clamp to [0,1]
    except Exception as e:
        print(f"Error calculating IoU: {str(e)}")
        return 0.0

def calculate_dice_coefficient(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    """Calculate Dice coefficient for binary segmentation."""
    try:
        pred_binary = (torch.sigmoid(pred) > threshold).float()
        target_binary = target.float()
        
        intersection = (pred_binary * target_binary).sum()
        dice = (2.0 * intersection) / (pred_binary.sum() + target_binary.sum() + 1e-8)
        
        return max(0.0, min(1.0, dice.item()))  # Clamp to [0,1]
    except Exception as e:
        print(f"Error calculating Dice: {str(e)}")
        return 0.0

def calculate_pixel_accuracy(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    """Calculate pixel-wise accuracy."""
    try:
        pred_binary = (torch.sigmoid(pred) > threshold).float()
        target_binary = target.float()
        
        correct = (pred_binary == target_binary).float().sum()
        total = target_binary.numel()
        
        if total == 0:
            return 0.0
        
        accuracy = (correct / total).item()
        return max(0.0, min(1.0, accuracy))  # Clamp to [0,1]
    except Exception as e:
        print(f"Error calculating pixel accuracy: {str(e)}")
        return 0.0

class MetricsTracker:
    """Track training and validation metrics."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.losses = []
        self.ious = []
        self.dice_scores = []
        self.pixel_accuracies = []
    
    def update(self, loss: float, iou: float, dice: float, pixel_acc: float):
        self.losses.append(loss)
        self.ious.append(iou)
        self.dice_scores.append(dice)
        self.pixel_accuracies.append(pixel_acc)
    
    def get_averages(self) -> Dict[str, float]:
        # Always return all expected keys, even if empty
        result = {
            'loss': np.mean(self.losses) if self.losses else float('nan'),
            'iou': np.mean(self.ious) if self.ious else float('nan'),
            'dice': np.mean(self.dice_scores) if self.dice_scores else float('nan'),
            'pixel_accuracy': np.mean(self.pixel_accuracies) if self.pixel_accuracies else float('nan')
        }
        # Convert NaN to 0.0 to avoid errors
        for key in result:
            if np.isnan(result[key]):
                result[key] = 0.0
        return result

# =======================================================================================
# 7. MODEL DEFINITION
# =======================================================================================

def create_model(architecture: str = 'unet', encoder_name: str = 'timm-efficientnet-b3', encoder_weights: str = 'imagenet', deep_supervision: bool = False) -> nn.Module:
    """Create segmentation model with advanced options."""
    if architecture == 'unet':
        model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=config.INPUT_CHANNELS,
            classes=config.OUTPUT_CHANNELS,
            activation=None,
            decoder_attention_type='scse',
        )
    elif architecture == 'unetplusplus':
        model = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=config.INPUT_CHANNELS,
            classes=config.OUTPUT_CHANNELS,
            activation=None,
            decoder_attention_type='scse',
        )
    elif architecture == 'fpn':
        model = smp.FPN(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=config.INPUT_CHANNELS,
            classes=config.OUTPUT_CHANNELS,
            activation=None,
        )
    elif architecture == 'deeplabv3+':
        model = smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=config.INPUT_CHANNELS,
            classes=config.OUTPUT_CHANNELS,
            activation=None,
        )
    elif architecture == 'segformer':
        try:
            import torch
            import timm
            from segmentation_models_pytorch.encoders import get_encoder
            from segmentation_models_pytorch.base import SegmentationHead
            encoder = get_encoder(
                encoder_name,
                in_channels=config.INPUT_CHANNELS,
                depth=5,
                weights=encoder_weights
            )
            class SegFormerHead(nn.Module):
                def __init__(self, in_channels, out_channels):
                    super().__init__()
                    self.head = SegmentationHead(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=1
                    )
                def forward(self, x):
                    return self.head(x)
            model = nn.Sequential(
                encoder,
                SegFormerHead(encoder.out_channels[-1], config.OUTPUT_CHANNELS)
            )
        except Exception as e:
            raise ImportError(f"SegFormer requires timm and SMP >= 0.3.0: {e}")
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
    if deep_supervision and hasattr(model, 'decoder'):
        model.decoder.deep_supervision = True
    return model

class ModelEnsemble(nn.Module):
    def __init__(self, models: list):
        super().__init__()
        self.models = nn.ModuleList(models)
    def forward(self, x):
        preds = [torch.sigmoid(m(x)) for m in self.models]
        return torch.mean(torch.stack(preds, dim=0), dim=0)

# =======================================================================================
# 8. TRAINING AND VALIDATION
# =======================================================================================

def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device
) -> Dict[str, float]:
    """Train model for one epoch."""
    
    model.train()
    metrics_tracker = MetricsTracker()
    
    pbar = tqdm(dataloader, desc="Training", leave=False)
    for batch_idx, (images, masks) in enumerate(pbar):
        try:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            if masks.dtype != torch.float32:
                masks = masks.float()
            
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = loss_fn(outputs, masks)
            
            # Check for invalid loss
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: Invalid loss detected: {loss.item()}")
                continue
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # Calculate metrics
            with torch.no_grad():
                iou = calculate_iou(outputs, masks)
                dice = calculate_dice_coefficient(outputs, masks)
                pixel_acc = calculate_pixel_accuracy(outputs, masks)
                metrics_tracker.update(loss.item(), iou, dice, pixel_acc)
                
                # Update progress bar less frequently to reduce overhead
                if batch_idx % 10 == 0:
                    pbar.set_postfix({
                        'loss': f"{loss.item():.4f}",
                        'iou': f"{iou:.4f}",
                        'dice': f"{dice:.4f}"
                    })
                    
        except Exception as e:
            print(f"Error in training batch {batch_idx}: {str(e)}")
            continue
    
    return metrics_tracker.get_averages()

def validate_model(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device
) -> Dict[str, float]:
    """Validate model."""
    
    model.eval()
    metrics_tracker = MetricsTracker()
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validation", leave=False)
        for batch_idx, (images, masks) in enumerate(pbar):
            try:
                images = images.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)
                if masks.dtype != torch.float32:
                    masks = masks.float()
                
                with torch.cuda.amp.autocast():
                    outputs = model(images)
                    loss = loss_fn(outputs, masks)
                
                # Check for invalid loss
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"Warning: Invalid validation loss detected: {loss.item()}")
                    continue
                
                # Calculate metrics
                iou = calculate_iou(outputs, masks)
                dice = calculate_dice_coefficient(outputs, masks)
                pixel_acc = calculate_pixel_accuracy(outputs, masks)
                metrics_tracker.update(loss.item(), iou, dice, pixel_acc)
                
                # Update progress bar less frequently
                if batch_idx % 10 == 0:
                    pbar.set_postfix({
                        'loss': f"{loss.item():.4f}",
                        'iou': f"{iou:.4f}",
                        'dice': f"{dice:.4f}"
                    })
                    
            except Exception as e:
                print(f"Error in validation batch {batch_idx}: {str(e)}")
                continue
    
    return metrics_tracker.get_averages()

# =======================================================================================
# 9. TRAINING LOOP
# =======================================================================================

class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(self, patience: int = 15, min_delta: float = 1e-4, mode: str = 'max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_score = None
        self.counter = 0
        self.early_stop = False
        
        if mode == 'max':
            self.is_better = lambda current, best: current > best + min_delta
        else:
            self.is_better = lambda current, best: current < best - min_delta
    
    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
        elif self.is_better(score, self.best_score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    loss_name: str,
    encoder_name: str,
    epochs: int = 100,
    lr: float = 1e-4,
    weight_decay: float = 1e-4
) -> Dict:
    """Train model with specified configuration."""
    
    # Setup
    device = DEVICE
    model = model.to(device)
    
    # Loss function and optimizer
    loss_fn = get_loss_function(loss_name)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Better learning rate scheduler
    scheduler = ReduceLROnPlateau(
        optimizer, 
        mode='max', 
        patience=3,  # Reduced patience for faster adaptation
        factor=0.7,  # Less aggressive reduction
        verbose=True,
        min_lr=1e-6
    )
    scaler = torch.cuda.amp.GradScaler()
    
    # Early stopping
    early_stopping = EarlyStopping(patience=config.PATIENCE, min_delta=config.MIN_DELTA, mode='max')
    
    # Training history
    history = {
        'train_loss': [], 'train_iou': [], 'train_dice': [], 'train_pixel_acc': [],
        'val_loss': [], 'val_iou': [], 'val_dice': [], 'val_pixel_acc': []
    }
    
    best_iou = 0.0
    model_name = f"unet_{encoder_name}_{loss_name}"
    model_path = config.MODEL_DIR / f"{model_name}.pth"
    
    print(f"\nTraining {model_name}...")
    print(f"Device: {device}")
    print(f"Loss function: {loss_name}")
    print(f"Encoder: {encoder_name}")
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        
        # Clear CUDA cache every few epochs
        if epoch % 5 == 0:
            clear_cuda_cache()
        
        # Training
        train_metrics = train_one_epoch(model, train_loader, optimizer, loss_fn, scaler, device)
        
        # Validation
        val_metrics = validate_model(model, val_loader, loss_fn, device)
        
        # Update history - ensure all keys exist
        for key in ['loss', 'iou', 'dice', 'pixel_accuracy']:
            train_key = f'train_{key}'
            val_key = f'val_{key}'
            if train_key not in history:
                history[train_key] = []
            if val_key not in history:
                history[val_key] = []
            
            train_value = train_metrics.get(key, float('nan'))
            val_value = val_metrics.get(key, float('nan'))
            history[train_key].append(train_value)
            history[val_key].append(val_value)
        
        # Learning rate scheduling
        current_val_iou = val_metrics.get('iou', 0.0)
        scheduler.step(current_val_iou)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Print epoch summary
        train_loss = train_metrics.get('loss', float('nan'))
        train_iou = train_metrics.get('iou', float('nan'))
        train_dice = train_metrics.get('dice', float('nan'))
        train_pixel_acc = train_metrics.get('pixel_accuracy', float('nan'))
        
        val_loss = val_metrics.get('loss', float('nan'))
        val_iou = val_metrics.get('iou', float('nan'))
        val_dice = val_metrics.get('dice', float('nan'))
        val_pixel_acc = val_metrics.get('pixel_accuracy', float('nan'))
        
        print(f"Train - Loss: {train_loss:.4f}, IoU: {train_iou:.4f}, "
              f"Dice: {train_dice:.4f}, Pixel Acc: {train_pixel_acc:.4f}")
        print(f"Val   - Loss: {val_loss:.4f}, IoU: {val_iou:.4f}, "
              f"Dice: {val_dice:.4f}, Pixel Acc: {val_pixel_acc:.4f}")
        print(f"LR: {current_lr:.2e}")
        
        # Save best model
        current_val_iou = val_metrics.get('iou', 0.0)
        if current_val_iou > best_iou:
            best_iou = current_val_iou
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_iou': best_iou,
                'history': history
            }, model_path)
            print(f"[OK] New best model saved! IoU: {best_iou:.4f}")
        
        # Early stopping
        if early_stopping(current_val_iou):
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break
    
    return {
        'model_name': model_name,
        'best_iou': best_iou,
        'history': history,
        'model_path': str(model_path)
    }

# =======================================================================================
# 10. EVALUATION AND VISUALIZATION
# =======================================================================================

def evaluate_on_test_set(
    model: nn.Module, 
    test_loader: DataLoader, 
    model_name: str,
    apply_postprocessing: bool = True,
    save_dir: Path = None,
    ensemble_models=None
) -> Dict:
    """Comprehensive evaluation on test set with per-image metrics and visualizations."""
    model.eval()
    device = DEVICE
    all_predictions = []
    all_targets = []
    all_probabilities = []
    per_image_metrics = []
    if save_dir is None:
        save_dir = config.RESULTS_DIR
    error_map_dir = save_dir / f"error_maps_{model_name}"
    error_map_dir.mkdir(exist_ok=True, parents=True)
    print(f"\nEvaluating {model_name} on test set...")
    with torch.no_grad():
        for idx, (images, masks) in enumerate(tqdm(test_loader, desc="Testing")):
            images = images.to(device)
            masks_np = masks.cpu().numpy()
            with torch.cuda.amp.autocast():
                probabilities = predict_with_tta(model, images, device, ensemble_models=ensemble_models)
                probabilities_np = probabilities.cpu().numpy()
            predictions = (probabilities_np > config.BINARY_THRESHOLD).astype(np.uint8)
            if apply_postprocessing:
                for i in range(predictions.shape[0]):
                    orig_img = images[i].cpu().numpy().transpose(1, 2, 0)
                    predictions[i, 0] = apply_morphological_postprocessing(predictions[i, 0], orig_image=orig_img)
            all_predictions.append(predictions)
            all_targets.append(masks_np)
            all_probabilities.append(probabilities_np)
            # Per-image metrics and error maps
            for i in range(predictions.shape[0]):
                pred = predictions[i, 0]
                gt = masks_np[i, 0] if masks_np.shape[1] == 1 else masks_np[i]
                iou = np.sum((pred & gt)) / (np.sum((pred | gt)) + 1e-8)
                dice = 2 * np.sum((pred & gt)) / (np.sum(pred) + np.sum(gt) + 1e-8)
                acc = np.mean(pred == gt)
                per_image_metrics.append({'idx': idx * test_loader.batch_size + i, 'iou': iou, 'dice': dice, 'pixel_acc': acc})
                # Save error map
                img = (images[i].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                plot_error_map(img, gt, pred, error_map_dir / f"error_map_{idx * test_loader.batch_size + i}.png")
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    all_probabilities = np.concatenate(all_probabilities, axis=0)
    pred_flat = all_predictions.flatten()
    target_flat = all_targets.flatten()
    prob_flat = all_probabilities.flatten()
    from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
    metrics = {
        'pixel_accuracy': accuracy_score(target_flat, pred_flat),
        'precision': precision_score(target_flat, pred_flat, zero_division=0),
        'recall': recall_score(target_flat, pred_flat, zero_division=0),
        'f1_score': f1_score(target_flat, pred_flat, zero_division=0),
        'iou': calculate_iou(
            torch.from_numpy(all_probabilities), 
            torch.from_numpy(all_targets), 
            config.BINARY_THRESHOLD
        ),
        'dice': calculate_dice_coefficient(
            torch.from_numpy(all_probabilities), 
            torch.from_numpy(all_targets), 
            config.BINARY_THRESHOLD
        )
    }
    if np.sum(target_flat) > 0:
        try:
            metrics['auc_roc'] = roc_auc_score(target_flat, prob_flat)
            metrics['auc_pr'] = average_precision_score(target_flat, prob_flat)
        except ValueError:
            metrics['auc_roc'] = 0.0
            metrics['auc_pr'] = 0.0
    else:
        metrics['auc_roc'] = 0.0
        metrics['auc_pr'] = 0.0
    # Save confusion matrix and classification report
    cm = confusion_matrix(target_flat, pred_flat)
    cr = classification_report(target_flat, pred_flat, output_dict=True)
    np.save(save_dir / f"confusion_matrix_{model_name}.npy", cm)
    with open(save_dir / f"classification_report_{model_name}.json", 'w') as f:
        json.dump(cr, f, indent=2)
    # Save per-image metrics
    with open(save_dir / f"per_image_metrics_{model_name}.json", 'w') as f:
        json.dump(per_image_metrics, f, indent=2)
    # Pixel accuracy plot
    plt.figure(figsize=(8, 4))
    plt.hist([m['pixel_acc'] for m in per_image_metrics], bins=20, color='blue', alpha=0.7)
    plt.title('Per-image Pixel Accuracy')
    plt.xlabel('Pixel Accuracy')
    plt.ylabel('Count')
    plt.savefig(save_dir / f"pixel_accuracy_hist_{model_name}.png", dpi=200)
    plt.close()
    return metrics

def plot_training_history(history: Dict, model_name: str, save_path: Path):
    """Plot training history."""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Training History - {model_name}', fontsize=16)
    
    metrics = ['loss', 'iou', 'dice', 'pixel_acc']
    titles = ['Loss', 'IoU', 'Dice Coefficient', 'Pixel Accuracy']
    
    for i, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[i // 2, i % 2]
        
        train_key = f'train_{metric}'
        val_key = f'val_{metric}'
        
        if train_key in history and val_key in history:
            epochs = range(1, len(history[train_key]) + 1)
            ax.plot(epochs, history[train_key], label='Train', marker='o', markersize=3)
            ax.plot(epochs, history[val_key], label='Validation', marker='s', markersize=3)
            ax.set_xlabel('Epoch')
            ax.set_ylabel(title)
            ax.set_title(title)
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def visualize_predictions(
    model: nn.Module,
    dataset: BinarySlumDataset,
    model_name: str,
    num_samples: int = 8,
    save_path: Optional[Path] = None
):
    """Visualize model predictions on test samples."""
    
    model.eval()
    device = DEVICE
    
    # Randomly select samples
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
    
    fig, axes = plt.subplots(len(indices), 4, figsize=(16, 4 * len(indices)))
    if len(indices) == 1:
        axes = axes.reshape(1, -1)
    
    with torch.no_grad():
        for i, idx in enumerate(indices):
            # Get original image (without normalization for display)
            image_path = dataset.image_paths[idx]
            mask_path = dataset.mask_paths[idx]
            
            # Load original image
            original_image = cv2.imread(str(image_path))
            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
            original_image = cv2.resize(original_image, (config.IMAGE_SIZE, config.IMAGE_SIZE))
            
            # Load ground truth mask
            gt_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            gt_mask = cv2.resize(gt_mask, (config.IMAGE_SIZE, config.IMAGE_SIZE))
            gt_binary_mask = create_binary_mask(gt_mask, config.SLUM_CLASS_ID)
            
            # Get model input
            image_tensor, mask_tensor = dataset[idx]
            image_input = image_tensor.unsqueeze(0).to(device)
            
            # Get prediction
            with torch.cuda.amp.autocast():
                output = model(image_input)
                prob_map = torch.sigmoid(output).squeeze().cpu().numpy()
                pred_mask = (prob_map > config.BINARY_THRESHOLD).astype(np.uint8)
            
            # Apply post-processing
            pred_mask_processed = apply_morphological_postprocessing(pred_mask)
            
            # Plot
            axes[i, 0].imshow(original_image)
            axes[i, 0].set_title('Original Image')
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(original_image)
            axes[i, 1].imshow(gt_binary_mask, alpha=0.6, cmap='Reds')
            axes[i, 1].set_title('Ground Truth')
            axes[i, 1].axis('off')
            
            axes[i, 2].imshow(original_image)
            axes[i, 2].imshow(prob_map, alpha=0.6, cmap='Reds', vmin=0, vmax=1)
            axes[i, 2].set_title('Probability Map')
            axes[i, 2].axis('off')
            
            axes[i, 3].imshow(original_image)
            axes[i, 3].imshow(pred_mask_processed, alpha=0.6, cmap='Reds')
            axes[i, 3].set_title('Final Prediction')
            axes[i, 3].axis('off')
    
    plt.suptitle(f'Predictions - {model_name}', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Prediction visualization saved to: {save_path}")
    
    plt.close()

# --- TTA FOR INFERENCE ---
def predict_with_tta(model: nn.Module, image: torch.Tensor, device: torch.device, scales=[1.0, 0.75, 1.25], ensemble_models=None) -> torch.Tensor:
    model.eval()
    predictions = []
    with torch.no_grad():
        models = ensemble_models if ensemble_models is not None else [model]
        for m in models:
            for scale in scales:
                if scale != 1.0:
                    size = [int(image.shape[2] * scale), int(image.shape[3] * scale)]
                    img_scaled = F.interpolate(image, size=size, mode='bilinear', align_corners=False)
                else:
                    img_scaled = image
                pred = m(img_scaled.to(device))
                pred = F.interpolate(pred, size=(image.shape[2], image.shape[3]), mode='bilinear', align_corners=False)
                predictions.append(torch.sigmoid(pred))
                # Flips
                for dims in [[3], [2], [2, 3]]:
                    pred_flip = m(torch.flip(img_scaled.to(device), dims=dims))
                    pred_flip = torch.flip(pred_flip, dims=dims)
                    pred_flip = F.interpolate(pred_flip, size=(image.shape[2], image.shape[3]), mode='bilinear', align_corners=False)
                    predictions.append(torch.sigmoid(pred_flip))
    final_pred = torch.mean(torch.stack(predictions), dim=0)
    return final_pred

# --- ROBUST POST-PROCESSING ---
def apply_morphological_postprocessing(mask: np.ndarray, remove_small_objects: int = None, fill_holes: bool = True, opening_kernel_size: int = 5, closing_kernel_size: int = 7, use_crf: bool = False, orig_image: np.ndarray = None) -> np.ndarray:
    # Adaptive area threshold
    if remove_small_objects is None:
        remove_small_objects = max(100, int(0.001 * mask.size))
    if mask.dtype != np.uint8:
        mask = (mask * 255).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < remove_small_objects:
            mask[labels == i] = 0
    if opening_kernel_size > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (opening_kernel_size, opening_kernel_size))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    if fill_holes and closing_kernel_size > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (closing_kernel_size, closing_kernel_size))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = (mask > 0).astype(np.uint8)
    # Optional CRF post-processing
    if use_crf and orig_image is not None:
        try:
            import pydensecrf.densecrf as dcrf
            from pydensecrf.utils import unary_from_softmax, create_pairwise_bilateral
            h, w = mask.shape
            d = dcrf.DenseCRF2D(w, h, 2)
            unary = unary_from_softmax(np.stack([1 - mask, mask], axis=0))
            d.setUnaryEnergy(unary)
            feats = create_pairwise_bilateral(sdims=(10, 10), schan=(13, 13, 13), img=orig_image, chdim=2)
            d.addPairwiseEnergy(feats, compat=10)
            Q = d.inference(5)
            mask = np.argmax(Q, axis=0).reshape((h, w)).astype(np.uint8)
        except Exception as e:
            print(f"[WARN] CRF post-processing failed: {e}")
    return mask

# --- ADVANCED VISUALIZATION: ERROR MAPS ---
def plot_error_map(image, gt_mask, pred_mask, save_path):
    error = np.abs(gt_mask.astype(np.float32) - pred_mask.astype(np.float32))
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title('Original')
    plt.axis('off')
    plt.subplot(1, 3, 2)
    plt.imshow(gt_mask, cmap='gray')
    plt.title('Ground Truth')
    plt.axis('off')
    plt.subplot(1, 3, 3)
    plt.imshow(error, cmap='hot')
    plt.title('Error Map')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

# --- MAIN: USE STRONGER ENCODER, LARGER INPUT, ADVANCED LOSS ---
def main():
    config.IMAGE_SIZE = 224
    config.BATCH_SIZE = 8
    architectures = ['unet', 'unetplusplus', 'fpn', 'deeplabv3+']
    encoders = ['timm-efficientnet-b3', 'resnet50', 'resnet34']
    loss_functions = ['advanced_combo', 'asym_focal_tversky', 'ohem_bce', 'bce_dice', 'focal']
    print("[INFO] Starting Binary Slum Detection Pipeline")
    print(f"Configuration: {config.IMAGE_SIZE}x{config.IMAGE_SIZE} images, batch size {config.BATCH_SIZE}")
    set_seed(config.SEED)
    print("\n[INFO] Creating data loaders...")
    train_transform = get_training_augmentation(config.IMAGE_SIZE)
    val_transform = get_validation_augmentation(config.IMAGE_SIZE)
    train_dataset = BinarySlumDataset(
        config.TRAIN_IMG_DIR,
        config.TRAIN_MASK_DIR,
        transform=train_transform,
        slum_class_id=config.SLUM_CLASS_ID,
        oversample_slum=True
    )
    val_dataset = BinarySlumDataset(
        config.VAL_IMG_DIR,
        config.VAL_MASK_DIR,
        transform=val_transform,
        slum_class_id=config.SLUM_CLASS_ID,
        oversample_slum=False
    )
    test_dataset = BinarySlumDataset(
        config.TEST_IMG_DIR,
        config.TEST_MASK_DIR,
        transform=val_transform,
        slum_class_id=config.SLUM_CLASS_ID,
        oversample_slum=False
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )
    print(f"[INFO] Data loaded: {len(train_dataset)} train, {len(val_dataset)} val, {len(test_dataset)} test")
    print("\n[INFO] Starting experiments...")
    all_results = []
    for arch in architectures:
        for encoder in encoders:
            for loss_fn in loss_functions:
                try:
                    print(f"\n[INFO] Experiment: {arch} + {encoder} + {loss_fn}")
                    model = create_model(architecture=arch, encoder_name=encoder, encoder_weights=config.ENCODER_WEIGHTS)
                    # SWA wrapper
                    from torch.optim.swa_utils import AveragedModel, SWALR
                    optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
                    scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)
                    scaler = torch.cuda.amp.GradScaler()
                    # Train model
                    result = train_model(
                        model=model,
                        train_loader=train_loader,
                        val_loader=val_loader,
                        loss_name=loss_fn,
                        encoder_name=encoder,
                        epochs=config.EPOCHS,
                        lr=config.LEARNING_RATE,
                        weight_decay=config.WEIGHT_DECAY
                    )
                    # SWA
                    swa_model = AveragedModel(model)
                    swa_scheduler = SWALR(optimizer, swa_lr=1e-5)
                    for _ in range(3):
                        train_one_epoch(swa_model, train_loader, optimizer, get_loss_function(loss_fn), scaler, DEVICE)
                        swa_scheduler.step()
                    # Ensemble: model + swa_model
                    ensemble = ModelEnsemble([model, swa_model])
                    # Evaluate
                    test_metrics = evaluate_on_test_set(ensemble, test_loader, f"{arch}_{encoder}_{loss_fn}", apply_postprocessing=True)
                    result['encoder'] = encoder
                    result['loss_function'] = loss_fn
                    result['test_metrics'] = test_metrics
                    all_results.append(result)
                    print(f"\n[OK] Completed: {arch} + {encoder} + {loss_fn}")
                    print(f"Best Val IoU: {result['best_iou']:.4f}")
                    print(f"Test IoU: {test_metrics['iou']:.4f}")
                    print(f"Test F1: {test_metrics['f1_score']:.4f}")
                except Exception as e:
                    print(f"[FAIL] Failed: {arch} + {encoder} + {loss_fn} - {str(e)}")
                    continue
    
    # Save comprehensive results
    print("\n[INFO] Saving comprehensive results...")
    
    # Create comparison table
    comparison_data = []
    for result in all_results:
        comparison_data.append({
            'Model': result['model_name'],
            'Encoder': result['encoder'],
            'Loss': result['loss_function'],
            'Val IoU': result['best_val_iou'],
            'Test IoU': result['test_metrics']['iou'],
            'Test Dice': result['test_metrics']['dice'],
            'Test F1': result['test_metrics']['f1_score'],
            'Test Precision': result['test_metrics']['precision'],
            'Test Recall': result['test_metrics']['recall'],
            'Test Pixel Acc': result['test_metrics']['pixel_accuracy'],
            'AUC-ROC': result['test_metrics']['auc_roc'],
            'AUC-PR': result['test_metrics']['auc_pr']
        })
    
    # Save results as JSON
    results_path = config.RESULTS_DIR / "experiment_results.json"
    with open(results_path, 'w') as f:
        json.dump(comparison_data, f, indent=2)
    
    # Create comparison visualization
    if comparison_data:
        import pandas as pd
        
        df = pd.DataFrame(comparison_data)
        
        # Plot comparison
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # IoU comparison
        ax1 = axes[0, 0]
        models = df['Model'].values
        test_iou = df['Test IoU'].values
        bars1 = ax1.bar(range(len(models)), test_iou)
        ax1.set_title('Test IoU by Model')
        ax1.set_ylabel('IoU')
        ax1.set_xticks(range(len(models)))
        ax1.set_xticklabels(models, rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, value in zip(bars1, test_iou):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
        
        # F1 Score comparison
        ax2 = axes[0, 1]
        test_f1 = df['Test F1'].values
        bars2 = ax2.bar(range(len(models)), test_f1)
        ax2.set_title('Test F1 Score by Model')
        ax2.set_ylabel('F1 Score')
        ax2.set_xticks(range(len(models)))
        ax2.set_xticklabels(models, rotation=45, ha='right')
        
        for bar, value in zip(bars2, test_f1):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
        
        # Precision vs Recall
        ax3 = axes[1, 0]
        precision = df['Test Precision'].values
        recall = df['Test Recall'].values
        scatter = ax3.scatter(recall, precision, s=100, alpha=0.7)
        ax3.set_xlabel('Recall')
        ax3.set_ylabel('Precision')
        ax3.set_title('Precision vs Recall')
        ax3.grid(True, alpha=0.3)
        
        # Add model labels
        for i, model in enumerate(models):
            ax3.annotate(model, (recall[i], precision[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # AUC comparison
        ax4 = axes[1, 1]
        auc_roc = df['AUC-ROC'].values
        auc_pr = df['AUC-PR'].values
        
        x = np.arange(len(models))
        width = 0.35
        
        bars3 = ax4.bar(x - width/2, auc_roc, width, label='AUC-ROC')
        bars4 = ax4.bar(x + width/2, auc_pr, width, label='AUC-PR')
        
        ax4.set_title('AUC Scores by Model')
        ax4.set_ylabel('AUC')
        ax4.set_xticks(x)
        ax4.set_xticklabels(models, rotation=45, ha='right')
        ax4.legend()
        
        plt.tight_layout()
        plt.savefig(config.RESULTS_DIR / "model_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Print summary table
        print("\n[INFO] EXPERIMENT SUMMARY")
        print("=" * 120)
        print(df.round(4).to_string(index=False))

        # Find best models robustly (handle empty or failed experiments)
        best_iou_idx = None
        best_f1_idx = None
        if not df.empty and 'Test IoU' in df and df['Test IoU'].notnull().any():
            best_iou_idx = df['Test IoU'].idxmax()
        if not df.empty and 'Test F1' in df and df['Test F1'].notnull().any():
            best_f1_idx = df['Test F1'].idxmax()

        print(f"\n[INFO] BEST MODELS:")
        if best_iou_idx is not None:
            print(f"Best IoU: {df.loc[best_iou_idx, 'Model']} (IoU: {df.loc[best_iou_idx, 'Test IoU']:.4f})")
        else:
            print("Best IoU: N/A")
        if best_f1_idx is not None:
            print(f"Best F1:  {df.loc[best_f1_idx, 'Model']} (F1: {df.loc[best_f1_idx, 'Test F1']:.4f})")
        else:
            print("Best F1: N/A")

    print(f"\n[OK] All experiments completed! Results saved to: {config.RESULTS_DIR}")

if __name__ == "__main__":
    main()
