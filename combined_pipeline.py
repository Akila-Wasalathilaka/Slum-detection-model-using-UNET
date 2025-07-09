"""
Combined production slum detection pipeline.
This file combines all modules for easy deployment as a single script.
"""
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
import argparse

# Deep Learning
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import OneCycleLR

# Data Augmentation
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Segmentation Models
import segmentation_models_pytorch as smp

# Suppress warnings
warnings.filterwarnings("ignore")
os.environ['PYTHONWARNINGS'] = 'ignore'

# =======================================================================================
# CONFIGURATION
# =======================================================================================

class ProductionConfig:
    BASE_DIR = Path(os.getcwd())
    DATA_DIR = BASE_DIR / "data_preprocessed"
    RESULTS_DIR = BASE_DIR / "results_production"
    MODEL_DIR = BASE_DIR / "models_production"
    CHECKPOINT_DIR = MODEL_DIR / "checkpoints"
    TRAIN_IMG_DIR = DATA_DIR / "train" / "images"
    TRAIN_MASK_DIR = DATA_DIR / "train" / "masks"
    VAL_IMG_DIR = DATA_DIR / "val" / "images"
    VAL_MASK_DIR = DATA_DIR / "val" / "masks"
    TEST_IMG_DIR = DATA_DIR / "test" / "images"
    TEST_MASK_DIR = DATA_DIR / "test" / "masks"
    IMAGE_SIZES = [120, 160, 192]
    PRIMARY_SIZE = 160
    INPUT_CHANNELS = 3
    OUTPUT_CHANNELS = 1
    SLUM_CLASS_ID = 2
    BINARY_THRESHOLDS = {'conservative': 0.5, 'balanced': 0.35, 'sensitive': 0.25}
    DEFAULT_THRESHOLD = 'balanced'
    BATCH_SIZE = 12
    EPOCHS = 80
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4
    WARMUP_EPOCHS = 5
    PATIENCE = 15
    MIN_DELTA = 1e-4
    GRAD_CLIP_VALUE = 1.0
    NUM_WORKERS = 4 if os.cpu_count() >= 4 else 2
    PIN_MEMORY = True
    SEED = 42
    LOSS_WEIGHTS = {'dice': 0.4, 'focal': 0.3, 'bce': 0.2, 'tversky': 0.1}
    FOCAL_ALPHA = 0.3
    FOCAL_GAMMA = 2.5
    TVERSKY_ALPHA = 0.6
    TVERSKY_BETA = 0.4
    ENCODERS = ['timm-efficientnet-b4', 'timm-efficientnet-b3', 'resnet50', 'timm-regnety_016']
    ENCODER_WEIGHTS = 'imagenet'
    TTA_TRANSFORMS = 8
    MIN_OBJECT_SIZE = 25
    MORPHOLOGY_KERNEL = 5
    ENSEMBLE_MODELS = 3
    UNCERTAINTY_THRESHOLD = 0.1

config = ProductionConfig()

# =======================================================================================
# LOSS FUNCTIONS
# =======================================================================================

class AdaptiveFocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.3, gamma: float = 2.5, adaptive: bool = True, reduction: str = 'mean'):
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
    def __init__(self, alpha: float = 0.6, beta: float = 0.4, smooth: float = 1e-6, class_weight: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.class_weight = class_weight
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(inputs)
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)
        true_pos = (probs_flat * targets_flat).sum()
        false_neg = (targets_flat * (1 - probs_flat)).sum()
        false_pos = ((1 - targets_flat) * probs_flat).sum() * self.class_weight
        tversky = (true_pos + self.smooth) / (true_pos + self.alpha * false_neg + self.beta * false_pos + self.smooth)
        return 1 - tversky

class ProductionLoss(nn.Module):
    def __init__(self, weights: Dict[str, float], config):
        super().__init__()
        self.weights = weights
        self.dice_loss = smp.losses.DiceLoss(mode='binary', from_logits=True, smooth=1e-6)
        self.focal_loss = AdaptiveFocalLoss(alpha=config.FOCAL_ALPHA, gamma=config.FOCAL_GAMMA, adaptive=True)
        self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([5.0]))
        self.tversky_loss = ImprovedTverskyLoss(alpha=config.TVERSKY_ALPHA, beta=config.TVERSKY_BETA, class_weight=2.0)
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.bce_loss.pos_weight.device != inputs.device:
            self.bce_loss.pos_weight = self.bce_loss.pos_weight.to(inputs.device)
        dice = self.dice_loss(inputs, targets)
        focal = self.focal_loss(inputs, targets)
        bce = self.bce_loss(inputs, targets)
        tversky = self.tversky_loss(inputs, targets)
        total_loss = (
            self.weights['dice'] * dice +
            self.weights['focal'] * focal +
            self.weights['bce'] * bce +
            self.weights['tversky'] * tversky
        )
        return total_loss

# =======================================================================================
# DATASET
# =======================================================================================

class ProductionDataset(Dataset):
    def __init__(self, image_dir: Path, mask_dir: Path, transform=None, image_size: int = 160):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.transform = transform
        self.image_size = image_size
        self.image_files = sorted(list(self.image_dir.glob("*.tif")))
        self.mask_files = sorted(list(self.mask_dir.glob("*.png")))
        assert len(self.image_files) == len(self.mask_files), f"Mismatch: {len(self.image_files)} images vs {len(self.mask_files)} masks"
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        mask_path = self.mask_files[idx]
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        binary_mask = (mask == config.SLUM_CLASS_ID).astype(np.float32)
        if image.shape[:2] != (self.image_size, self.image_size):
            image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
            binary_mask = cv2.resize(binary_mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        if self.transform:
            augmented = self.transform(image=image, mask=binary_mask)
            image = augmented['image']
            binary_mask = augmented['mask']
        if len(binary_mask.shape) == 3:
            binary_mask = binary_mask[:, :, 0]
        return image, binary_mask.unsqueeze(0)

def get_production_augmentations(image_size: int, phase: str = 'train'):
    if phase == 'train':
        return A.Compose([
            A.RandomResizedCrop(image_size, image_size, scale=(0.8, 1.2), interpolation=cv2.INTER_LINEAR, p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=30, p=0.6),
            A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.3),
            A.OneOf([
                A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=1.0),
                A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1.0),
                A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1.0)
            ], p=0.7),
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
            A.OneOf([
                A.RandomShadow(shadow_roi=(0, 0.5, 1, 1), num_shadows_lower=1, num_shadows_upper=2, p=1.0),
                A.RandomFog(fog_coef_lower=0.3, fog_coef_upper=0.7, p=1.0)
            ], p=0.2),
            A.CoarseDropout(max_holes=8, max_height=16, max_width=16, min_holes=1, min_height=4, min_width=4, fill_value=0, p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(image_size, image_size, interpolation=cv2.INTER_LINEAR),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

# =======================================================================================
# MODEL
# =======================================================================================

class AttentionGate(nn.Module):
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
    def __init__(self, encoder_name: str = 'timm-efficientnet-b4', encoder_weights: str = 'imagenet'):
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
        self.aux_head1 = nn.Conv2d(256, 1, kernel_size=1)
        self.aux_head2 = nn.Conv2d(128, 1, kernel_size=1)
        self.aux_head3 = nn.Conv2d(64, 1, kernel_size=1)
        self.dropout = nn.Dropout2d(p=0.2)
    
    def forward(self, x):
        features = self.backbone.encoder(x)
        decoder_output = self.backbone.decoder(*features)
        main_output = self.backbone.segmentation_head(decoder_output)
        if self.training:
            aux1 = F.interpolate(self.aux_head1(decoder_output), size=x.shape[-2:], mode='bilinear', align_corners=False)
            aux2 = F.interpolate(self.aux_head2(decoder_output), size=x.shape[-2:], mode='bilinear', align_corners=False)
            aux3 = F.interpolate(self.aux_head3(decoder_output), size=x.shape[-2:], mode='bilinear', align_corners=False)
            return main_output, aux1, aux2, aux3
        return main_output

# =======================================================================================
# UTILITIES
# =======================================================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def setup_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True
        print(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    return device

def create_directories(config):
    for dir_path in [config.RESULTS_DIR, config.MODEL_DIR, config.CHECKPOINT_DIR]:
        dir_path.mkdir(exist_ok=True, parents=True)

# =======================================================================================
# TRAINER
# =======================================================================================

class ProductionTrainer:
    def __init__(self, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, device: torch.device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.criterion = ProductionLoss(config.LOSS_WEIGHTS, config)
        self.optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY, eps=1e-8)
        self.scheduler = OneCycleLR(self.optimizer, max_lr=config.LEARNING_RATE * 10, epochs=config.EPOCHS, steps_per_epoch=len(train_loader), pct_start=0.1, div_factor=10, final_div_factor=100)
        self.scaler = torch.cuda.amp.GradScaler()
        self.train_losses = []
        self.val_losses = []
        self.val_ious = []
        self.val_dices = []
        self.best_score = 0.0
        self.patience_counter = 0
    
    def train_epoch(self) -> float:
        self.model.train()
        epoch_loss = 0.0
        num_batches = len(self.train_loader)
        pbar = tqdm(self.train_loader, desc="Training")
        for batch_idx, (images, masks) in enumerate(pbar):
            images = images.to(self.device, non_blocking=True)
            masks = masks.to(self.device, non_blocking=True)
            self.optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = self.model(images)
                if isinstance(outputs, tuple):
                    main_output, aux1, aux2, aux3 = outputs
                    main_loss = self.criterion(main_output, masks)
                    aux_loss1 = self.criterion(aux1, masks) * 0.4
                    aux_loss2 = self.criterion(aux2, masks) * 0.3
                    aux_loss3 = self.criterion(aux3, masks) * 0.2
                    loss = main_loss + aux_loss1 + aux_loss2 + aux_loss3
                else:
                    loss = self.criterion(outputs, masks)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), config.GRAD_CLIP_VALUE)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            epoch_loss += loss.item()
            if batch_idx % 10 == 0:
                pbar.set_postfix({'loss': f"{loss.item():.4f}", 'lr': f"{self.scheduler.get_last_lr()[0]:.2e}"})
        return epoch_loss / num_batches
    
    def validate_epoch(self) -> Tuple[float, float, float]:
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
                        outputs = outputs[0]
                    loss = self.criterion(outputs, masks)
                val_loss += loss.item()
                preds = torch.sigmoid(outputs)
                binary_preds = (preds > config.BINARY_THRESHOLDS[config.DEFAULT_THRESHOLD]).float()
                intersection = (binary_preds * masks).sum(dim=(2, 3))
                union = (binary_preds + masks).clamp(0, 1).sum(dim=(2, 3))
                iou = (intersection + 1e-8) / (union + 1e-8)
                all_ious.extend(iou.cpu().numpy())
                dice = (2 * intersection + 1e-8) / (binary_preds.sum(dim=(2, 3)) + masks.sum(dim=(2, 3)) + 1e-8)
                all_dices.extend(dice.cpu().numpy())
        avg_val_loss = val_loss / len(self.val_loader)
        avg_iou = np.mean(all_ious)
        avg_dice = np.mean(all_dices)
        return avg_val_loss, avg_iou, avg_dice
    
    def train(self) -> None:
        print("Starting production training...")
        for epoch in range(config.EPOCHS):
            print(f"\nEpoch {epoch + 1}/{config.EPOCHS}")
            train_loss = self.train_epoch()
            val_loss, val_iou, val_dice = self.validate_epoch()
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_ious.append(val_iou)
            self.val_dices.append(val_dice)
            combined_score = 0.5 * val_iou + 0.5 * val_dice
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}, IoU: {val_iou:.4f}, Dice: {val_dice:.4f}")
            print(f"Combined Score: {combined_score:.4f}")
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
            if self.patience_counter >= config.PATIENCE:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break

# =======================================================================================
# INFERENCE
# =======================================================================================

class ProductionInference:
    def __init__(self, model_path: str, device: torch.device, use_tta: bool = True):
        self.device = device
        self.use_tta = use_tta
        checkpoint = torch.load(model_path, map_location=device)
        self.model = ProductionUNet()
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(device)
        self.model.eval()
        self.morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (config.MORPHOLOGY_KERNEL, config.MORPHOLOGY_KERNEL))
    
    def preprocess_image(self, image_path: str, size: int = None) -> torch.Tensor:
        if size is None:
            size = config.PRIMARY_SIZE
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)
        transform = A.Compose([A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ToTensorV2()])
        transformed = transform(image=image)
        return transformed['image'].unsqueeze(0)
    
    def postprocess_prediction(self, prediction: np.ndarray, threshold: str = 'balanced') -> np.ndarray:
        thresh_value = config.BINARY_THRESHOLDS[threshold]
        binary_pred = (prediction > thresh_value).astype(np.uint8)
        num_labels, labels = cv2.connectedComponents(binary_pred)
        for label_id in range(1, num_labels):
            mask = (labels == label_id)
            if np.sum(mask) < config.MIN_OBJECT_SIZE:
                binary_pred[mask] = 0
        binary_pred = cv2.morphologyEx(binary_pred, cv2.MORPH_OPEN, self.morph_kernel)
        binary_pred = cv2.morphologyEx(binary_pred, cv2.MORPH_CLOSE, self.morph_kernel)
        return binary_pred
    
    def predict_single(self, image_path: str, threshold: str = 'balanced') -> Tuple[np.ndarray, np.ndarray]:
        image_tensor = self.preprocess_image(image_path).to(self.device)
        with torch.no_grad():
            output = self.model(image_tensor)
            if isinstance(output, tuple):
                output = output[0]
            prediction = torch.sigmoid(output).squeeze().cpu()
        prediction_np = prediction.numpy()
        binary_prediction = self.postprocess_prediction(prediction_np, threshold)
        return prediction_np, binary_prediction
    
    def predict_batch(self, image_paths: List[str], threshold: str = 'balanced') -> List[Tuple[np.ndarray, np.ndarray]]:
        results = []
        for image_path in tqdm(image_paths, desc="Processing images"):
            pred, binary_pred = self.predict_single(image_path, threshold)
            results.append((pred, binary_pred))
        return results

# =======================================================================================
# MAIN FUNCTIONS
# =======================================================================================

def main_training():
    set_seed(config.SEED)
    device = setup_device()
    create_directories(config)
    
    print("=== PRODUCTION SLUM DETECTION TRAINING ===")
    print(f"Device: {device}")
    print(f"Image size: {config.PRIMARY_SIZE}")
    print(f"Batch size: {config.BATCH_SIZE}")
    print(f"Epochs: {config.EPOCHS}")
    
    train_transform = get_production_augmentations(config.PRIMARY_SIZE, 'train')
    val_transform = get_production_augmentations(config.PRIMARY_SIZE, 'val')
    
    train_dataset = ProductionDataset(config.TRAIN_IMG_DIR, config.TRAIN_MASK_DIR, train_transform, config.PRIMARY_SIZE)
    val_dataset = ProductionDataset(config.VAL_IMG_DIR, config.VAL_MASK_DIR, val_transform, config.PRIMARY_SIZE)
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    model = ProductionUNet(encoder_name=config.ENCODERS[0], encoder_weights=config.ENCODER_WEIGHTS).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    trainer = ProductionTrainer(model, train_loader, val_loader, device)
    trainer.train()
    
    print("Training completed!")

def main_inference():
    model_path = config.MODEL_DIR / 'best_production_model.pth'
    
    if not model_path.exists():
        print("No trained model found. Please train the model first.")
        return
    
    device = setup_device()
    inference = ProductionInference(str(model_path), device, use_tta=False)
    test_images = list(config.TEST_IMG_DIR.glob("*.tif"))[:10]
    
    print(f"Running inference on {len(test_images)} test images...")
    
    results = inference.predict_batch([str(img) for img in test_images])
    
    results_dir = config.RESULTS_DIR / "inference_results"
    results_dir.mkdir(exist_ok=True)
    
    for i, (pred, binary_pred) in enumerate(results):
        prob_map = (pred * 255).astype('uint8')
        cv2.imwrite(str(results_dir / f"prob_map_{i:03d}.png"), prob_map)
        binary_map = (binary_pred * 255).astype('uint8')
        cv2.imwrite(str(results_dir / f"binary_pred_{i:03d}.png"), binary_map)
    
    print(f"Results saved to {results_dir}")

def main():
    parser = argparse.ArgumentParser(description='Production Slum Detection Pipeline - Combined Version')
    parser.add_argument('--mode', choices=['train', 'infer', 'both'], default='both', help='Operation mode: train, infer, or both')
    parser.add_argument('--model_path', type=str, help='Path to model for inference (optional)')
    
    args = parser.parse_args()
    
    if args.mode in ['train', 'both']:
        main_training()
    
    if args.mode in ['infer', 'both']:
        main_inference()

if __name__ == "__main__":
    main()
