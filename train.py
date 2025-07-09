# File: train.py
# =======================================================================================
#
#  COMPLETE SLUM DETECTION PIPELINE (V12.1 - Corrected GradScaler)
#
# =======================================================================================

import os
import glob
import random
import warnings
from tqdm import tqdm
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import albumentations as albu
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import segmentation_models_pytorch as smp

warnings.filterwarnings("ignore")

# --- 1. Configuration ---
class Config:
    PREPROCESSED_DIR = os.path.join(os.getcwd(), "data_preprocessed")
    RESULTS_DIR = os.path.join(os.getcwd(), "results")
    MODEL_DIR = os.path.join(os.getcwd(), "models")
    
    TRAIN_IMG_DIR = os.path.join(PREPROCESSED_DIR, "train", "images")
    TRAIN_MASK_DIR = os.path.join(PREPROCESSED_DIR, "train", "masks")
    VAL_IMG_DIR = os.path.join(PREPROCESSED_DIR, "val", "images")
    VAL_MASK_DIR = os.path.join(PREPROCESSED_DIR, "val", "masks")
    TEST_IMG_DIR = os.path.join(PREPROCESSED_DIR, "test", "images")
    TEST_MASK_DIR = os.path.join(PREPROCESSED_DIR, "test", "masks")
    
    IMAGESIZE = 224
    CLASSES = 7
    CLASS_NAMES = ['vegetation', 'built-up', 'informal settlements', 'impervious surfaces', 'barren', 'water', 'Unlabelled']
    CLASS_COLORS = [
        (0, 128, 0), (128, 128, 128), (255, 0, 0), (0, 0, 128), 
        (165, 42, 42), (0, 255, 255), (128, 0, 128)
    ]
    
    ENCODER = 'timm-efficientnet-b3'
    ENCODER_WEIGHTS = 'imagenet'
    
    BATCH_SIZE = 16
    EPOCHS = 50
    LEARNING_RATE = 1e-3
    SEED = 42
    NUM_WORKERS = 4

config = Config()
os.makedirs(config.RESULTS_DIR, exist_ok=True)
os.makedirs(config.MODEL_DIR, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 2. Data Pipeline & Helpers ---
def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True

class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_paths = sorted(glob.glob(os.path.join(image_dir, "*.tif")))
        self.mask_paths = sorted(glob.glob(os.path.join(mask_dir, "*.png")))
        self.transform = transform
    def __len__(self): return len(self.image_paths)
    def __getitem__(self, idx):
        image = cv2.cvtColor(cv2.imread(self.image_paths[idx]), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image, mask = augmented['image'], augmented['mask']
        return image, mask.long()

def get_transforms(image_size):
    train_transform = albu.Compose([
        albu.Resize(image_size, image_size),
        albu.HorizontalFlip(p=0.5),
        albu.VerticalFlip(p=0.5),
        albu.Normalize(),
        ToTensorV2(),
    ])
    val_transform = albu.Compose([
        albu.Resize(image_size, image_size),
        albu.Normalize(),
        ToTensorV2(),
    ])
    return train_transform, val_transform

# --- 3. Training & Evaluation Logic ---
def iou_score(pred, target, n_classes):
    pred = torch.argmax(pred, dim=1)
    iou = 0.0
    for cls in range(n_classes):
        pred_inds, target_inds = (pred == cls), (target == cls)
        intersection = (pred_inds & target_inds).long().sum().item()
        union = (pred_inds | target_inds).long().sum().item()
        iou += (intersection / union) if union != 0 else 1.0
    return iou / n_classes

def train_one_epoch(loader, model, optimizer, loss_fn, scaler):
    model.train(); loop = tqdm(loader, desc="Training")
    total_loss = 0
    for images, masks in loop:
        images, masks = images.to(DEVICE), masks.to(DEVICE, dtype=torch.long)
        with torch.amp.autocast(device_type=DEVICE.type):
            predictions = model(images)
            loss = loss_fn(predictions, masks)
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())
    return total_loss / len(loader)

def evaluate_model(loader, model, loss_fn):
    model.eval(); loop = tqdm(loader, desc="Validating")
    total_loss, total_iou = 0, 0
    with torch.no_grad():
        for images, masks in loop:
            images, masks = images.to(DEVICE), masks.to(DEVICE, dtype=torch.long)
            with torch.amp.autocast(device_type=DEVICE.type):
                predictions = model(images)
                loss = loss_fn(predictions, masks)
            iou = iou_score(predictions, masks, config.CLASSES)
            total_loss += loss.item()
            total_iou += iou
            loop.set_postfix(loss=loss.item(), iou=iou)
    return total_loss / len(loader), total_iou / len(loader)

# --- 4. Visualization & Reporting ---
def save_metrics_report(loader, model, save_path_cm, save_path_report):
    print("Generating final report on test set...")
    model.eval()
    all_preds, all_gts = [], []
    with torch.no_grad():
        for images, masks in tqdm(loader, desc="Testing"):
            images = images.to(DEVICE)
            with torch.amp.autocast(device_type=DEVICE.type):
                predictions = model(images)
            all_preds.append(torch.argmax(predictions, dim=1).cpu().numpy())
            all_gts.append(masks.numpy())

    all_preds = np.concatenate([p.flatten() for p in all_preds])
    all_gts = np.concatenate([g.flatten() for g in all_gts])

    cm = confusion_matrix(all_gts, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=config.CLASS_NAMES, yticklabels=config.CLASS_NAMES)
    plt.title('Confusion Matrix'); plt.ylabel('Actual Class'); plt.xlabel('Predicted Class'); plt.xticks(rotation=45, ha='right')
    plt.savefig(save_path_cm, dpi=300, bbox_inches='tight'); plt.close()
    print(f"Confusion matrix saved to: {save_path_cm}")

    report = classification_report(all_gts, all_preds, target_names=config.CLASS_NAMES, zero_division=0)
    print("\n--- Detailed Test Set Classification Report ---\n", report)
    with open(save_path_report, 'w') as f: f.write(report); print(f"Classification report saved to: {save_path_report}")
    
def visualize_test_predictions(dataset, model, save_path, num_samples=10):
    print("--- Visualizing Test Set Predictions ---")
    model.eval()
    plt.figure(figsize=(20, 6 * num_samples))
    indices = random.sample(range(len(dataset)), k=min(num_samples, len(dataset)))
    
    slum_class_idx = config.CLASS_NAMES.index('informal settlements')
    
    for i, idx in enumerate(indices):
        raw_image = cv2.cvtColor(cv2.imread(dataset.image_paths[idx]), cv2.COLOR_BGR2RGB)
        image, gt_mask = dataset[idx] 
        
        with torch.no_grad(), torch.amp.autocast(device_type=DEVICE.type):
            logits = model(image.unsqueeze(0).to(DEVICE))
            probabilities = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
            pred_mask = np.argmax(probabilities, axis=0)
        
        slum_prob_map = probabilities[slum_class_idx, :, :]

        # Plot Original Image
        plt.subplot(num_samples, 4, i * 4 + 1); plt.imshow(raw_image); plt.title("Original Image"); plt.axis("off")
        
        # Plot Ground Truth
        gt_display_mask = np.ma.masked_where(gt_mask.numpy() != slum_class_idx, gt_mask.numpy())
        plt.subplot(num_samples, 4, i * 4 + 2); plt.imshow(raw_image); plt.imshow(gt_display_mask, cmap='Reds', alpha=0.6)
        plt.title("Ground Truth (Slum)"); plt.axis("off")
        
        # Plot Prediction Heatmap
        plt.subplot(num_samples, 4, i * 4 + 3); plt.imshow(raw_image); plt.imshow(slum_prob_map, cmap='Reds', alpha=0.6)
        plt.title("Slum Probability Heatmap"); plt.axis("off")
        
        # Plot Final Prediction
        pred_display_mask = np.ma.masked_where(pred_mask != slum_class_idx, pred_mask)
        plt.subplot(num_samples, 4, i * 4 + 4); plt.imshow(raw_image); plt.imshow(pred_display_mask, cmap='Reds', alpha=0.6)
        plt.title("Final Prediction (Slum)"); plt.axis("off")

    plt.tight_layout(); plt.savefig(save_path, dpi=300); plt.close()
    print(f"Test prediction visualization saved to: {save_path}")

# --- 5. Main Workflow ---
def main():
    set_seed(config.SEED)
    print(f"Using device: {DEVICE}")
    
    train_transform, val_transform = get_transforms(config.IMAGESIZE)
    train_dataset = SegmentationDataset(config.TRAIN_IMG_DIR, config.TRAIN_MASK_DIR, transform=train_transform)
    val_dataset = SegmentationDataset(config.VAL_IMG_DIR, config.VAL_MASK_DIR, transform=val_transform)
    test_dataset = SegmentationDataset(config.TEST_IMG_DIR, config.TEST_MASK_DIR, transform=val_transform)
    
    if not train_dataset.image_paths:
        print(f"FATAL: Training data not found. Ensure data exists in '{config.PREPROCESSED_DIR}'")
        return

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=True)

    model = smp.Unet(
        encoder_name=config.ENCODER, encoder_weights=config.ENCODER_WEIGHTS,
        in_channels=3, classes=config.CLASSES,
    ).to(DEVICE)
    
    loss_fn = smp.losses.DiceLoss(mode='multiclass', from_logits=True)
    optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5, verbose=False)
    
    # CORRECTED: GradScaler does not take any arguments here.
    scaler = torch.amp.GradScaler()
    
    best_iou = 0.0
    history = {'train_loss': [], 'val_loss': [], 'train_iou': [], 'val_iou': []}
    model_path = os.path.join(config.MODEL_DIR, "best_model.pth")
    
    for epoch in range(config.EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{config.EPOCHS} ---")
        train_loss = train_one_epoch(train_loader, model, optimizer, loss_fn, scaler)
        val_loss, val_iou = evaluate_model(val_loader, model, loss_fn)
        
        history['train_loss'].append(train_loss); history['val_loss'].append(val_iou)
        history['train_iou'].append(0); history['val_iou'].append(val_iou) # Placeholder for train_iou
        
        scheduler.step(val_iou)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1} -> Train Loss:{train_loss:.4f} | Val Loss:{val_loss:.4f} | Val IoU:{val_iou:.4f} | LR: {current_lr:.1e}")
        
        if val_iou > best_iou:
            best_iou = val_iou; torch.save(model.state_dict(), model_path)
            print(f"✅ Model Saved! New best validation IoU: {best_iou:.4f}")

    print("\n--- Final Evaluation on Test Set ---")
    model.load_state_dict(torch.load(model_path))
    save_metrics_report(test_loader, model, 
        os.path.join(config.RESULTS_DIR, "test_confusion_matrix.png"),
        os.path.join(config.RESULTS_DIR, "test_classification_report.txt")
    )
    visualize_test_predictions(
        test_dataset, model, os.path.join(config.RESULTS_DIR, "test_predictions_visualization.png")
    )

if __name__ == '__main__':
    main()