# File: run_pipeline.py
# =======================================================================================
#
#  COMPLETE SLUM DETECTION PIPELINE (V10 - HIGH-PERFORMANCE DATA LOADING)
#
# =======================================================================================

import os
import glob
import random
import time
import warnings
from tqdm import tqdm
import numpy as np
import cv2
import tifffile
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset as BaseDataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import albumentations as albu
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# =======================================================================================
# 1. CONFIGURATION
# =======================================================================================
class Config:
    BASE_DIR = os.getcwd()
    DATA_DIR = os.path.join(BASE_DIR, "data")
    RESULTS_DIR = os.path.join(BASE_DIR, "results")
    MODEL_DIR = os.path.join(BASE_DIR, "models")
    TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train", "images")
    TRAIN_MASK_DIR = os.path.join(DATA_DIR, "train", "masks")
    VAL_IMG_DIR = os.path.join(DATA_DIR, "val", "images")
    VAL_MASK_DIR = os.path.join(DATA_DIR, "val", "masks")
    TEST_IMG_DIR = os.path.join(DATA_DIR, "test", "images")
    TEST_MASK_DIR = os.path.join(DATA_DIR, "test", "masks")

    IMAGESIZE = 120
    CLASSES = 7
    CLASS_NAMES = ['vegetation', 'built-up', 'informal settlements', 'impervious surfaces', 'barren', 'water', 'Unlabelled']
    
    # ================== IMPORTANT! UPDATE THIS LIST ==================
    # Use the output from check_colors.py to set the correct (R, G, B) values for each class.
    # The order must match the order in CLASS_NAMES.
    CLASS_COLORS = [
        (0, 128, 0),      # Class 0: vegetation (Green)
        (128, 128, 128),  # Class 1: built-up (Gray)
        (255, 0, 0),      # Class 2: informal settlements (Red)
        (0, 0, 128),      # Class 3: impervious surfaces (Blue)
        (165, 42, 42),    # Class 4: barren (Brown)
        (0, 255, 255),    # Class 5: water (Cyan)
        (128, 0, 128),    # Class 6: Unlabelled (Purple)
    ]
    # ===============================================================

    MODEL_NAME = "UNet"
    BATCH_SIZE = 32
    EPOCHS = 75
    LEARNING_RATE = 1e-4
    SEED = 42
    NUM_WORKERS = 2 

config = Config()

for path in [config.RESULTS_DIR, config.MODEL_DIR]:
    os.makedirs(path, exist_ok=True)

# =======================================================================================
# 2. HELPERS & DEVICE SETUP
# =======================================================================================
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEVICE = get_device()

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

set_seed(config.SEED)

# =======================================================================================
# 3. U-NET MODEL
# =======================================================================================
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.conv(x)

class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__(); self.pool = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_ch, out_ch))
    def forward(self, x): return self.pool(x)

class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]; diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__(); self.conv = nn.Conv2d(in_ch, out_ch, 1)
    def forward(self, x): return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super().__init__()
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128); self.down2 = Down(128, 256); self.down3 = Down(256, 512); self.down4 = Down(512, 1024)
        self.up1 = Up(1024, 512); self.up2 = Up(512, 256); self.up3 = Up(256, 128); self.up4 = Up(128, 64)
        self.outc = OutConv(64, n_classes)
    def forward(self, x):
        x1=self.inc(x); x2=self.down1(x1); x3=self.down2(x2); x4=self.down3(x3); x5=self.down4(x4)
        x = self.up1(x5, x4); x = self.up2(x, x3); x = self.up3(x, x2); x = self.up4(x, x1)
        return self.outc(x)

# =======================================================================================
# 4. DATA PIPELINE
# =======================================================================================
def load_and_convert_mask(mask_path, color_map):
    mask_rgb = cv2.cvtColor(cv2.imread(mask_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    h, w, _ = mask_rgb.shape
    class_mask = np.zeros((h, w), dtype=np.uint8)
    for i, color in enumerate(color_map):
        condition = np.all(mask_rgb == color, axis=-1)
        class_mask[condition] = i
    return class_mask

class SegmentationDataset(BaseDataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_paths = sorted(glob.glob(os.path.join(image_dir, "*.tif")))
        self.mask_paths = sorted(glob.glob(os.path.join(mask_dir, "*.png")))
        self.transform = transform
        assert len(self.image_paths) == len(self.mask_paths), f"Mismatch in {image_dir} and {mask_dir}"
    def __len__(self): return len(self.image_paths)
    def __getitem__(self, idx):
        image = cv2.cvtColor(cv2.imread(self.image_paths[idx]), cv2.COLOR_BGR2RGB)
        mask = load_and_convert_mask(self.mask_paths[idx], config.CLASS_COLORS)
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image, mask = augmented['image'], augmented['mask']
        return image, mask.long()

def get_transforms():
    imagenet_mean, imagenet_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    train_transform = albu.Compose([
        albu.Resize(config.IMAGESIZE, config.IMAGESIZE),
        albu.HorizontalFlip(p=0.5), albu.VerticalFlip(p=0.5), albu.RandomRotate90(p=0.5),
        albu.Normalize(mean=imagenet_mean, std=imagenet_std), ToTensorV2(),
    ])
    val_transform = albu.Compose([
        albu.Resize(config.IMAGESIZE, config.IMAGESIZE),
        albu.Normalize(mean=imagenet_mean, std=imagenet_std), ToTensorV2(),
    ])
    return train_transform, val_transform

# =======================================================================================
# 5. TRAINING, METRICS, AND TESTING
# =======================================================================================
def iou_score(pred, target, n_classes):
    pred = torch.argmax(pred, dim=1); iou = 0.0
    for cls in range(n_classes):
        pred_inds, target_inds = (pred == cls), (target == cls)
        intersection = (pred_inds & target_inds).long().sum().item()
        union = (pred_inds | target_inds).long().sum().item()
        iou += (intersection / union) if union != 0 else 1.0
    return iou / n_classes

def train_one_epoch(loader, model, optimizer, loss_fn, scaler):
    model.train(); loop = tqdm(loader, desc="Training")
    total_loss, total_iou = 0, 0
    for images, masks in loop:
        images, masks = images.to(DEVICE), masks.to(DEVICE)
        with torch.cuda.amp.autocast():
            predictions = model(images)
            loss = loss_fn(predictions, masks)
        optimizer.zero_grad(); scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
        iou = iou_score(predictions, masks, config.CLASSES)
        total_loss += loss.item(); total_iou += iou
        loop.set_postfix(loss=loss.item(), iou=iou)
    return total_loss / len(loader), total_iou / len(loader)

def evaluate_model(loader, model, loss_fn):
    model.eval(); loop = tqdm(loader, desc="Evaluating")
    total_loss, total_iou = 0, 0
    all_preds, all_gts = [], []
    with torch.no_grad():
        for images, masks in loop:
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            with torch.cuda.amp.autocast():
                predictions = model(images)
                loss = loss_fn(predictions, masks)
            iou = iou_score(predictions, masks, config.CLASSES)
            total_loss += loss.item(); total_iou += iou
            all_preds.append(torch.argmax(predictions, dim=1).cpu().numpy())
            all_gts.append(masks.cpu().numpy())
            loop.set_postfix(loss=loss.item(), iou=iou)
    all_preds = np.concatenate(all_preds).flatten(); all_gts = np.concatenate(all_gts).flatten()
    return total_loss / len(loader), total_iou / len(loader), all_gts, all_preds

def calculate_pixel_weights(mask_dir):
    print("Calculating pixel weights..."); mask_paths = glob.glob(os.path.join(mask_dir, "*.png"))
    if not mask_paths:
        print("Warning: No masks found."); return torch.ones(config.CLASSES).to(DEVICE)
    counts = np.zeros(config.CLASSES)
    for path in tqdm(mask_paths, desc="Analyzing masks"):
        mask = load_and_convert_mask(path, config.CLASS_COLORS)
        counts += np.bincount(mask.flatten(), minlength=config.CLASSES)
    total_pixels = counts.sum()
    weights = total_pixels / (config.CLASSES * counts); weights[np.isinf(weights)|np.isnan(weights)] = 0
    weights = torch.tensor(weights, dtype=torch.float32).to(DEVICE)
    print(f"Calculated weights: {weights.cpu().numpy()}"); return weights

# =======================================================================================
# 6. VISUALIZATION & REPORTING
# =======================================================================================
def plot_history(history, save_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5)); fig.suptitle('Training History')
    ax1.plot(history['train_loss'], label='Train Loss'); ax1.plot(history['val_loss'], label='Validation Loss')
    ax1.set_title('Loss Over Epochs'); ax1.legend()
    ax2.plot(history['train_iou'], label='Train IoU'); ax2.plot(history['val_iou'], label='Validation IoU')
    ax2.set_title('Mean IoU Over Epochs'); ax2.legend()
    plt.savefig(save_path, dpi=300); plt.close(); print(f"Training history plot saved to: {save_path}")

def save_metrics_report(gts, preds, save_path_cm, save_path_report):
    cm = confusion_matrix(gts, preds); plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=config.CLASS_NAMES, yticklabels=config.CLASS_NAMES)
    plt.title('Confusion Matrix'); plt.ylabel('Actual Class'); plt.xlabel('Predicted Class'); plt.xticks(rotation=45, ha='right')
    plt.savefig(save_path_cm, dpi=300, bbox_inches='tight'); plt.close(); print(f"Confusion matrix saved to: {save_path_cm}")
    report = classification_report(gts, preds, target_names=config.CLASS_NAMES, zero_division=0)
    print("\n--- Detailed Test Set Classification Report ---\n", report)
    with open(save_path_report, 'w') as f: f.write(report); print(f"Classification report saved to: {save_path_report}")
    
def visualize_test_predictions(dataset, model, save_path, num_samples=10):
    print("--- Visualizing Test Set Predictions ---")
    cmap = ListedColormap(['green', 'gray', 'red', 'blue', 'brown', 'cyan', 'purple'])
    model.eval(); plt.figure(figsize=(20, 5 * num_samples))
    indices = random.sample(range(len(dataset)), k=min(num_samples, len(dataset)))
    
    for i, idx in enumerate(indices):
        raw_image = cv2.cvtColor(cv2.imread(dataset.image_paths[idx]), cv2.COLOR_BGR2RGB)
        image, gt_mask = dataset[idx] 
        with torch.no_grad(), torch.cuda.amp.autocast():
            logits = model(image.unsqueeze(0).to(DEVICE))
            pred_mask = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()
        
        plt.subplot(num_samples, 3, i * 3 + 1); plt.imshow(raw_image); plt.title("Original Image"); plt.axis("off")
        plt.subplot(num_samples, 3, i * 3 + 2); plt.imshow(raw_image); plt.imshow(gt_mask.numpy(), cmap=cmap, alpha=0.6, vmin=0, vmax=config.CLASSES-1)
        plt.title("Ground Truth"); plt.axis("off")
        plt.subplot(num_samples, 3, i * 3 + 3); plt.imshow(raw_image); plt.imshow(pred_mask, cmap=cmap, alpha=0.6, vmin=0, vmax=config.CLASSES-1)
        plt.title("Model Prediction"); plt.axis("off")
    plt.tight_layout(); plt.savefig(save_path, dpi=300); plt.close()
    print(f"Test prediction visualization saved to: {save_path}")

# =======================================================================================
# 7. MAIN EXECUTION BLOCK
# =======================================================================================
if __name__ == '__main__':
    if DEVICE.type == 'cpu':
        print("="*60 + "\nFATAL ERROR: No CUDA-enabled GPU was found.\n" + "="*60); sys.exit()

    train_dataset = SegmentationDataset(config.TRAIN_IMG_DIR, config.TRAIN_MASK_DIR, transform=get_transforms()[0])
    val_dataset = SegmentationDataset(config.VAL_IMG_DIR, config.VAL_MASK_DIR, transform=get_transforms()[1])
    test_dataset = SegmentationDataset(config.TEST_IMG_DIR, config.TEST_MASK_DIR, transform=get_transforms()[1])

    if not train_dataset.image_paths:
        print(f"FATAL: Training data not found in {config.TRAIN_IMG_DIR}.")
    else:
        train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=True)

        model = UNet(n_channels=3, n_classes=config.CLASSES).to(DEVICE)
        loss_weights = calculate_pixel_weights(config.TRAIN_MASK_DIR)
        loss_fn = nn.CrossEntropyLoss(weight=loss_weights)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
        scheduler = ReduceLROnPlateau(optimizer, 'max', patience=5, factor=0.1, verbose=True)
        scaler = torch.cuda.amp.GradScaler()
        
        best_iou = 0.0
        history = {'train_loss': [], 'val_loss': [], 'train_iou': [], 'val_iou': []}
        model_path = os.path.join(config.MODEL_DIR, f"{config.MODEL_NAME}_best_model.pth")

        for epoch in range(config.EPOCHS):
            print(f"\n--- Epoch {epoch+1}/{config.EPOCHS} ---")
            train_loss, train_iou = train_one_epoch(train_loader, model, optimizer, loss_fn, scaler)
            val_loss, val_iou, _, _ = evaluate_model(val_loader, model, loss_fn)
            
            history['train_loss'].append(train_loss); history['val_loss'].append(val_loss)
            history['train_iou'].append(train_iou); history['val_iou'].append(val_iou)
            
            scheduler.step(val_iou)
            print(f"Epoch {epoch+1} -> Train Loss:{train_loss:.4f}, IoU:{train_iou:.4f} | Val Loss:{val_loss:.4f}, IoU:{val_iou:.4f}")
            
            if val_iou > best_iou:
                best_iou = val_iou; torch.save(model.state_dict(), model_path)
                print(f"✅ Model Saved! New best validation IoU: {best_iou:.4f}")

        print("\n--- Final Evaluation on Test Set ---")
        model.load_state_dict(torch.load(model_path))
        test_loss, test_iou, test_gts, test_preds = evaluate_model(test_loader, model, loss_fn)
        print(f"Final Test Metrics -> Loss: {test_loss:.4f}, Mean IoU: {test_iou:.4f}")

        plot_history(history, os.path.join(config.RESULTS_DIR, "training_curves.png"))
        save_metrics_report(
            test_gts, test_preds,
            os.path.join(config.RESULTS_DIR, "test_confusion_matrix.png"),
            os.path.join(config.RESULTS_DIR, "test_classification_report.txt")
        )
        visualize_test_predictions(
            test_dataset, model, os.path.join(config.RESULTS_DIR, "test_predictions_visualization.png")
        )
    print("\n\n--- Pipeline Finished ---")