import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, jaccard_score
from tqdm import tqdm
import segmentation_models_pytorch as smp
import warnings
warnings.filterwarnings('ignore')

# Check for CUDA
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

class SlumDataset(Dataset):
    """Dataset class for slum detection"""
    
    def __init__(self, image_dir, mask_dir, transform=None, target_size=(120, 120)):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.target_size = target_size
        
        # Get image files
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith('.tif')]
        self.image_files.sort()
        
        # Class mapping based on analysis
        self.slum_rgb = (250, 235, 185)  # Informal settlements
        self.class_colors = {
            (40, 120, 240): 0,    # Water
            (80, 140, 50): 0,     # Vegetation  
            (100, 100, 150): 0,   # Water/Other
            (200, 160, 40): 0,    # Built-up/Impervious
            (200, 200, 200): 0,   # Built-up/Impervious
            (250, 235, 185): 1,   # Informal settlements (SLUMS) - TARGET
            (0, 0, 0): 0,         # Unlabelled/Background
        }
        
        print(f"Dataset initialized with {len(self.image_files)} images")
        print(f"Target RGB for slums: {self.slum_rgb}")
    
    def __len__(self):
        return len(self.image_files)
    
    def rgb_to_binary_mask(self, rgb_mask):
        """Convert RGB mask to binary slum mask"""
        binary_mask = np.zeros(rgb_mask.shape[:2], dtype=np.uint8)
        
        # Check each pixel for slum class
        for color, class_id in self.class_colors.items():
            mask = np.all(rgb_mask == color, axis=2)
            binary_mask[mask] = class_id
        
        return binary_mask
    
    def __getitem__(self, idx):
        # Load image
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = np.array(Image.open(img_path).convert('RGB'))
        
        # Load mask
        mask_file = self.image_files[idx].replace('.tif', '.png')
        mask_path = os.path.join(self.mask_dir, mask_file)
        rgb_mask = np.array(Image.open(mask_path).convert('RGB'))
        
        # Convert RGB mask to binary
        binary_mask = self.rgb_to_binary_mask(rgb_mask)
        
        # Ensure target size
        if image.shape[:2] != self.target_size:
            image = cv2.resize(image, self.target_size)
            binary_mask = cv2.resize(binary_mask, self.target_size, interpolation=cv2.INTER_NEAREST)
        
        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image, mask=binary_mask)
            image = augmented['image']
            binary_mask = augmented['mask']
        
        # Convert to tensors
        if not torch.is_tensor(image):
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
        if not torch.is_tensor(binary_mask):
            binary_mask = torch.from_numpy(binary_mask).long()
        
        return image, binary_mask

def get_transforms(phase='train'):
    """Get augmentation transforms"""
    
    if phase == 'train':
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Rotate(limit=30, p=0.3),
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.3
            ),
            A.ElasticTransform(
                alpha=1, sigma=50, alpha_affine=50,
                p=0.2
            ),
            A.GridDistortion(
                num_steps=5, distort_limit=0.1,
                p=0.2
            ),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    
    def __init__(self, alpha=0.75, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class TverskyLoss(nn.Module):
    """Tversky Loss for imbalanced segmentation"""
    
    def __init__(self, alpha=0.7, beta=0.3, smooth=1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
    
    def forward(self, inputs, targets):
        inputs = torch.softmax(inputs, dim=1)
        inputs = inputs[:, 1, :, :]  # Take slum class
        targets = targets.float()
        
        # Flatten
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        # True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()
        FP = ((1 - targets) * inputs).sum()
        FN = (targets * (1 - inputs)).sum()
        
        tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
        
        return 1 - tversky

class DiceLoss(nn.Module):
    """Dice Loss for segmentation"""
    
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, inputs, targets):
        inputs = torch.softmax(inputs, dim=1)
        inputs = inputs[:, 1, :, :]  # Take slum class
        targets = targets.float()
        
        # Flatten
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        
        return 1 - dice

class CombinedLoss(nn.Module):
    """Combined BCE + Dice Loss"""
    
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.CrossEntropyLoss()
        self.dice = DiceLoss()
    
    def forward(self, inputs, targets):
        bce_loss = self.bce(inputs, targets)
        dice_loss = self.dice(inputs, targets)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss

def calculate_metrics(predictions, targets):
    """Calculate evaluation metrics"""
    
    # Convert to numpy
    if torch.is_tensor(predictions):
        predictions = predictions.cpu().numpy()
    if torch.is_tensor(targets):
        targets = targets.cpu().numpy()
    
    # Flatten
    predictions = predictions.flatten()
    targets = targets.flatten()
    
    # Calculate metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        targets, predictions, average='binary', zero_division=0
    )
    
    iou = jaccard_score(targets, predictions, average='binary', zero_division=0)
    accuracy = (predictions == targets).mean()
    
    # Dice coefficient
    intersection = (predictions * targets).sum()
    dice = (2. * intersection) / (predictions.sum() + targets.sum() + 1e-6)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'iou': iou,
        'dice': dice
    }

def create_model(architecture='unet', encoder='resnet34', pretrained=True):
    """Create segmentation model"""
    
    if architecture == 'unet':
        model = smp.Unet(
            encoder_name=encoder,
            encoder_weights='imagenet' if pretrained else None,
            in_channels=3,
            classes=2,  # Background and slum
            activation=None,  # We'll use softmax in loss
        )
    elif architecture == 'unetplusplus':
        model = smp.UnetPlusPlus(
            encoder_name=encoder,
            encoder_weights='imagenet' if pretrained else None,
            in_channels=3,
            classes=2,
            activation=None,
        )
    elif architecture == 'deeplabv3plus':
        model = smp.DeepLabV3Plus(
            encoder_name=encoder,
            encoder_weights='imagenet' if pretrained else None,
            in_channels=3,
            classes=2,
            activation=None,
        )
    
    return model

def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """Train for one epoch"""
    
    model.train()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch} [Train]')
    
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(images)
        loss = criterion(outputs, masks)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        # Get predictions
        preds = torch.argmax(outputs, dim=1)
        
        all_preds.append(preds.cpu())
        all_targets.append(masks.cpu())
        
        pbar.set_postfix({'loss': loss.item()})
    
    # Calculate metrics
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    metrics = calculate_metrics(all_preds, all_targets)
    
    avg_loss = running_loss / len(dataloader)
    
    return avg_loss, metrics

def validate_epoch(model, dataloader, criterion, device, epoch):
    """Validate for one epoch"""
    
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch} [Val]')
    
    with torch.no_grad():
        for images, masks in pbar:
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            running_loss += loss.item()
            
            # Get predictions
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.append(preds.cpu())
            all_targets.append(masks.cpu())
            
            pbar.set_postfix({'loss': loss.item()})
    
    # Calculate metrics
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    metrics = calculate_metrics(all_preds, all_targets)
    
    avg_loss = running_loss / len(dataloader)
    
    return avg_loss, metrics

def save_sample_predictions(model, dataset, device, save_path='sample_predictions.png', num_samples=6):
    """Save sample predictions for visualization"""
    
    model.eval()
    
    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4*num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    indices = np.linspace(0, len(dataset)-1, num_samples, dtype=int)
    
    with torch.no_grad():
        for i, idx in enumerate(indices):
            image, mask = dataset[idx]
            
            # Add batch dimension
            image_batch = image.unsqueeze(0).to(device)
            
            # Get prediction
            output = model(image_batch)
            pred = torch.argmax(output, dim=1).squeeze().cpu().numpy()
            
            # Convert image for visualization
            img_vis = image.permute(1, 2, 0).cpu().numpy()
            img_vis = (img_vis * np.array([0.229, 0.224, 0.225])) + np.array([0.485, 0.456, 0.406])
            img_vis = np.clip(img_vis, 0, 1)
            
            # Plot
            axes[i, 0].imshow(img_vis)
            axes[i, 0].set_title('Input Image')
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(mask.cpu().numpy(), cmap='gray')
            axes[i, 1].set_title('Ground Truth')
            axes[i, 1].axis('off')
            
            axes[i, 2].imshow(pred, cmap='gray')
            axes[i, 2].set_title('Prediction')
            axes[i, 2].axis('off')
            
            # Overlay
            overlay = img_vis.copy()
            overlay[pred == 1] = [1, 0, 0]  # Red for slums
            axes[i, 3].imshow(overlay)
            axes[i, 3].set_title('Overlay')
            axes[i, 3].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    print("UNet Slum Detection Model Training")
    print("="*50)
