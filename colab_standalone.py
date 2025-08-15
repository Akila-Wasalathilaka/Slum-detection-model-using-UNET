# Install requirements
!pip install torch torchvision segmentation-models-pytorch opencv-python albumentations scikit-image tqdm

# Create minimal model
import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

# Enhanced UNet
class EnhancedUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.unet = smp.Unet(encoder_name="resnet34", encoder_weights="imagenet", in_channels=6, classes=1)
        
        # Modify first conv for 6 channels
        first_conv = self.unet.encoder.conv1
        new_conv = nn.Conv2d(6, first_conv.out_channels, kernel_size=first_conv.kernel_size, 
                           stride=first_conv.stride, padding=first_conv.padding, bias=False)
        with torch.no_grad():
            new_conv.weight[:, :3] = first_conv.weight
            nn.init.kaiming_normal_(new_conv.weight[:, 3:], mode='fan_out', nonlinearity='relu')
        self.unet.encoder.conv1 = new_conv
    
    def forward(self, x):
        return self.unet(x)

# Dataset
class SlumDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.transform = transform
        self.image_paths = list(self.images_dir.glob("*.tif")) + list(self.images_dir.glob("*.png"))
        self.mask_paths = [self.masks_dir / f.name for f in self.image_paths]
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        image = cv2.imread(str(self.image_paths[idx]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Add texture channels
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient = np.sqrt(grad_x**2 + grad_y**2)
        
        # 6-channel input: RGB + gradient + gray + laplacian
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        image_6ch = np.stack([
            image[:,:,0], image[:,:,1], image[:,:,2], 
            gradient, gray, laplacian
        ], axis=-1).astype(np.float32)
        
        # Load mask
        mask = cv2.imread(str(self.mask_paths[idx]), cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype(np.float32)
        
        if self.transform:
            transformed = self.transform(image=image_6ch, mask=mask)
            image_6ch = transformed['image']
            mask = transformed['mask']
        else:
            image_6ch = torch.from_numpy(image_6ch.transpose(2,0,1)) / 255.0
            mask = torch.from_numpy(mask)
        
        return image_6ch, mask

# Transforms
def get_transforms():
    return A.Compose([
        A.Resize(256, 256),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406, 0.5, 0.5, 0.5], 
                   std=[0.229, 0.224, 0.225, 0.25, 0.25, 0.25]),
        ToTensorV2()
    ])

# Loss function
class ComboLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
    
    def dice_loss(self, pred, target):
        pred = torch.sigmoid(pred)
        smooth = 1e-6
        intersection = (pred * target).sum()
        return 1 - (2 * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    
    def forward(self, pred, target):
        bce = self.bce(pred, target)
        dice = self.dice_loss(pred, target)
        return bce + dice, {'bce': bce.item(), 'dice': dice.item()}

# Training setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

model = EnhancedUNet().to(device)
criterion = ComboLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

# Create datasets (you need to upload your data folder)
train_dataset = SlumDataset('data/train/images', 'data/train/masks', get_transforms())
val_dataset = SlumDataset('data/val/images', 'data/val/masks', get_transforms())

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=2)

print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

# Training loop
best_val_loss = float('inf')
for epoch in range(50):
    # Train
    model.train()
    train_loss = 0
    for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss, _ = criterion(outputs, masks)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        train_loss += loss.item()
    
    # Validate
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss, _ = criterion(outputs, masks)
            val_loss += loss.item()
    
    train_loss /= len(train_loader)
    val_loss /= len(val_loader)
    
    print(f"Epoch {epoch+1}: Train={train_loss:.4f}, Val={val_loss:.4f}")
    
    # Save best
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_model.pth')
        print("✅ Best model saved!")

print("🎉 Training complete!")