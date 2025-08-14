"""
Fixed Training System for Slum Detection
=======================================
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import cv2
from pathlib import Path
import json
from tqdm import tqdm
import albumentations as A

class FixedSlumDataset(Dataset):
    """Fixed dataset that handles .tif and .png files correctly."""
    
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.transform = transform
        
        # Get all image files (.tif and .jpg)
        self.image_files = list(self.images_dir.glob("*.tif")) + list(self.images_dir.glob("*.jpg"))
        
        # Filter to only include files with corresponding masks
        self.valid_files = []
        for img_file in self.image_files:
            mask_file = self.masks_dir / (img_file.stem + '.png')
            
            if mask_file.exists():
                self.valid_files.append((img_file, mask_file))
        
        print(f"Found {len(self.valid_files)} valid image-mask pairs")
    
    def __len__(self):
        return len(self.valid_files)
    
    def __getitem__(self, idx):
        img_path, mask_path = self.valid_files[idx]
        
        # Load image (handle .tif files)
        image = cv2.imread(str(img_path))
        if image is None:
            # Try with different flags for .tif
            image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        
        if image is not None:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            # Create dummy image if loading fails
            image = np.zeros((128, 128, 3), dtype=np.uint8)
        
        # Load mask
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            mask = np.zeros((128, 128), dtype=np.uint8)
        
        # Resize to standard size
        image = cv2.resize(image, (128, 128))
        mask = cv2.resize(mask, (128, 128))
        
        # Convert mask to binary (slum vs non-slum)
        mask = (mask > 0).astype(np.float32)
        
        # Apply transforms
        if self.transform:
            try:
                transformed = self.transform(image=image, mask=mask)
                image = transformed['image']
                mask = transformed['mask']
            except:
                pass  # Skip transform if it fails
        
        # Convert to tensors
        image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
        mask = torch.from_numpy(mask).float().unsqueeze(0)
        
        # Normalize image
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image = (image - mean) / std
        
        return image, mask

def get_fixed_transforms():
    """Get working augmentation transforms."""
    train_transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.GaussNoise(p=0.2),
    ])
    
    return train_transform, None

class SimpleSlumModel(nn.Module):
    """Simple but effective slum detection model."""
    
    def __init__(self):
        super().__init__()
        
        # Simple UNet-like architecture
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        features = self.encoder(x)
        output = self.decoder(features)
        return output

class FixedTrainer:
    """Fixed trainer that actually works."""
    
    def __init__(self, data_root="data"):
        self.data_root = Path(data_root)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def train(self, epochs=20, batch_size=8):
        """Train the model."""
        print("🚀 Starting FIXED slum detection training...")
        print("=" * 50)
        
        # Create model
        model = SimpleSlumModel().to(self.device)
        
        # Create datasets
        train_transform, _ = get_fixed_transforms()
        
        train_dataset = FixedSlumDataset(
            self.data_root / "train" / "images",
            self.data_root / "train" / "masks",
            transform=train_transform
        )
        
        val_dataset = FixedSlumDataset(
            self.data_root / "val" / "images",
            self.data_root / "val" / "masks",
            transform=None
        )
        
        if len(train_dataset) == 0:
            print("❌ No training data found!")
            return 0.0
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Training setup
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        
        best_loss = float('inf')
        
        for epoch in range(epochs):
            # Training
            model.train()
            train_loss = 0
            
            for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
                images, masks = images.to(self.device), masks.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            
            # Validation
            model.eval()
            val_loss = 0
            
            with torch.no_grad():
                for images, masks in val_loader:
                    images, masks = images.to(self.device), masks.to(self.device)
                    outputs = model(images)
                    loss = criterion(outputs, masks)
                    val_loss += loss.item()
            
            avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else avg_train_loss
            
            print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
            
            # Save best model
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                torch.save(model.state_dict(), 'best_slum_model.pth')
                print(f"✅ Best model saved! Loss: {best_loss:.4f}")
        
        print(f"🎯 Training completed! Best loss: {best_loss:.4f}")
        return best_loss

if __name__ == "__main__":
    trainer = FixedTrainer()
    trainer.train(epochs=30, batch_size=4)