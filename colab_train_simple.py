# Step 1: Clone repo (run this first in Colab)
!git clone https://github.com/Akila-Wasalathilaka/Slum-detection-model-using-UNET.git
%cd Slum-detection-model-using-UNET
!pip install -r requirements.txt

# Step 2: Train model
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader

from models.enhanced_unet import create_enhanced_model
from models.global_losses import ComboLossV2
from utils.global_transforms import get_global_train_transforms, get_global_val_transforms
from utils.dataset import SlumDataset

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# Model
model = create_enhanced_model(encoder="resnet34", in_channels=6).to(device)
criterion = ComboLossV2()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

# Data
train_dataset = SlumDataset('data/train/images', 'data/train/masks', get_global_train_transforms())
val_dataset = SlumDataset('data/val/images', 'data/val/masks', get_global_val_transforms())
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
        
        if isinstance(outputs, dict):
            main_loss, _ = criterion(outputs['main'], masks)
            boundary_loss = nn.BCEWithLogitsLoss()(outputs['boundary'], masks)
            loss = main_loss + 0.3 * boundary_loss
        else:
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
            if isinstance(outputs, dict):
                outputs = outputs['main']
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

print("🎉 Training done!")