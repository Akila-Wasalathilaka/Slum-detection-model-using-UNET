# 🏘️ COMPLETE KAGGLE PIPELINE: Train UNET + Predictions + Evaluation
# Copy and paste this entire code into a single Kaggle cell

import os, sys, subprocess, torch, numpy as np, matplotlib.pyplot as plt, cv2, json
import torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
import seaborn as sns
from tqdm import tqdm
import warnings; warnings.filterwarnings('ignore')

print("🚀 COMPLETE SLUM DETECTION PIPELINE - Train + Evaluate + Predict")
print("="*70)

# Setup and clone repository
print("📥 Cloning repository...")
os.chdir('/kaggle/working')
subprocess.run("git clone https://github.com/Akila-Wasalathilaka/Slum-detection-model-using-UNET.git", shell=True, capture_output=True)
os.chdir('/kaggle/working/Slum-detection-model-using-UNET')
sys.path.insert(0, '/kaggle/working/Slum-detection-model-using-UNET')

# Install dependencies
print("📦 Installing packages...")
subprocess.run("pip install segmentation-models-pytorch albumentations opencv-python scikit-learn tqdm seaborn", shell=True, capture_output=True)

# Create comprehensive dataset for training
print("🖼️ Creating training dataset (500 images)...")

class SlumDataset(Dataset):
    def __init__(self, num_samples=500, image_size=120):
        self.num_samples = num_samples
        self.image_size = image_size
        self.images, self.masks = self._generate_dataset()
    
    def _generate_dataset(self):
        images, masks = [], []
        
        for i in range(self.num_samples):
            np.random.seed(42 + i)
            
            # Create realistic satellite image
            img = np.random.uniform(0.2, 0.7, (self.image_size, self.image_size, 3))
            mask = np.zeros((self.image_size, self.image_size), dtype=np.float32)
            
            # Add terrain variety
            terrain_type = i % 4
            if terrain_type == 0:  # Urban
                img *= [0.4, 0.4, 0.4]  # Darker urban
            elif terrain_type == 1:  # Suburban
                img *= [0.6, 0.5, 0.4]  # Brown suburban
            elif terrain_type == 2:  # Rural
                img *= [0.3, 0.6, 0.3]  # Green rural
            else:  # Mixed
                pass  # Keep original
            
            # Add slum areas (60% of images have slums)
            if i < int(self.num_samples * 0.6):
                num_slums = np.random.randint(1, 4)
                
                for _ in range(num_slums):
                    # Random slum location and size
                    center_x = np.random.randint(15, self.image_size - 15)
                    center_y = np.random.randint(15, self.image_size - 15)
                    slum_size = np.random.randint(8, 20)
                    
                    # Create irregular slum shape
                    for dx in range(-slum_size, slum_size):
                        for dy in range(-slum_size, slum_size):
                            x, y = center_x + dx, center_y + dy
                            
                            if (0 <= x < self.image_size and 0 <= y < self.image_size):
                                distance = np.sqrt(dx**2 + dy**2)
                                if distance < slum_size and np.random.random() > 0.3:
                                    # Slum appearance
                                    img[y, x] = [
                                        np.random.uniform(0.4, 0.8),  # Brownish
                                        np.random.uniform(0.3, 0.6),  # Orange
                                        np.random.uniform(0.2, 0.5)   # Reddish
                                    ]
                                    mask[y, x] = 1.0  # Mark as slum in mask
            
            # Add roads and infrastructure
            if np.random.random() > 0.3:
                road_y = np.random.randint(5, self.image_size - 5)
                img[road_y:road_y+2, :] = [0.1, 0.1, 0.1]  # Road
            
            # Normalize and add to dataset
            img = np.clip(img, 0, 1)
            images.append(img)
            masks.append(mask)
        
        return images, masks
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.masks[idx]
        
        # Convert to tensors
        image = torch.FloatTensor(image.transpose(2, 0, 1))  # CHW
        mask = torch.FloatTensor(mask).unsqueeze(0)  # Add channel dim
        
        return image, mask

# Create datasets
print("📊 Creating train/validation split...")
train_dataset = SlumDataset(num_samples=400)
val_dataset = SlumDataset(num_samples=100)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

print(f"✅ Training samples: {len(train_dataset)}")
print(f"✅ Validation samples: {len(val_dataset)}")

# Define UNET model
print("🧠 Creating UNET model...")

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()
        
        # Encoder
        self.enc1 = self.conv_block(in_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)
        
        # Bottleneck
        self.bottleneck = self.conv_block(512, 1024)
        
        # Decoder
        self.upconv4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = self.conv_block(1024, 512)
        self.upconv3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = self.conv_block(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = self.conv_block(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = self.conv_block(128, 64)
        
        self.final = nn.Conv2d(64, out_channels, 1)
        
    def conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(nn.MaxPool2d(2)(e1))
        e3 = self.enc3(nn.MaxPool2d(2)(e2))
        e4 = self.enc4(nn.MaxPool2d(2)(e3))
        
        # Bottleneck
        b = self.bottleneck(nn.MaxPool2d(2)(e4))
        
        # Decoder
        d4 = self.upconv4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)
        
        d3 = self.upconv3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.upconv2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.upconv1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        
        return torch.sigmoid(self.final(d1))

# Initialize model, loss, optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet().to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print(f"✅ Model created on {device}")
print(f"📊 Parameters: {sum(p.numel() for p in model.parameters()):,}")

# Training loop
print("\n🏋️ Starting training...")
num_epochs = 15
train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []

for epoch in range(num_epochs):
    # Training
    model.train()
    train_loss, train_acc = 0.0, 0.0
    
    for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        images, masks = images.to(device), masks.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        
        # Calculate accuracy
        predicted = (outputs > 0.5).float()
        accuracy = (predicted == masks).float().mean()
        train_acc += accuracy.item()
    
    # Validation
    model.eval()
    val_loss, val_acc = 0.0, 0.0
    
    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            val_loss += loss.item()
            
            predicted = (outputs > 0.5).float()
            accuracy = (predicted == masks).float().mean()
            val_acc += accuracy.item()
    
    # Store metrics
    train_losses.append(train_loss / len(train_loader))
    val_losses.append(val_loss / len(val_loader))
    train_accuracies.append(train_acc / len(train_loader))
    val_accuracies.append(val_acc / len(val_loader))
    
    print(f"Epoch {epoch+1}: Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}, "
          f"Train Acc: {train_accuracies[-1]:.4f}, Val Acc: {val_accuracies[-1]:.4f}")

print("✅ Training completed!")

# Create comprehensive evaluation charts
print("\n📊 Creating evaluation charts...")

# 1. Training History
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

# Loss curves
ax1.plot(train_losses, label='Train Loss', linewidth=2, color='blue')
ax1.plot(val_losses, label='Val Loss', linewidth=2, color='red')
ax1.set_title('Training & Validation Loss', fontsize=14, fontweight='bold')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Accuracy curves
ax2.plot(train_accuracies, label='Train Accuracy', linewidth=2, color='green')
ax2.plot(val_accuracies, label='Val Accuracy', linewidth=2, color='orange')
ax2.set_title('Training & Validation Accuracy', fontsize=14, fontweight='bold')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Model evaluation on validation set
print("🔍 Evaluating model performance...")
model.eval()
all_predictions, all_targets = [], []

with torch.no_grad():
    for images, masks in val_loader:
        images, masks = images.to(device), masks.to(device)
        outputs = model(images)
        
        all_predictions.extend(outputs.cpu().numpy().flatten())
        all_targets.extend(masks.cpu().numpy().flatten())

all_predictions = np.array(all_predictions)
all_targets = np.array(all_targets)

# ROC Curve
fpr, tpr, _ = roc_curve(all_targets, all_predictions)
roc_auc = auc(fpr, tpr)

ax3.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {roc_auc:.3f})')
ax3.plot([0, 1], [0, 1], 'k--', linewidth=1)
ax3.set_xlim([0.0, 1.0])
ax3.set_ylim([0.0, 1.05])
ax3.set_xlabel('False Positive Rate')
ax3.set_ylabel('True Positive Rate')
ax3.set_title(f'ROC Curve (AUC = {roc_auc:.3f})', fontsize=14, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(all_targets, all_predictions)
pr_auc = auc(recall, precision)

ax4.plot(recall, precision, linewidth=2, label=f'PR Curve (AUC = {pr_auc:.3f})')
ax4.set_xlabel('Recall')
ax4.set_ylabel('Precision')
ax4.set_title(f'Precision-Recall Curve (AUC = {pr_auc:.3f})', fontsize=14, fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Confusion Matrix
binary_predictions = (all_predictions > 0.5).astype(int)
cm = confusion_matrix(all_targets.astype(int), binary_predictions)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['No Slum', 'Slum'], 
            yticklabels=['No Slum', 'Slum'])
plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# Generate test predictions
print("\n🔍 Generating 20 test predictions...")
test_images = []
for i in range(20):
    np.random.seed(100 + i)
    img = np.random.uniform(0.3, 0.6, (120, 120, 3))
    
    if i < 12:  # Add slums to first 12
        for _ in range(np.random.randint(1, 3)):
            cx, cy = np.random.randint(20, 100, 2)
            size = np.random.randint(10, 25)
            for dx in range(-size, size):
                for dy in range(-size, size):
                    if 0 <= cx+dx < 120 and 0 <= cy+dy < 120 and np.random.random() > 0.4:
                        img[cy+dy, cx+dx] = [np.random.uniform(0.4, 0.7), np.random.uniform(0.3, 0.5), np.random.uniform(0.2, 0.4)]
    
    test_images.append(np.clip(img, 0, 1))

# Run predictions
predictions, probabilities = [], []
model.eval()

with torch.no_grad():
    for img in test_images:
        img_tensor = torch.FloatTensor(img.transpose(2, 0, 1)).unsqueeze(0).to(device)
        output = model(img_tensor)
        prob = output.squeeze().cpu().numpy()
        pred = (prob > 0.5).astype(np.uint8)
        
        predictions.append(pred)
        probabilities.append(prob)

# Visualize predictions
print("📊 Creating prediction visualizations...")
fig, axes = plt.subplots(4, 10, figsize=(25, 10))
fig.suptitle('TRAINED UNET - 20 Slum Detection Predictions', fontsize=16, fontweight='bold')

for i in range(20):
    row, col = (i // 10) * 2, i % 10
    
    # Original
    axes[row, col].imshow(test_images[i])
    axes[row, col].set_title(f'Test {i+1}', fontsize=10)
    axes[row, col].axis('off')
    
    # Prediction
    axes[row + 1, col].imshow(test_images[i])
    slum_mask = predictions[i] > 0
    if slum_mask.sum() > 0:
        overlay = np.zeros((*predictions[i].shape, 3))
        overlay[slum_mask] = [1, 0, 0]
        axes[row + 1, col].imshow(overlay, alpha=0.7)
    
    slum_pct = (predictions[i].sum() / 14400) * 100
    conf = probabilities[i].max()
    axes[row + 1, col].set_title(f'Slum: {slum_pct:.1f}%\nConf: {conf:.3f}', fontsize=9)
    axes[row + 1, col].axis('off')

plt.tight_layout()
plt.show()

# Final Results Summary
print("\n" + "="*70)
print("🏆 COMPLETE PIPELINE RESULTS")
print("="*70)

slum_detected = sum(1 for p in predictions if p.sum() > 0)
print(f"📊 TRAINING METRICS:")
print(f"   Final Train Accuracy: {train_accuracies[-1]:.3f}")
print(f"   Final Validation Accuracy: {val_accuracies[-1]:.3f}")
print(f"   ROC AUC Score: {roc_auc:.3f}")
print(f"   Precision-Recall AUC: {pr_auc:.3f}")

print(f"\n📊 TEST PREDICTIONS:")
print(f"   Images Analyzed: 20")
print(f"   Slums Detected: {slum_detected}")
print(f"   Detection Rate: {slum_detected/20*100:.1f}%")

print(f"\n📋 INDIVIDUAL RESULTS:")
for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
    slum_pct = (pred.sum() / 14400) * 100
    status = "🔴 SLUM" if pred.sum() > 0 else "🟢 CLEAR"
    conf_level = "HIGH" if prob.max() > 0.7 else "MED" if prob.max() > 0.5 else "LOW"
    print(f"Test {i+1:2d}: {status} | Area: {slum_pct:5.1f}% | Conf: {prob.max():.3f} ({conf_level})")

print(f"\n🎉 PIPELINE COMPLETE!")
print(f"📁 Repository: /kaggle/working/Slum-detection-model-using-UNET")
print(f"🧠 Model trained for {num_epochs} epochs with {sum(p.numel() for p in model.parameters()):,} parameters")
