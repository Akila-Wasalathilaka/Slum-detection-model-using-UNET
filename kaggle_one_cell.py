# 🏘️ KAGGLE SLUM DETECTION - ONE CELL EXECUTION
# Copy and paste this entire cell into Kaggle notebook

import os, sys, subprocess, torch, numpy as np, matplotlib.pyplot as plt
from pathlib import Path
import warnings; warnings.filterwarnings('ignore')

# Setup
os.chdir('/kaggle/working')
print("🚀 Cloning repository...")
subprocess.run("git clone https://github.com/nichula01/Slum-Detection-Using-Unet-Architecture.git", shell=True)
os.chdir('/kaggle/working/Slum-Detection-Using-Unet-Architecture')
sys.path.insert(0, '/kaggle/working/Slum-Detection-Using-Unet-Architecture')

# Install dependencies
print("📦 Installing dependencies...")
subprocess.run("pip install segmentation-models-pytorch albumentations opencv-python", shell=True)

# Create sample data
print("🖼️ Creating sample images...")
os.makedirs('kaggle_samples', exist_ok=True)
sample_images = []
for i in range(20):
    np.random.seed(42 + i)
    img = np.random.normal(0.4, 0.1, (120, 120, 3))
    if i < 10:  # Add slum-like areas to first 10
        x, y = np.random.randint(20, 100, 2)
        size = np.random.randint(15, 30)
        for dx in range(-size//2, size//2):
            for dy in range(-size//2, size//2):
                if 0 <= x+dx < 120 and 0 <= y+dy < 120 and np.random.random() > 0.3:
                    img[x+dx, y+dy] = [0.6, 0.5, 0.3]
    img = np.clip(img, 0, 1)
    sample_images.append((img * 255).astype(np.uint8))

# Load model
print("🧠 Loading model...")
from models.unet import create_model
model = create_model("unet", "resnet34", pretrained=True, num_classes=1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device).eval()

# Predict
print("🔍 Running predictions...")
predictions, probabilities = [], []
with torch.no_grad():
    for i, img in enumerate(sample_images):
        # Preprocess
        img_norm = img.astype(np.float32) / 255.0
        img_norm = (img_norm - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        tensor = torch.from_numpy(img_norm.transpose(2, 0, 1)).unsqueeze(0).to(device)
        
        # Predict
        output = torch.sigmoid(model(tensor))
        prob = output.squeeze().cpu().numpy()
        pred = (prob > 0.5).astype(np.uint8)
        
        predictions.append(pred)
        probabilities.append(prob)
        if (i + 1) % 5 == 0: print(f"✅ Processed {i + 1}/20")

# Visualize
print("📊 Creating visualizations...")
fig, axes = plt.subplots(4, 10, figsize=(25, 10))
fig.suptitle('Kaggle Slum Detection Results - 20 Predictions', fontsize=16, weight='bold')

for i in range(20):
    row, col = (i // 10) * 2, i % 10
    
    # Original
    axes[row, col].imshow(sample_images[i])
    axes[row, col].set_title(f'Image {i+1}', fontsize=10)
    axes[row, col].axis('off')
    
    # Prediction
    axes[row + 1, col].imshow(sample_images[i])
    slum_mask = predictions[i] > 0
    if slum_mask.sum() > 0:
        overlay = np.zeros((*predictions[i].shape, 3))
        overlay[slum_mask] = [1, 0, 0]
        axes[row + 1, col].imshow(overlay, alpha=0.6)
    
    slum_pct = (predictions[i].sum() / 14400) * 100
    axes[row + 1, col].set_title(f'Slum: {slum_pct:.1f}%\nMax: {probabilities[i].max():.3f}', fontsize=9)
    axes[row + 1, col].axis('off')

plt.tight_layout()
plt.show()

# Results
print("\n" + "="*60)
print("🏆 SLUM DETECTION RESULTS")
print("="*60)
slum_detected = sum(1 for p in predictions if p.sum() > 0)
print(f"📊 Images Analyzed: 20")
print(f"🏘️ Slums Detected: {slum_detected}")
print(f"📈 Detection Rate: {slum_detected/20*100:.1f}%")

print("\n📋 INDIVIDUAL RESULTS:")
for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
    slum_pct = (pred.sum() / 14400) * 100
    status = "🔴 SLUM" if pred.sum() > 0 else "🟢 CLEAR"
    print(f"Image {i+1:2d}: {status} | Coverage: {slum_pct:5.1f}% | Max Prob: {prob.max():.3f}")

print(f"\n🎉 KAGGLE SLUM DETECTION COMPLETE!")
print(f"Average Coverage: {np.mean([(p.sum()/14400)*100 for p in predictions]):.2f}%")
print(f"Average Max Prob: {np.mean([p.max() for p in probabilities]):.3f}")
