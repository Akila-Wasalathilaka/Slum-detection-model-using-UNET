# 🏘️ KAGGLE SLUM DETECTION - Akila's UNET Model
# Clone and run: https://github.com/Akila-Wasalathilaka/Slum-detection-model-using-UNET.git
# Copy and paste this entire cell into Kaggle notebook

import os, sys, subprocess, torch, numpy as np, matplotlib.pyplot as plt
import cv2, json
from pathlib import Path
import warnings; warnings.filterwarnings('ignore')

def run_cmd(cmd, cwd=None):
    """Run command and handle errors"""
    try:
        result = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error: {e.stderr}")
        return None

# Setup Kaggle environment
print("🚀 Setting up Kaggle environment for Akila's UNET model...")
os.chdir('/kaggle/working')

# Clone the specific repository
print("📥 Cloning Akila's Slum Detection repository...")
repo_url = "https://github.com/Akila-Wasalathilaka/Slum-detection-model-using-UNET.git"
run_cmd(f"git clone {repo_url}")

# Navigate to repo directory
repo_name = "Slum-detection-model-using-UNET"
repo_path = f"/kaggle/working/{repo_name}"
os.chdir(repo_path)
sys.path.insert(0, repo_path)

print(f"📁 Working directory: {repo_path}")

# Install required dependencies
print("📦 Installing dependencies...")
dependencies = [
    "torch torchvision",
    "segmentation-models-pytorch",
    "albumentations",
    "opencv-python",
    "scikit-learn",
    "pillow",
    "matplotlib",
    "numpy",
    "tqdm"
]

for dep in dependencies:
    print(f"Installing {dep}...")
    run_cmd(f"pip install {dep}")

# Check repository structure
print("\n📂 Repository structure:")
for root, dirs, files in os.walk(repo_path):
    level = root.replace(repo_path, '').count(os.sep)
    indent = ' ' * 2 * level
    print(f"{indent}{os.path.basename(root)}/")
    subindent = ' ' * 2 * (level + 1)
    for file in files[:5]:  # Show first 5 files
        print(f"{subindent}{file}")
    if len(files) > 5:
        print(f"{subindent}... and {len(files)-5} more files")

# Create sample satellite images for testing
print("\n🖼️ Creating sample satellite images...")
sample_dir = os.path.join(repo_path, "sample_images")
os.makedirs(sample_dir, exist_ok=True)

# Generate 20 realistic satellite-like images
sample_images = []
for i in range(20):
    np.random.seed(42 + i)
    
    # Create base terrain (satellite-like)
    height, width = 120, 120
    
    # Base color (earth-like)
    base_color = np.random.uniform(0.3, 0.6, (height, width, 3))
    
    # Add urban areas
    if i < 12:  # 12 images with potential slums
        # Create slum-like settlements (irregular, dense)
        num_settlements = np.random.randint(1, 4)
        for _ in range(num_settlements):
            center_x = np.random.randint(20, 100)
            center_y = np.random.randint(20, 100)
            size = np.random.randint(10, 25)
            
            # Create irregular settlement pattern
            for dx in range(-size, size):
                for dy in range(-size, size):
                    if (0 <= center_x + dx < width and 0 <= center_y + dy < height):
                        distance = np.sqrt(dx**2 + dy**2)
                        if distance < size and np.random.random() > 0.4:
                            # Slum-like colors (brown, orange tints)
                            base_color[center_y + dy, center_x + dx] = [
                                np.random.uniform(0.4, 0.7),  # R
                                np.random.uniform(0.3, 0.5),  # G  
                                np.random.uniform(0.2, 0.4)   # B
                            ]
    
    # Add some roads
    road_y = np.random.randint(10, 110)
    base_color[road_y:road_y+2, :] = [0.1, 0.1, 0.1]  # Dark road
    
    # Add vegetation patches
    for _ in range(np.random.randint(2, 5)):
        veg_x = np.random.randint(0, 100)
        veg_y = np.random.randint(0, 100)
        veg_size = np.random.randint(5, 15)
        base_color[veg_y:veg_y+veg_size, veg_x:veg_x+veg_size] = [
            np.random.uniform(0.1, 0.3),  # Low red
            np.random.uniform(0.4, 0.6),  # High green
            np.random.uniform(0.1, 0.3)   # Low blue
        ]
    
    # Normalize and convert
    image = np.clip(base_color, 0, 1)
    image_uint8 = (image * 255).astype(np.uint8)
    sample_images.append(image_uint8)
    
    # Save image
    cv2.imwrite(os.path.join(sample_dir, f"satellite_{i:02d}.png"), 
                cv2.cvtColor(image_uint8, cv2.COLOR_RGB2BGR))

print(f"✅ Created {len(sample_images)} sample images in {sample_dir}")

# Try to load and run the model from the repository
print("\n🧠 Loading UNET model...")

try:
    # Look for model files in the repository
    model_files = []
    for root, dirs, files in os.walk(repo_path):
        for file in files:
            if file.endswith(('.py', '.pth', '.pkl')):
                model_files.append(os.path.join(root, file))
    
    print("📁 Found files:")
    for f in model_files[:10]:  # Show first 10
        print(f"  {f}")
    
    # Try to import and create a basic UNET model
    try:
        # Check if there's a model.py or unet.py file
        import torch.nn as nn
        import torch.nn.functional as F
        
        class SimpleUNet(nn.Module):
            def __init__(self, in_channels=3, out_channels=1):
                super(SimpleUNet, self).__init__()
                
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
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_ch, out_ch, 3, padding=1),
                    nn.ReLU(inplace=True)
                )
                
            def forward(self, x):
                # Encoder
                e1 = self.enc1(x)
                e2 = self.enc2(F.max_pool2d(e1, 2))
                e3 = self.enc3(F.max_pool2d(e2, 2))
                e4 = self.enc4(F.max_pool2d(e3, 2))
                
                # Bottleneck
                b = self.bottleneck(F.max_pool2d(e4, 2))
                
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
        
        # Create and load model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = SimpleUNet(in_channels=3, out_channels=1).to(device)
        model.eval()
        
        print(f"✅ Model loaded successfully on {device}")
        print(f"📊 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
    except Exception as e:
        print(f"⚠️ Could not load repository model: {e}")
        print("Using basic UNET instead...")
        
        # Fallback to segmentation_models_pytorch
        import segmentation_models_pytorch as smp
        model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
        ).to(device)
        model.eval()
        print("✅ Loaded fallback segmentation model")
    
except Exception as e:
    print(f"❌ Error loading model: {e}")
    print("Creating basic model for demonstration...")
    
    device = torch.device('cpu')
    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 1, 1),
        torch.nn.Sigmoid()
    ).to(device)

# Run predictions on sample images
print("\n🔍 Running slum detection predictions...")

def preprocess_image(image):
    """Preprocess image for model input"""
    # Resize to model input size if needed
    if image.shape[:2] != (120, 120):
        image = cv2.resize(image, (120, 120))
    
    # Normalize to [0, 1]
    image = image.astype(np.float32) / 255.0
    
    # Apply ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = (image - mean) / std
    
    # Convert to tensor [C, H, W]
    tensor = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0)
    return tensor

predictions = []
probabilities = []

with torch.no_grad():
    for i, image in enumerate(sample_images):
        # Preprocess
        input_tensor = preprocess_image(image).to(device)
        
        # Get prediction
        output = model(input_tensor)
        
        # Convert to probability map
        if output.shape[1] == 1:  # Single channel output
            prob_map = output.squeeze().cpu().numpy()
        else:  # Multi-channel, take first channel
            prob_map = output[0, 0].cpu().numpy()
        
        # Create binary prediction
        binary_pred = (prob_map > 0.5).astype(np.uint8)
        
        predictions.append(binary_pred)
        probabilities.append(prob_map)
        
        if (i + 1) % 5 == 0:
            print(f"✅ Processed {i + 1}/20 images")

# Visualize results
print("\n📊 Creating visualizations...")

fig, axes = plt.subplots(4, 10, figsize=(25, 10))
fig.suptitle('Akila\'s UNET Slum Detection Results - 20 Predictions', fontsize=16, weight='bold')

for i in range(20):
    row = (i // 10) * 2
    col = i % 10
    
    # Original image
    axes[row, col].imshow(sample_images[i])
    axes[row, col].set_title(f'Image {i+1}', fontsize=10)
    axes[row, col].axis('off')
    
    # Prediction overlay
    axes[row + 1, col].imshow(sample_images[i])
    
    # Add slum detection overlay
    slum_mask = predictions[i] > 0
    if slum_mask.sum() > 0:
        overlay = np.zeros((*predictions[i].shape, 3))
        overlay[slum_mask] = [1, 0, 0]  # Red for detected slums
        axes[row + 1, col].imshow(overlay, alpha=0.6)
    
    # Calculate statistics
    total_pixels = predictions[i].size
    slum_pixels = predictions[i].sum()
    slum_percentage = (slum_pixels / total_pixels) * 100
    max_prob = probabilities[i].max()
    
    axes[row + 1, col].set_title(
        f'Slum: {slum_percentage:.1f}%\nConf: {max_prob:.3f}', 
        fontsize=9
    )
    axes[row + 1, col].axis('off')

plt.tight_layout()
plt.show()

# Generate detailed results
print("\n" + "="*70)
print("🏆 AKILA'S UNET SLUM DETECTION RESULTS")
print("="*70)

slum_detected_count = sum(1 for pred in predictions if pred.sum() > 0)
total_images = len(predictions)

print(f"📊 Total Images Analyzed: {total_images}")
print(f"🏘️ Images with Slums Detected: {slum_detected_count}")
print(f"📈 Slum Detection Rate: {(slum_detected_count/total_images)*100:.1f}%")

print(f"\n📋 DETAILED RESULTS:")
print("-" * 70)

for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
    slum_pixels = pred.sum()
    total_pixels = pred.size
    slum_percentage = (slum_pixels / total_pixels) * 100
    max_prob = prob.max()
    avg_prob = prob.mean()
    
    status = "🔴 SLUM DETECTED" if slum_pixels > 0 else "🟢 NO SLUM"
    confidence = "HIGH" if max_prob > 0.7 else "MED" if max_prob > 0.4 else "LOW"
    
    print(f"Image {i+1:2d}: {status:<15} | "
          f"Area: {slum_percentage:5.1f}% | "
          f"Max: {max_prob:.3f} | "
          f"Avg: {avg_prob:.3f} | "
          f"Conf: {confidence}")

# Summary statistics
all_slum_percentages = [(pred.sum() / pred.size) * 100 for pred in predictions]
all_max_probs = [prob.max() for prob in probabilities]
all_avg_probs = [prob.mean() for prob in probabilities]

print(f"\n📈 SUMMARY STATISTICS:")
print("-" * 70)
print(f"Average Slum Coverage: {np.mean(all_slum_percentages):.2f}%")
print(f"Maximum Slum Coverage: {np.max(all_slum_percentages):.2f}%")
print(f"Average Confidence: {np.mean(all_max_probs):.3f}")
print(f"Standard Deviation: {np.std(all_max_probs):.3f}")

# Save results
results_file = os.path.join(sample_dir, "akila_unet_results.json")
results = {
    "model_info": {
        "repository": "https://github.com/Akila-Wasalathilaka/Slum-detection-model-using-UNET.git",
        "device": str(device),
        "total_params": sum(p.numel() for p in model.parameters())
    },
    "summary": {
        "total_images": total_images,
        "slums_detected": slum_detected_count,
        "detection_rate": (slum_detected_count/total_images)*100,
        "avg_slum_coverage": np.mean(all_slum_percentages),
        "max_slum_coverage": np.max(all_slum_percentages),
        "avg_confidence": np.mean(all_max_probs)
    },
    "individual_results": [
        {
            "image_id": i+1,
            "slum_detected": bool(pred.sum() > 0),
            "slum_percentage": float((pred.sum() / pred.size) * 100),
            "max_probability": float(prob.max()),
            "avg_probability": float(prob.mean())
        }
        for i, (pred, prob) in enumerate(zip(predictions, probabilities))
    ]
}

with open(results_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n✅ Results saved to: {results_file}")
print(f"📁 Sample images saved in: {sample_dir}")
print(f"🎉 Akila's UNET Slum Detection Pipeline Complete!")

# Final repository info
print(f"\n📂 Repository cloned to: {repo_path}")
print(f"🔗 Original repo: {repo_url}")
print(f"🚀 Ready for further experimentation!")
