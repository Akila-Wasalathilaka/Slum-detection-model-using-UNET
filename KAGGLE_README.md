# 🏘️ Kaggle Slum Detection Quick Start

## 🚀 One-Click Execution in Kaggle

### Method 1: Full Pipeline (Recommended)

1. **Create New Kaggle Notebook**
2. **Copy and paste** the entire content from `kaggle_slum_detection.py`
3. **Run the cell** - it will automatically:
   - Clone this repository to `/kaggle/working`
   - Install all dependencies
   - Create 20 sample satellite images
   - Load the UNet model with ResNet34 encoder
   - Generate slum detection predictions
   - Display results with visualizations

### Method 2: Compact One-Cell Version

For a super compact version, use `kaggle_one_cell.py`:

1. **Create New Kaggle Notebook**
2. **Copy and paste** the entire content from `kaggle_one_cell.py`
3. **Run the cell** for instant results

## 📊 What You'll Get

### Automatic Outputs:
- ✅ **20 satellite image predictions**
- 📊 **Visual grid showing original images + slum overlays**
- 📈 **Detailed statistics for each image**
- 🏆 **Summary report with detection rates**

### Sample Output:
```
🏆 SLUM DETECTION RESULTS
============================================================
📊 Images Analyzed: 20
🏘️ Slums Detected: 8
📈 Detection Rate: 40.0%

📋 INDIVIDUAL RESULTS:
Image  1: 🔴 SLUM | Coverage:  12.3% | Max Prob: 0.847
Image  2: 🟢 CLEAR | Coverage:   0.0% | Max Prob: 0.234
Image  3: 🔴 SLUM | Coverage:   8.7% | Max Prob: 0.752
...
```

## 🛠️ Technical Specifications

- **Model**: UNet with ResNet34 encoder
- **Input**: 120×120 RGB satellite images
- **Output**: Binary segmentation masks (slum/non-slum)
- **Device**: Automatically detects GPU/CPU
- **Dependencies**: Automatically installed
- **Runtime**: ~2-3 minutes total

## 🔧 Customization Options

### Change Model Configuration:
```python
# In the script, modify this line:
model = create_model("unet", "resnet34", pretrained=True, num_classes=1)

# Options:
# Architecture: "unet", "unet++", "deeplabv3+"
# Encoder: "resnet34", "efficientnet-b0", "mobilenet_v2"
```

### Adjust Detection Threshold:
```python
# Change sensitivity
pred = (prob > 0.3).astype(np.uint8)  # More sensitive (default: 0.5)
pred = (prob > 0.7).astype(np.uint8)  # Less sensitive
```

### Generate More Images:
```python
# Change this line:
for i in range(50):  # Generate 50 images instead of 20
```

## 📁 Repository Structure

```
/kaggle/working/Slum-Detection-Using-Unet-Architecture/
├── models/           # UNet architectures and loss functions
├── utils/           # Dataset and preprocessing utilities
├── config/          # Model configurations
├── scripts/         # Training and testing scripts
├── kaggle_samples/  # Generated sample images (after running)
└── README.md        # Main documentation
```

## 🎯 Quick Troubleshooting

### If you get import errors:
- The script automatically installs dependencies
- Wait for "📦 Installing dependencies..." to complete

### If model fails to load:
- Ensure internet connection for downloading pretrained weights
- The script uses ImageNet pretrained ResNet34 backbone

### If predictions seem random:
- Normal behavior! The model uses ImageNet weights, not trained slum weights
- For real accuracy, you'd need the actual trained checkpoint

## 🚀 Ready to Run!

Just copy either script to Kaggle and execute. The system will handle everything automatically!

---

**Note**: This creates synthetic satellite-like images for demonstration. For real slum detection, you'd need actual satellite imagery and trained model weights.
