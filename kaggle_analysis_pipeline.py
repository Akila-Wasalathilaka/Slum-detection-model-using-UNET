#!/usr/bin/env python3
"""
Kaggle Analysis Pipeline for Slum Detection
==========================================
Minimal analysis and prediction pipeline for Kaggle environment
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

# Setup paths
sys.path.append('/kaggle/working')

def setup_kaggle_environment():
    """Setup Kaggle environment"""
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    return device

def create_charts(y_true, y_pred, y_prob):
    """Create analysis charts"""
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true.flatten(), y_prob.flatten())
    roc_auc = auc(fpr, tpr)
    
    axes[0, 0].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    axes[0, 0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    axes[0, 0].set_xlim([0.0, 1.0])
    axes[0, 0].set_ylim([0.0, 1.05])
    axes[0, 0].set_xlabel('False Positive Rate')
    axes[0, 0].set_ylabel('True Positive Rate')
    axes[0, 0].set_title('ROC Curve')
    axes[0, 0].legend(loc="lower right")
    axes[0, 0].grid(True)
    
    # Confusion Matrix
    cm = confusion_matrix(y_true.flatten(), y_pred.flatten())
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 1])
    axes[0, 1].set_title('Confusion Matrix')
    axes[0, 1].set_xlabel('Predicted')
    axes[0, 1].set_ylabel('Actual')
    
    # Prediction Distribution
    axes[1, 0].hist(y_prob.flatten(), bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    axes[1, 0].set_title('Prediction Probability Distribution')
    axes[1, 0].set_xlabel('Probability')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].grid(True)
    
    # Threshold Analysis
    thresholds = np.linspace(0, 1, 100)
    precisions, recalls, f1s = [], [], []
    
    for thresh in thresholds:
        pred_thresh = (y_prob > thresh).astype(int)
        tp = np.sum((pred_thresh == 1) & (y_true == 1))
        fp = np.sum((pred_thresh == 1) & (y_true == 0))
        fn = np.sum((pred_thresh == 0) & (y_true == 1))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
    
    axes[1, 1].plot(thresholds, precisions, label='Precision', color='red')
    axes[1, 1].plot(thresholds, recalls, label='Recall', color='blue')
    axes[1, 1].plot(thresholds, f1s, label='F1-Score', color='green')
    axes[1, 1].set_title('Threshold Analysis')
    axes[1, 1].set_xlabel('Threshold')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('/kaggle/working/analysis_charts.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return roc_auc

def make_predictions(model, data_loader, device):
    """Make predictions and collect results"""
    model.eval()
    all_preds = []
    all_probs = []
    all_targets = []
    
    with torch.no_grad():
        for images, masks in data_loader:
            images, masks = images.to(device), masks.to(device)
            
            if len(masks.shape) == 3:
                masks = masks.unsqueeze(1)
            
            outputs = model(images)
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            
            all_preds.append(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            all_targets.append(masks.cpu().numpy())
    
    return np.concatenate(all_preds), np.concatenate(all_probs), np.concatenate(all_targets)

def create_prediction_samples(model, data_loader, device, num_samples=6):
    """Create prediction visualization samples"""
    model.eval()
    fig, axes = plt.subplots(3, num_samples, figsize=(20, 12))
    
    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(data_loader):
            if batch_idx >= num_samples:
                break
                
            images, masks = images.to(device), masks.to(device)
            if len(masks.shape) == 3:
                masks = masks.unsqueeze(1)
            
            outputs = model(images)
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            
            # Show first image from batch
            img = images[0].cpu().permute(1, 2, 0).numpy()
            mask = masks[0, 0].cpu().numpy()
            pred = preds[0, 0].cpu().numpy()
            
            # Normalize image for display
            img = (img - img.min()) / (img.max() - img.min())
            
            axes[0, batch_idx].imshow(img)
            axes[0, batch_idx].set_title(f'Image {batch_idx+1}')
            axes[0, batch_idx].axis('off')
            
            axes[1, batch_idx].imshow(mask, cmap='gray')
            axes[1, batch_idx].set_title('Ground Truth')
            axes[1, batch_idx].axis('off')
            
            axes[2, batch_idx].imshow(pred, cmap='gray')
            axes[2, batch_idx].set_title('Prediction')
            axes[2, batch_idx].axis('off')
    
    plt.tight_layout()
    plt.savefig('/kaggle/working/prediction_samples.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main analysis pipeline"""
    print("📊 Kaggle Analysis Pipeline")
    print("=" * 40)
    
    # Setup environment
    device = setup_kaggle_environment()
    
    # Import modules after path setup
    from models import create_model
    from config import get_model_config, get_data_config
    from utils.dataset import SlumDataset, create_data_loaders
    from utils.transforms import get_val_transforms
    
    # Load configs
    model_config = get_model_config('balanced')
    data_config = get_data_config('standard')
    
    # Create model and load weights
    model = create_model(
        architecture=model_config.architecture,
        encoder=model_config.encoder,
        pretrained=False,
        num_classes=model_config.num_classes
    ).to(device)
    
    # Load trained weights
    if os.path.exists('/kaggle/working/slum_model.pth'):
        model.load_state_dict(torch.load('/kaggle/working/slum_model.pth', map_location=device))
        print("✅ Model weights loaded")
    else:
        print("⚠️ No trained model found, using random weights")
    
    # Create test dataset
    paths = data_config.get_paths()
    val_transforms = get_val_transforms(data_config)
    
    test_dataset = SlumDataset(
        images_dir=paths['val_images'],  # Using val as test for demo
        masks_dir=paths['val_masks'],
        transform=val_transforms,
        slum_rgb=data_config.slum_rgb,
        image_size=data_config.image_size
    )
    
    test_loader = create_data_loaders(
        train_dataset=None,
        val_dataset=test_dataset,
        batch_size=8,
        num_workers=2
    )['val']
    
    print("🔮 Making predictions...")
    y_pred, y_prob, y_true = make_predictions(model, test_loader, device)
    
    print("📈 Creating analysis charts...")
    roc_auc = create_charts(y_true, y_pred, y_prob)
    
    print("🖼️ Creating prediction samples...")
    create_prediction_samples(model, test_loader, device)
    
    # Performance summary
    print("\n📊 Performance Summary:")
    print(f"AUC-ROC: {roc_auc:.4f}")
    
    # Calculate metrics
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    print("\n✅ Analysis complete!")
    print("📁 Charts saved to /kaggle/working/")

if __name__ == "__main__":
    main()