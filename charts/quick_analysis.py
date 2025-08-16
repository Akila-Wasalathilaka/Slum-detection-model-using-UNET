"""
Quick Model Analysis - Generate Key Charts After Training
=========================================================

Simplified analysis script for immediate post-training evaluation.
Generates essential charts: confusion matrix, ROC curve, and performance metrics.
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import project modules
from models import create_model
from config import get_model_config, get_data_config
from utils.dataset import SlumDataset
from utils.transforms import get_test_transforms
from utils.checkpoint import load_checkpoint

# Import analysis libraries
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, precision_recall_curve,
    precision_score, recall_score, f1_score, accuracy_score
)


def quick_model_analysis(checkpoint_path, output_dir="charts", show_plots=False):
    """
    Quick analysis of trained model with essential visualizations.
    
    Args:
        checkpoint_path: Path to the trained model checkpoint
        output_dir: Directory to save charts
        show_plots: Whether to display plots interactively
    """
    print("🚀 QUICK MODEL ANALYSIS")
    print("=" * 30)
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    charts_dir = Path(output_dir)
    charts_dir.mkdir(parents=True, exist_ok=True)
    
    # Set plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Load configurations
    model_config = get_model_config('balanced')
    data_config = get_data_config('standard')
    
    print(f"🔧 Device: {device}")
    print(f"📁 Charts directory: {charts_dir}")
    
    # Load model
    print("🔄 Loading model...")
    model = create_model(
        architecture=model_config.architecture,
        encoder=model_config.encoder,
        pretrained=False,
        num_classes=model_config.num_classes
    )
    
    checkpoint = load_checkpoint(checkpoint_path, model, device=device)
    model = model.to(device)
    model.eval()
    
    # Load test dataset
    print("📊 Loading test data...")
    paths = data_config.get_paths()
    test_transforms = get_test_transforms(data_config)
    
    test_dataset = SlumDataset(
        images_dir=paths['test_images'],
        masks_dir=paths['test_masks'],
        transform=test_transforms,
        slum_rgb=data_config.slum_rgb,
        image_size=data_config.image_size,
        use_tile_masks_only=data_config.use_tile_masks_only,
        min_slum_pixels=0,
        cache_masks=True
    )
    
    print(f"✅ Test dataset: {len(test_dataset)} samples")
    
    # Get predictions
    print("🔍 Generating predictions...")
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=16, shuffle=False,
        num_workers=4, pin_memory=True
    )
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for images, masks in test_loader:
            images, masks = images.to(device), masks.to(device)
            
            if len(masks.shape) == 3:
                masks = masks.unsqueeze(1)
            
            outputs = model(images)
            predictions = torch.sigmoid(outputs)
            
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(masks.cpu().numpy())
    
    predictions = np.concatenate(all_predictions, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    
    # Remove channel dimension if present
    if len(predictions.shape) == 4:
        predictions = predictions.squeeze(1)
    if len(targets.shape) == 4:
        targets = targets.squeeze(1)
    
    print(f"✅ Predictions shape: {predictions.shape}")
    
    # 1. ROC Curve
    print("📈 Creating ROC curve...")
    fpr, tpr, thresholds = roc_curve(targets.flatten(), predictions.flatten())
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC Curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
             label='Random Classifier')
    
    # Mark optimal threshold
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    plt.plot(fpr[optimal_idx], tpr[optimal_idx], 'ro', markersize=8,
             label=f'Optimal = {optimal_threshold:.3f}')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Slum Detection Model')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(charts_dir / "quick_roc_curve.png", dpi=300, bbox_inches='tight')
    if show_plots:
        plt.show()
    else:
        plt.close()
    
    # 2. Confusion Matrix
    print("📊 Creating confusion matrix...")
    y_pred_binary = (predictions > optimal_threshold).astype(int)
    cm = confusion_matrix(targets.flatten(), y_pred_binary.flatten())
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=['Non-Slum', 'Slum'],
               yticklabels=['Non-Slum', 'Slum'])
    plt.title(f'Confusion Matrix (Threshold = {optimal_threshold:.3f})')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    plt.tight_layout()
    plt.savefig(charts_dir / "quick_confusion_matrix.png", dpi=300, bbox_inches='tight')
    if show_plots:
        plt.show()
    else:
        plt.close()
    
    # 3. Performance Metrics
    print("📋 Calculating metrics...")
    accuracy = accuracy_score(targets.flatten(), y_pred_binary.flatten())
    precision = precision_score(targets.flatten(), y_pred_binary.flatten(), zero_division=0)
    recall = recall_score(targets.flatten(), y_pred_binary.flatten(), zero_division=0)
    f1 = f1_score(targets.flatten(), y_pred_binary.flatten(), zero_division=0)
    
    # Calculate additional metrics from confusion matrix
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    metrics = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'Specificity': specificity
    }
    
    # Plot metrics
    plt.figure(figsize=(10, 6))
    bars = plt.bar(metrics.keys(), metrics.values(), 
                   color=['skyblue', 'lightcoral', 'lightgreen', 'gold', 'lightpink'],
                   alpha=0.8, edgecolor='black')
    
    plt.title('Model Performance Metrics')
    plt.ylabel('Score')
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, (metric, value) in zip(bars, metrics.items()):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(charts_dir / "quick_performance_metrics.png", dpi=300, bbox_inches='tight')
    if show_plots:
        plt.show()
    else:
        plt.close()
    
    # 4. Precision-Recall Curve
    print("📈 Creating Precision-Recall curve...")
    precision_curve, recall_curve, _ = precision_recall_curve(targets.flatten(), predictions.flatten())
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall_curve, precision_curve, color='blue', lw=2)
    plt.fill_between(recall_curve, precision_curve, alpha=0.2, color='blue')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(charts_dir / "quick_precision_recall.png", dpi=300, bbox_inches='tight')
    if show_plots:
        plt.show()
    else:
        plt.close()
    
    # Print summary
    print("\n✅ ANALYSIS COMPLETE!")
    print("=" * 30)
    print(f"📊 AUC-ROC: {roc_auc:.4f}")
    print(f"🎯 Optimal Threshold: {optimal_threshold:.3f}")
    print(f"📈 Accuracy: {accuracy:.4f}")
    print(f"📈 Precision: {precision:.4f}")
    print(f"📈 Recall: {recall:.4f}")
    print(f"📈 F1-Score: {f1:.4f}")
    print(f"📈 Specificity: {specificity:.4f}")
    print(f"\n📁 Charts saved to: {charts_dir}")
    
    # Save metrics to file
    metrics_summary = {
        'AUC_ROC': float(roc_auc),
        'Optimal_Threshold': float(optimal_threshold),
        'Accuracy': float(accuracy),
        'Precision': float(precision),
        'Recall': float(recall),
        'F1_Score': float(f1),
        'Specificity': float(specificity),
        'True_Positives': int(tp),
        'True_Negatives': int(tn),
        'False_Positives': int(fp),
        'False_Negatives': int(fn)
    }
    
    import json
    with open(charts_dir / "quick_analysis_metrics.json", 'w') as f:
        json.dump(metrics_summary, f, indent=2)
    
    return metrics_summary


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Quick Model Analysis')
    parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint')
    parser.add_argument('--output', default='charts', help='Output directory')
    parser.add_argument('--show', action='store_true', help='Show plots interactively')
    args = parser.parse_args()
    
    results = quick_model_analysis(args.checkpoint, args.output, args.show)
