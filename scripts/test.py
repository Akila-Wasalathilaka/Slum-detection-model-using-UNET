"""
Test Script for Slum Detection Model
====================================

Comprehensive testing and evaluation script for trained slum detection models.
Supports multiple models, Test Time Augmentation, and detailed analysis.
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import argparse
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import project modules
from models import create_model
from models.metrics import create_metrics, MetricsTracker
from config import get_model_config, get_data_config
from utils.dataset import SlumDataset, create_data_loaders
from utils.transforms import get_test_transforms, get_tta_transforms
from utils.checkpoint import load_checkpoint
from utils.visualization import visualize_predictions, TrainingVisualizer


def load_model_from_checkpoint(checkpoint_path, model_config, device):
    """Load model from checkpoint file."""
    print(f"📂 Loading model from: {checkpoint_path}")
    
    # Create model
    model = create_model(
        architecture=model_config.architecture,
        encoder=model_config.encoder,
        pretrained=False,  # We're loading trained weights
        num_classes=model_config.num_classes
    )
    
    # Load checkpoint
    checkpoint = load_checkpoint(checkpoint_path, model, device=device)
    model = model.to(device)
    model.eval()
    
    return model, checkpoint


def test_time_augmentation(model, images, tta_transforms, device):
    """Apply Test Time Augmentation for better predictions."""
    all_predictions = []
    
    with torch.no_grad():
        for transform in tta_transforms:
            # Apply transform to batch
            tta_images = []
            for img in images:
                img_np = img.cpu().numpy().transpose(1, 2, 0)
                # Denormalize for transform
                img_np = (img_np * np.array([0.229, 0.224, 0.225]) + 
                         np.array([0.485, 0.456, 0.406]))
                img_np = (img_np * 255).astype(np.uint8)
                
                # Apply transform
                transformed = transform(image=img_np)
                tta_images.append(transformed['image'])
            
            tta_batch = torch.stack(tta_images).to(device)
            
            # Get predictions
            outputs = model(tta_batch)
            predictions = torch.sigmoid(outputs)
            all_predictions.append(predictions)
    
    # Average all predictions
    avg_predictions = torch.stack(all_predictions).mean(dim=0)
    return avg_predictions


def evaluate_model(model, test_loader, device, use_tta=False, tta_transforms=None):
    """Evaluate model on test dataset."""
    model.eval()
    
    metrics_calc = create_metrics(device=device)
    metrics_tracker = MetricsTracker(['iou', 'dice', 'precision', 'recall', 'f1', 'accuracy'])
    
    all_predictions = []
    all_targets = []
    all_images = []
    
    print("🔍 Evaluating model...")
    
    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(test_loader):
            images, masks = images.to(device), masks.to(device)
            
            # Add channel dimension to masks if needed
            if len(masks.shape) == 3:
                masks = masks.unsqueeze(1)
            
            # Get predictions
            if use_tta and tta_transforms:
                predictions = test_time_augmentation(model, images, tta_transforms, device)
            else:
                outputs = model(images)
                predictions = torch.sigmoid(outputs)
            
            # Calculate metrics
            metrics = metrics_calc(torch.logit(predictions), masks)
            metrics_tracker.update(metrics)
            
            # Store for detailed analysis
            all_predictions.append(predictions.cpu())
            all_targets.append(masks.cpu())
            all_images.append(images.cpu())
            
            # Progress update
            if batch_idx % 10 == 0:
                print(f"Batch {batch_idx}/{len(test_loader)} - "
                      f"IoU: {metrics['iou']:.4f} Dice: {metrics['dice']:.4f}")
    
    # Final metrics
    final_metrics = metrics_tracker.get_average()
    
    # Concatenate all results
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    all_images = torch.cat(all_images, dim=0)
    
    return final_metrics, all_predictions, all_targets, all_images


def detailed_analysis(predictions, targets, save_dir):
    """Perform detailed analysis of predictions."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert to numpy
    pred_np = predictions.numpy()
    target_np = targets.numpy()
    
    # Remove channel dimension if present
    if len(pred_np.shape) == 4:
        pred_np = pred_np.squeeze(1)
    if len(target_np.shape) == 4:
        target_np = target_np.squeeze(1)
    
    print("📊 Performing detailed analysis...")
    
    # Create visualizer
    visualizer = TrainingVisualizer(save_dir)
    
    # Threshold analysis
    thresholds = np.arange(0.1, 1.0, 0.1)
    threshold_results = []
    
    from sklearn.metrics import precision_score, recall_score, f1_score
    
    for thresh in thresholds:
        pred_binary = (pred_np > thresh).astype(int)
        
        # Calculate metrics for each threshold
        precision = precision_score(target_np.flatten(), pred_binary.flatten(), zero_division=0)
        recall = recall_score(target_np.flatten(), pred_binary.flatten(), zero_division=0)
        f1 = f1_score(target_np.flatten(), pred_binary.flatten(), zero_division=0)
        
        threshold_results.append({
            'threshold': thresh,
            'precision': precision,
            'recall': recall,
            'f1': f1
        })
    
    # Plot threshold analysis
    thresholds_plot = [r['threshold'] for r in threshold_results]
    precisions = [r['precision'] for r in threshold_results]
    recalls = [r['recall'] for r in threshold_results]
    f1_scores = [r['f1'] for r in threshold_results]
    
    plt.figure(figsize=(12, 8))
    plt.plot(thresholds_plot, precisions, 'b-', label='Precision', linewidth=2)
    plt.plot(thresholds_plot, recalls, 'r-', label='Recall', linewidth=2)
    plt.plot(thresholds_plot, f1_scores, 'g-', label='F1-Score', linewidth=2)
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title('Performance vs Threshold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_dir / 'threshold_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Confusion matrix and ROC curves
    visualizer.plot_confusion_matrix(target_np, pred_np, threshold=0.5)
    visualizer.plot_roc_pr_curves(target_np, pred_np)
    
    # Save threshold analysis
    with open(save_dir / 'threshold_analysis.json', 'w') as f:
        json.dump(threshold_results, f, indent=2)
    
    return threshold_results


def save_test_results(metrics, threshold_analysis, checkpoint_info, save_path):
    """Save comprehensive test results."""
    results = {
        'test_metrics': metrics,
        'threshold_analysis': threshold_analysis,
        'model_info': {
            'epoch': checkpoint_info.get('epoch'),
            'training_metrics': checkpoint_info.get('metrics', {}),
            'timestamp': checkpoint_info.get('timestamp'),
            'pytorch_version': checkpoint_info.get('pytorch_version')
        },
        'test_timestamp': datetime.now().isoformat()
    }
    
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"📄 Test results saved: {save_path}")


def main():
    """Main testing function."""
    parser = argparse.ArgumentParser(description='Test Slum Detection Model')
    parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint')
    parser.add_argument('--config', default='balanced', help='Model configuration preset')
    parser.add_argument('--data', default='standard', help='Data configuration preset')
    parser.add_argument('--output', default='test_results', help='Output directory for results')
    parser.add_argument('--tta', action='store_true', help='Use Test Time Augmentation')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for testing')
    parser.add_argument('--visualize', type=int, default=8, help='Number of samples to visualize')
    args = parser.parse_args()
    
    print("🧪 SLUM DETECTION MODEL TESTING")
    print("=" * 50)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Model Config: {args.config}")
    print(f"Data Config: {args.data}")
    print(f"Use TTA: {args.tta}")
    print()
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔧 Device: {device}")
    
    # Load configurations
    model_config = get_model_config(args.config)
    data_config = get_data_config(args.data)
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    model, checkpoint_info = load_model_from_checkpoint(args.checkpoint, model_config, device)
    
    # Create test dataset
    print("📊 Creating test dataset...")
    paths = data_config.get_paths()
    
    test_transforms = get_test_transforms(data_config)
    
    test_dataset = SlumDataset(
        images_dir=paths['test_images'],
        masks_dir=paths['test_masks'],
        transform=test_transforms,
        slum_rgb=data_config.slum_rgb,
        image_size=data_config.image_size,
        use_tile_masks_only=data_config.use_tile_masks_only,
        min_slum_pixels=0,  # Include all test samples
        cache_masks=True
    )
    
    if len(test_dataset) == 0:
        print("❌ No test samples found!")
        return
    
    # Create test loader
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Prepare TTA if requested
    tta_transforms = None
    if args.tta:
        print("🔄 Preparing Test Time Augmentation...")
        tta_transforms = get_tta_transforms(data_config)
        print(f"   TTA variants: {len(tta_transforms)}")
    
    # Evaluate model
    metrics, predictions, targets, images = evaluate_model(
        model, test_loader, device, args.tta, tta_transforms
    )
    
    # Print results
    print("\n🏆 TEST RESULTS")
    print("=" * 30)
    for metric_name, value in metrics.items():
        print(f"{metric_name.upper()}: {value:.4f}")
    
    # Detailed analysis
    print("\n📊 Performing detailed analysis...")
    threshold_analysis = detailed_analysis(predictions, targets, output_dir / "analysis")
    
    # Find optimal threshold
    best_f1_idx = max(range(len(threshold_analysis)), 
                     key=lambda i: threshold_analysis[i]['f1'])
    best_threshold = threshold_analysis[best_f1_idx]
    
    print(f"\n🎯 Optimal Threshold: {best_threshold['threshold']:.2f}")
    print(f"   Precision: {best_threshold['precision']:.4f}")
    print(f"   Recall: {best_threshold['recall']:.4f}")
    print(f"   F1-Score: {best_threshold['f1']:.4f}")
    
    # Visualize predictions
    if args.visualize > 0:
        print(f"\n🎨 Creating visualizations for {args.visualize} samples...")
        
        # Select samples for visualization
        num_viz = min(args.visualize, len(images))
        viz_indices = np.linspace(0, len(images)-1, num_viz, dtype=int)
        
        viz_images = images[viz_indices]
        viz_predictions = predictions[viz_indices]
        viz_targets = targets[viz_indices]
        
        visualize_predictions(
            viz_images, viz_targets, viz_predictions,
            save_path=output_dir / "prediction_samples.png",
            num_samples=num_viz,
            threshold=best_threshold['threshold']
        )
        
        print(f"   Saved: {output_dir}/prediction_samples.png")
    
    # Save comprehensive results
    save_test_results(
        metrics, threshold_analysis, checkpoint_info,
        output_dir / "test_results.json"
    )
    
    # Save model configuration for reference
    model_config.save(output_dir / "model_config.json")
    data_config.save(output_dir / "data_config.json")
    
    print(f"\n✅ Testing completed!")
    print(f"📁 Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
