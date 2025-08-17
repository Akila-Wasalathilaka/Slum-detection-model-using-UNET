"""
Comprehensive Model Analysis and Chart Generation
================================================

Advanced analysis script for generating detailed charts, confusion matrices,
ROC curves, and performance visualizations after model training.
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
from models.metrics import create_metrics
from config import get_model_config, get_data_config
from utils.dataset import SlumDataset
from utils.transforms import get_test_transforms
from utils.checkpoint import load_checkpoint

# Import analysis libraries
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score,
    precision_score, recall_score, f1_score, accuracy_score
)
import pandas as pd


class ModelAnalyzer:
    """Comprehensive model analysis and visualization."""
    
    def __init__(self, charts_dir="charts"):
        self.charts_dir = Path(charts_dir)
        self.charts_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        
        # Create subdirectories
        (self.charts_dir / "confusion_matrices").mkdir(exist_ok=True)
        (self.charts_dir / "roc_curves").mkdir(exist_ok=True)
        (self.charts_dir / "precision_recall").mkdir(exist_ok=True)
        (self.charts_dir / "threshold_analysis").mkdir(exist_ok=True)
        (self.charts_dir / "performance_metrics").mkdir(exist_ok=True)
        (self.charts_dir / "predictions").mkdir(exist_ok=True)
        
        print(f"📊 Model Analyzer initialized - Charts will be saved to: {self.charts_dir}")
    
    def load_model_and_data(self, checkpoint_path, model_config, data_config, device):
        """Load trained model and test dataset."""
        print("🔄 Loading model and test data...")
        
        # Load model
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
        
        print(f"✅ Model loaded - Test dataset: {len(test_dataset)} samples")
        return model, test_dataset, checkpoint
    
    def get_predictions(self, model, test_dataset, device, batch_size=16):
        """Get model predictions on test dataset."""
        print("🔍 Generating predictions...")
        
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False,
            num_workers=4, pin_memory=True
        )
        
        all_predictions = []
        all_targets = []
        all_images = []
        
        with torch.no_grad():
            for images, masks in test_loader:
                images, masks = images.to(device), masks.to(device)
                
                if len(masks.shape) == 3:
                    masks = masks.unsqueeze(1)
                
                outputs = model(images)
                predictions = torch.sigmoid(outputs)
                
                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(masks.cpu().numpy())
                all_images.append(images.cpu().numpy())
        
        predictions = np.concatenate(all_predictions, axis=0)
        targets = np.concatenate(all_targets, axis=0)
        images = np.concatenate(all_images, axis=0)
        
        # Remove channel dimension if present
        if len(predictions.shape) == 4:
            predictions = predictions.squeeze(1)
        if len(targets.shape) == 4:
            targets = targets.squeeze(1)
        
        print(f"✅ Predictions generated - Shape: {predictions.shape}")
        return predictions, targets, images
    
    def create_confusion_matrix(self, y_true, y_pred, threshold=0.5, title="Confusion Matrix"):
        """Create and save confusion matrix."""
        print(f"📊 Creating confusion matrix (threshold={threshold})...")
        
        # Convert to binary predictions
        y_pred_binary = (y_pred > threshold).astype(int)
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true.flatten(), y_pred_binary.flatten())
        
        # Calculate percentages
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Raw counts
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                   xticklabels=['Non-Slum', 'Slum'],
                   yticklabels=['Non-Slum', 'Slum'])
        ax1.set_title(f'{title} - Raw Counts')
        ax1.set_xlabel('Predicted')
        ax1.set_ylabel('Actual')
        
        # Percentages
        sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Blues', ax=ax2,
                   xticklabels=['Non-Slum', 'Slum'],
                   yticklabels=['Non-Slum', 'Slum'])
        ax2.set_title(f'{title} - Percentages')
        ax2.set_xlabel('Predicted')
        ax2.set_ylabel('Actual')
        
        plt.tight_layout()
        
        # Save
        save_path = self.charts_dir / "confusion_matrices" / f"confusion_matrix_t{threshold:.2f}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Calculate metrics
        tn, fp, fn, tp = cm.ravel()
        metrics = {
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_positives': int(tp),
            'accuracy': accuracy_score(y_true.flatten(), y_pred_binary.flatten()),
            'precision': precision_score(y_true.flatten(), y_pred_binary.flatten(), zero_division=0),
            'recall': recall_score(y_true.flatten(), y_pred_binary.flatten(), zero_division=0),
            'f1_score': f1_score(y_true.flatten(), y_pred_binary.flatten(), zero_division=0),
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
            'threshold': threshold
        }
        
        print(f"   Saved: {save_path}")
        return metrics
    
    def create_roc_curve(self, y_true, y_pred, title="ROC Curve"):
        """Create and save ROC curve."""
        print("📈 Creating ROC curve...")
        
        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(y_true.flatten(), y_pred.flatten())
        roc_auc = auc(fpr, tpr)
        
        # Create plot
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC Curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                label='Random Classifier')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'{title}')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        # Add optimal threshold point
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        plt.plot(fpr[optimal_idx], tpr[optimal_idx], 'ro', markersize=8,
                label=f'Optimal Threshold = {optimal_threshold:.3f}')
        plt.legend(loc="lower right")
        
        # Save
        save_path = self.charts_dir / "roc_curves" / "roc_curve.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   Saved: {save_path}")
        print(f"   AUC: {roc_auc:.4f}")
        print(f"   Optimal Threshold: {optimal_threshold:.3f}")
        
        return {'auc': roc_auc, 'optimal_threshold': optimal_threshold}
    
    def create_precision_recall_curve(self, y_true, y_pred, title="Precision-Recall Curve"):
        """Create and save Precision-Recall curve."""
        print("📈 Creating Precision-Recall curve...")
        
        # Calculate PR curve
        precision, recall, thresholds = precision_recall_curve(y_true.flatten(), y_pred.flatten())
        ap_score = average_precision_score(y_true.flatten(), y_pred.flatten())
        
        # Create plot
        plt.figure(figsize=(10, 8))
        plt.plot(recall, precision, color='blue', lw=2,
                label=f'PR Curve (AP = {ap_score:.3f})')
        
        # Baseline (random classifier)
        positive_ratio = np.mean(y_true.flatten())
        plt.axhline(y=positive_ratio, color='red', linestyle='--', lw=2,
                   label=f'Random Classifier (AP = {positive_ratio:.3f})')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'{title}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Find optimal F1 threshold
        f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-8)
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx]
        
        plt.plot(recall[optimal_idx], precision[optimal_idx], 'ro', markersize=8,
                label=f'Optimal F1 Threshold = {optimal_threshold:.3f}')
        plt.legend()
        
        # Save
        save_path = self.charts_dir / "precision_recall" / "precision_recall_curve.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   Saved: {save_path}")
        print(f"   Average Precision: {ap_score:.4f}")
        print(f"   Optimal F1 Threshold: {optimal_threshold:.3f}")
        
        return {'average_precision': ap_score, 'optimal_f1_threshold': optimal_threshold}
    
    def threshold_analysis(self, y_true, y_pred):
        """Comprehensive threshold analysis."""
        print("🔍 Performing threshold analysis...")
        
        thresholds = np.arange(0.1, 1.0, 0.05)
        results = []
        
        for thresh in thresholds:
            y_pred_binary = (y_pred > thresh).astype(int)
            
            metrics = {
                'threshold': thresh,
                'accuracy': accuracy_score(y_true.flatten(), y_pred_binary.flatten()),
                'precision': precision_score(y_true.flatten(), y_pred_binary.flatten(), zero_division=0),
                'recall': recall_score(y_true.flatten(), y_pred_binary.flatten(), zero_division=0),
                'f1_score': f1_score(y_true.flatten(), y_pred_binary.flatten(), zero_division=0)
            }
            results.append(metrics)
        
        # Convert to DataFrame for easier plotting
        df = pd.DataFrame(results)
        
        # Create comprehensive threshold plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Threshold Analysis', fontsize=16, fontweight='bold')
        
        # Accuracy vs Threshold
        axes[0, 0].plot(df['threshold'], df['accuracy'], 'b-', linewidth=2, marker='o')
        axes[0, 0].set_title('Accuracy vs Threshold')
        axes[0, 0].set_xlabel('Threshold')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Precision vs Threshold
        axes[0, 1].plot(df['threshold'], df['precision'], 'r-', linewidth=2, marker='o')
        axes[0, 1].set_title('Precision vs Threshold')
        axes[0, 1].set_xlabel('Threshold')
        axes[0, 1].set_ylabel('Precision')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Recall vs Threshold
        axes[1, 0].plot(df['threshold'], df['recall'], 'g-', linewidth=2, marker='o')
        axes[1, 0].set_title('Recall vs Threshold')
        axes[1, 0].set_xlabel('Threshold')
        axes[1, 0].set_ylabel('Recall')
        axes[1, 0].grid(True, alpha=0.3)
        
        # F1-Score vs Threshold
        axes[1, 1].plot(df['threshold'], df['f1_score'], 'purple', linewidth=2, marker='o')
        axes[1, 1].set_title('F1-Score vs Threshold')
        axes[1, 1].set_xlabel('Threshold')
        axes[1, 1].set_ylabel('F1-Score')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Mark optimal F1 threshold
        best_f1_idx = df['f1_score'].idxmax()
        best_threshold = df.loc[best_f1_idx, 'threshold']
        best_f1 = df.loc[best_f1_idx, 'f1_score']
        
        axes[1, 1].plot(best_threshold, best_f1, 'ro', markersize=10,
                       label=f'Best F1: {best_f1:.3f} @ {best_threshold:.2f}')
        axes[1, 1].legend()
        
        plt.tight_layout()
        save_path = self.charts_dir / "threshold_analysis" / "threshold_analysis.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Combined metrics plot
        plt.figure(figsize=(12, 8))
        plt.plot(df['threshold'], df['accuracy'], 'b-', linewidth=2, label='Accuracy', marker='o')
        plt.plot(df['threshold'], df['precision'], 'r-', linewidth=2, label='Precision', marker='s')
        plt.plot(df['threshold'], df['recall'], 'g-', linewidth=2, label='Recall', marker='^')
        plt.plot(df['threshold'], df['f1_score'], 'purple', linewidth=2, label='F1-Score', marker='d')
        
        # Mark optimal points
        plt.axvline(x=best_threshold, color='red', linestyle='--', alpha=0.7,
                   label=f'Optimal F1 Threshold: {best_threshold:.2f}')
        
        plt.xlabel('Threshold')
        plt.ylabel('Score')
        plt.title('All Metrics vs Threshold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xlim(0.1, 0.95)
        plt.ylim(0, 1)
        
        save_path = self.charts_dir / "threshold_analysis" / "combined_metrics.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   Saved threshold analysis charts")
        print(f"   Best F1 Threshold: {best_threshold:.3f} (F1: {best_f1:.4f})")
        
        # Save threshold data
        df.to_csv(self.charts_dir / "threshold_analysis" / "threshold_data.csv", index=False)
        
        return df, best_threshold, best_f1
    
    def create_performance_summary(self, metrics_dict):
        """Create performance summary visualization."""
        print("📊 Creating performance summary...")
        
        # Extract key metrics
        key_metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        values = [metrics_dict.get(metric, 0) for metric in key_metrics]
        
        # Create bar plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Bar plot
        bars = ax1.bar(key_metrics, values, color=['skyblue', 'lightcoral', 'lightgreen', 'gold'],
                      alpha=0.8, edgecolor='black')
        ax1.set_title('Model Performance Metrics', fontweight='bold')
        ax1.set_ylabel('Score')
        ax1.set_ylim(0, 1)
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Radar chart
        angles = np.linspace(0, 2 * np.pi, len(key_metrics), endpoint=False).tolist()
        values_radar = values + [values[0]]  # Close the plot
        angles += angles[:1]
        
        ax2 = plt.subplot(122, projection='polar')
        ax2.plot(angles, values_radar, 'o-', linewidth=2, color='blue')
        ax2.fill(angles, values_radar, alpha=0.25, color='blue')
        ax2.set_xticks(angles[:-1])
        ax2.set_xticklabels(key_metrics)
        ax2.set_ylim(0, 1)
        ax2.set_title('Performance Radar Chart', pad=20, fontweight='bold')
        
        plt.tight_layout()
        save_path = self.charts_dir / "performance_metrics" / "performance_summary.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   Saved: {save_path}")
    
    def create_prediction_samples(self, images, y_true, y_pred, threshold=0.5, num_samples=8):
        """Create sample predictions visualization."""
        print(f"🎨 Creating prediction samples (n={num_samples})...")
        
        # Select random samples
        indices = np.random.choice(len(images), min(num_samples, len(images)), replace=False)
        
        fig, axes = plt.subplots(num_samples, 4, figsize=(16, num_samples * 4))
        if num_samples == 1:
            axes = axes.reshape(1, -1)
        
        for i, idx in enumerate(indices):
            # Denormalize image
            img = images[idx].transpose(1, 2, 0)
            img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            img = np.clip(img, 0, 1)
            
            # Original image
            axes[i, 0].imshow(img)
            axes[i, 0].set_title('Original Image')
            axes[i, 0].axis('off')
            
            # Ground truth
            axes[i, 1].imshow(y_true[idx], cmap='Reds', alpha=0.8)
            axes[i, 1].set_title('Ground Truth')
            axes[i, 1].axis('off')
            
            # Prediction probability
            axes[i, 2].imshow(y_pred[idx], cmap='Blues', vmin=0, vmax=1)
            axes[i, 2].set_title('Prediction Prob')
            axes[i, 2].axis('off')
            
            # Binary prediction
            pred_binary = (y_pred[idx] > threshold).astype(float)
            axes[i, 3].imshow(pred_binary, cmap='Greens', alpha=0.8)
            axes[i, 3].set_title(f'Binary (>{threshold:.2f})')
            axes[i, 3].axis('off')
        
        plt.tight_layout()
        save_path = self.charts_dir / "predictions" / "prediction_samples.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   Saved: {save_path}")
    
    def generate_classification_report(self, y_true, y_pred, threshold=0.5):
        """Generate detailed classification report."""
        print("📋 Generating classification report...")
        
        y_pred_binary = (y_pred > threshold).astype(int)
        
        # Generate report
        report = classification_report(
            y_true.flatten(), y_pred_binary.flatten(),
            target_names=['Non-Slum', 'Slum'],
            output_dict=True
        )
        
        # Convert to DataFrame for better visualization
        df_report = pd.DataFrame(report).transpose()
        
        # Create visualization
        plt.figure(figsize=(10, 6))
        
        # Plot precision, recall, f1-score for each class
        metrics = ['precision', 'recall', 'f1-score']
        classes = ['Non-Slum', 'Slum']
        
        x = np.arange(len(classes))
        width = 0.25
        
        for i, metric in enumerate(metrics):
            values = [df_report.loc[cls, metric] for cls in classes]
            plt.bar(x + i * width, values, width, label=metric, alpha=0.8)
        
        plt.xlabel('Classes')
        plt.ylabel('Score')
        plt.title('Classification Report - Per Class Metrics')
        plt.xticks(x + width, classes)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)
        
        # Add value labels
        for i, metric in enumerate(metrics):
            values = [df_report.loc[cls, metric] for cls in classes]
            for j, value in enumerate(values):
                plt.text(j + i * width, value + 0.01, f'{value:.3f}',
                        ha='center', va='bottom', fontweight='bold')
        
        save_path = self.charts_dir / "performance_metrics" / "classification_report.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save report as CSV
        df_report.to_csv(self.charts_dir / "performance_metrics" / "classification_report.csv")
        
        print(f"   Saved: {save_path}")
        return report
    
    def run_complete_analysis(self, checkpoint_path, model_config, data_config, device='cuda'):
        """Run complete model analysis pipeline."""
        print("🚀 STARTING COMPREHENSIVE MODEL ANALYSIS")
        print("=" * 50)
        
        # Resolve device if 'auto'
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        print(f"🔧 Using device: {device}")
        
        # Load model and data
        model, test_dataset, checkpoint_info = self.load_model_and_data(
            checkpoint_path, model_config, data_config, device
        )
        
        # Get predictions
        predictions, targets, images = self.get_predictions(model, test_dataset, device)
        
        # Initialize results dictionary
        analysis_results = {
            'model_info': {
                'checkpoint': str(checkpoint_path),
                'architecture': model_config.architecture,
                'encoder': model_config.encoder,
                'epoch': checkpoint_info.get('epoch', 'unknown'),
                'training_metrics': checkpoint_info.get('metrics', {})
            },
            'test_dataset': {
                'num_samples': len(test_dataset),
                'positive_ratio': float(np.mean(targets))
            }
        }
        
        # ROC Curve Analysis
        roc_results = self.create_roc_curve(targets, predictions)
        analysis_results['roc_analysis'] = roc_results
        
        # Precision-Recall Curve
        pr_results = self.create_precision_recall_curve(targets, predictions)
        analysis_results['pr_analysis'] = pr_results
        
        # Threshold Analysis
        threshold_df, best_threshold, best_f1 = self.threshold_analysis(targets, predictions)
        analysis_results['threshold_analysis'] = {
            'best_threshold': float(best_threshold),
            'best_f1_score': float(best_f1)
        }
        
        # Confusion Matrices for different thresholds
        thresholds_to_test = [0.3, 0.5, best_threshold, 0.7]
        confusion_results = {}
        
        for thresh in thresholds_to_test:
            cm_metrics = self.create_confusion_matrix(
                targets, predictions, thresh, 
                f"Confusion Matrix (Threshold={thresh:.2f})"
            )
            confusion_results[f'threshold_{thresh:.2f}'] = cm_metrics
        
        analysis_results['confusion_matrices'] = confusion_results
        
        # Classification Report
        classification_rep = self.generate_classification_report(
            targets, predictions, best_threshold
        )
        analysis_results['classification_report'] = classification_rep
        
        # Performance Summary
        best_metrics = confusion_results[f'threshold_{best_threshold:.2f}']
        self.create_performance_summary(best_metrics)
        
        # Prediction Samples
        self.create_prediction_samples(images, targets, predictions, best_threshold)
        
        # Save complete analysis results
        with open(self.charts_dir / "complete_analysis_results.json", 'w', encoding='utf-8') as f:
            json.dump(analysis_results, f, indent=2, default=str)
        
        # Create summary report
        self.create_analysis_summary_report(analysis_results)
        
        print("\n✅ ANALYSIS COMPLETE!")
        print(f"📁 All charts saved to: {self.charts_dir}")
        print(f"📊 Best threshold: {best_threshold:.3f} (F1: {best_f1:.4f})")
        print(f"📈 AUC-ROC: {roc_results['auc']:.4f}")
        print(f"📉 Average Precision: {pr_results['average_precision']:.4f}")
        
        return analysis_results
    
    def create_analysis_summary_report(self, results):
        """Create comprehensive analysis summary report."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        report = f"""
MODEL ANALYSIS SUMMARY REPORT
============================

Generated: {timestamp}

MODEL INFORMATION
================
Checkpoint: {results['model_info']['checkpoint']}
Architecture: {results['model_info']['architecture']}
Encoder: {results['model_info']['encoder']}
Training Epoch: {results['model_info']['epoch']}

DATASET INFORMATION
==================
Test Samples: {results['test_dataset']['num_samples']}
Positive Class Ratio: {results['test_dataset']['positive_ratio']:.3f}

PERFORMANCE METRICS
==================
AUC-ROC: {results['roc_analysis']['auc']:.4f}
Average Precision: {results['pr_analysis']['average_precision']:.4f}
Optimal Threshold: {results['threshold_analysis']['best_threshold']:.3f}
Best F1-Score: {results['threshold_analysis']['best_f1_score']:.4f}

CONFUSION MATRIX ANALYSIS (Best Threshold)
=========================================
"""
        
        best_cm = results['confusion_matrices'][f'threshold_{results["threshold_analysis"]["best_threshold"]:.2f}']
        
        report += f"""
True Positives: {best_cm['true_positives']}
True Negatives: {best_cm['true_negatives']}
False Positives: {best_cm['false_positives']}
False Negatives: {best_cm['false_negatives']}

Accuracy: {best_cm['accuracy']:.4f}
Precision: {best_cm['precision']:.4f}
Recall: {best_cm['recall']:.4f}
F1-Score: {best_cm['f1_score']:.4f}
Specificity: {best_cm['specificity']:.4f}

CHARTS GENERATED
===============
✅ Confusion Matrices (multiple thresholds)
✅ ROC Curve with optimal threshold
✅ Precision-Recall Curve
✅ Threshold Analysis
✅ Performance Summary
✅ Classification Report
✅ Prediction Samples

All charts saved to: {self.charts_dir}

===============================
Analysis completed successfully!
"""
        
        with open(self.charts_dir / "ANALYSIS_SUMMARY_REPORT.txt", 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"📄 Summary report saved: {self.charts_dir}/ANALYSIS_SUMMARY_REPORT.txt")


def main():
    """Main analysis function."""
    parser = argparse.ArgumentParser(description='Comprehensive Model Analysis')
    parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint')
    parser.add_argument('--model_config', default='balanced', help='Model configuration preset')
    parser.add_argument('--data_config', default='standard', help='Data configuration preset')
    parser.add_argument('--output_dir', default='charts', help='Output directory for charts')
    parser.add_argument('--device', default='auto', help='Device to use (auto/cuda/cpu)')
    args = parser.parse_args()
    
    # Setup device
    if args.device == 'auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"🔧 Using device: {device}")
    
    # Load configurations
    model_config = get_model_config(args.model_config)
    data_config = get_data_config(args.data_config)
    
    # Run analysis
    analyzer = ModelAnalyzer(args.output_dir)
    results = analyzer.run_complete_analysis(
        args.checkpoint, model_config, data_config, device
    )
    
    return results


if __name__ == "__main__":
    main()
