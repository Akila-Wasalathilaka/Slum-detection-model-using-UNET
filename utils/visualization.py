"""
Visualization Utilities for Slum Detection
==========================================

Comprehensive visualization tools for training monitoring, result analysis,
and model interpretability for slum detection models.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
import torch
import cv2
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import json
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve


class TrainingVisualizer:
    """Visualizer for training progress and metrics."""
    
    def __init__(self, save_dir: str):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def plot_training_history(self, history: Dict[str, List[float]], epoch: int):
        """Plot comprehensive training history."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Training Progress - Epoch {epoch}', fontsize=16, fontweight='bold')
        
        # Loss plot
        axes[0, 0].plot(history['train_loss'], label='Train Loss', linewidth=2)
        axes[0, 0].plot(history['val_loss'], label='Val Loss', linewidth=2)
        axes[0, 0].set_title('Loss', fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # IoU plot
        axes[0, 1].plot(history['train_iou'], label='Train IoU', linewidth=2)
        axes[0, 1].plot(history['val_iou'], label='Val IoU', linewidth=2)
        axes[0, 1].set_title('IoU Score', fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('IoU')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Dice plot
        axes[0, 2].plot(history['train_dice'], label='Train Dice', linewidth=2)
        axes[0, 2].plot(history['val_dice'], label='Val Dice', linewidth=2)
        axes[0, 2].set_title('Dice Score', fontweight='bold')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Dice')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # F1 plot
        axes[1, 0].plot(history['train_f1'], label='Train F1', linewidth=2)
        axes[1, 0].plot(history['val_f1'], label='Val F1', linewidth=2)
        axes[1, 0].set_title('F1 Score', fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('F1')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Learning rate plot
        if history['learning_rates']:
            axes[1, 1].plot(history['learning_rates'], linewidth=2, color='red')
            axes[1, 1].set_title('Learning Rate', fontweight='bold')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].set_yscale('log')
            axes[1, 1].grid(True, alpha=0.3)
        
        # Metrics comparison
        if len(history['val_iou']) > 0:
            latest_metrics = {
                'IoU': history['val_iou'][-1],
                'Dice': history['val_dice'][-1],
                'F1': history['val_f1'][-1]
            }
            
            bars = axes[1, 2].bar(latest_metrics.keys(), latest_metrics.values(), 
                                 color=['#1f77b4', '#ff7f0e', '#2ca02c'], alpha=0.7)
            axes[1, 2].set_title('Latest Validation Metrics', fontweight='bold')
            axes[1, 2].set_ylabel('Score')
            axes[1, 2].set_ylim(0, 1)
            axes[1, 2].grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, value in zip(bars, latest_metrics.values()):
                axes[1, 2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                               f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.save_dir / f'training_history_epoch_{epoch}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, 
                            threshold: float = 0.5):
        """Plot confusion matrix for binary classification."""
        # Convert predictions to binary
        y_pred_binary = (y_pred > threshold).astype(int)
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true.flatten(), y_pred_binary.flatten())
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Non-Slum', 'Slum'],
                   yticklabels=['Non-Slum', 'Slum'])
        plt.title('Confusion Matrix', fontweight='bold')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig(self.save_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_roc_pr_curves(self, y_true: np.ndarray, y_pred: np.ndarray):
        """Plot ROC and Precision-Recall curves."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # ROC Curve
        from sklearn.metrics import roc_auc_score
        fpr, tpr, _ = roc_curve(y_true.flatten(), y_pred.flatten())
        auc_score = roc_auc_score(y_true.flatten(), y_pred.flatten())
        
        ax1.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {auc_score:.3f})')
        ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_title('ROC Curve')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Precision-Recall Curve
        from sklearn.metrics import average_precision_score
        precision, recall, _ = precision_recall_curve(y_true.flatten(), y_pred.flatten())
        ap_score = average_precision_score(y_true.flatten(), y_pred.flatten())
        
        ax2.plot(recall, precision, linewidth=2, label=f'PR Curve (AP = {ap_score:.3f})')
        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')
        ax2.set_title('Precision-Recall Curve')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'roc_pr_curves.png', dpi=300, bbox_inches='tight')
        plt.close()


def visualize_predictions(
    images: torch.Tensor,
    masks: torch.Tensor,
    predictions: torch.Tensor,
    save_path: Optional[str] = None,
    num_samples: int = 4,
    threshold: float = 0.5,
    denormalize: bool = True,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
) -> None:
    """
    Visualize model predictions alongside ground truth.
    
    Args:
        images: Input images tensor [B, C, H, W]
        masks: Ground truth masks tensor [B, H, W] or [B, 1, H, W]
        predictions: Model predictions tensor [B, 1, H, W]
        save_path: Path to save the visualization
        num_samples: Number of samples to visualize
        threshold: Threshold for binary predictions
        denormalize: Whether to denormalize images
        mean: Normalization mean for denormalization
        std: Normalization std for denormalization
    """
    # Convert to numpy and select samples
    num_samples = min(num_samples, images.size(0))
    
    # Move to CPU and convert to numpy
    images = images[:num_samples].cpu().numpy()
    masks = masks[:num_samples].cpu().numpy()
    predictions = torch.sigmoid(predictions[:num_samples]).cpu().numpy()
    
    # Remove channel dimension from masks if present
    if len(masks.shape) == 4:
        masks = masks.squeeze(1)
    if len(predictions.shape) == 4:
        predictions = predictions.squeeze(1)
    
    # Denormalize images
    if denormalize:
        for i in range(3):
            images[:, i] = images[:, i] * std[i] + mean[i]
        images = np.clip(images, 0, 1)
    
    # Create figure
    fig, axes = plt.subplots(num_samples, 4, figsize=(16, num_samples * 4))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_samples):
        # Original image
        img = np.transpose(images[i], (1, 2, 0))
        axes[i, 0].imshow(img)
        axes[i, 0].set_title('Original Image')
        axes[i, 0].axis('off')
        
        # Ground truth mask
        axes[i, 1].imshow(masks[i], cmap='Reds', alpha=0.7)
        axes[i, 1].set_title('Ground Truth')
        axes[i, 1].axis('off')
        
        # Prediction probability
        axes[i, 2].imshow(predictions[i], cmap='Reds', vmin=0, vmax=1)
        axes[i, 2].set_title('Prediction (Prob)')
        axes[i, 2].axis('off')
        
        # Binary prediction
        pred_binary = (predictions[i] > threshold).astype(float)
        axes[i, 3].imshow(pred_binary, cmap='Reds', alpha=0.7)
        axes[i, 3].set_title(f'Prediction (>{threshold})')
        axes[i, 3].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def visualize_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    prediction: Optional[np.ndarray] = None,
    alpha: float = 0.5,
    save_path: Optional[str] = None
) -> None:
    """
    Visualize mask overlay on image.
    
    Args:
        image: RGB image [H, W, 3]
        mask: Binary mask [H, W]
        prediction: Optional prediction mask [H, W]
        alpha: Overlay transparency
        save_path: Path to save the visualization
    """
    fig, axes = plt.subplots(1, 3 if prediction is not None else 2, 
                           figsize=(15 if prediction is not None else 10, 5))
    
    # Original image
    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Ground truth overlay
    axes[1].imshow(image)
    axes[1].imshow(mask, cmap='Reds', alpha=alpha)
    axes[1].set_title('Ground Truth Overlay')
    axes[1].axis('off')
    
    # Prediction overlay
    if prediction is not None:
        axes[2].imshow(image)
        axes[2].imshow(prediction, cmap='Blues', alpha=alpha)
        axes[2].set_title('Prediction Overlay')
        axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_class_distribution(stats: Dict[str, Any], save_path: Optional[str] = None):
    """Plot class distribution statistics."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Dataset Class Distribution Analysis', fontsize=16, fontweight='bold')
    
    # Slum percentage distribution
    slum_percentages = [s['slum_percentage'] for s in stats.values()]
    
    axes[0, 0].hist(slum_percentages, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].set_title('Slum Percentage Distribution')
    axes[0, 0].set_xlabel('Slum Percentage')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Slum pixel count distribution
    slum_pixels = [s['slum_pixels'] for s in stats.values()]
    
    axes[0, 1].hist(slum_pixels, bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
    axes[0, 1].set_title('Slum Pixel Count Distribution')
    axes[0, 1].set_xlabel('Slum Pixels')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Binary classification
    has_slums = [1 if p > 0 else 0 for p in slum_percentages]
    slum_counts = [sum(has_slums), len(has_slums) - sum(has_slums)]
    
    axes[1, 0].pie(slum_counts, labels=['With Slums', 'No Slums'], autopct='%1.1f%%',
                   colors=['orange', 'lightblue'])
    axes[1, 0].set_title('Images with/without Slums')
    
    # Summary statistics
    stats_text = f"""
    Total Images: {len(stats)}
    Images with Slums: {sum(has_slums)}
    Average Slum %: {np.mean(slum_percentages):.2%}
    Max Slum %: {np.max(slum_percentages):.2%}
    Total Slum Pixels: {sum(slum_pixels):,}
    """
    
    axes[1, 1].text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5))
    axes[1, 1].set_xlim(0, 1)
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].axis('off')
    axes[1, 1].set_title('Dataset Statistics')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def create_model_comparison_plot(results: Dict[str, Dict[str, float]], 
                               save_path: Optional[str] = None):
    """Create comparison plot for multiple models."""
    models = list(results.keys())
    metrics = list(results[models[0]].keys())
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    x = np.arange(len(metrics))
    width = 0.8 / len(models)
    
    for i, model in enumerate(models):
        values = [results[model][metric] for metric in metrics]
        ax.bar(x + i * width, values, width, label=model, alpha=0.8)
    
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Score')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, model in enumerate(models):
        values = [results[model][metric] for metric in metrics]
        for j, value in enumerate(values):
            ax.text(x[j] + i * width, value + 0.01, f'{value:.3f}',
                   ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def create_overlay(image: np.ndarray, mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    Create an overlay of prediction mask on original image.
    
    Args:
        image: Original image array (H, W, 3)
        mask: Prediction mask array (H, W) with values 0-1
        alpha: Transparency for overlay
        
    Returns:
        Overlay image as numpy array
    """
    # Ensure image is in correct format
    if image.dtype != np.uint8:
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
    
    # Ensure mask is in correct format
    if mask.dtype != np.uint8:
        mask = (mask * 255).astype(np.uint8)
    
    # Resize mask to match image if needed
    if mask.shape != image.shape[:2]:
        import cv2
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
    
    # Create colored overlay (red for slums)
    overlay = np.zeros_like(image)
    overlay[:, :, 0] = mask  # Red channel
    
    # Blend with original image
    result = cv2.addWeighted(image, 1-alpha, overlay, alpha, 0)
    
    return result


def save_prediction_grid(images: List[np.ndarray], predictions: List[np.ndarray], 
                        ground_truth: List[np.ndarray], save_path: str, 
                        titles: Optional[List[str]] = None):
    """
    Save a grid of predictions for visualization.
    
    Args:
        images: List of original images
        predictions: List of prediction masks
        ground_truth: List of ground truth masks
        save_path: Path to save the grid
        titles: Optional titles for each sample
    """
    n_samples = len(images)
    fig, axes = plt.subplots(n_samples, 3, figsize=(12, 4 * n_samples))
    
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(n_samples):
        # Original image
        axes[i, 0].imshow(images[i])
        axes[i, 0].set_title(f"Original {i+1}" if titles is None else f"{titles[i]} - Original")
        axes[i, 0].axis('off')
        
        # Ground truth
        axes[i, 1].imshow(ground_truth[i], cmap='gray', vmin=0, vmax=1)
        axes[i, 1].set_title("Ground Truth")
        axes[i, 1].axis('off')
        
        # Prediction
        axes[i, 2].imshow(predictions[i], cmap='gray', vmin=0, vmax=1)
        axes[i, 2].set_title("Prediction")
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    # Test visualization functions
    print("Testing visualization utilities...")
    
    # Create test data
    test_images = torch.randn(4, 3, 120, 120)
    test_masks = torch.randint(0, 2, (4, 120, 120)).float()
    test_predictions = torch.randn(4, 1, 120, 120)
    
    # Test prediction visualization
    visualize_predictions(
        test_images, test_masks, test_predictions,
        save_path="test_predictions.png",
        num_samples=2
    )
    print("Test prediction visualization saved")
    
    # Test training visualizer
    visualizer = TrainingVisualizer("test_plots")
    
    # Create dummy history
    test_history = {
        'train_loss': [0.5, 0.4, 0.3, 0.2],
        'val_loss': [0.6, 0.5, 0.4, 0.3],
        'train_iou': [0.5, 0.6, 0.7, 0.8],
        'val_iou': [0.4, 0.5, 0.6, 0.7],
        'train_dice': [0.5, 0.6, 0.7, 0.8],
        'val_dice': [0.4, 0.5, 0.6, 0.7],
        'train_f1': [0.5, 0.6, 0.7, 0.8],
        'val_f1': [0.4, 0.5, 0.6, 0.7],
        'learning_rates': [1e-4, 8e-5, 6e-5, 4e-5]
    }
    
    visualizer.plot_training_history(test_history, 4)
    print("Test training history plot saved")
    
    print("Visualization tests completed!")
