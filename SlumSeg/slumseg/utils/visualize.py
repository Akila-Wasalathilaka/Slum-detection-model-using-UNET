"""Visualization utilities for slum segmentation."""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import torch
from typing import List, Tuple, Optional, Union
import rasterio
from rasterio.plot import show


def create_red_overlay(
    rgb_image: np.ndarray, 
    mask: np.ndarray, 
    alpha: float = 0.4
) -> np.ndarray:
    """
    Create red overlay on RGB image where mask is positive.
    
    Args:
        rgb_image: RGB image (H, W, 3) in 0-255 range
        mask: Binary mask (H, W) with 0/1 values
        alpha: Transparency of overlay (0=transparent, 1=opaque)
        
    Returns:
        RGB image with red overlay
    """
    overlay = rgb_image.copy().astype(np.float32)
    
    # Create red mask
    red_mask = np.zeros_like(overlay)
    red_mask[..., 2] = 255  # Red channel
    
    # Apply overlay where mask is positive
    mask_3d = np.repeat(mask[..., None], 3, axis=2).astype(bool)
    overlay[mask_3d] = (1 - alpha) * overlay[mask_3d] + alpha * red_mask[mask_3d]
    
    return overlay.astype(np.uint8)


def create_sample_grid(
    images_dir: str,
    masks_dir: str, 
    image_ids: List[str],
    n_samples: int = 9,
    figsize: Tuple[int, int] = (15, 15),
    save_path: Optional[str] = None
) -> Optional[plt.Figure]:
    """Create a grid showing sample images and masks."""
    
    # Select random samples
    if len(image_ids) > n_samples:
        selected_ids = np.random.choice(image_ids, n_samples, replace=False)
    else:
        selected_ids = image_ids[:n_samples]
    
    # Calculate grid size
    grid_size = int(np.ceil(np.sqrt(n_samples)))
    
    fig, axes = plt.subplots(grid_size, grid_size, figsize=figsize)
    if grid_size == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for i, img_id in enumerate(selected_ids):
        if i >= len(axes):
            break
            
        # Load image
        img_path = Path(images_dir) / f"{img_id}.tif"
        mask_path = Path(masks_dir) / f"{img_id}.tif"
        
        try:
            # Load RGB image
            with rasterio.open(img_path) as src:
                image = src.read([1, 2, 3])
                image = np.transpose(image, (1, 2, 0))
                image = (image / image.max() * 255).astype(np.uint8)
            
            # Load mask
            if mask_path.exists():
                with rasterio.open(mask_path) as src:
                    mask = src.read(1)
                    mask = (mask > 0).astype(np.uint8)
            else:
                mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
            
            # Create overlay
            overlay = create_red_overlay(image, mask, alpha=0.4)
            
            # Plot
            axes[i].imshow(overlay)
            axes[i].set_title(f"{img_id}\nSlum: {(mask > 0).sum()}/{mask.size} px")
            axes[i].axis('off')
            
        except (rasterio.RasterioIOError, FileNotFoundError) as e:
            axes[i].text(0.5, 0.5, f"Error loading\n{img_id}\n{str(e)[:20]}...", 
                        ha='center', va='center', transform=axes[i].transAxes)
            axes[i].axis('off')
    
    # Hide unused subplots
    for i in range(len(selected_ids), len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        return None
    
    return fig


def plot_class_distribution(
    class_counts: dict,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6)
) -> Optional[plt.Figure]:
    """Plot class distribution."""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Bar plot
    classes = list(class_counts.keys())
    counts = list(class_counts.values())
    
    bars = ax1.bar(classes, counts, alpha=0.7)
    ax1.set_ylabel('Pixel Count')
    ax1.set_title('Class Distribution')
    ax1.set_yscale('log')
    
    # Add count labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{count:,}', ha='center', va='bottom')
    
    # Pie chart
    ax2.pie(counts, labels=classes, autopct='%1.1f%%', startangle=90)
    ax2.set_title('Class Distribution (%)')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        return None
    
    return fig


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    train_metrics: List[float],
    val_metrics: List[float],
    metric_name: str = "IoU",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 5)
) -> Optional[plt.Figure]:
    """Plot training curves."""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    epochs = range(1, len(train_losses) + 1)
    
    # Loss curves
    ax1.plot(epochs, train_losses, 'b-', label='Train Loss', alpha=0.7)
    ax1.plot(epochs, val_losses, 'r-', label='Val Loss', alpha=0.7)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Metric curves  
    ax2.plot(epochs, train_metrics, 'b-', label=f'Train {metric_name}', alpha=0.7)
    ax2.plot(epochs, val_metrics, 'r-', label=f'Val {metric_name}', alpha=0.7)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel(metric_name)
    ax2.set_title(f'Training and Validation {metric_name}')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        return None
    
    return fig


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str] = ["Background", "Slum"],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 6)
) -> Optional[plt.Figure]:
    """Plot confusion matrix."""
    
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true.flatten(), y_pred.flatten())
    
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        return None
    
    return fig


def save_prediction_overlay(
    image: np.ndarray,
    prediction: np.ndarray, 
    output_path: str,
    threshold: float = 0.5,
    alpha: float = 0.4
):
    """Save prediction overlay as PNG."""
    
    # Convert prediction to binary mask
    if prediction.max() <= 1.0:
        # Probability map
        mask = (prediction > threshold).astype(np.uint8)
    else:
        # Already binary
        mask = (prediction > 0).astype(np.uint8)
    
    # Ensure image is in correct format
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    
    # Create overlay
    overlay = create_red_overlay(image, mask, alpha=alpha)
    
    # Save as PNG
    cv2.imwrite(output_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
