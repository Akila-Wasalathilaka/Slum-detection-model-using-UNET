"""
Utility functions for production slum detection pipeline.
"""
import random
import numpy as np
import torch
import logging
import warnings
from pathlib import Path
import matplotlib.pyplot as plt
import cv2
from typing import Dict, List, Tuple, Any
import json
import os

def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def setup_logging(log_level=logging.INFO):
    """Setup logging configuration."""
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('training.log'),
            logging.StreamHandler()
        ]
    )

def setup_ultra_logging(log_level=logging.INFO):
    """Setup ultra-accurate logging configuration."""
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('ultra_training.log'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger('UltraAccurateSlumDetection')
    return logger

def suppress_warnings():
    """Suppress unnecessary warnings."""
    warnings.filterwarnings("ignore")
    import os
    os.environ['PYTHONWARNINGS'] = 'ignore'
    logging.getLogger().setLevel(logging.ERROR)

def setup_device():
    """Setup and configure device."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True
        print(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    return device

def create_directories(config):
    """Create necessary directories."""
    for dir_path in [config.RESULTS_DIR, config.MODEL_DIR, config.CHECKPOINT_DIR]:
        dir_path.mkdir(exist_ok=True, parents=True)

def count_parameters(model):
    """Count model parameters."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def save_experiment_info(config, model, save_path):
    """Save experiment configuration and model info."""
    import json
    
    total_params, trainable_params = count_parameters(model)
    
    info = {
        'config': {
            'image_size': config.PRIMARY_SIZE,
            'batch_size': config.BATCH_SIZE,
            'epochs': config.EPOCHS,
            'learning_rate': config.LEARNING_RATE,
            'weight_decay': config.WEIGHT_DECAY,
            'loss_weights': config.LOSS_WEIGHTS,
            'encoders': config.ENCODERS,
            'encoder_weights': config.ENCODER_WEIGHTS
        },
        'model': {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'architecture': 'ProductionUNet'
        }
    }
    
    with open(save_path, 'w') as f:
        json.dump(info, f, indent=2)

def calculate_ultra_metrics(predictions: np.ndarray, targets: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
    """Calculate comprehensive metrics for ultra-accurate evaluation."""
    from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
    
    # Flatten arrays for metric calculation
    pred_flat = predictions.flatten()
    target_flat = targets.flatten()
    
    # Convert to binary
    pred_binary = (pred_flat > 0.5).astype(int)
    target_binary = target_flat.astype(int)
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(target_binary, pred_binary),
        'precision': precision_score(target_binary, pred_binary, zero_division=0),
        'recall': recall_score(target_binary, pred_binary, zero_division=0),
        'f1_score': f1_score(target_binary, pred_binary, zero_division=0),
        'dice_score': dice_coefficient(predictions, targets),
        'iou_score': iou_score(predictions, targets),
        'avg_confidence': np.mean(scores) if len(scores) > 0 else 0.0
    }
    
    return metrics

def dice_coefficient(pred: np.ndarray, target: np.ndarray, smooth: float = 1e-6) -> float:
    """Calculate Dice coefficient."""
    pred_flat = pred.flatten()
    target_flat = target.flatten()
    
    intersection = np.sum(pred_flat * target_flat)
    union = np.sum(pred_flat) + np.sum(target_flat)
    
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice

def iou_score(pred: np.ndarray, target: np.ndarray, smooth: float = 1e-6) -> float:
    """Calculate IoU score."""
    pred_flat = pred.flatten()
    target_flat = target.flatten()
    
    intersection = np.sum(pred_flat * target_flat)
    union = np.sum(pred_flat) + np.sum(target_flat) - intersection
    
    iou = (intersection + smooth) / (union + smooth)
    return iou

def save_training_plots(train_losses: List[float], val_losses: List[float], 
                       train_metrics: List[float], val_metrics: List[float],
                       save_path: str):
    """Save training plots."""
    fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    ax1.plot(train_losses, label='Train Loss', color='blue')
    ax1.plot(val_losses, label='Val Loss', color='red')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Metrics plot
    ax2.plot(train_metrics, label='Train Dice', color='blue')
    ax2.plot(val_metrics, label='Val Dice', color='red')
    ax2.set_title('Training and Validation Dice Score')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Dice Score')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def visualize_ultra_results(images: np.ndarray, predictions: np.ndarray, 
                           targets: np.ndarray, save_path: str, num_samples: int = 4):
    """Visualize ultra-accurate results."""
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))
    
    for i in range(num_samples):
        if i >= len(images):
            break
            
        # Original image
        axes[i, 0].imshow(images[i].transpose(1, 2, 0))
        axes[i, 0].set_title('Original Image')
        axes[i, 0].axis('off')
        
        # Prediction
        axes[i, 1].imshow(predictions[i], cmap='gray')
        axes[i, 1].set_title('Prediction')
        axes[i, 1].axis('off')
        
        # Ground truth
        axes[i, 2].imshow(targets[i], cmap='gray')
        axes[i, 2].set_title('Ground Truth')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
