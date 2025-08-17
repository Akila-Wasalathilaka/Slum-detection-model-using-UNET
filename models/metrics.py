"""
Comprehensive Evaluation Metrics for Slum Detection
===================================================

Advanced metrics for binary segmentation evaluation, specifically optimized
for slum detection tasks with detailed analysis capabilities.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import precision_recall_curve, roc_auc_score, average_precision_score


class IoUScore(nn.Module):
    """
    Intersection over Union (IoU) / Jaccard Index for binary segmentation.
    Primary metric for segmentation evaluation.
    """
    
    def __init__(self, threshold: float = 0.5, smooth: float = 1e-6):
        super(IoUScore, self).__init__()
        self.threshold = threshold
        self.smooth = smooth
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = torch.sigmoid(pred)
        pred_binary = (pred > self.threshold).float()
        
        # Flatten tensors
        pred_flat = pred_binary.view(-1)
        target_flat = target.view(-1)
        
        # Calculate intersection and union
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum() - intersection
        
        iou = (intersection + self.smooth) / (union + self.smooth)
        return iou


class DiceScore(nn.Module):
    """
    Dice Coefficient / F1-Score for binary segmentation.
    Measures overlap between prediction and ground truth.
    """
    
    def __init__(self, threshold: float = 0.5, smooth: float = 1e-6):
        super(DiceScore, self).__init__()
        self.threshold = threshold
        self.smooth = smooth
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = torch.sigmoid(pred)
        pred_binary = (pred > self.threshold).float()
        
        # Flatten tensors
        pred_flat = pred_binary.view(-1)
        target_flat = target.view(-1)
        
        # Calculate intersection
        intersection = (pred_flat * target_flat).sum()
        total = pred_flat.sum() + target_flat.sum()
        
        dice = (2.0 * intersection + self.smooth) / (total + self.smooth)
        return dice


class PixelAccuracy(nn.Module):
    """
    Pixel-wise accuracy for binary segmentation.
    """
    
    def __init__(self, threshold: float = 0.5):
        super(PixelAccuracy, self).__init__()
        self.threshold = threshold
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = torch.sigmoid(pred)
        pred_binary = (pred > self.threshold).float()
        
        correct = (pred_binary == target).float()
        accuracy = correct.mean()
        return accuracy


class Precision(nn.Module):
    """
    Precision score for binary segmentation.
    TP / (TP + FP) - How many detected slums are actually slums?
    """
    
    def __init__(self, threshold: float = 0.5, smooth: float = 1e-6):
        super(Precision, self).__init__()
        self.threshold = threshold
        self.smooth = smooth
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = torch.sigmoid(pred)
        pred_binary = (pred > self.threshold).float()
        
        # Flatten tensors
        pred_flat = pred_binary.view(-1)
        target_flat = target.view(-1)
        
        # Calculate true positives and false positives
        tp = (pred_flat * target_flat).sum()
        fp = (pred_flat * (1 - target_flat)).sum()
        
        precision = (tp + self.smooth) / (tp + fp + self.smooth)
        return precision


class Recall(nn.Module):
    """
    Recall/Sensitivity score for binary segmentation.
    TP / (TP + FN) - How many actual slums are detected?
    """
    
    def __init__(self, threshold: float = 0.5, smooth: float = 1e-6):
        super(Recall, self).__init__()
        self.threshold = threshold
        self.smooth = smooth
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = torch.sigmoid(pred)
        pred_binary = (pred > self.threshold).float()
        
        # Flatten tensors
        pred_flat = pred_binary.view(-1)
        target_flat = target.view(-1)
        
        # Calculate true positives and false negatives
        tp = (pred_flat * target_flat).sum()
        fn = ((1 - pred_flat) * target_flat).sum()
        
        recall = (tp + self.smooth) / (tp + fn + self.smooth)
        return recall


class F1Score(nn.Module):
    """
    F1-Score (harmonic mean of precision and recall).
    Balanced measure of model performance.
    """
    
    def __init__(self, threshold: float = 0.5, smooth: float = 1e-6):
        super(F1Score, self).__init__()
        self.precision = Precision(threshold, smooth)
        self.recall = Recall(threshold, smooth)
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        prec = self.precision(pred, target)
        rec = self.recall(pred, target)
        
        f1 = 2 * (prec * rec) / (prec + rec + 1e-6)
        return f1


class Specificity(nn.Module):
    """
    Specificity score for binary segmentation.
    TN / (TN + FP) - How well does the model avoid false alarms?
    """
    
    def __init__(self, threshold: float = 0.5, smooth: float = 1e-6):
        super(Specificity, self).__init__()
        self.threshold = threshold
        self.smooth = smooth
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = torch.sigmoid(pred)
        pred_binary = (pred > self.threshold).float()
        
        # Flatten tensors
        pred_flat = pred_binary.view(-1)
        target_flat = target.view(-1)
        
        # Calculate true negatives and false positives
        tn = ((1 - pred_flat) * (1 - target_flat)).sum()
        fp = (pred_flat * (1 - target_flat)).sum()
        
        specificity = (tn + self.smooth) / (tn + fp + self.smooth)
        return specificity


class SegmentationMetrics:
    """
    Comprehensive metrics calculator for segmentation evaluation.
    Computes all relevant metrics and provides detailed analysis.
    """
    
    def __init__(self, threshold: float = 0.5, device: str = 'cuda'):
        self.threshold = threshold
        self.device = device
        
        # Initialize metric calculators
        self.iou = IoUScore(threshold).to(device)
        self.dice = DiceScore(threshold).to(device)
        self.accuracy = PixelAccuracy(threshold).to(device)
        self.precision = Precision(threshold).to(device)
        self.recall = Recall(threshold).to(device)
        self.f1 = F1Score(threshold).to(device)
        self.specificity = Specificity(threshold).to(device)
    
    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
        """
        Calculate all metrics for the given prediction and target.
        
        Args:
            pred: Model predictions (logits)
            target: Ground truth binary masks
        
        Returns:
            Dictionary of metric names and values
        """
        with torch.no_grad():
            metrics = {
                'iou': self.iou(pred, target).item(),
                'dice': self.dice(pred, target).item(),
                'accuracy': self.accuracy(pred, target).item(),
                'precision': self.precision(pred, target).item(),
                'recall': self.recall(pred, target).item(),
                'f1': self.f1(pred, target).item(),
                'specificity': self.specificity(pred, target).item()
            }
        
        return metrics
    
    def detailed_analysis(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor
    ) -> Dict[str, any]:
        """
        Perform detailed analysis including confusion matrix and threshold analysis.
        """
        pred_probs = torch.sigmoid(pred).cpu().numpy()
        target_np = target.cpu().numpy()
        
        # Flatten for sklearn metrics
        pred_flat = pred_probs.flatten()
        target_flat = target_np.flatten()
        
        # Calculate advanced metrics
        try:
            auc_score = roc_auc_score(target_flat, pred_flat)
            ap_score = average_precision_score(target_flat, pred_flat)
        except:
            auc_score = 0.0
            ap_score = 0.0
        
        # Confusion matrix components
        pred_binary = (pred_flat > self.threshold).astype(int)
        tp = np.sum((pred_binary == 1) & (target_flat == 1))
        tn = np.sum((pred_binary == 0) & (target_flat == 0))
        fp = np.sum((pred_binary == 1) & (target_flat == 0))
        fn = np.sum((pred_binary == 0) & (target_flat == 1))
        
        return {
            'confusion_matrix': {'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn},
            'auc_roc': auc_score,
            'average_precision': ap_score,
            'positive_pixel_ratio': np.mean(target_flat),
            'predicted_positive_ratio': np.mean(pred_binary),
            'total_pixels': len(target_flat)
        }


class MetricsTracker:
    """
    Track metrics across multiple batches and epochs.
    Provides running averages and history.
    """
    
    def __init__(self, metric_names: List[str]):
        self.metric_names = metric_names
        self.reset()
    
    def reset(self):
        """Reset all tracked metrics."""
        self.values = {name: [] for name in self.metric_names}
        self.running_sum = {name: 0.0 for name in self.metric_names}
        self.count = 0
    
    def update(self, metrics: Dict[str, float]):
        """Update metrics with new values."""
        for name in self.metric_names:
            if name in metrics:
                self.values[name].append(metrics[name])
                self.running_sum[name] += metrics[name]
        self.count += 1
    
    def get_average(self) -> Dict[str, float]:
        """Get average values across all updates."""
        if self.count == 0:
            return {name: 0.0 for name in self.metric_names}
        
        return {name: self.running_sum[name] / self.count for name in self.metric_names}
    
    def get_history(self) -> Dict[str, List[float]]:
        """Get full history of metric values."""
        return self.values.copy()


def create_metrics(threshold: float = 0.5, device: str = 'cuda') -> SegmentationMetrics:
    """
    Factory function to create metrics calculator.
    
    Args:
        threshold: Binary threshold for predictions
        device: Device to run calculations on
    
    Returns:
        Configured metrics calculator
    """
    return SegmentationMetrics(threshold=threshold, device=device)


# Metric configurations for different evaluation scenarios
METRIC_CONFIGS = {
    "standard": {
        "threshold": 0.5,
        "metrics": ["iou", "dice", "accuracy", "precision", "recall", "f1"],
        "description": "Standard segmentation metrics"
    },
    "conservative": {
        "threshold": 0.7,
        "metrics": ["iou", "dice", "precision", "specificity"],
        "description": "Conservative threshold, emphasis on precision"
    },
    "sensitive": {
        "threshold": 0.3,
        "metrics": ["iou", "dice", "recall", "f1"],
        "description": "Sensitive threshold, emphasis on recall"
    },
    "comprehensive": {
        "threshold": 0.5,
        "metrics": ["iou", "dice", "accuracy", "precision", "recall", "f1", "specificity"],
        "description": "All available metrics"
    }
}


def get_metrics_info():
    """Print available metric configurations."""
    print("Available Metric Configurations:")
    print("=" * 40)
    for name, config in METRIC_CONFIGS.items():
        print(f"{name.upper()}:")
        print(f"  Threshold: {config['threshold']}")
        print(f"  Metrics: {', '.join(config['metrics'])}")
        print(f"  Description: {config['description']}")
        print()


if __name__ == "__main__":
    # Test metrics
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Testing metrics on device: {device}")
    
    # Create test data
    pred = torch.randn(2, 1, 120, 120).to(device)
    target = torch.randint(0, 2, (2, 1, 120, 120)).float().to(device)
    
    # Test metrics calculator
    metrics_calc = create_metrics(threshold=0.5, device=device)
    metrics = metrics_calc(pred, target)
    
    print("\nMetric Results:")
    print("=" * 20)
    for name, value in metrics.items():
        print(f"{name.upper()}: {value:.4f}")
    
    # Test detailed analysis
    detailed = metrics_calc.detailed_analysis(pred, target)
    print(f"\nAUC-ROC: {detailed['auc_roc']:.4f}")
    print(f"Average Precision: {detailed['average_precision']:.4f}")
    
    print("\n")
    get_metrics_info()
