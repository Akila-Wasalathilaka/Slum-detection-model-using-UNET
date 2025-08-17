"""Metrics for segmentation evaluation."""

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class SegmentationMetrics:
    """Comprehensive metrics for binary segmentation."""
    
    def __init__(self, threshold: float = 0.5, eps: float = 1e-6):
        self.threshold = threshold
        self.eps = eps
        self.reset()
    
    def reset(self):
        """Reset accumulated metrics."""
        self.predictions = []
        self.targets = []
        self.logits = []
    
    def update(self, logits: torch.Tensor, targets: torch.Tensor):
        """Update metrics with batch results."""
        # Convert to numpy and flatten
        logits_np = logits.detach().cpu().numpy().flatten()
        targets_np = targets.detach().cpu().numpy().flatten()
        
        self.logits.extend(logits_np)
        self.targets.extend(targets_np)
        
        # Predictions
        probs = torch.sigmoid(logits).detach().cpu().numpy().flatten()
        preds = (probs > self.threshold).astype(np.uint8)
        self.predictions.extend(preds)
    
    def compute(self) -> Dict[str, float]:
        """Compute all metrics."""
        if not self.targets:
            return {}
        
        targets = np.array(self.targets)
        predictions = np.array(self.predictions)
        logits = np.array(self.logits)
        probs = 1 / (1 + np.exp(-logits))  # sigmoid
        
        metrics = {}
        
        # Basic pixel-wise metrics
        tp = np.sum((predictions == 1) & (targets == 1))
        fp = np.sum((predictions == 1) & (targets == 0))
        fn = np.sum((predictions == 0) & (targets == 1))
        tn = np.sum((predictions == 0) & (targets == 0))
        
        # IoU (Jaccard)
        iou = tp / (tp + fp + fn + self.eps)
        metrics['iou'] = iou
        
        # Dice/F1
        dice = 2 * tp / (2 * tp + fp + fn + self.eps)
        metrics['dice'] = dice
        metrics['f1'] = dice  # Same as dice for binary
        
        # Precision, Recall
        precision = tp / (tp + fp + self.eps)
        recall = tp / (tp + fn + self.eps)
        metrics['precision'] = precision
        metrics['recall'] = recall
        
        # Specificity
        specificity = tn / (tn + fp + self.eps)
        metrics['specificity'] = specificity
        
        # Accuracy
        accuracy = (tp + tn) / (tp + tn + fp + fn + self.eps)
        metrics['accuracy'] = accuracy
        
        # ROC AUC
        try:
            if len(np.unique(targets)) > 1:  # Need both classes
                auc = roc_auc_score(targets, probs)
                metrics['auc'] = auc
        except Exception as e:
            logger.warning(f"Could not compute AUC: {e}")
        
        # Average Precision (PR AUC)
        try:
            if len(np.unique(targets)) > 1:
                ap = average_precision_score(targets, probs)
                metrics['avg_precision'] = ap
        except Exception as e:
            logger.warning(f"Could not compute AP: {e}")
        
        return metrics
    
    def get_confusion_matrix(self) -> np.ndarray:
        """Get confusion matrix."""
        if not self.targets:
            return np.array([[0, 0], [0, 0]])
        
        targets = np.array(self.targets)
        predictions = np.array(self.predictions)
        
        return confusion_matrix(targets, predictions, labels=[0, 1])


def compute_iou(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    """Compute IoU for a single prediction."""
    pred = (torch.sigmoid(pred) > threshold).float()
    target = target.float()
    
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    
    iou = intersection / (union + 1e-6)
    return iou.item()


def compute_dice(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    """Compute Dice coefficient."""
    pred = (torch.sigmoid(pred) > threshold).float()
    target = target.float()
    
    intersection = (pred * target).sum()
    dice = (2.0 * intersection) / (pred.sum() + target.sum() + 1e-6)
    
    return dice.item()


def threshold_sweep(logits: torch.Tensor, targets: torch.Tensor, 
                   thresholds: Optional[List[float]] = None) -> Dict[str, List[float]]:
    """Perform threshold sweep to find optimal threshold."""
    
    if thresholds is None:
        thresholds = np.arange(0.1, 0.95, 0.05).tolist()
    
    metrics = {'threshold': [], 'iou': [], 'dice': [], 'f1': []}
    
    for thresh in thresholds:
        # Compute metrics at this threshold
        pred = (torch.sigmoid(logits) > thresh).float()
        target = targets.float()
        
        # IoU
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum() - intersection
        iou = intersection / (union + 1e-6)
        
        # Dice
        dice = (2.0 * intersection) / (pred.sum() + target.sum() + 1e-6)
        
        metrics['threshold'].append(thresh)
        metrics['iou'].append(iou.item())
        metrics['dice'].append(dice.item())
        metrics['f1'].append(dice.item())
    
    return metrics


def compute_pixel_metrics(predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    """Compute comprehensive pixel-wise metrics."""
    
    # Flatten arrays
    pred_flat = predictions.flatten()
    targ_flat = targets.flatten()
    
    # Confusion matrix components
    tp = np.sum((pred_flat == 1) & (targ_flat == 1))
    fp = np.sum((pred_flat == 1) & (targ_flat == 0))
    fn = np.sum((pred_flat == 0) & (targ_flat == 1))
    tn = np.sum((pred_flat == 0) & (targ_flat == 0))
    
    eps = 1e-6
    
    metrics = {
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
        'iou': tp / (tp + fp + fn + eps),
        'dice': 2 * tp / (2 * tp + fp + fn + eps),
        'precision': tp / (tp + fp + eps),
        'recall': tp / (tp + fn + eps),
        'specificity': tn / (tn + fp + eps),
        'accuracy': (tp + tn) / (tp + tn + fp + fn + eps),
    }
    
    return metrics


class RegionMetrics:
    """Compute metrics per region/city."""
    
    def __init__(self):
        self.region_data = {}
    
    def update(self, logits: torch.Tensor, targets: torch.Tensor, regions: List[str]):
        """Update with region information."""
        for i, region in enumerate(regions):
            if region not in self.region_data:
                self.region_data[region] = {'logits': [], 'targets': []}
            
            self.region_data[region]['logits'].append(logits[i].detach().cpu())
            self.region_data[region]['targets'].append(targets[i].detach().cpu())
    
    def compute(self, threshold: float = 0.5) -> Dict[str, Dict[str, float]]:
        """Compute metrics per region."""
        region_metrics = {}
        
        for region, data in self.region_data.items():
            # Concatenate all data for this region
            logits = torch.cat(data['logits'], dim=0)
            targets = torch.cat(data['targets'], dim=0)
            
            # Compute metrics
            pred = (torch.sigmoid(logits) > threshold).float()
            
            intersection = (pred * targets).sum()
            union = pred.sum() + targets.sum() - intersection
            
            iou = intersection / (union + 1e-6)
            dice = (2.0 * intersection) / (pred.sum() + targets.sum() + 1e-6)
            
            region_metrics[region] = {
                'iou': iou.item(),
                'dice': dice.item(),
                'samples': len(data['logits'])
            }
        
        return region_metrics
