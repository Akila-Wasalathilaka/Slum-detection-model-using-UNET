"""
Advanced Loss Functions for Slum Detection
==========================================

Comprehensive collection of loss functions optimized for binary segmentation
with class imbalance, specifically designed for slum detection tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class DiceLoss(nn.Module):
    """
    Dice Loss for binary segmentation.
    Better for overlapping regions and handles class imbalance well.
    """
    
    def __init__(self, smooth: float = 1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = torch.sigmoid(pred)
        target = target.float()  # Ensure target is float
        
        # Flatten tensors
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        # Calculate intersection and union
        intersection = (pred_flat * target_flat).sum()
        total = pred_flat.sum() + target_flat.sum()
        
        dice = (2.0 * intersection + self.smooth) / (total + self.smooth)
        return 1.0 - dice


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance in binary segmentation.
    Focuses learning on hard examples.
    """
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = 'mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Ensure target is float
        target = target.float()
        
        # Apply sigmoid to get probabilities
        pred_prob = torch.sigmoid(pred)
        
        # Calculate binary cross entropy
        bce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        
        # Calculate focal weight
        pt = torch.where(target == 1, pred_prob, 1 - pred_prob)
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        
        # Apply focal weight
        focal_loss = focal_weight * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class TverskyLoss(nn.Module):
    """
    Tversky Loss - generalization of Dice loss.
    Allows control over false positives vs false negatives.
    
    alpha: weight for false negatives (higher = penalize missing slums more)
    beta: weight for false positives (higher = penalize false detections more)
    """
    
    def __init__(self, alpha: float = 0.7, beta: float = 0.3, smooth: float = 1e-6):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = torch.sigmoid(pred)
        
        # Flatten tensors
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        # True positives, false positives, false negatives
        tp = (pred_flat * target_flat).sum()
        fp = (pred_flat * (1 - target_flat)).sum()
        fn = ((1 - pred_flat) * target_flat).sum()
        
        tversky = (tp + self.smooth) / (tp + self.alpha * fn + self.beta * fp + self.smooth)
        return 1.0 - tversky


class IoULoss(nn.Module):
    """
    IoU (Jaccard) Loss for binary segmentation.
    Directly optimizes the IoU metric.
    """
    
    def __init__(self, smooth: float = 1e-6):
        super(IoULoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = torch.sigmoid(pred)
        
        # Flatten tensors
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        # Calculate intersection and union
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum() - intersection
        
        iou = (intersection + self.smooth) / (union + self.smooth)
        return 1.0 - iou


class CombinedLoss(nn.Module):
    """
    Combined loss function mixing multiple loss types.
    Leverages strengths of different loss functions.
    """
    
    def __init__(
        self,
        bce_weight: float = 0.5,
        dice_weight: float = 0.3,
        focal_weight: float = 0.2,
        focal_alpha: float = 1.0,
        focal_gamma: float = 2.0
    ):
        super(CombinedLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss()
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = 0.0
        
        # Ensure target is float and in correct range
        target = target.float()
        
        if self.bce_weight > 0:
            loss += self.bce_weight * self.bce_loss(pred, target)
        
        if self.dice_weight > 0:
            loss += self.dice_weight * self.dice_loss(pred, target)
        
        if self.focal_weight > 0:
            loss += self.focal_weight * self.focal_loss(pred, target)
        
        return loss


class WeightedBCELoss(nn.Module):
    """
    Weighted Binary Cross Entropy Loss.
    Handles class imbalance by weighting positive/negative samples.
    """
    
    def __init__(self, pos_weight: Optional[float] = None):
        super(WeightedBCELoss, self).__init__()
        self.pos_weight = pos_weight
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.pos_weight is not None:
            pos_weight = torch.tensor(self.pos_weight, device=pred.device)
            return F.binary_cross_entropy_with_logits(pred, target, pos_weight=pos_weight)
        else:
            return F.binary_cross_entropy_with_logits(pred, target)


# Loss function factory
def create_loss(
    loss_type: str = "combined",
    class_weights: Optional[dict] = None,
    **kwargs
) -> nn.Module:
    """
    Factory function to create different loss functions.
    
    Args:
        loss_type: Type of loss ('bce', 'dice', 'focal', 'tversky', 'combined', 'iou')
        class_weights: Class weights for handling imbalance
        **kwargs: Additional parameters for specific loss functions
    
    Returns:
        Loss function ready for training
    """
    
    if loss_type.lower() == "bce":
        pos_weight = class_weights.get('pos_weight') if class_weights else None
        return WeightedBCELoss(pos_weight=pos_weight)
    
    elif loss_type.lower() == "dice":
        return DiceLoss(**kwargs)
    
    elif loss_type.lower() == "focal":
        return FocalLoss(**kwargs)
    
    elif loss_type.lower() == "tversky":
        return TverskyLoss(**kwargs)
    
    elif loss_type.lower() == "iou":
        return IoULoss(**kwargs)
    
    elif loss_type.lower() == "combined":
        return CombinedLoss(**kwargs)
    
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


# Predefined loss configurations for different scenarios
LOSS_CONFIGS = {
    "balanced": {
        "loss_type": "combined",
        "bce_weight": 0.5,
        "dice_weight": 0.5,
        "focal_weight": 0.0,
        "description": "Balanced BCE + Dice for general use"
    },
    "imbalanced": {
        "loss_type": "focal",
        "alpha": 1.0,
        "gamma": 2.0,
        "description": "Focal loss for heavily imbalanced data"
    },
    "precision_focused": {
        "loss_type": "tversky",
        "alpha": 0.3,  # Lower alpha = less penalty for FN
        "beta": 0.7,   # Higher beta = more penalty for FP
        "description": "Emphasizes precision over recall"
    },
    "recall_focused": {
        "loss_type": "tversky",
        "alpha": 0.7,  # Higher alpha = more penalty for FN
        "beta": 0.3,   # Lower beta = less penalty for FP
        "description": "Emphasizes recall over precision"
    },
    "overlap_focused": {
        "loss_type": "dice",
        "description": "Pure Dice loss for overlap optimization"
    }
}


def get_loss_info():
    """Print available loss configurations and their characteristics."""
    print("Available Loss Configurations:")
    print("=" * 50)
    for name, config in LOSS_CONFIGS.items():
        print(f"{name.upper()}:")
        print(f"  Type: {config['loss_type']}")
        print(f"  Description: {config['description']}")
        # Print specific parameters
        for key, value in config.items():
            if key not in ['loss_type', 'description']:
                print(f"  {key}: {value}")
        print()


if __name__ == "__main__":
    # Test loss functions
    pred = torch.randn(4, 1, 120, 120)
    target = torch.randint(0, 2, (4, 1, 120, 120)).float()
    
    print("Testing Loss Functions:")
    print("=" * 30)
    
    # Test each loss type
    for loss_name in ["bce", "dice", "focal", "tversky", "combined"]:
        loss_fn = create_loss(loss_name)
        loss_value = loss_fn(pred, target)
        print(f"{loss_name.upper()} Loss: {loss_value.item():.4f}")
    
    print("\n")
    get_loss_info()
