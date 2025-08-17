"""Loss functions for segmentation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict
import logging

logger = logging.getLogger(__name__)


class DiceLoss(nn.Module):
    """Dice loss for segmentation."""
    
    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        
        # Flatten
        probs = probs.view(-1)
        targets = targets.view(-1)
        
        intersection = (probs * targets).sum()
        union = probs.sum() + targets.sum()
        
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice


class FocalLoss(nn.Module):
    """Focal loss for addressing class imbalance."""
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        # Validate reduction parameter
        if reduction not in ['mean', 'sum', 'none']:
            raise ValueError(f"Invalid reduction: {reduction}. Must be 'mean', 'sum', or 'none'")
        self.reduction = reduction
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        elif self.reduction == 'none':
            return focal_loss
        else:
            raise ValueError(f"Invalid reduction: {self.reduction}")


class TverskyLoss(nn.Module):
    """Tversky loss - generalization of Dice loss."""
    
    def __init__(self, alpha: float = 0.7, beta: float = 0.3, smooth: float = 1e-6):
        super().__init__()
        self.alpha = alpha  # False positive penalty
        self.beta = beta    # False negative penalty
        self.smooth = smooth
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        
        # Flatten
        probs = probs.view(-1)
        targets = targets.view(-1)
        
        true_pos = (probs * targets).sum()
        false_neg = (targets * (1 - probs)).sum()
        false_pos = ((1 - targets) * probs).sum()
        
        tversky = (true_pos + self.smooth) / (
            true_pos + self.alpha * false_pos + self.beta * false_neg + self.smooth
        )
        
        return 1 - tversky


class CombinedLoss(nn.Module):
    """Combination of multiple losses."""
    
    def __init__(
        self,
        loss_type: str = "bce_dice",
        pos_weight: Optional[float] = None,
        alpha: float = 0.7,
        gamma: float = 2.0,
        beta: float = 0.7,
        bce_weight: float = 1.0,
        dice_weight: float = 1.0
    ):
        super().__init__()
        self.loss_type = loss_type.lower()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        
        # Initialize component losses
        self.dice_loss = DiceLoss()
        
        if pos_weight is not None:
            self.pos_weight = torch.tensor(pos_weight)
        else:
            self.pos_weight = None
            
        if loss_type == "focal_dice":
            self.focal_loss = FocalLoss(alpha=alpha, gamma=gamma)
        elif loss_type == "tversky":
            self.tversky_loss = TverskyLoss(alpha=alpha, beta=beta)
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Move pos_weight to correct device
        if self.pos_weight is not None:
            pos_weight = self.pos_weight.to(logits.device)
        else:
            pos_weight = None
        
        if self.loss_type == "bce_dice":
            bce = F.binary_cross_entropy_with_logits(
                logits, targets, pos_weight=pos_weight
            )
            dice = self.dice_loss(logits, targets)
            return self.bce_weight * bce + self.dice_weight * dice
            
        elif self.loss_type == "focal_dice":
            focal = self.focal_loss(logits, targets)
            dice = self.dice_loss(logits, targets)
            return self.bce_weight * focal + self.dice_weight * dice
            
        elif self.loss_type == "tversky":
            return self.tversky_loss(logits, targets)
            
        elif self.loss_type == "bce":
            return F.binary_cross_entropy_with_logits(
                logits, targets, pos_weight=pos_weight
            )
            
        elif self.loss_type == "dice":
            return self.dice_loss(logits, targets)
            
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")


def create_loss_function(config: Dict) -> nn.Module:
    """Create loss function from config."""
    
    loss_config = config.get("loss", {})
    loss_name = loss_config.get("name", "bce_dice")
    pos_weight = loss_config.get("pos_weight")
    
    loss_fn = CombinedLoss(
        loss_type=loss_name,
        pos_weight=pos_weight,
        alpha=loss_config.get("alpha", 0.7),
        gamma=loss_config.get("gamma", 2.0),
        beta=loss_config.get("beta", 0.7)
    )
    
    # Sanitize loss_name for logging to prevent log injection
    safe_loss_name = str(loss_name).replace('\n', '').replace('\r', '')
    logger.info(f"Created loss function: {safe_loss_name}")
    if pos_weight is not None:
        # Sanitize pos_weight for logging
        safe_pos_weight = str(pos_weight).replace('\n', '').replace('\r', '')
        logger.info(f"Using positive class weight: {safe_pos_weight}")
    
    return loss_fn


class WeightedBCELoss(nn.Module):
    """BCE loss with automatic positive weight calculation."""
    
    def __init__(self, pos_weight: Optional[float] = None):
        super().__init__()
        self.pos_weight = pos_weight
        self._auto_pos_weight = None
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Calculate pos_weight automatically if not provided
        if self.pos_weight is None and self._auto_pos_weight is None:
            pos_count = targets.sum()
            neg_count = targets.numel() - pos_count
            if pos_count > 0:
                self._auto_pos_weight = neg_count / pos_count
            else:
                self._auto_pos_weight = 1.0
        
        pos_weight = self.pos_weight or self._auto_pos_weight
        
        if pos_weight is not None:
            pos_weight_tensor = torch.tensor(pos_weight, device=logits.device)
        else:
            pos_weight_tensor = None
        
        return F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=pos_weight_tensor
        )
