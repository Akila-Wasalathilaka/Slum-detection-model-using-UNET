"""
Ultra-accurate custom loss functions for slum detection.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from typing import Dict
import cv2
import numpy as np

class UltraFocalLoss(nn.Module):
    """Ultra-optimized Focal Loss for slum detection."""
    
    def __init__(self, alpha: float = 0.25, gamma: float = 3.0, adaptive: bool = True, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction
        self.adaptive = adaptive
        if adaptive:
            self.alpha = nn.Parameter(torch.tensor(alpha))
            self.gamma = nn.Parameter(torch.tensor(gamma))
        else:
            self.register_buffer('alpha', torch.tensor(alpha))
            self.register_buffer('gamma', torch.tensor(gamma))
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        alpha = torch.clamp(self.alpha, 0.1, 0.9)
        gamma = torch.clamp(self.gamma, 1.0, 5.0)
        
        # Use label smoothing for better generalization
        smooth_targets = targets * 0.95 + 0.025
        
        probs = torch.sigmoid(inputs)
        bce_loss = F.binary_cross_entropy_with_logits(inputs, smooth_targets, reduction='none')
        
        p_t = smooth_targets * probs + (1 - smooth_targets) * (1 - probs)
        alpha_t = smooth_targets * alpha + (1 - smooth_targets) * (1 - alpha)
        focal_weight = alpha_t * (1 - p_t) ** gamma
        
        focal_loss = focal_weight * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

class BoundaryLoss(nn.Module):
    """Boundary-aware loss for precise slum edge detection."""
    
    def __init__(self, theta0: float = 3, theta: float = 5):
        super().__init__()
        self.theta0 = theta0
        self.theta = theta
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(inputs)
        
        # Calculate distance transforms for boundary emphasis
        boundary_loss = 0
        batch_size = targets.shape[0]
        
        for i in range(batch_size):
            target_np = targets[i, 0].cpu().numpy().astype(np.uint8)
            pred_np = probs[i, 0].detach().cpu().numpy()
            
            # Calculate distance transform from boundaries
            contours, _ = cv2.findContours(target_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            boundary_mask = np.zeros_like(target_np)
            cv2.drawContours(boundary_mask, contours, -1, 1, thickness=3)
            
            # Distance-weighted loss
            dist_transform = cv2.distanceTransform(1 - boundary_mask, cv2.DIST_L2, 3)
            weight = np.exp(-dist_transform / self.theta0) + 0.1
            
            # Boundary-weighted BCE (using logits-safe version)
            target_torch = torch.from_numpy(target_np).float().to(inputs.device)
            pred_torch = torch.from_numpy(pred_np).float().to(inputs.device)
            weight_torch = torch.from_numpy(weight).float().to(inputs.device)
            
            # Convert to logits for safe autocast
            pred_logits = torch.logit(torch.clamp(pred_torch, 1e-7, 1-1e-7))
            bce = F.binary_cross_entropy_with_logits(pred_logits, target_torch, reduction='none')
            weighted_bce = (bce * weight_torch).mean()
            boundary_loss += weighted_bce
        
        return boundary_loss / batch_size

class UltraTverskyLoss(nn.Module):
    """Ultra-optimized Tversky Loss with dynamic weighting."""
    
    def __init__(self, alpha: float = 0.7, beta: float = 0.3, smooth: float = 1e-6, class_weight: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.class_weight = class_weight
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(inputs)
        
        # Dynamic alpha/beta based on class imbalance in batch
        pos_ratio = targets.mean()
        dynamic_alpha = self.alpha * (1 + pos_ratio)
        dynamic_beta = self.beta * (2 - pos_ratio)
        
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)
        
        true_pos = (probs_flat * targets_flat).sum()
        false_neg = (targets_flat * (1 - probs_flat)).sum()
        false_pos = ((1 - targets_flat) * probs_flat).sum() * self.class_weight
        
        tversky = (true_pos + self.smooth) / (
            true_pos + dynamic_alpha * false_neg + dynamic_beta * false_pos + self.smooth
        )
        
        return 1 - tversky

class UltraSlumDetectionLoss(nn.Module):
    """Ultra-accurate combined loss function for slum detection."""
    
    def __init__(self, weights: Dict[str, float] = None, config = None):
        super().__init__()
        
        # Default weights if not provided
        if weights is None:
            weights = {
                'dice': 0.35,
                'focal': 0.25, 
                'bce': 0.15,
                'tversky': 0.15,
                'boundary': 0.1
            }
        
        # Default config values if not provided
        if config is None:
            from config import config as default_config
            config = default_config
            
        self.weights = weights
        
        # Initialize ultra-optimized loss components
        self.dice_loss = smp.losses.DiceLoss(
            mode='binary', 
            from_logits=True, 
            smooth=1e-6,
            eps=1e-7  # Better numerical stability
        )
        
        self.focal_loss = UltraFocalLoss(
            alpha=config.FOCAL_ALPHA,
            gamma=config.FOCAL_GAMMA,
            adaptive=True
        )
        
        self.bce_loss = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([7.0])  # Higher weight for slum class
        )
        
        self.tversky_loss = UltraTverskyLoss(
            alpha=config.TVERSKY_ALPHA,
            beta=config.TVERSKY_BETA,
            class_weight=3.0  # Higher weight for slums
        )
        
        self.boundary_loss = BoundaryLoss(theta0=3, theta=5)
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Move weights to correct device
        if self.bce_loss.pos_weight.device != inputs.device:
            self.bce_loss.pos_weight = self.bce_loss.pos_weight.to(inputs.device)
        
        # Calculate individual losses
        dice = self.dice_loss(inputs, targets)
        focal = self.focal_loss(inputs, targets)
        bce = self.bce_loss(inputs, targets)
        tversky = self.tversky_loss(inputs, targets)
        boundary = self.boundary_loss(inputs, targets)
        
        # Combine with optimized weights
        total_loss = (
            self.weights['dice'] * dice +
            self.weights['focal'] * focal +
            self.weights['bce'] * bce +
            self.weights['tversky'] * tversky +
            self.weights['boundary'] * boundary
        )
        
        return total_loss
