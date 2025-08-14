"""
Global Loss Functions for Domain Generalization
==============================================
Advanced losses for robust training across domains
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt
import numpy as np

class BoundaryLoss(nn.Module):
    """Boundary loss using distance transform"""
    def __init__(self, theta0=3, theta=5):
        super().__init__()
        self.theta0 = theta0
        self.theta = theta
    
    def forward(self, pred, target):
        """
        pred: (B, 1, H, W) - predictions
        target: (B, 1, H, W) - ground truth
        """
        pred_sigmoid = torch.sigmoid(pred)
        
        # Compute distance transform for each sample in batch
        batch_size = target.shape[0]
        boundary_losses = []
        
        for i in range(batch_size):
            gt_np = target[i, 0].cpu().numpy()
            
            # Distance transform
            gt_dist = distance_transform_edt(gt_np)
            gt_dist = torch.from_numpy(gt_dist).float().to(pred.device)
            
            # Boundary loss
            boundary_loss = gt_dist * (pred_sigmoid[i, 0] - target[i, 0]) ** 2
            boundary_losses.append(boundary_loss.mean())
        
        return torch.stack(boundary_losses).mean()

class TverskyLoss(nn.Module):
    """Tversky loss for handling class imbalance"""
    def __init__(self, alpha=0.7, beta=0.3, smooth=1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
    
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        
        # Flatten
        pred = pred.view(-1)
        target = target.view(-1)
        
        # True positives, false positives, false negatives
        tp = (pred * target).sum()
        fp = (pred * (1 - target)).sum()
        fn = ((1 - pred) * target).sum()
        
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return 1 - tversky

class FocalTverskyLoss(nn.Module):
    """Focal Tversky loss for extremely hard examples"""
    def __init__(self, alpha=0.7, beta=0.3, gamma=0.75, smooth=1e-6):
        super().__init__()
        self.tversky = TverskyLoss(alpha, beta, smooth)
        self.gamma = gamma
    
    def forward(self, pred, target):
        tversky = self.tversky(pred, target)
        focal_tversky = torch.pow(tversky, self.gamma)
        return focal_tversky

class LovaszSoftmaxLoss(nn.Module):
    """Lovász-Softmax loss for IoU optimization"""
    def __init__(self):
        super().__init__()
    
    def lovasz_grad(self, gt_sorted):
        """Compute gradient of Lovász extension w.r.t sorted errors"""
        p = len(gt_sorted)
        gts = gt_sorted.sum()
        intersection = gts - gt_sorted.float().cumsum(0)
        union = gts + (1 - gt_sorted).float().cumsum(0)
        jaccard = 1. - intersection / union
        if p > 1:
            jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
        return jaccard
    
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        
        # Binary case
        pred = pred.view(-1)
        target = target.view(-1)
        
        # Sort by prediction confidence
        errors = (pred - target).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        target_sorted = target[perm]
        
        grad = self.lovasz_grad(target_sorted)
        loss = torch.dot(errors_sorted, grad)
        return loss

class OHEMLoss(nn.Module):
    """Online Hard Example Mining Loss"""
    def __init__(self, base_loss=nn.BCEWithLogitsLoss(reduction='none'), top_k=0.3):
        super().__init__()
        self.base_loss = base_loss
        self.top_k = top_k
    
    def forward(self, pred, target):
        # Compute per-pixel loss
        pixel_losses = self.base_loss(pred, target)
        
        # Flatten
        pixel_losses = pixel_losses.view(-1)
        
        # Select top-k hardest examples
        num_pixels = pixel_losses.numel()
        num_hard = int(num_pixels * self.top_k)
        
        hard_losses, _ = torch.topk(pixel_losses, num_hard)
        return hard_losses.mean()

class ComboLossV2(nn.Module):
    """Enhanced combination loss for global generalization"""
    def __init__(self, 
                 bce_weight=1.0,
                 dice_weight=1.0, 
                 focal_weight=1.0,
                 tversky_weight=0.5,
                 boundary_weight=0.3,
                 lovasz_weight=0.2):
        super().__init__()
        
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.tversky_weight = tversky_weight
        self.boundary_weight = boundary_weight
        self.lovasz_weight = lovasz_weight
        
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.focal = FocalLoss()
        self.tversky = TverskyLoss()
        self.boundary = BoundaryLoss()
        self.lovasz = LovaszSoftmaxLoss()
    
    def forward(self, pred, target, boundary_target=None):
        losses = {}
        
        # Main losses
        if self.bce_weight > 0:
            losses['bce'] = self.bce_weight * self.bce(pred, target)
        
        if self.dice_weight > 0:
            losses['dice'] = self.dice_weight * self.dice(pred, target)
        
        if self.focal_weight > 0:
            losses['focal'] = self.focal_weight * self.focal(pred, target)
        
        if self.tversky_weight > 0:
            losses['tversky'] = self.tversky_weight * self.tversky(pred, target)
        
        if self.boundary_weight > 0:
            losses['boundary'] = self.boundary_weight * self.boundary(pred, target)
        
        if self.lovasz_weight > 0:
            losses['lovasz'] = self.lovasz_weight * self.lovasz(pred, target)
        
        total_loss = sum(losses.values())
        return total_loss, losses

class DiceLoss(nn.Module):
    """Dice loss"""
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        
        pred = pred.view(-1)
        target = target.view(-1)
        
        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        return 1 - dice

class FocalLoss(nn.Module):
    """Focal loss for class imbalance"""
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, pred, target):
        bce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss