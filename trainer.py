"""
Ultra-accurate training pipeline for slum detection.
"""
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from pathlib import Path
from typing import Tuple
from losses import UltraSlumDetectionLoss
from config import config

class UltraAccurateTrainer:
    """Ultra-accurate training pipeline with advanced optimizations."""
    
    def __init__(self, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, device: torch.device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Ultra-optimized loss function
        self.criterion = UltraSlumDetectionLoss(config.LOSS_WEIGHTS, config)
        
        # Advanced optimizer with better parameters
        self.optimizer = AdamW(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
            eps=1e-8,
            betas=(0.9, 0.999)
        )
        
        # Enhanced scheduler with warm restarts
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=20,  # First restart every 20 epochs
            T_mult=2,  # Double the period after each restart
            eta_min=1e-6
        )
        
        # Mixed precision scaler with device detection
        if self.device.type == 'cuda':
            self.scaler = torch.amp.GradScaler('cuda')
            self.use_amp = True
        else:
            self.scaler = None
            self.use_amp = False
        
        # Enhanced metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.val_ious = []
        self.val_dices = []
        self.val_precisions = []
        self.val_recalls = []
        self.val_f1s = []
        
        # Memory management
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        
        # Ultra-accurate model tracking
        self.best_score = 0.0
        self.best_iou = 0.0
        self.best_dice = 0.0
        self.patience_counter = 0
        
        # Learning rate warmup
        self.warmup_steps = len(train_loader) * config.WARMUP_EPOCHS
        self.current_step = 0
    
    def _get_warmup_lr(self):
        """Calculate learning rate during warmup."""
        if self.current_step < self.warmup_steps:
            return config.LEARNING_RATE * (self.current_step / self.warmup_steps)
        return config.LEARNING_RATE
    
    def _calculate_metrics(self, predictions, targets, threshold=None):
        """Calculate comprehensive metrics."""
        if threshold is None:
            threshold = config.BINARY_THRESHOLDS[config.DEFAULT_THRESHOLD]
        
        preds = torch.sigmoid(predictions)
        binary_preds = (preds > threshold).float()
        
        # Calculate metrics
        tp = (binary_preds * targets).sum()
        fp = (binary_preds * (1 - targets)).sum()
        fn = ((1 - binary_preds) * targets).sum()
        tn = ((1 - binary_preds) * (1 - targets)).sum()
        
        # IoU (Intersection over Union)
        intersection = tp
        union = tp + fp + fn
        iou = (intersection + 1e-8) / (union + 1e-8)
        
        # Dice Score
        dice = (2 * tp + 1e-8) / (2 * tp + fp + fn + 1e-8)
        
        # Precision, Recall, F1
        precision = (tp + 1e-8) / (tp + fp + 1e-8)
        recall = (tp + 1e-8) / (tp + fn + 1e-8)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
        
        return {
            'iou': iou.item(),
            'dice': dice.item(),
            'precision': precision.item(),
            'recall': recall.item(),
            'f1': f1.item()
        }
    
    def train_epoch(self) -> float:
        """Memory-efficient training for one epoch."""
        self.model.train()
        epoch_loss = 0.0
        num_batches = len(self.train_loader)
        
        # Enable gradient checkpointing if available
        if hasattr(self.model, 'backbone') and hasattr(self.model.backbone, 'encoder'):
            self.model.backbone.encoder.gradient_checkpointing_enable()
        
        pbar = tqdm(self.train_loader, desc="Training")
        for batch_idx, (images, masks) in enumerate(pbar):
            # Efficient GPU memory transfer
            images = images.to(self.device, non_blocking=True, memory_format=torch.channels_last)
            masks = masks.to(self.device, non_blocking=True)
            
            # Clear GPU cache periodically
            if batch_idx % 10 == 0:
                torch.cuda.empty_cache()
            
            # Warmup learning rate
            if self.current_step < self.warmup_steps:
                lr = self._get_warmup_lr()
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr
            
            self.optimizer.zero_grad()
            
            if self.use_amp:
                with torch.amp.autocast('cuda'):
                    outputs = self.model(images)
                    if isinstance(outputs, tuple):
                        main_output, aux1, aux2, aux3 = outputs
                        # Enhanced deep supervision with progressive weighting
                        main_loss = self.criterion(main_output, masks)
                        aux_loss1 = self.criterion(aux1, masks) * 0.5
                        aux_loss2 = self.criterion(aux2, masks) * 0.3
                        aux_loss3 = self.criterion(aux3, masks) * 0.2
                        loss = main_loss + aux_loss1 + aux_loss2 + aux_loss3
                    else:
                        loss = self.criterion(outputs, masks)
            else:
                outputs = self.model(images)
                if isinstance(outputs, tuple):
                    main_output, aux1, aux2, aux3 = outputs
                    # Enhanced deep supervision with progressive weighting
                    main_loss = self.criterion(main_output, masks)
                    aux_loss1 = self.criterion(aux1, masks) * 0.5
                    aux_loss2 = self.criterion(aux2, masks) * 0.3
                    aux_loss3 = self.criterion(aux3, masks) * 0.2
                    loss = main_loss + aux_loss1 + aux_loss2 + aux_loss3
                else:
                    loss = self.criterion(outputs, masks)
            
            if self.use_amp:
                self.scaler.scale(loss).backward()
                
                # Enhanced gradient clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), config.GRAD_CLIP_VALUE)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), config.GRAD_CLIP_VALUE)
                self.optimizer.step()
            
            # Update scheduler after warmup
            if self.current_step >= self.warmup_steps:
                self.scheduler.step(epoch_loss / (batch_idx + 1))
            
            epoch_loss += loss.item()
            self.current_step += 1
            
            # Memory cleanup for small GPU
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            
            if batch_idx % 10 == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'lr': f"{current_lr:.2e}"
                })
        
        return epoch_loss / num_batches
    
    @torch.cuda.amp.autocast()
    def validate_epoch(self) -> Tuple[float, dict]:
        """Memory-efficient validation."""
        self.model.eval()
        val_loss = 0.0
        metrics_sum = {
            'iou': 0., 'dice': 0., 'precision': 0., 
            'recall': 0., 'f1': 0.
        }
        num_batches = len(self.val_loader)
        
        # Disable gradient checkpointing for validation
        if hasattr(self.model, 'backbone') and hasattr(self.model.backbone, 'encoder'):
            self.model.backbone.encoder.gradient_checkpointing_disable()
        
        with torch.no_grad(), torch.cuda.amp.autocast():
            for images, masks in tqdm(self.val_loader, desc="Validation"):
                # Efficient GPU memory transfer
                images = images.to(self.device, non_blocking=True, memory_format=torch.channels_last)
                masks = masks.to(self.device, non_blocking=True)
                
                # Clear cache periodically
                torch.cuda.empty_cache()
                
                if self.use_amp:
                    with torch.amp.autocast('cuda'):
                        outputs = self.model(images)
                        if isinstance(outputs, tuple):
                            outputs = outputs[0]  # Use main output for validation
                        
                        loss = self.criterion(outputs, masks)
                else:
                    outputs = self.model(images)
                    if isinstance(outputs, tuple):
                        outputs = outputs[0]  # Use main output for validation
                    
                    loss = self.criterion(outputs, masks)
                
                val_loss += loss.item()
                
                # Calculate and accumulate metrics
                metrics = self._calculate_metrics(outputs, masks)
                for k, v in metrics.items():
                    metrics_sum[k] += v
                
                val_loss += loss.item()
        
        # Calculate averages
        avg_val_loss = val_loss / num_batches
        avg_metrics = {k: v / num_batches for k, v in metrics_sum.items()}
        
        return avg_val_loss, avg_metrics
    
    def train(self) -> dict:
        """Main ultra-accurate training loop."""
        print("Starting ultra-accurate slum detection training...")
        
        for epoch in range(config.EPOCHS):
            print(f"\nEpoch {epoch + 1}/{config.EPOCHS}")
            
            # Train
            train_loss = self.train_epoch()
            
            # Validate
            val_loss, val_metrics = self.validate_epoch()
            
            # Track metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_ious.append(val_metrics['iou'])
            self.val_dices.append(val_metrics['dice'])
            self.val_precisions.append(val_metrics['precision'])
            self.val_recalls.append(val_metrics['recall'])
            self.val_f1s.append(val_metrics['f1'])
            
            # Ultra-accurate combined score with multiple metrics
            combined_score = (
                0.3 * val_metrics['iou'] + 
                0.3 * val_metrics['dice'] + 
                0.2 * val_metrics['f1'] + 
                0.2 * val_metrics['precision']
            )
            
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            print(f"IoU: {val_metrics['iou']:.4f}, Dice: {val_metrics['dice']:.4f}")
            print(f"Precision: {val_metrics['precision']:.4f}, Recall: {val_metrics['recall']:.4f}")
            print(f"F1: {val_metrics['f1']:.4f}")
            print(f"Combined Score: {combined_score:.4f}")
            
            # Save best model with multiple criteria
            is_best = False
            if combined_score > self.best_score:
                self.best_score = combined_score
                self.best_iou = val_metrics['iou']
                self.best_dice = val_metrics['dice']
                self.patience_counter = 0
                is_best = True
            else:
                self.patience_counter += 1
            
            if is_best:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'best_score': self.best_score,
                    'best_iou': self.best_iou,
                    'best_dice': self.best_dice,
                    'val_metrics': val_metrics,
                    'config': config
                }, config.MODEL_DIR / 'ultra_accurate_slum_model.pth')
                
                print(f"🎯 New BEST model saved! Score: {self.best_score:.4f}")
                print(f"   Best IoU: {self.best_iou:.4f}, Best Dice: {self.best_dice:.4f}")
            
            # Early stopping with patience
            if self.patience_counter >= config.PATIENCE:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break
            
            # Save checkpoint every 20 epochs
            if (epoch + 1) % 20 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'train_losses': self.train_losses,
                    'val_losses': self.val_losses,
                    'val_metrics': {
                        'iou': self.val_ious,
                        'dice': self.val_dices,
                        'precision': self.val_precisions,
                        'recall': self.val_recalls,
                        'f1': self.val_f1s
                    }
                }, config.CHECKPOINT_DIR / f'ultra_checkpoint_epoch_{epoch + 1}.pth')
        
        print(f"\n🎉 Training completed!")
        print(f"🏆 Best Score: {self.best_score:.4f}")
        print(f"🎯 Best IoU: {self.best_iou:.4f}")
        print(f"🎯 Best Dice: {self.best_dice:.4f}")
        
        # Return training results
        return {
            'best_score': self.best_score,
            'best_iou': self.best_iou,
            'best_dice': self.best_dice,
            'epochs_trained': len(self.train_losses),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_ious': self.val_ious,
            'val_dices': self.val_dices,
            'val_precisions': self.val_precisions,
            'val_recalls': self.val_recalls,
            'val_f1s': self.val_f1s
        }
