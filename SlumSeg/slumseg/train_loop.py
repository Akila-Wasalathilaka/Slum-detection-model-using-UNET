"""Training loop utilities with mixed precision and optimizations."""

import torch
import torch.nn.functional as F
from tqdm import tqdm
from typing import Dict, Any
import numpy as np


def train_epoch(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    scaler: torch.cuda.amp.GradScaler,
    metrics,
    device: torch.device,
    config: Dict[str, Any]
) -> Dict[str, float]:
    """Train for one epoch with mixed precision."""
    
    model.train()
    metrics.reset()
    
    running_loss = 0.0
    num_batches = len(train_loader)
    
    # Training configuration
    use_amp = config['train'].get('amp', False)
    channels_last = config['train'].get('channels_last', False)
    accumulate_batches = config['train'].get('accumulate_grad_batches', 1)
    
    optimizer.zero_grad()
    
    pbar = tqdm(train_loader, desc="Training")
    for batch_idx, batch in enumerate(pbar):
        images = batch['image'].to(device, non_blocking=True)
        targets = batch['mask'].to(device, non_blocking=True)
        
        # Convert to channels last if enabled
        if channels_last:
            images = images.to(memory_format=torch.channels_last)
        
        # Forward pass with mixed precision
        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, targets)
            
            # Scale loss for gradient accumulation
            if accumulate_batches > 1:
                loss = loss / accumulate_batches
        
        # Backward pass
        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Update metrics
        with torch.inference_mode():
            metrics.update(outputs, targets)
        
        # Optimizer step
        if (batch_idx + 1) % accumulate_batches == 0:
            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            
            optimizer.zero_grad()
        
        # Update learning rate for step-based schedulers
        if scheduler and (hasattr(scheduler, 'step_update') or scheduler.__class__.__name__ == 'OneCycleLR'):
            scheduler.step()
        
        # Update running loss
        running_loss += loss.item() * accumulate_batches
        
        # Update progress bar
        current_lr = optimizer.param_groups[0]['lr']
        pbar.set_postfix({
            'loss': f'{running_loss/(batch_idx+1):.4f}',
            'lr': f'{current_lr:.2e}'
        })
    
    # Step scheduler if not OneCycleLR
    if scheduler and scheduler.__class__.__name__ != 'OneCycleLR':
        scheduler.step()
    
    # Compute final metrics
    final_metrics = metrics.compute()
    final_metrics['loss'] = running_loss / num_batches
    
    return final_metrics


def validate_epoch(
    model: torch.nn.Module,
    val_loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    metrics,
    device: torch.device,
    config: Dict[str, Any]
) -> Dict[str, float]:
    """Validate for one epoch."""
    
    model.eval()
    metrics.reset()
    
    running_loss = 0.0
    num_batches = len(val_loader)
    
    # Validation configuration
    use_amp = config['train'].get('amp', False)
    channels_last = config['train'].get('channels_last', False)
    use_tta = config['eval'].get('tta', False)
    
    pbar = tqdm(val_loader, desc="Validation")
    
    with torch.inference_mode():
        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(device, non_blocking=True)
            targets = batch['mask'].to(device, non_blocking=True)
            
            # Convert to channels last if enabled
            if channels_last:
                images = images.to(memory_format=torch.channels_last)
            
            # Forward pass with mixed precision
            with torch.cuda.amp.autocast(enabled=use_amp):
                if use_tta:
                    outputs = test_time_augmentation(model, images)
                else:
                    outputs = model(images)
                
                loss = criterion(outputs, targets)
            
            # Update metrics
            metrics.update(outputs, targets)
            
            # Update running loss
            running_loss += loss.item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{running_loss/(batch_idx+1):.4f}'
            })
    
    # Compute final metrics
    final_metrics = metrics.compute()
    final_metrics['loss'] = running_loss / num_batches
    
    return final_metrics


def test_time_augmentation(model: torch.nn.Module, images: torch.Tensor) -> torch.Tensor:
    """Apply test time augmentation (TTA)."""
    
    batch_size = images.size(0)
    device = images.device
    
    # Original prediction
    outputs = model(images)
    
    # Horizontal flip
    images_hflip = torch.flip(images, dims=[3])
    outputs_hflip = model(images_hflip)
    outputs_hflip = torch.flip(outputs_hflip, dims=[3])
    
    # Vertical flip  
    images_vflip = torch.flip(images, dims=[2])
    outputs_vflip = model(images_vflip)
    outputs_vflip = torch.flip(outputs_vflip, dims=[2])
    
    # Both flips
    images_hvflip = torch.flip(torch.flip(images, dims=[2]), dims=[3])
    outputs_hvflip = model(images_hvflip)
    outputs_hvflip = torch.flip(torch.flip(outputs_hvflip, dims=[3]), dims=[2])
    
    # Average predictions
    final_outputs = (outputs + outputs_hflip + outputs_vflip + outputs_hvflip) / 4.0
    
    return final_outputs


def predict_batch(
    model: torch.nn.Module,
    images: torch.Tensor,
    device: torch.device,
    use_tta: bool = False,
    use_amp: bool = True
) -> torch.Tensor:
    """Predict on a batch of images."""
    
    model.eval()
    
    with torch.inference_mode():
        images = images.to(device, non_blocking=True)
        
        with torch.cuda.amp.autocast(enabled=use_amp):
            if use_tta:
                outputs = test_time_augmentation(model, images)
            else:
                outputs = model(images)
    
    return outputs
