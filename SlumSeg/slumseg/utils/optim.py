"""Optimizer and scheduler utilities."""

import torch
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR, StepLR
from typing import Dict, Any


def get_optimizer(model: torch.nn.Module, config: Dict[str, Any]) -> torch.optim.Optimizer:
    """Create optimizer from config."""
    optimizer_name = config.get('optimizer', 'adamw').lower()
    lr = config.get('lr', 1e-3)
    weight_decay = config.get('weight_decay', 1e-4)
    
    if optimizer_name == 'adam':
        optimizer = Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            eps=1e-8
        )
    elif optimizer_name == 'adamw':
        optimizer = AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            eps=1e-8
        )
    elif optimizer_name == 'sgd':
        momentum = config.get('momentum', 0.9)
        optimizer = SGD(
            model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=True
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    
    return optimizer


def get_scheduler(
    optimizer: torch.optim.Optimizer, 
    config: Dict[str, Any], 
    steps_per_epoch: int
) -> torch.optim.lr_scheduler._LRScheduler:
    """Create learning rate scheduler from config."""
    scheduler_name = config.get('scheduler', 'onecycle').lower()
    epochs = config.get('epochs', 100)
    
    if scheduler_name == 'onecycle':
        max_lr = config.get('lr', 1e-3)
        scheduler = OneCycleLR(
            optimizer,
            max_lr=max_lr,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            pct_start=0.1,
            anneal_strategy='cos',
            div_factor=25.0,
            final_div_factor=1e4
        )
    elif scheduler_name == 'cosine':
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=epochs,
            eta_min=1e-6
        )
    elif scheduler_name == 'step':
        step_size = config.get('step_size', epochs // 3)
        gamma = config.get('gamma', 0.1)
        scheduler = StepLR(
            optimizer,
            step_size=step_size,
            gamma=gamma
        )
    elif scheduler_name == 'none':
        scheduler = None
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")
    
    return scheduler
