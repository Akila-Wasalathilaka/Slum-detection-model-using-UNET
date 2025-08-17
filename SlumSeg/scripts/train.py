#!/usr/bin/env python3
"""
Training script for slum segmentation models.
Supports single model training with all optimizations.
"""

import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
from tqdm import tqdm
import wandb
from omegaconf import OmegaConf

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from slumseg.data.dataset import SlumDataset, get_train_transforms, get_valid_transforms, get_image_ids_from_dir, split_dataset, calculate_class_weights
from slumseg.models.factory import make_model
from slumseg.models.losses import get_loss_function
from slumseg.utils.metrics import SegmentationMetrics
from slumseg.utils.optim import get_optimizer, get_scheduler
from slumseg.utils.seed import set_seed
from slumseg.train_loop import train_epoch, validate_epoch


def setup_hardware_optimizations(config):
    """Setup hardware optimizations for training."""
    # CUDNN optimizations
    if config.get('hardware', {}).get('cudnn_benchmark', True):
        torch.backends.cudnn.benchmark = True
    
    # TensorFloat-32 optimizations
    if config.get('hardware', {}).get('tf32', True):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    # Deterministic training
    if config.get('hardware', {}).get('deterministic', False):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def create_data_loaders(config, device):
    """Create train and validation data loaders."""
    data_config = config['data']
    train_config = config['train']
    
    # Get image IDs
    train_images_dir = os.path.join(data_config['root'], 'train', data_config['images_dir'])
    train_masks_dir = os.path.join(data_config['root'], 'train', data_config['masks_dir'])
    
    all_image_ids = get_image_ids_from_dir(train_images_dir)
    train_ids, val_ids = split_dataset(
        all_image_ids, 
        val_ratio=data_config.get('val_ratio', 0.15),
        random_state=config.get('seed', 42)
    )
    
    print(f"Train images: {len(train_ids)}")
    print(f"Validation images: {len(val_ids)}")
    
    # Calculate class weights if needed
    pos_weight = None
    loss_config = config.get('loss', {})
    if loss_config.get('pos_weight') is None:
        print("Calculating class weights...")
        class_weights = calculate_class_weights(train_masks_dir, train_ids)
        pos_weight = class_weights[0].item()
        print(f"Calculated pos_weight: {pos_weight:.3f}")
    else:
        pos_weight = loss_config['pos_weight']
    
    # Create datasets
    train_transforms = get_train_transforms(config)
    val_transforms = get_valid_transforms()
    
    train_dataset = SlumDataset(
        train_images_dir, train_masks_dir, train_ids, 
        transform=train_transforms, cache_images=False
    )
    
    val_dataset = SlumDataset(
        train_images_dir, train_masks_dir, val_ids,
        transform=val_transforms, cache_images=False
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config['batch_size'],
        shuffle=True,
        num_workers=train_config.get('num_workers', 4),
        pin_memory=train_config.get('pin_memory', True),
        persistent_workers=train_config.get('persistent_workers', True),
        prefetch_factor=train_config.get('prefetch_factor', 2)
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config['batch_size'],
        shuffle=False,
        num_workers=train_config.get('num_workers', 4),
        pin_memory=train_config.get('pin_memory', True),
        persistent_workers=train_config.get('persistent_workers', True),
        prefetch_factor=train_config.get('prefetch_factor', 2)
    )
    
    return train_loader, val_loader, pos_weight


def main():
    parser = argparse.ArgumentParser(description='Train slum segmentation model')
    parser.add_argument('--config', type=str, required=True, help='Config file path')
    parser.add_argument('--tiles', type=str, help='Tiles directory (optional)')
    parser.add_argument('--resume', type=str, help='Resume from checkpoint')
    parser.add_argument('--wandb', action='store_true', help='Use Weights & Biases logging')
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set random seed
    set_seed(config.get('seed', 42))
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Setup hardware optimizations
    setup_hardware_optimizations(config)
    
    # Initialize wandb if requested
    if args.wandb:
        wandb.init(
            project=config.get('project_name', 'SlumSeg'),
            config=config,
            name=f"{config['model']['arch']}_{config['model']['encoder']}"
        )
    
    # Create output directories
    output_dir = Path('outputs')
    checkpoints_dir = output_dir / 'checkpoints'
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader, val_loader, pos_weight = create_data_loaders(config, device)
    
    # Create model
    print("Creating model...")
    model = make_model(
        arch=config['model']['arch'],
        encoder=config['model']['encoder'],
        classes=config['model']['classes'],
        in_channels=config['model']['in_channels'],
        pretrained=config['model']['pretrained']
    )
    
    # Apply optimizations
    if config['train'].get('channels_last', False):
        model = model.to(memory_format=torch.channels_last)
    
    # Compile model if requested
    if config['train'].get('compile_mode'):
        try:
            model = torch.compile(model, mode=config['train']['compile_mode'])
            print(f"Model compiled with mode: {config['train']['compile_mode']}")
        except Exception as e:
            print(f"Failed to compile model: {e}")
    
    model = model.to(device)
    
    # Create loss function
    loss_config = config.get('loss', {})
    loss_config['pos_weight'] = pos_weight
    criterion = get_loss_function(loss_config, device)
    
    # Create optimizer and scheduler
    optimizer = get_optimizer(model, config['train'])
    scheduler = get_scheduler(optimizer, config['train'], len(train_loader))
    
    # Create metrics
    metrics = SegmentationMetrics(device=device)
    
    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler(enabled=config['train'].get('amp', False))
    
    # Training loop
    best_val_iou = 0.0
    patience_counter = 0
    patience = config['train'].get('early_stopping_patience', 10)
    
    print("Starting training...")
    for epoch in range(config['train']['epochs']):
        print(f"\nEpoch {epoch+1}/{config['train']['epochs']}")
        
        # Training
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, scheduler, 
            scaler, metrics, device, config
        )
        
        # Validation
        val_metrics = validate_epoch(
            model, val_loader, criterion, metrics, device, config
        )
        
        # Logging
        print(f"Train Loss: {train_metrics['loss']:.4f}, Train IoU: {train_metrics['iou']:.4f}")
        print(f"Val Loss: {val_metrics['loss']:.4f}, Val IoU: {val_metrics['iou']:.4f}")
        
        if args.wandb:
            wandb.log({
                'epoch': epoch,
                'train/loss': train_metrics['loss'],
                'train/iou': train_metrics['iou'],
                'train/dice': train_metrics['dice'],
                'val/loss': val_metrics['loss'],
                'val/iou': val_metrics['iou'],
                'val/dice': val_metrics['dice'],
                'lr': optimizer.param_groups[0]['lr']
            })
        
        # Save best model
        if val_metrics['iou'] > best_val_iou:
            best_val_iou = val_metrics['iou']
            patience_counter = 0
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'best_val_iou': best_val_iou,
                'config': config
            }
            
            torch.save(checkpoint, checkpoints_dir / 'best.ckpt')
            print(f"New best model saved! Val IoU: {best_val_iou:.4f}")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping triggered after {patience} epochs without improvement")
            break
        
        # Save periodic checkpoint
        if (epoch + 1) % 5 == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'val_iou': val_metrics['iou'],
                'config': config
            }
            torch.save(checkpoint, checkpoints_dir / f'epoch_{epoch+1}.ckpt')
    
    print(f"\nTraining completed! Best validation IoU: {best_val_iou:.4f}")
    
    if args.wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
