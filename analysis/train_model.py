import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import json
import os
from datetime import datetime
import argparse

from unet_slum_detection import (
    SlumDataset, get_transforms, create_model,
    FocalLoss, TverskyLoss, DiceLoss, CombinedLoss,
    train_epoch, validate_epoch, save_sample_predictions,
    device
)

def main():
    # Configuration
    config = {
        'data_root': 'data',
        'batch_size': 16,
        'learning_rate': 1e-4,
        'num_epochs': 100,
        'early_stopping_patience': 15,
        'architecture': 'unet',  # unet, unetplusplus, deeplabv3plus
        'encoder': 'resnet34',   # resnet34, efficientnet-b0, resnet50
        'target_size': (120, 120),
        'loss_functions': ['bce', 'focal', 'tversky', 'combined'],
        'save_dir': 'experiments'
    }
    
    print(f"Training Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print(f"Device: {device}")
    print("-" * 50)
    
    # Create save directory
    os.makedirs(config['save_dir'], exist_ok=True)
    
    # Prepare datasets
    print("Loading datasets...")
    
    train_dataset = SlumDataset(
        image_dir=os.path.join(config['data_root'], 'train', 'images'),
        mask_dir=os.path.join(config['data_root'], 'train', 'masks'),
        transform=get_transforms('train'),
        target_size=config['target_size']
    )
    
    val_dataset = SlumDataset(
        image_dir=os.path.join(config['data_root'], 'val', 'images'),
        mask_dir=os.path.join(config['data_root'], 'val', 'masks'),
        transform=get_transforms('val'),
        target_size=config['target_size']
    )
    
    # Data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Check class distribution in training set
    print("\nAnalyzing class distribution...")
    slum_pixels = 0
    total_pixels = 0
    
    for i in range(min(100, len(train_dataset))):  # Sample 100 images
        _, mask = train_dataset[i]
        total_pixels += mask.numel()
        slum_pixels += (mask == 1).sum().item()
    
    slum_ratio = slum_pixels / total_pixels
    print(f"Slum pixels: {slum_pixels:,} ({slum_ratio:.2%})")
    print(f"Non-slum pixels: {total_pixels - slum_pixels:,} ({1-slum_ratio:.2%})")
    print(f"Class imbalance ratio: 1:{(total_pixels - slum_pixels)/slum_pixels:.1f}")
    
    # Train with different loss functions
    results = {}
    
    for loss_name in config['loss_functions']:
        print(f"\n{'='*60}")
        print(f"Training with {loss_name.upper()} loss")
        print(f"{'='*60}")
        
        # Create model
        model = create_model(
            architecture=config['architecture'],
            encoder=config['encoder'],
            pretrained=True
        ).to(device)
        
        # Loss function
        if loss_name == 'bce':
            criterion = torch.nn.CrossEntropyLoss()
        elif loss_name == 'focal':
            criterion = FocalLoss(alpha=0.75, gamma=2.0)
        elif loss_name == 'tversky':
            criterion = TverskyLoss(alpha=0.7, beta=0.3)
        elif loss_name == 'combined':
            criterion = CombinedLoss(bce_weight=0.5, dice_weight=0.5)
        
        # Optimizer and scheduler
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=1e-4
        )
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        # Training history
        history = {
            'train_loss': [], 'val_loss': [],
            'train_iou': [], 'val_iou': [],
            'train_dice': [], 'val_dice': [],
            'train_f1': [], 'val_f1': [],
            'lr': []
        }
        
        best_val_iou = 0.0
        patience_counter = 0
        
        print(f"Starting training with {loss_name} loss...")
        
        for epoch in range(1, config['num_epochs'] + 1):
            # Train
            train_loss, train_metrics = train_epoch(
                model, train_loader, criterion, optimizer, device, epoch
            )
            
            # Validate
            val_loss, val_metrics = validate_epoch(
                model, val_loader, criterion, device, epoch
            )
            
            # Update scheduler
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]['lr']
            
            # Save history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_iou'].append(train_metrics['iou'])
            history['val_iou'].append(val_metrics['iou'])
            history['train_dice'].append(train_metrics['dice'])
            history['val_dice'].append(val_metrics['dice'])
            history['train_f1'].append(train_metrics['f1'])
            history['val_f1'].append(val_metrics['f1'])
            history['lr'].append(current_lr)
            
            # Print progress
            print(f"Epoch {epoch:3d} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Train IoU: {train_metrics['iou']:.4f} | "
                  f"Val IoU: {val_metrics['iou']:.4f} | "
                  f"Val Dice: {val_metrics['dice']:.4f} | "
                  f"LR: {current_lr:.2e}")
            
            # Save best model
            if val_metrics['iou'] > best_val_iou:
                best_val_iou = val_metrics['iou']
                patience_counter = 0
                
                # Save model
                model_path = os.path.join(config['save_dir'], f'best_model_{loss_name}.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_iou': best_val_iou,
                    'config': config
                }, model_path)
                
                print(f"  → New best IoU: {best_val_iou:.4f} (saved)")
                
                # Save sample predictions
                save_sample_predictions(
                    model, val_dataset, device,
                    save_path=os.path.join(config['save_dir'], f'predictions_{loss_name}.png')
                )
                
            else:
                patience_counter += 1
                
            # Early stopping
            if patience_counter >= config['early_stopping_patience']:
                print(f"Early stopping triggered after {epoch} epochs")
                break
        
        # Save training history
        results[loss_name] = {
            'best_val_iou': best_val_iou,
            'final_epoch': epoch,
            'history': history
        }
        
        with open(os.path.join(config['save_dir'], f'history_{loss_name}.json'), 'w') as f:
            json.dump(history, f, indent=2)
        
        print(f"\n{loss_name.upper()} training completed!")
        print(f"Best validation IoU: {best_val_iou:.4f}")
        
        # Clean up GPU memory
        del model
        torch.cuda.empty_cache()
    
    # Compare results
    print(f"\n{'='*60}")
    print("FINAL RESULTS COMPARISON")
    print(f"{'='*60}")
    
    comparison_data = []
    for loss_name, result in results.items():
        best_iou = result['best_val_iou']
        final_epoch = result['final_epoch']
        
        print(f"{loss_name.upper():12} | Best IoU: {best_iou:.4f} | Epochs: {final_epoch:3d}")
        comparison_data.append((loss_name, best_iou, final_epoch))
    
    # Find best performing model
    best_loss, best_iou, _ = max(comparison_data, key=lambda x: x[1])
    print(f"\n🏆 Best performing loss: {best_loss.upper()} (IoU: {best_iou:.4f})")
    
    if best_iou >= 0.70:
        print(f"✅ Target accuracy achieved! ({best_iou:.1%} IoU)")
    else:
        print(f"⚠️  Target accuracy not reached. Best: {best_iou:.1%} IoU")
    
    # Create comparison plots
    plot_comparison(results, config['save_dir'])
    
    # Save final comparison
    with open(os.path.join(config['save_dir'], 'final_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nAll results saved to: {config['save_dir']}")
    
    return results

def plot_comparison(results, save_dir):
    """Plot comparison of different loss functions"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    metrics = ['iou', 'dice', 'f1', 'loss']
    titles = ['IoU Score', 'Dice Score', 'F1 Score', 'Loss']
    
    for i, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[i // 2, i % 2]
        
        for loss_name, result in results.items():
            history = result['history']
            
            if metric == 'loss':
                epochs = range(1, len(history['val_loss']) + 1)
                ax.plot(epochs, history['val_loss'], label=f'{loss_name} (val)', marker='o', markersize=3)
            else:
                epochs = range(1, len(history[f'val_{metric}']) + 1)
                ax.plot(epochs, history[f'val_{metric}'], label=f'{loss_name} (val)', marker='o', markersize=3)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel(title)
        ax.set_title(f'Validation {title}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'loss_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print("Comparison plots saved!")

if __name__ == "__main__":
    results = main()
