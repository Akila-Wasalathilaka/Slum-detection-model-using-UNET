"""
Main Training Script for Slum Detection Model
=============================================

Comprehensive training script with experiment management, multiple model
comparisons, and advanced training features for optimal slum detection.
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import argparse
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import project modules
from models import create_model, create_loss
from models.metrics import create_metrics, MetricsTracker
from config import get_model_config, get_training_config, get_data_config
from utils.dataset import SlumDataset, create_data_loaders, verify_dataset_setup
from utils.transforms import get_train_transforms, get_val_transforms
from utils.checkpoint import CheckpointManager
from utils.visualization import TrainingVisualizer

# Configure environment
def setup_environment(config):
    """Setup training environment and device."""
    # Set random seeds for reproducibility
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)
    
    # Determine device
    if config.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(config.device)
    
    print(f"🔧 Training Environment:")
    print(f"  Device: {device}")
    print(f"  PyTorch Version: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"  CUDA Version: {torch.version.cuda}")
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory // 1024**3} GB")
    print()
    
    return device


def create_experiment_dir(config):
    """Create experiment directory with timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"{config.experiment_name}_{timestamp}"
    exp_dir = Path("experiments") / exp_name
    
    # Create directories
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "checkpoints").mkdir(exist_ok=True)
    (exp_dir / "logs").mkdir(exist_ok=True)
    (exp_dir / "plots").mkdir(exist_ok=True)
    (exp_dir / "configs").mkdir(exist_ok=True)
    
    return exp_dir


def save_configs(exp_dir, model_config, training_config, data_config):
    """Save all configurations to experiment directory."""
    configs_dir = exp_dir / "configs"
    
    model_config.save(configs_dir / "model_config.json")
    training_config.save(configs_dir / "training_config.json")
    data_config.save(configs_dir / "data_config.json")
    
    print(f"📁 Configs saved to: {configs_dir}")


def create_optimizer(model, config):
    """Create optimizer based on configuration."""
    if config.optimizer.lower() == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            betas=config.betas,
            weight_decay=config.weight_decay
        )
    elif config.optimizer.lower() == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            betas=config.betas,
            weight_decay=config.weight_decay
        )
    elif config.optimizer.lower() == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=config.learning_rate,
            momentum=config.momentum,
            weight_decay=config.weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer: {config.optimizer}")
    
    return optimizer


def create_scheduler(optimizer, config):
    """Create learning rate scheduler."""
    if config.scheduler.lower() == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.scheduler_params.get("T_max", config.epochs),
            eta_min=config.scheduler_params.get("eta_min", 1e-6)
        )
    elif config.scheduler.lower() == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=config.mode,
            patience=config.scheduler_params.get("patience", 10),
            factor=config.scheduler_params.get("factor", 0.5),
            min_lr=config.scheduler_params.get("min_lr", 1e-6)
        )
    elif config.scheduler.lower() == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.scheduler_params.get("step_size", 30),
            gamma=config.scheduler_params.get("gamma", 0.1)
        )
    elif config.scheduler.lower() == "exponential":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=config.scheduler_params.get("gamma", 0.95)
        )
    else:
        scheduler = None
    
    return scheduler


def train_epoch(model, train_loader, criterion, optimizer, device, config, epoch):
    """Train for one epoch."""
    model.train()
    
    # Initialize metrics tracking
    metrics_calc = create_metrics(device=device)
    metrics_tracker = MetricsTracker(['loss', 'iou', 'dice', 'precision', 'recall', 'f1'])
    
    # Enable mixed precision if configured
    scaler = torch.cuda.amp.GradScaler() if config.use_amp else None
    
    for batch_idx, (images, masks) in enumerate(train_loader):
        images, masks = images.to(device), masks.to(device)
        
        # Add channel dimension to masks if needed
        if len(masks.shape) == 3:
            masks = masks.unsqueeze(1)
        
        # Ensure masks are float type for loss computation
        masks = masks.float()
        
        optimizer.zero_grad()
        
        # Forward pass with mixed precision
        if config.use_amp:
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, masks)
        else:
            outputs = model(images)
            loss = criterion(outputs, masks)
        
        # Backward pass
        if config.use_amp:
            scaler.scale(loss).backward()
            if config.grad_clip_norm:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if config.grad_clip_norm:
                nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
            optimizer.step()
        
        # Calculate metrics
        with torch.no_grad():
            metrics = metrics_calc(outputs, masks)
            metrics['loss'] = loss.item()
            metrics_tracker.update(metrics)
        
        # Log progress
        if batch_idx % config.log_frequency == 0:
            print(f"Epoch {epoch}/{config.epochs} [{batch_idx}/{len(train_loader)}] "
                  f"Loss: {loss.item():.4f} IoU: {metrics['iou']:.4f} "
                  f"Dice: {metrics['dice']:.4f}")
    
    return metrics_tracker.get_average()


def validate_epoch(model, val_loader, criterion, device):
    """Validate for one epoch."""
    model.eval()
    
    metrics_calc = create_metrics(device=device)
    metrics_tracker = MetricsTracker(['loss', 'iou', 'dice', 'precision', 'recall', 'f1'])
    
    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)
            
            # Add channel dimension to masks if needed
            if len(masks.shape) == 3:
                masks = masks.unsqueeze(1)
            
            # Ensure masks are float type for loss computation
            masks = masks.float()
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            # Calculate metrics
            metrics = metrics_calc(outputs, masks)
            metrics['loss'] = loss.item()
            metrics_tracker.update(metrics)
    
    return metrics_tracker.get_average()


class EarlyStopping:
    """Early stopping utility."""
    
    def __init__(self, patience=15, min_delta=1e-4, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_score = None
        self.counter = 0
        self.early_stop = False
    
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif self.mode == 'max':
            if score > self.best_score + self.min_delta:
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1
        else:  # mode == 'min'
            if score < self.best_score - self.min_delta:
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1
        
        if self.counter >= self.patience:
            self.early_stop = True
        
        return self.early_stop


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train Slum Detection Model')
    parser.add_argument('--model', default='balanced', help='Model configuration preset')
    parser.add_argument('--training', default='development', help='Training configuration preset')
    parser.add_argument('--data', default='standard', help='Data configuration preset')
    parser.add_argument('--experiment', default=None, help='Experiment name override')
    args = parser.parse_args()
    
    print("🏗️  SLUM DETECTION MODEL TRAINING")
    print("=" * 50)
    print(f"Model Config: {args.model}")
    print(f"Training Config: {args.training}")
    print(f"Data Config: {args.data}")
    print()
    
    # Load configurations
    model_config = get_model_config(args.model)
    training_config = get_training_config(args.training)
    data_config = get_data_config(args.data)
    
    # Override experiment name if provided
    if args.experiment:
        training_config.experiment_name = args.experiment
    
    # Setup environment
    device = setup_environment(training_config)
    
    # Create experiment directory
    exp_dir = create_experiment_dir(training_config)
    print(f"📁 Experiment directory: {exp_dir}")
    
    # Save configurations
    save_configs(exp_dir, model_config, training_config, data_config)
    
    # Verify dataset setup
    if not verify_dataset_setup(data_config):
        print("❌ Dataset verification failed!")
        return
    
    # Create datasets
    print("📊 Creating datasets...")
    paths = data_config.get_paths()
    
    # Get transforms
    train_transforms = get_train_transforms(data_config)
    val_transforms = get_val_transforms(data_config)
    
    # Create datasets
    train_dataset = SlumDataset(
        images_dir=paths['train_images'],
        masks_dir=paths['train_masks'],
        transform=train_transforms,
        slum_rgb=data_config.slum_rgb,
        image_size=data_config.image_size,
        use_tile_masks_only=data_config.use_tile_masks_only,
        min_slum_pixels=data_config.min_slum_pixels,
        max_slum_percentage=data_config.max_slum_percentage,
        min_slum_percentage=data_config.min_slum_percentage
    )
    
    val_dataset = SlumDataset(
        images_dir=paths['val_images'],
        masks_dir=paths['val_masks'],
        transform=val_transforms,
        slum_rgb=data_config.slum_rgb,
        image_size=data_config.image_size,
        use_tile_masks_only=data_config.use_tile_masks_only,
        min_slum_pixels=data_config.min_slum_pixels,
        max_slum_percentage=data_config.max_slum_percentage,
        min_slum_percentage=data_config.min_slum_percentage
    )
    
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        print("❌ No valid samples found in dataset!")
        return
    
    # Create data loaders
    data_loaders = create_data_loaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=training_config.batch_size,
        num_workers=training_config.num_workers,
        pin_memory=training_config.pin_memory,
        use_weighted_sampling=training_config.use_weighted_sampling
    )
    
    # Create model
    print("🏗️  Creating model...")
    model = create_model(
        architecture=model_config.architecture,
        encoder=model_config.encoder,
        pretrained=model_config.pretrained,
        num_classes=model_config.num_classes
    )
    model = model.to(device)
    
    # Calculate model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model Parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    # Create loss function with class weights
    class_weights = train_dataset.get_class_weights()
    if training_config.class_weights:
        class_weights.update(training_config.class_weights)
    
    criterion = create_loss(
        loss_type=training_config.loss_type,
        class_weights=class_weights,
        **training_config.loss_params
    )
    
    # Create optimizer and scheduler
    optimizer = create_optimizer(model, training_config)
    scheduler = create_scheduler(optimizer, training_config)
    
    # Create checkpoint manager
    checkpoint_manager = CheckpointManager(
        exp_dir / "checkpoints",
        max_checkpoints=training_config.max_checkpoints
    )
    
    # Create visualizer
    visualizer = TrainingVisualizer(exp_dir / "plots")
    
    # Early stopping
    early_stopping = None
    if training_config.early_stopping:
        early_stopping = EarlyStopping(
            patience=training_config.patience,
            min_delta=training_config.min_delta,
            mode=training_config.mode
        )
    
    # Training history
    history = {
        'train_loss': [], 'val_loss': [],
        'train_iou': [], 'val_iou': [],
        'train_dice': [], 'val_dice': [],
        'train_precision': [], 'val_precision': [],
        'train_recall': [], 'val_recall': [],
        'train_f1': [], 'val_f1': [],
        'learning_rates': []
    }
    
    # Training loop
    print("🚀 Starting training...")
    best_metric = 0.0 if training_config.mode == 'max' else float('inf')
    
    for epoch in range(1, training_config.epochs + 1):
        print(f"\nEpoch {epoch}/{training_config.epochs}")
        print("-" * 30)
        
        # Training
        train_metrics = train_epoch(
            model, data_loaders['train'], criterion, optimizer, device, training_config, epoch
        )
        
        # Validation
        if epoch % training_config.val_frequency == 0:
            val_metrics = validate_epoch(model, data_loaders['val'], criterion, device)
            
            # Update history - safely handle missing keys
            for key in train_metrics:
                train_key = f'train_{key}'
                if train_key in history:
                    history[train_key].append(train_metrics[key])
                    
            for key in val_metrics:
                val_key = f'val_{key}'
                if val_key in history:
                    history[val_key].append(val_metrics[key])
            
            # Learning rate
            current_lr = optimizer.param_groups[0]['lr']
            history['learning_rates'].append(current_lr)
            
            print(f"Train - Loss: {train_metrics['loss']:.4f}, IoU: {train_metrics['iou']:.4f}, "
                  f"Dice: {train_metrics['dice']:.4f}")
            print(f"Val   - Loss: {val_metrics['loss']:.4f}, IoU: {val_metrics['iou']:.4f}, "
                  f"Dice: {val_metrics['dice']:.4f}")
            print(f"Learning Rate: {current_lr:.6f}")
            
            # Monitor metric for checkpointing and early stopping
            monitor_value = val_metrics[training_config.monitor_metric.replace('val_', '')]
            
            # Save checkpoint
            is_best = False
            if training_config.mode == 'max' and monitor_value > best_metric:
                best_metric = monitor_value
                is_best = True
            elif training_config.mode == 'min' and monitor_value < best_metric:
                best_metric = monitor_value
                is_best = True
            
            if is_best or not training_config.save_best_only:
                checkpoint_manager.save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    metrics=val_metrics,
                    is_best=is_best
                )
            
            # Early stopping
            if early_stopping:
                if early_stopping(monitor_value):
                    print(f"Early stopping triggered at epoch {epoch}")
                    break
            
            # Update scheduler
            if scheduler:
                if training_config.scheduler.lower() == "plateau":
                    scheduler.step(monitor_value)
                else:
                    scheduler.step()
            
            # Plot training progress
            if epoch % training_config.plot_frequency == 0:
                visualizer.plot_training_history(history, epoch)
        else:
            # Update history with training metrics only
            for key in train_metrics:
                if f'train_{key}' in history:
                    history[f'train_{key}'].append(train_metrics[key])
            
            # Update scheduler (non-plateau schedulers)
            if scheduler and training_config.scheduler.lower() != "plateau":
                scheduler.step()
    
    # Final visualizations
    print("\n📊 Creating final plots...")
    visualizer.plot_training_history(history, training_config.epochs)
    
    # Save training history
    with open(exp_dir / "training_history.json", 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\n✅ Training completed!")
    print(f"📁 Results saved to: {exp_dir}")
    print(f"🏆 Best {training_config.monitor_metric}: {best_metric:.4f}")
    
    # Run post-training analysis
    best_checkpoint = checkpoint_manager.best_checkpoint
    if best_checkpoint and Path(best_checkpoint).exists():
        print(f"\n🎯 Running post-training analysis...")
        try:
            from charts.post_training_analysis import run_post_training_analysis
            
            analysis_results = run_post_training_analysis(
                checkpoint_path=str(best_checkpoint),
                analysis_type="quick",  # Use quick analysis by default
                output_dir=str(exp_dir / "charts"),
                model_config_name=args.model,
                data_config_name=args.data
            )
            
            print(f"📊 Analysis results saved to: {exp_dir / 'charts'}")
            
        except Exception as e:
            print(f"⚠️  Post-training analysis failed: {str(e)}")
            print("   You can run it manually later using:")
            print(f"   python charts/post_training_analysis.py --checkpoint {best_checkpoint}")
        
        # Generate advanced predictions
        print(f"\n🔮 Generating advanced predictions...")
        try:
            from advanced_slum_detection import AdvancedSlumDetector
            
            # Create advanced predictions directory
            advanced_dir = exp_dir / "advanced_predictions"
            advanced_dir.mkdir(exist_ok=True)
            
            # Initialize detector
            detector = AdvancedSlumDetector(
                checkpoint_path=str(best_checkpoint),
                model_config=args.model
            )
            
            # Generate predictions on test set
            test_images_dir = data_config.data_root / "test" / "images"
            if test_images_dir.exists():
                batch_results = detector.predict_batch(
                    image_dir=str(test_images_dir),
                    output_dir=str(advanced_dir),
                    threshold=0.3,
                    save_visualizations=True
                )
                
                print(f"🏘️  Advanced predictions completed:")
                print(f"   - Processed: {batch_results['summary_stats']['total_images']} images")
                print(f"   - Slum detected: {batch_results['summary_stats']['slum_detected']} images")
                print(f"   - Results saved to: {advanced_dir}")
            else:
                print(f"⚠️  Test images directory not found: {test_images_dir}")
                
        except Exception as e:
            print(f"⚠️  Advanced prediction generation failed: {str(e)}")
            print("   You can run it manually later using:")
            print(f"   python advanced_slum_detection/advanced_detector.py --checkpoint {best_checkpoint} --input data/test/images --output {exp_dir}/advanced_predictions --batch")
    else:
        print("⚠️  No best checkpoint found for analysis")


if __name__ == "__main__":
    main()
