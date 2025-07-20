"""
Checkpoint Management for Slum Detection Models
===============================================

Comprehensive checkpoint management system for saving, loading,
and managing model states during training and inference.
"""

import os
import torch
import json
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import glob


class CheckpointManager:
    """
    Advanced checkpoint manager for model training.
    
    Features:
    - Automatic best model tracking
    - Configurable checkpoint retention
    - Metadata storage
    - Resume training support
    """
    
    def __init__(
        self,
        checkpoint_dir: str,
        max_checkpoints: int = 5,
        save_optimizer: bool = True,
        save_scheduler: bool = True
    ):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory to save checkpoints
            max_checkpoints: Maximum number of checkpoints to keep
            save_optimizer: Whether to save optimizer state
            save_scheduler: Whether to save scheduler state
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_checkpoints = max_checkpoints
        self.save_optimizer = save_optimizer
        self.save_scheduler = save_scheduler
        
        # Track saved checkpoints
        self.checkpoint_history = []
        self.best_checkpoint = None
        self.best_metric = None
        
        print(f"📁 Checkpoint manager initialized: {self.checkpoint_dir}")
    
    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        epoch: int,
        metrics: Dict[str, float],
        is_best: bool = False,
        extra_data: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Save a checkpoint with model state and metadata.
        
        Args:
            model: PyTorch model to save
            optimizer: Optimizer state
            scheduler: Learning rate scheduler state
            epoch: Current epoch number
            metrics: Training metrics
            is_best: Whether this is the best checkpoint so far
            extra_data: Additional data to save
        
        Returns:
            Path to saved checkpoint
        """
        # Create checkpoint data
        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'metrics': metrics,
            'timestamp': datetime.now().isoformat(),
            'pytorch_version': torch.__version__
        }
        
        # Add optimizer state
        if self.save_optimizer and optimizer is not None:
            checkpoint_data['optimizer_state_dict'] = optimizer.state_dict()
        
        # Add scheduler state
        if self.save_scheduler and scheduler is not None:
            checkpoint_data['scheduler_state_dict'] = scheduler.state_dict()
        
        # Add extra data
        if extra_data:
            checkpoint_data.update(extra_data)
        
        # Generate checkpoint filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_name = f"checkpoint_epoch_{epoch:03d}_{timestamp}.pth"
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        
        # Save checkpoint
        torch.save(checkpoint_data, checkpoint_path)
        
        # Update history
        checkpoint_info = {
            'path': str(checkpoint_path),
            'epoch': epoch,
            'metrics': metrics,
            'timestamp': checkpoint_data['timestamp'],
            'is_best': is_best
        }
        self.checkpoint_history.append(checkpoint_info)
        
        # Handle best checkpoint
        if is_best:
            self.best_checkpoint = checkpoint_path
            self.best_metric = metrics
            
            # Save best checkpoint copy
            best_path = self.checkpoint_dir / "best_checkpoint.pth"
            shutil.copy2(checkpoint_path, best_path)
            
            print(f"🏆 Best checkpoint saved: {checkpoint_name}")
        else:
            print(f"💾 Checkpoint saved: {checkpoint_name}")
        
        # Save metadata
        self._save_metadata()
        
        # Clean up old checkpoints
        self._cleanup_checkpoints()
        
        return str(checkpoint_path)
    
    def load_checkpoint(
        self,
        checkpoint_path: str,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: str = 'cuda'
    ) -> Dict[str, Any]:
        """
        Load a checkpoint and restore model state.
        
        Args:
            checkpoint_path: Path to checkpoint file
            model: Model to load state into
            optimizer: Optimizer to load state into
            scheduler: Scheduler to load state into
            device: Device to load checkpoint on
        
        Returns:
            Dictionary with loaded metadata
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        print(f"📂 Loading checkpoint: {checkpoint_path}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state
        if scheduler and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        print(f"✅ Checkpoint loaded successfully")
        print(f"   Epoch: {checkpoint.get('epoch', 'unknown')}")
        print(f"   Metrics: {checkpoint.get('metrics', {})}")
        
        return checkpoint
    
    def load_best_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: str = 'cuda'
    ) -> Optional[Dict[str, Any]]:
        """Load the best checkpoint."""
        best_path = self.checkpoint_dir / "best_checkpoint.pth"
        
        if best_path.exists():
            return self.load_checkpoint(str(best_path), model, optimizer, scheduler, device)
        else:
            print("⚠️  No best checkpoint found")
            return None
    
    def get_latest_checkpoint(self) -> Optional[str]:
        """Get path to the most recent checkpoint."""
        checkpoints = glob.glob(str(self.checkpoint_dir / "checkpoint_*.pth"))
        if checkpoints:
            # Sort by modification time
            latest = max(checkpoints, key=os.path.getmtime)
            return latest
        return None
    
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """List all available checkpoints with metadata."""
        return self.checkpoint_history.copy()
    
    def get_checkpoint_info(self, checkpoint_path: str) -> Dict[str, Any]:
        """Get information about a specific checkpoint."""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        return {
            'epoch': checkpoint.get('epoch'),
            'metrics': checkpoint.get('metrics', {}),
            'timestamp': checkpoint.get('timestamp'),
            'pytorch_version': checkpoint.get('pytorch_version'),
            'file_size': os.path.getsize(checkpoint_path),
            'file_path': checkpoint_path
        }
    
    def _save_metadata(self):
        """Save checkpoint metadata to JSON file."""
        metadata = {
            'checkpoint_history': self.checkpoint_history,
            'best_checkpoint': str(self.best_checkpoint) if self.best_checkpoint else None,
            'best_metric': self.best_metric,
            'total_checkpoints': len(self.checkpoint_history)
        }
        
        metadata_path = self.checkpoint_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def _cleanup_checkpoints(self):
        """Remove old checkpoints to maintain max_checkpoints limit."""
        if len(self.checkpoint_history) <= self.max_checkpoints:
            return
        
        # Sort by epoch (keep most recent)
        sorted_checkpoints = sorted(self.checkpoint_history, key=lambda x: x['epoch'])
        
        # Remove oldest checkpoints
        to_remove = sorted_checkpoints[:-self.max_checkpoints]
        
        for checkpoint_info in to_remove:
            checkpoint_path = Path(checkpoint_info['path'])
            if checkpoint_path.exists() and not checkpoint_info.get('is_best', False):
                checkpoint_path.unlink()
                print(f"🗑️  Removed old checkpoint: {checkpoint_path.name}")
        
        # Update history
        self.checkpoint_history = sorted_checkpoints[-self.max_checkpoints:]
    
    def export_model(
        self,
        model: torch.nn.Module,
        export_path: str,
        format: str = 'torch'
    ):
        """
        Export model for deployment.
        
        Args:
            model: Trained model
            export_path: Path to save exported model
            format: Export format ('torch', 'onnx', 'torchscript')
        """
        export_path = Path(export_path)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        
        model.eval()
        
        if format.lower() == 'torch':
            # Save just model state dict
            torch.save(model.state_dict(), export_path)
            print(f"🚀 Model exported (PyTorch): {export_path}")
        
        elif format.lower() == 'torchscript':
            # Export as TorchScript
            dummy_input = torch.randn(1, 3, 120, 120)
            scripted_model = torch.jit.trace(model, dummy_input)
            scripted_model.save(str(export_path))
            print(f"🚀 Model exported (TorchScript): {export_path}")
        
        elif format.lower() == 'onnx':
            # Export as ONNX
            try:
                import onnx
                dummy_input = torch.randn(1, 3, 120, 120)
                torch.onnx.export(
                    model, dummy_input, str(export_path),
                    export_params=True,
                    opset_version=11,
                    input_names=['image'],
                    output_names=['mask'],
                    dynamic_axes={
                        'image': {0: 'batch_size'},
                        'mask': {0: 'batch_size'}
                    }
                )
                print(f"🚀 Model exported (ONNX): {export_path}")
            except ImportError:
                print("❌ ONNX export requires 'onnx' package")
        
        else:
            raise ValueError(f"Unknown export format: {format}")


def save_checkpoint(
    filepath: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: int = 0,
    metrics: Optional[Dict[str, float]] = None,
    **kwargs
):
    """
    Simple checkpoint saving function.
    
    Args:
        filepath: Path to save checkpoint
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch
        metrics: Training metrics
        **kwargs: Additional data to save
    """
    checkpoint_data = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'timestamp': datetime.now().isoformat(),
        'pytorch_version': torch.__version__
    }
    
    if optimizer:
        checkpoint_data['optimizer_state_dict'] = optimizer.state_dict()
    
    if metrics:
        checkpoint_data['metrics'] = metrics
    
    checkpoint_data.update(kwargs)
    
    # Create directory if needed
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    torch.save(checkpoint_data, filepath)
    print(f"💾 Checkpoint saved: {filepath}")


def load_checkpoint(
    filepath: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = 'cuda'
) -> Dict[str, Any]:
    """
    Simple checkpoint loading function.
    
    Args:
        filepath: Path to checkpoint
        model: Model to load into
        optimizer: Optimizer to load into
        device: Device to load on
    
    Returns:
        Checkpoint data dictionary
    """
    # Handle device mapping properly
    device_str = str(device) if hasattr(device, 'type') else device
    if device_str == 'cpu' or 'cpu' in device_str:
        map_location = 'cpu'
    elif 'cuda' in device_str:
        map_location = device_str
    else:
        map_location = None
    
    try:
        checkpoint = torch.load(filepath, map_location=map_location, weights_only=False)
    except RuntimeError as e:
        if "tagged with auto" in str(e):
            # Fallback to CPU loading
            print("⚠️  Loading checkpoint on CPU due to device mismatch")
            checkpoint = torch.load(filepath, map_location='cpu', weights_only=False)
        else:
            raise e
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"📂 Checkpoint loaded: {filepath}")
    return checkpoint


if __name__ == "__main__":
    # Test checkpoint manager
    print("Testing checkpoint manager...")
    
    # Create dummy model
    import torch.nn as nn
    
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 1, 3, 1, 1)
        
        def forward(self, x):
            return self.conv(x)
    
    model = DummyModel()
    optimizer = torch.optim.Adam(model.parameters())
    
    # Test checkpoint manager
    manager = CheckpointManager("test_checkpoints", max_checkpoints=3)
    
    # Save test checkpoints
    for epoch in range(1, 6):
        metrics = {
            'loss': 1.0 / epoch,
            'iou': epoch * 0.1,
            'dice': epoch * 0.1
        }
        
        is_best = epoch == 3  # Epoch 3 is best
        
        manager.save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=None,
            epoch=epoch,
            metrics=metrics,
            is_best=is_best
        )
    
    # Test loading
    best_checkpoint = manager.load_best_checkpoint(model, optimizer)
    print(f"Best checkpoint epoch: {best_checkpoint['epoch']}")
    
    # List checkpoints
    checkpoints = manager.list_checkpoints()
    print(f"Total checkpoints: {len(checkpoints)}")
    
    # Test export
    manager.export_model(model, "test_model.pth", format='torch')
    
    print("Checkpoint manager tests completed!")
    
    # Cleanup
    import shutil
    shutil.rmtree("test_checkpoints", ignore_errors=True)
    if os.path.exists("test_model.pth"):
        os.remove("test_model.pth")
