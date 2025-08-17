"""Model factory for various segmentation architectures."""

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


def make_model(
    arch: str = "unet",
    encoder: str = "resnet34", 
    classes: int = 1,
    in_channels: int = 3,
    pretrained: bool = True
) -> nn.Module:
    """Create model with direct parameters - alias for create_model."""
    config = {
        "model": {
            "arch": arch,
            "encoder": encoder,
            "classes": classes,
            "in_channels": in_channels,
            "pretrained": pretrained
        }
    }
    return create_model(config)


def create_model(config: Dict) -> nn.Module:
    """Create segmentation model from config."""
    
    model_config = config["model"]
    arch = model_config["arch"].lower()
    encoder = model_config["encoder"]
    pretrained = model_config.get("pretrained", True)
    in_channels = model_config.get("in_channels", 3)
    classes = model_config.get("classes", 1)
    
    # Model mapping
    model_classes = {
        "unet": smp.Unet,
        "unet++": smp.UnetPlusPlus,
        "unetplusplus": smp.UnetPlusPlus,
        "deeplabv3": smp.DeepLabV3,
        "deeplabv3+": smp.DeepLabV3Plus,
        "deeplabv3plus": smp.DeepLabV3Plus,
        "fpn": smp.FPN,
        "pspnet": smp.PSPNet,
        "linknet": smp.Linknet,
        "manet": smp.MAnet,
        "pan": smp.PAN
    }
    
    if arch not in model_classes:
        raise ValueError(f"Unsupported architecture: {arch}. Available: {list(model_classes.keys())}")
    
    # Create model
    ModelClass = model_classes[arch]
    
    try:
        model = ModelClass(
            encoder_name=encoder,
            encoder_weights="imagenet" if pretrained else None,
            in_channels=in_channels,
            classes=classes,
            activation=None  # We'll apply sigmoid in loss
        )
        
        logger.info(f"Created {arch} model with {encoder} encoder")
        logger.info(f"Channels: {in_channels} -> {classes}, Pretrained: {pretrained}")
        
        return model
        
    except (ValueError, RuntimeError, ImportError) as e:
        logger.error(f"Failed to create model {arch}/{encoder}: {e}")
        # Fallback to basic UNet with ResNet34
        logger.info("Falling back to UNet with ResNet34")
        try:
            model = smp.Unet(
                encoder_name="resnet34",
                encoder_weights="imagenet" if pretrained else None,
                in_channels=in_channels,
                classes=classes,
                activation=None
            )
            return model
        except Exception as fallback_error:
            logger.error(f"Fallback model creation failed: {fallback_error}")
            raise


class ModelWrapper(nn.Module):
    """Wrapper for additional features like gradient checkpointing."""
    
    def __init__(self, model: nn.Module, use_gradient_checkpointing: bool = False):
        super().__init__()
        self.model = model
        self.use_gradient_checkpointing = use_gradient_checkpointing
        
        if use_gradient_checkpointing:
            self._enable_gradient_checkpointing()
    
    def _enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for memory efficiency."""
        
        def apply_gradient_checkpointing(module):
            if hasattr(module, 'gradient_checkpointing') and callable(module.gradient_checkpointing):
                module.gradient_checkpointing = True
            
            # For encoder blocks
            if hasattr(module, 'layer1'):  # ResNet-style
                for layer in [module.layer1, module.layer2, module.layer3, module.layer4]:
                    if hasattr(layer, '__iter__'):
                        for block in layer:
                            if hasattr(block, 'gradient_checkpointing'):
                                block.gradient_checkpointing = True
        
        self.model.apply(apply_gradient_checkpointing)
        logger.info("Enabled gradient checkpointing")
    
    def forward(self, x):
        if self.use_gradient_checkpointing and self.training:
            return torch.utils.checkpoint.checkpoint(self.model, x, use_reentrant=False)
        return self.model(x)


def get_model_summary(model: nn.Module, input_size: tuple = (3, 512, 512)) -> Dict:
    """Get model parameter summary."""
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Estimate model size in MB
    param_size = total_params * 4 / (1024 * 1024)  # Assuming float32
    
    # Test forward pass to get output shape
    model.eval()
    with torch.inference_mode():
        dummy_input = torch.randn(1, *input_size)
        try:
            output = model(dummy_input)
            output_shape = tuple(output.shape)
        except Exception as e:
            logger.warning(f"Could not run forward pass: {e}")
            output_shape = "Unknown"
    
    summary = {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "model_size_mb": param_size,
        "input_shape": input_size,
        "output_shape": output_shape
    }
    
    return summary


def setup_model_for_training(model: nn.Module, config: Dict) -> nn.Module:
    """Setup model with training optimizations."""
    
    train_config = config.get("train", {})
    hardware_config = config.get("hardware", {})
    
    # Gradient checkpointing
    if train_config.get("grad_ckpt", False):
        model = ModelWrapper(model, use_gradient_checkpointing=True)
    
    # Channels last format for better performance
    if train_config.get("channels_last", False):
        model = model.to(memory_format=torch.channels_last)
        logger.info("Using channels_last memory format")
    
    # Torch compile (PyTorch 2.0+)
    compile_mode = train_config.get("compile_mode")
    if compile_mode and hasattr(torch, 'compile'):
        try:
            model = torch.compile(model, mode=compile_mode)
            logger.info(f"Compiled model with mode: {compile_mode}")
        except Exception as e:
            logger.warning(f"Failed to compile model: {e}")
    
    # Hardware optimizations
    if hardware_config.get("cudnn_benchmark", True):
        torch.backends.cudnn.benchmark = True
    
    # Only set TF32 if CUDA is available
    if hardware_config.get("tf32", True) and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    return model


def load_checkpoint(model: nn.Module, checkpoint_path: str, strict: bool = True) -> Dict:
    """Load model checkpoint."""
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # Remove 'model.' prefix if present (from ModelWrapper)
    if any(key.startswith('model.') for key in state_dict.keys()):
        state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
    
    # Load state dict
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=strict)
    
    if missing_keys:
        logger.warning(f"Missing keys: {missing_keys}")
    if unexpected_keys:
        logger.warning(f"Unexpected keys: {unexpected_keys}")
    
    # Sanitize checkpoint path for logging
    safe_path = str(checkpoint_path).replace('\n', '').replace('\r', '')
    logger.info(f"Loaded checkpoint from {safe_path}")
    
    return checkpoint
