"""
Data Configuration for Slum Detection
=====================================

Configuration for data loading, preprocessing, and augmentation
specifically optimized for satellite imagery slum detection.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple, Optional
import json
import os


@dataclass
class DataConfig:
    """
    Configuration class for data processing and augmentation.
    """
    
    # Dataset Paths
    data_root: str = "data"
    train_images_dir: str = "train/images"
    train_masks_dir: str = "train/masks"
    val_images_dir: str = "val/images"
    val_masks_dir: str = "val/masks"
    test_images_dir: str = "test/images"
    test_masks_dir: str = "test/masks"
    
    # Image Properties
    image_size: Tuple[int, int] = (120, 120)  # Height, Width
    input_channels: int = 3  # RGB
    output_channels: int = 1  # Binary mask
    
    # Class Mapping (RGB to Binary)
    slum_rgb: Tuple[int, int, int] = (250, 235, 185)  # Slum class RGB
    background_value: int = 0   # Non-slum pixels
    slum_value: int = 1        # Slum pixels
    
    # Data Loading
    batch_size: int = 16
    num_workers: int = 4
    pin_memory: bool = True
    shuffle_train: bool = True
    drop_last: bool = True
    
    # Data Splits
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    
    # Preprocessing
    normalize: bool = True
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)  # ImageNet stats
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    
    # Data Filtering
    min_slum_pixels: int = 0      # Minimum slum pixels to include image
    max_slum_percentage: float = 1.0  # Maximum slum percentage
    min_slum_percentage: float = 0.0  # Minimum slum percentage
    use_tile_masks_only: bool = True  # Use only tile_* masks (contain slums)
    
    # Augmentation Parameters
    use_augmentation: bool = True
    augmentation_probability: float = 0.8
    
    # Geometric Augmentations
    horizontal_flip: bool = True
    vertical_flip: bool = True
    rotation_limit: int = 45
    shift_limit: float = 0.1
    scale_limit: float = 0.1
    
    # Color Augmentations
    brightness_limit: float = 0.2
    contrast_limit: float = 0.2
    saturation_limit: float = 0.2
    hue_shift_limit: int = 20
    
    # Noise and Blur
    gaussian_noise: bool = True
    gaussian_noise_limit: Tuple[float, float] = (0.0, 0.05)
    gaussian_blur: bool = True
    blur_limit: Tuple[int, int] = (3, 5)
    
    # Advanced Augmentations
    elastic_transform: bool = True
    elastic_alpha: float = 120
    elastic_sigma: float = 6
    grid_distortion: bool = True
    grid_num_steps: int = 5
    grid_distort_limit: float = 0.3
    
    # Cutout/CoarseDropout
    cutout: bool = True
    cutout_holes: int = 8
    cutout_size: Tuple[int, int] = (8, 8)
    
    # Test Time Augmentation
    use_tta: bool = False
    tta_transforms: List[str] = field(default_factory=lambda: [
        "original", "hflip", "vflip", "rotate90"
    ])
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'data_root': self.data_root,
            'train_images_dir': self.train_images_dir,
            'train_masks_dir': self.train_masks_dir,
            'val_images_dir': self.val_images_dir,
            'val_masks_dir': self.val_masks_dir,
            'test_images_dir': self.test_images_dir,
            'test_masks_dir': self.test_masks_dir,
            'image_size': self.image_size,
            'input_channels': self.input_channels,
            'output_channels': self.output_channels,
            'slum_rgb': self.slum_rgb,
            'background_value': self.background_value,
            'slum_value': self.slum_value,
            'batch_size': self.batch_size,
            'num_workers': self.num_workers,
            'pin_memory': self.pin_memory,
            'shuffle_train': self.shuffle_train,
            'drop_last': self.drop_last,
            'train_split': self.train_split,
            'val_split': self.val_split,
            'test_split': self.test_split,
            'normalize': self.normalize,
            'mean': self.mean,
            'std': self.std,
            'min_slum_pixels': self.min_slum_pixels,
            'max_slum_percentage': self.max_slum_percentage,
            'min_slum_percentage': self.min_slum_percentage,
            'use_tile_masks_only': self.use_tile_masks_only,
            'use_augmentation': self.use_augmentation,
            'augmentation_probability': self.augmentation_probability,
            'horizontal_flip': self.horizontal_flip,
            'vertical_flip': self.vertical_flip,
            'rotation_limit': self.rotation_limit,
            'shift_limit': self.shift_limit,
            'scale_limit': self.scale_limit,
            'brightness_limit': self.brightness_limit,
            'contrast_limit': self.contrast_limit,
            'saturation_limit': self.saturation_limit,
            'hue_shift_limit': self.hue_shift_limit,
            'gaussian_noise': self.gaussian_noise,
            'gaussian_noise_limit': self.gaussian_noise_limit,
            'gaussian_blur': self.gaussian_blur,
            'blur_limit': self.blur_limit,
            'elastic_transform': self.elastic_transform,
            'elastic_alpha': self.elastic_alpha,
            'elastic_sigma': self.elastic_sigma,
            'grid_distortion': self.grid_distortion,
            'grid_num_steps': self.grid_num_steps,
            'grid_distort_limit': self.grid_distort_limit,
            'cutout': self.cutout,
            'cutout_holes': self.cutout_holes,
            'cutout_size': self.cutout_size,
            'use_tta': self.use_tta,
            'tta_transforms': self.tta_transforms
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'DataConfig':
        """Create config from dictionary."""
        return cls(**config_dict)
    
    def save(self, filepath: str):
        """Save configuration to JSON file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'DataConfig':
        """Load configuration from JSON file."""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
    
    def get_paths(self) -> Dict[str, str]:
        """Get full paths for all data directories."""
        return {
            'train_images': os.path.join(self.data_root, self.train_images_dir),
            'train_masks': os.path.join(self.data_root, self.train_masks_dir),
            'val_images': os.path.join(self.data_root, self.val_images_dir),
            'val_masks': os.path.join(self.data_root, self.val_masks_dir),
            'test_images': os.path.join(self.data_root, self.test_images_dir),
            'test_masks': os.path.join(self.data_root, self.test_masks_dir)
        }


# Predefined data configurations
PRESET_DATA_CONFIGS = {
    "minimal": DataConfig(
        batch_size=8,
        use_augmentation=False,
        num_workers=2,
        min_slum_pixels=100,  # Only images with some slums
        use_tile_masks_only=True
    ),
    
    "light_augmentation": DataConfig(
        batch_size=16,
        use_augmentation=True,
        augmentation_probability=0.5,
        horizontal_flip=True,
        vertical_flip=True,
        rotation_limit=15,
        brightness_limit=0.1,
        contrast_limit=0.1,
        gaussian_noise=False,
        elastic_transform=False,
        cutout=False,
        use_tile_masks_only=True
    ),
    
    "standard": DataConfig(
        batch_size=16,
        use_augmentation=True,
        augmentation_probability=0.8,
        horizontal_flip=True,
        vertical_flip=True,
        rotation_limit=45,
        brightness_limit=0.2,
        contrast_limit=0.2,
        gaussian_noise=True,
        elastic_transform=True,
        cutout=True,
        use_tile_masks_only=True
    ),
    
    "heavy_augmentation": DataConfig(
        batch_size=16,
        use_augmentation=True,
        augmentation_probability=0.9,
        horizontal_flip=True,
        vertical_flip=True,
        rotation_limit=90,
        shift_limit=0.2,
        scale_limit=0.2,
        brightness_limit=0.3,
        contrast_limit=0.3,
        saturation_limit=0.3,
        hue_shift_limit=30,
        gaussian_noise=True,
        gaussian_blur=True,
        elastic_transform=True,
        grid_distortion=True,
        cutout=True,
        cutout_holes=12,
        use_tile_masks_only=True
    ),
    
    "production": DataConfig(
        batch_size=32,
        use_augmentation=True,
        augmentation_probability=0.8,
        num_workers=8,
        use_tta=True,
        tta_transforms=["original", "hflip", "vflip", "rotate90", "rotate180"],
        use_tile_masks_only=True
    ),
    
    "all_data": DataConfig(
        batch_size=16,
        use_augmentation=True,
        augmentation_probability=0.8,
        use_tile_masks_only=False,  # Include all masks
        min_slum_pixels=0
    )
}


def get_data_config(preset: str = "standard") -> DataConfig:
    """
    Get a predefined data configuration.
    
    Args:
        preset: Configuration preset name
    
    Returns:
        DataConfig instance
    """
    if preset not in PRESET_DATA_CONFIGS:
        available = list(PRESET_DATA_CONFIGS.keys())
        raise ValueError(f"Unknown preset '{preset}'. Available: {available}")
    
    return PRESET_DATA_CONFIGS[preset]


def list_data_presets() -> List[str]:
    """List all available data configuration presets."""
    return list(PRESET_DATA_CONFIGS.keys())


def get_augmentation_info() -> Dict[str, Dict[str, Any]]:
    """Get information about available augmentation techniques."""
    return {
        "geometric": {
            "transforms": ["horizontal_flip", "vertical_flip", "rotation", "shift", "scale"],
            "description": "Spatial transformations preserving semantic content"
        },
        "color": {
            "transforms": ["brightness", "contrast", "saturation", "hue"],
            "description": "Color space manipulations"
        },
        "noise": {
            "transforms": ["gaussian_noise", "gaussian_blur"],
            "description": "Noise and blur for robustness"
        },
        "advanced": {
            "transforms": ["elastic_transform", "grid_distortion", "cutout"],
            "description": "Advanced augmentations for better generalization"
        }
    }


def print_data_config_info():
    """Print information about available data configurations."""
    print("📊 DATA CONFIGURATION OPTIONS")
    print("=" * 50)
    
    print("\n📋 Available Presets:")
    for name, config in PRESET_DATA_CONFIGS.items():
        print(f"  {name.upper()}:")
        print(f"    Batch Size: {config.batch_size}")
        print(f"    Augmentation: {config.use_augmentation}")
        if config.use_augmentation:
            print(f"    Aug Probability: {config.augmentation_probability}")
        print(f"    Tile Masks Only: {config.use_tile_masks_only}")
        print(f"    Workers: {config.num_workers}")
        print()
    
    print("🔄 Augmentation Categories:")
    aug_info = get_augmentation_info()
    for category, info in aug_info.items():
        print(f"  {category.upper()}: {info['description']}")
        print(f"    Transforms: {', '.join(info['transforms'])}")
    print()
    
    print("🎯 Class Mapping:")
    config = get_data_config("standard")
    print(f"  Slum RGB: {config.slum_rgb}")
    print(f"  Slum Value: {config.slum_value}")
    print(f"  Background Value: {config.background_value}")


if __name__ == "__main__":
    # Test configuration
    print("Testing Data Configuration...")
    
    # Create and test config
    config = get_data_config("standard")
    print(f"Loaded config with batch size: {config.batch_size}")
    print(f"Augmentation enabled: {config.use_augmentation}")
    
    # Test paths
    paths = config.get_paths()
    print(f"Train images path: {paths['train_images']}")
    
    # Test save/load
    config.save("test_data_config.json")
    loaded_config = DataConfig.load("test_data_config.json")
    print(f"Saved and loaded successfully: {loaded_config.image_size}")
    
    # Clean up
    if os.path.exists("test_data_config.json"):
        os.remove("test_data_config.json")
    
    print("\n")
    print_data_config_info()
