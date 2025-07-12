"""
Data Transforms for Slum Detection
==================================

Comprehensive data augmentation and preprocessing transforms
optimized for satellite imagery slum detection.
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2


def get_train_transforms(config):
    """
    Get training transforms with data augmentation.
    
    Args:
        config: DataConfig instance
    
    Returns:
        Albumentations Compose transform
    """
    transforms = []
    
    if config.use_augmentation:
        aug_transforms = []
        
        # Geometric augmentations
        if config.horizontal_flip:
            aug_transforms.append(A.HorizontalFlip(p=0.5))
        
        if config.vertical_flip:
            aug_transforms.append(A.VerticalFlip(p=0.5))
        
        if config.rotation_limit > 0:
            aug_transforms.append(A.Rotate(
                limit=config.rotation_limit, 
                interpolation=cv2.INTER_LINEAR,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                mask_value=0,
                p=0.5
            ))
        
        if config.shift_limit > 0 or config.scale_limit > 0:
            aug_transforms.append(A.ShiftScaleRotate(
                shift_limit=config.shift_limit,
                scale_limit=config.scale_limit,
                rotate_limit=0,  # Already handled above
                interpolation=cv2.INTER_LINEAR,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                mask_value=0,
                p=0.5
            ))
        
        # Color augmentations
        color_transforms = []
        if config.brightness_limit > 0:
            color_transforms.append(A.RandomBrightnessContrast(
                brightness_limit=config.brightness_limit,
                contrast_limit=config.contrast_limit,
                p=0.5
            ))
        
        if config.saturation_limit > 0 or config.hue_shift_limit > 0:
            color_transforms.append(A.HueSaturationValue(
                hue_shift_limit=config.hue_shift_limit,
                sat_shift_limit=int(config.saturation_limit * 100),
                val_shift_limit=0,
                p=0.5
            ))
        
        if color_transforms:
            aug_transforms.extend(color_transforms)
        
        # Noise and blur
        if config.gaussian_noise:
            aug_transforms.append(A.GaussNoise(
                var_limit=config.gaussian_noise_limit,
                p=0.3
            ))
        
        if config.gaussian_blur:
            aug_transforms.append(A.GaussianBlur(
                blur_limit=config.blur_limit,
                p=0.3
            ))
        
        # Advanced augmentations
        if config.elastic_transform:
            aug_transforms.append(A.ElasticTransform(
                alpha=config.elastic_alpha,
                sigma=config.elastic_sigma,
                alpha_affine=0,
                interpolation=cv2.INTER_LINEAR,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                mask_value=0,
                p=0.3
            ))
        
        if config.grid_distortion:
            aug_transforms.append(A.GridDistortion(
                num_steps=config.grid_num_steps,
                distort_limit=config.grid_distort_limit,
                interpolation=cv2.INTER_LINEAR,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                mask_value=0,
                p=0.3
            ))
        
        if config.cutout:
            aug_transforms.append(A.CoarseDropout(
                max_holes=config.cutout_holes,
                max_height=config.cutout_size[0],
                max_width=config.cutout_size[1],
                min_holes=1,
                min_height=4,
                min_width=4,
                fill_value=0,
                mask_fill_value=0,
                p=0.3
            ))
        
        # Apply augmentations with probability
        if aug_transforms:
            transforms.append(A.OneOf(aug_transforms, p=config.augmentation_probability))
    
    # Resize to target size
    transforms.append(A.Resize(
        height=config.image_size[0],
        width=config.image_size[1],
        interpolation=cv2.INTER_LINEAR
    ))
    
    # Normalization
    if config.normalize:
        transforms.append(A.Normalize(
            mean=config.mean,
            std=config.std,
            max_pixel_value=255.0
        ))
    else:
        transforms.append(A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0
        ))
    
    # Convert to tensor
    transforms.append(ToTensorV2())
    
    return A.Compose(transforms)


def get_val_transforms(config):
    """
    Get validation transforms (no augmentation).
    
    Args:
        config: DataConfig instance
    
    Returns:
        Albumentations Compose transform
    """
    transforms = []
    
    # Resize to target size
    transforms.append(A.Resize(
        height=config.image_size[0],
        width=config.image_size[1],
        interpolation=cv2.INTER_LINEAR
    ))
    
    # Normalization
    if config.normalize:
        transforms.append(A.Normalize(
            mean=config.mean,
            std=config.std,
            max_pixel_value=255.0
        ))
    else:
        transforms.append(A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0
        ))
    
    # Convert to tensor
    transforms.append(ToTensorV2())
    
    return A.Compose(transforms)


def get_test_transforms(config):
    """
    Get test transforms (same as validation).
    
    Args:
        config: DataConfig instance
    
    Returns:
        Albumentations Compose transform
    """
    return get_val_transforms(config)


def get_tta_transforms(config):
    """
    Get Test Time Augmentation transforms.
    
    Args:
        config: DataConfig instance
    
    Returns:
        List of Albumentations Compose transforms
    """
    base_transforms = [
        A.Resize(
            height=config.image_size[0],
            width=config.image_size[1],
            interpolation=cv2.INTER_LINEAR
        )
    ]
    
    if config.normalize:
        base_transforms.append(A.Normalize(
            mean=config.mean,
            std=config.std,
            max_pixel_value=255.0
        ))
    else:
        base_transforms.append(A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0
        ))
    
    base_transforms.append(ToTensorV2())
    
    tta_transforms = []
    
    for tta_name in config.tta_transforms:
        if tta_name == "original":
            transforms = base_transforms.copy()
        elif tta_name == "hflip":
            transforms = [A.HorizontalFlip(p=1.0)] + base_transforms
        elif tta_name == "vflip":
            transforms = [A.VerticalFlip(p=1.0)] + base_transforms
        elif tta_name == "rotate90":
            transforms = [A.Rotate(limit=90, p=1.0)] + base_transforms
        elif tta_name == "rotate180":
            transforms = [A.Rotate(limit=180, p=1.0)] + base_transforms
        elif tta_name == "rotate270":
            transforms = [A.Rotate(limit=270, p=1.0)] + base_transforms
        else:
            continue  # Skip unknown transforms
        
        tta_transforms.append(A.Compose(transforms))
    
    return tta_transforms


def create_custom_transform(
    image_size=(120, 120),
    normalize=True,
    augment=False,
    **aug_params
):
    """
    Create a custom transform with specified parameters.
    
    Args:
        image_size: Target image size (H, W)
        normalize: Whether to normalize images
        augment: Whether to apply augmentations
        **aug_params: Additional augmentation parameters
    
    Returns:
        Albumentations Compose transform
    """
    transforms = []
    
    if augment:
        # Basic augmentations
        transforms.extend([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=45, p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.GaussNoise(p=0.3)
        ])
        
        # Add custom augmentations
        if 'elastic' in aug_params and aug_params['elastic']:
            transforms.append(A.ElasticTransform(p=0.3))
        
        if 'cutout' in aug_params and aug_params['cutout']:
            transforms.append(A.CoarseDropout(
                max_holes=8, max_height=8, max_width=8, p=0.3
            ))
    
    # Resize
    transforms.append(A.Resize(
        height=image_size[0],
        width=image_size[1],
        interpolation=cv2.INTER_LINEAR
    ))
    
    # Normalize
    if normalize:
        transforms.append(A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0
        ))
    
    # To tensor
    transforms.append(ToTensorV2())
    
    return A.Compose(transforms)


def print_transforms_info():
    """Print information about available transforms."""
    print("🔄 DATA TRANSFORMS INFO")
    print("=" * 30)
    
    print("\n📋 Transform Categories:")
    print("  GEOMETRIC: HorizontalFlip, VerticalFlip, Rotate, ShiftScaleRotate")
    print("  COLOR: RandomBrightnessContrast, HueSaturationValue")
    print("  NOISE: GaussNoise, GaussianBlur")
    print("  ADVANCED: ElasticTransform, GridDistortion, CoarseDropout")
    print("  PREPROCESSING: Resize, Normalize, ToTensorV2")
    
    print("\n🎯 Usage:")
    print("  TRAINING: Heavy augmentation for robustness")
    print("  VALIDATION: Only preprocessing (resize, normalize)")
    print("  TESTING: Same as validation + optional TTA")
    print("  TTA: Multiple transform variants for ensemble prediction")


if __name__ == "__main__":
    # Test transforms
    from config.data_config import get_data_config
    import numpy as np
    
    print("Testing transforms...")
    
    config = get_data_config("standard")
    
    # Create test data
    test_image = np.random.randint(0, 255, (120, 120, 3), dtype=np.uint8)
    test_mask = np.random.randint(0, 2, (120, 120), dtype=np.uint8)
    
    # Test training transforms
    train_transform = get_train_transforms(config)
    result = train_transform(image=test_image, mask=test_mask)
    print(f"Train transform result: image {result['image'].shape}, mask {result['mask'].shape}")
    
    # Test validation transforms
    val_transform = get_val_transforms(config)
    result = val_transform(image=test_image, mask=test_mask)
    print(f"Val transform result: image {result['image'].shape}, mask {result['mask'].shape}")
    
    # Test TTA transforms
    if config.use_tta:
        tta_transforms = get_tta_transforms(config)
        print(f"TTA transforms: {len(tta_transforms)} variants")
    
    print("\n")
    print_transforms_info()
