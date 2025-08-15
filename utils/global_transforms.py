"""
Global Domain Generalization Transforms (stable)
================================================
Augment on RGB (3ch), then add texture channels to get 6ch (RGB + gradient + entropy + laplacian),
normalize all 6 channels, and convert to tensors.
"""

import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from skimage import filters, morphology


def add_texture_channels(image, **kwargs):
    """Compute 3 texture channels from RGB and append to form 6 channels.
    Returns HxWx6 uint8 or float (Albumentations will handle normalization next).
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected RGB image before adding texture channels")

    # Ensure uint8 for rank filters
    if image.dtype != np.uint8:
        img8 = np.clip(image, 0, 255).astype(np.uint8)
    else:
        img8 = image

    gray = cv2.cvtColor(img8, cv2.COLOR_RGB2GRAY)

    eps = 1e-6
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    grad_mag = (grad_mag / max(float(grad_mag.max()), eps) * 255).astype(np.uint8)

    entropy = filters.rank.entropy(gray, morphology.disk(3))
    entropy = (entropy / max(float(entropy.max()), eps) * 255).astype(np.uint8)

    lap = cv2.Laplacian(gray, cv2.CV_32F)
    lap = (np.abs(lap) / max(float(np.abs(lap).max()), eps) * 255).astype(np.uint8)

    img6 = np.concatenate([img8, grad_mag[..., None], entropy[..., None], lap[..., None]], axis=2)
    return img6


def get_global_train_transforms(image_size=120):
    """Training transforms using only stable Albumentations args.
    Order: CLAHE -> geometric/color -> resize -> add textures -> normalize(6ch) -> tensor
    """
    return A.Compose([
        A.CLAHE(clip_limit=(2, 2), p=1.0),
        A.RandomScale(scale_limit=0.2, p=0.5),
        A.Rotate(limit=30, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=20, val_shift_limit=15, p=0.5),
        A.RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=0.2),
        A.GaussNoise(var_limit=(5.0, 25.0), p=0.2),
        A.MotionBlur(blur_limit=3, p=0.2),
        A.ImageCompression(quality_range=(70, 100), p=0.2),
        A.CoarseDropout(max_holes=6, max_height=16, max_width=16, p=0.2),
        A.Resize(image_size, image_size),
        A.Lambda(image=add_texture_channels),
        A.Normalize(mean=[0.485, 0.456, 0.406, 0.5, 0.5, 0.5],
                    std=[0.229, 0.224, 0.225, 0.25, 0.25, 0.25]),
        ToTensorV2(),
    ])


def get_global_val_transforms(image_size=120):
    """Validation transforms (deterministic)."""
    return A.Compose([
        A.CLAHE(clip_limit=(2, 2), p=1.0),
        A.Resize(image_size, image_size),
        A.Lambda(image=add_texture_channels),
        A.Normalize(mean=[0.485, 0.456, 0.406, 0.5, 0.5, 0.5],
                    std=[0.229, 0.224, 0.225, 0.25, 0.25, 0.25]),
        ToTensorV2(),
    ])