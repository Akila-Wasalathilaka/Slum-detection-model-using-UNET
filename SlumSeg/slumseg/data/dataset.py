"""Dataset utilities for slum segmentation."""

import os
import cv2
import numpy as np
import pandas as pd
import rasterio
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from sklearn.model_selection import GroupKFold, train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
from torch.utils.data import Dataset, DataLoader
import logging

logger = logging.getLogger(__name__)


class SlumDataset(Dataset):
    """Dataset class for slum segmentation with caching and transforms."""
    
    def __init__(
        self,
        image_paths: List[str],
        mask_paths: List[str],
        transform: Optional[A.Compose] = None,
        cache_dir: Optional[str] = None,
    ):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.cache_dir = Path(cache_dir) if cache_dir else None
        
        if len(image_paths) != len(mask_paths):
            raise ValueError(f"Mismatch: {len(image_paths)} images vs {len(mask_paths)} masks")
        
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Load image and mask
        image = self._load_image(self.image_paths[idx])
        mask = self._load_mask(self.mask_paths[idx])
        
        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]
        
        # Ensure proper types
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).float()
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask).float()
            
        # Normalize image to [0, 1] if needed
        if image.max() > 1:
            image = image / 255.0
            
        return {
            "image": image,
            "mask": mask.unsqueeze(0) if mask.dim() == 2 else mask,
            "image_path": self.image_paths[idx],
            "mask_path": self.mask_paths[idx]
        }
    
    def _load_image(self, path: str) -> np.ndarray:
        """Load image with caching support."""
        if self.cache_dir:
            cache_path = self.cache_dir / f"{Path(path).stem}_img.npy"
            if cache_path.exists():
                return np.load(cache_path)
        
        # Load with rasterio (handles TIFF/GeoTIFF)
        try:
            with rasterio.open(path) as src:
                image = src.read()  # (C, H, W)
                if image.shape[0] > 3:  # More than RGB
                    image = image[:3]  # Take first 3 channels
                image = np.transpose(image, (1, 2, 0))  # (H, W, C)
        except Exception as e:
            # Fallback to OpenCV
            # Sanitize path for logging
            safe_path = str(path).replace('\n', '').replace('\r', '')
            logger.warning(f"Rasterio failed for {safe_path}, using OpenCV: {e}")
            image = cv2.imread(path)
            if image is None:
                raise ValueError(f"Could not load image: {path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Cache if enabled
        if self.cache_dir:
            # Sanitize filename to prevent path traversal
            safe_stem = os.path.basename(Path(path).stem).replace('..', '')
            cache_path = self.cache_dir / f"{safe_stem}_img.npy"
            np.save(cache_path, image)
            
        return image.astype(np.uint8)
    
    def _load_mask(self, path: str) -> np.ndarray:
        """Load mask with caching support."""
        if self.cache_dir:
            cache_path = self.cache_dir / f"{Path(path).stem}_mask.npy"
            if cache_path.exists():
                return np.load(cache_path)
        
        # Load mask
        try:
            with rasterio.open(path) as src:
                mask = src.read(1)  # Single channel
        except Exception as e:
            logger.warning(f"Rasterio failed for mask {path}, using OpenCV: {e}")
            mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise ValueError(f"Could not load mask: {path}")
        
        # Binarize (assume slum=1, background=0)
        mask = (mask > 0).astype(np.uint8)
        
        # Cache if enabled
        if self.cache_dir:
            # Sanitize filename to prevent path traversal
            safe_stem = os.path.basename(Path(path).stem).replace('..', '')
            cache_path = self.cache_dir / f"{safe_stem}_mask.npy"
            np.save(cache_path, mask)
            
        return mask


def get_transforms(config: Dict, is_train: bool = True) -> A.Compose:
    """Get albumentations transforms."""
    
    if is_train and "train" in config.get("augment", {}):
        aug_config = config["augment"]["train"]
        transforms = [
            A.HorizontalFlip(p=aug_config.get("hflip", 0.5)),
            A.VerticalFlip(p=aug_config.get("vflip", 0.2)),
            A.Rotate(limit=aug_config.get("rotate_deg", 10), p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=aug_config.get("brightness", 0.15),
                contrast_limit=aug_config.get("contrast", 0.15),
                p=0.5
            ),
            A.Blur(blur_limit=3, p=aug_config.get("blur", 0.1)),
            A.CoarseDropout(
                max_holes=aug_config.get("cutout_max_holes", 8),
                max_height=aug_config.get("cutout_max_h_size", 32),
                max_width=aug_config.get("cutout_max_w_size", 32),
                p=aug_config.get("cutout_prob", 0.2)
            ),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]
    else:
        # Validation transforms
        transforms = [
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]
    
    return A.Compose(transforms)


def compute_class_weights(mask_paths: List[str], num_samples: int = 1000) -> float:
    """Compute positive class weight for imbalanced data."""
    logger.info(f"Computing class weights from {len(mask_paths)} masks...")
    
    # Sample subset for efficiency
    if len(mask_paths) > num_samples:
        sample_paths = np.random.choice(mask_paths, num_samples, replace=False)
    else:
        sample_paths = mask_paths
    
    total_pixels = 0
    positive_pixels = 0
    
    for mask_path in sample_paths:
        try:
            with rasterio.open(mask_path) as src:
                mask = src.read(1)
        except:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if mask is not None:
            mask_binary = (mask > 0).astype(np.uint8)
            total_pixels += mask_binary.size
            positive_pixels += mask_binary.sum()
    
    if positive_pixels == 0:
        logger.warning("No positive pixels found! Using default weight.")
        return 2.0
    
    pos_weight = (total_pixels - positive_pixels) / positive_pixels
    logger.info(f"Computed positive class weight: {pos_weight:.3f}")
    
    return float(pos_weight)


def create_data_splits(
    data_root: str,
    val_ratio: float = 0.15,
    regions_file: Optional[str] = None,
    seed: int = 42
) -> Tuple[List[str], List[str], List[str], List[str]]:
    """Create train/val splits with optional region-based grouping."""
    
    # Find all image/mask pairs
    train_img_dir = Path(data_root) / "train" / "images"
    train_mask_dir = Path(data_root) / "train" / "masks"
    val_img_dir = Path(data_root) / "val" / "images"
    val_mask_dir = Path(data_root) / "val" / "masks"
    
    # If val folder exists, use it
    if val_img_dir.exists() and val_mask_dir.exists():
        logger.info("Using existing train/val split from folders")
        
        train_images = sorted(list(train_img_dir.glob("*.tif")) + list(train_img_dir.glob("*.png")))
        train_masks = sorted(list(train_mask_dir.glob("*.tif")) + list(train_mask_dir.glob("*.png")))
        val_images = sorted(list(val_img_dir.glob("*.tif")) + list(val_img_dir.glob("*.png")))
        val_masks = sorted(list(val_mask_dir.glob("*.tif")) + list(val_mask_dir.glob("*.png")))
        
        return (
            [str(p) for p in train_images],
            [str(p) for p in train_masks],
            [str(p) for p in val_images],
            [str(p) for p in val_masks]
        )
    
    # Otherwise create split from train folder
    logger.info("Creating train/val split from train folder")
    
    train_images = sorted(list(train_img_dir.glob("*.tif")) + list(train_img_dir.glob("*.png")))
    train_masks = sorted(list(train_mask_dir.glob("*.tif")) + list(train_mask_dir.glob("*.png")))
    
    if len(train_images) != len(train_masks):
        raise ValueError(f"Mismatch: {len(train_images)} images vs {len(train_masks)} masks")
    
    # Extract region info for grouped split
    if regions_file and Path(regions_file).exists():
        logger.info(f"Using region-based split from {regions_file}")
        regions_df = pd.read_csv(regions_file)
        groups = []
        for img_path in train_images:
            img_name = Path(img_path).stem
            region = regions_df[regions_df['image'] == img_name]['region'].iloc[0]
            groups.append(region)
        
        # Group K-Fold split
        gkf = GroupKFold(n_splits=int(1 / val_ratio))
        train_idx, val_idx = next(gkf.split(train_images, groups=groups))
        
    else:
        # Simple random split
        logger.info("Using random train/val split")
        train_idx, val_idx = train_test_split(
            range(len(train_images)), 
            test_size=val_ratio, 
            random_state=seed
        )
    
    # Split the data
    train_imgs = [str(train_images[i]) for i in train_idx]
    train_msks = [str(train_masks[i]) for i in train_idx]
    val_imgs = [str(train_images[i]) for i in val_idx]
    val_msks = [str(train_masks[i]) for i in val_idx]
    
    logger.info(f"Split: {len(train_imgs)} train, {len(val_imgs)} val")
    
    return train_imgs, train_msks, val_imgs, val_msks


def get_image_ids_from_dir(images_dir: str) -> List[str]:
    """Get image IDs (without extension) from directory."""
    images_path = Path(images_dir)
    if not images_path.exists():
        return []
    
    image_files = list(images_path.glob("*.tif")) + list(images_path.glob("*.png")) + list(images_path.glob("*.jpg"))
    return [f.stem for f in image_files]


def get_valid_transforms() -> A.Compose:
    """Get validation transforms."""
    return A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def calculate_class_weights(mask_paths: List[str]) -> float:
    """Calculate class weights - alias for compute_class_weights."""
    return compute_class_weights(mask_paths)


def create_dataloader(
    image_paths: List[str],
    mask_paths: List[str],
    config: Dict,
    is_train: bool = True,
    cache_dir: Optional[str] = None
) -> DataLoader:
    """Create a DataLoader with proper settings."""
    
    dataset = SlumDataset(
        image_paths=image_paths,
        mask_paths=mask_paths,
        transform=get_transforms(config, is_train),
        cache_dir=cache_dir
    )
    
    train_config = config.get("train", {})
    
    return DataLoader(
        dataset,
        batch_size=train_config.get("batch_size", 16),
        shuffle=is_train,
        num_workers=train_config.get("num_workers", 4),
        pin_memory=train_config.get("pin_memory", True),
        persistent_workers=train_config.get("persistent_workers", True),
        prefetch_factor=train_config.get("prefetch_factor", 2),
        drop_last=is_train
    )
