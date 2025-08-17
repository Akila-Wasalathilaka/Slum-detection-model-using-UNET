"""
Advanced Dataset Implementation for Slum Detection
==================================================

High-performance dataset class with intelligent data loading, class mapping,
and built-in filtering for optimal slum detection training.
"""

import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import cv2
from typing import Dict, List, Tuple, Optional, Callable
import albumentations as A
from albumentations.pytorch import ToTensorV2
import glob
from pathlib import Path


class SlumDataset(Dataset):
    """
    Advanced dataset class for slum detection from satellite imagery.
    
    Features:
    - RGB to binary mask conversion
    - Intelligent data filtering
    - Built-in augmentation support
    - Efficient caching
    - Class balancing options
    """
    
    def __init__(
        self,
        images_dir: str,
        masks_dir: str,
        transform: Optional[Callable] = None,
        slum_rgb: Tuple[int, int, int] = (250, 235, 185),
        image_size: Tuple[int, int] = (120, 120),
        use_tile_masks_only: bool = True,
        min_slum_pixels: int = 0,
        max_slum_percentage: float = 1.0,
        min_slum_percentage: float = 0.0,
        cache_masks: bool = True
    ):
        """
        Initialize slum detection dataset.
        
        Args:
            images_dir: Directory containing RGB satellite images
            masks_dir: Directory containing RGB masks
            transform: Albumentations transform pipeline
            slum_rgb: RGB value representing slum class
            image_size: Target image size (H, W)
            use_tile_masks_only: Only use tile_* masks (contain slums)
            min_slum_pixels: Minimum slum pixels to include image
            max_slum_percentage: Maximum slum percentage to include
            min_slum_percentage: Minimum slum percentage to include
            cache_masks: Cache processed binary masks in memory
        """
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.transform = transform
        self.slum_rgb = slum_rgb
        self.image_size = image_size
        self.use_tile_masks_only = use_tile_masks_only
        self.min_slum_pixels = min_slum_pixels
        self.max_slum_percentage = max_slum_percentage
        self.min_slum_percentage = min_slum_percentage
        self.cache_masks = cache_masks
        
        # Initialize data structures
        self.image_paths = []
        self.mask_paths = []
        self.cached_masks = {} if cache_masks else None
        self.slum_stats = {}  # Store slum statistics for each image
        
        # Load and filter dataset
        self._load_dataset()
        self._filter_dataset()
        
        print(f"Dataset initialized with {len(self.image_paths)} samples")
        if len(self.image_paths) > 0:
            self._print_dataset_stats()
    
    def _load_dataset(self):
        """Load all valid image-mask pairs."""
        print("Loading dataset...")
        
        # Get all image files
        image_extensions = ['*.tif', '*.png', '*.jpg', '*.jpeg']
        all_images = []
        for ext in image_extensions:
            all_images.extend(glob.glob(str(self.images_dir / ext)))
        
        # Get all mask files
        mask_extensions = ['*.png', '*.tif']
        all_masks = []
        for ext in mask_extensions:
            all_masks.extend(glob.glob(str(self.masks_dir / ext)))
        
        print(f"Found {len(all_images)} images and {len(all_masks)} masks")
        
        # Match images with masks
        for img_path in all_images:
            img_name = Path(img_path).stem
            
            # Filter by tile prefix if specified
            if self.use_tile_masks_only and not img_name.startswith('tile_'):
                continue
            
            # Find corresponding mask
            mask_path = None
            for mask_p in all_masks:
                mask_name = Path(mask_p).stem
                if mask_name == img_name:
                    mask_path = mask_p
                    break
            
            if mask_path:
                self.image_paths.append(img_path)
                self.mask_paths.append(mask_path)
        
        print(f"Matched {len(self.image_paths)} image-mask pairs")
    
    def _filter_dataset(self):
        """Filter dataset based on slum content criteria."""
        if (self.min_slum_pixels == 0 and 
            self.max_slum_percentage == 1.0 and 
            self.min_slum_percentage == 0.0):
            print("No filtering applied")
            return
        
        print("Filtering dataset based on slum content...")
        
        filtered_images = []
        filtered_masks = []
        
        for img_path, mask_path in zip(self.image_paths, self.mask_paths):
            # Load and process mask
            binary_mask = self._load_binary_mask(mask_path)
            
            # Calculate slum statistics
            total_pixels = binary_mask.size
            slum_pixels = np.sum(binary_mask)
            slum_percentage = slum_pixels / total_pixels
            
            # Store statistics
            img_name = Path(img_path).stem
            self.slum_stats[img_name] = {
                'slum_pixels': slum_pixels,
                'total_pixels': total_pixels,
                'slum_percentage': slum_percentage
            }
            
            # Apply filters
            if (slum_pixels >= self.min_slum_pixels and
                slum_percentage >= self.min_slum_percentage and
                slum_percentage <= self.max_slum_percentage):
                filtered_images.append(img_path)
                filtered_masks.append(mask_path)
        
        self.image_paths = filtered_images
        self.mask_paths = filtered_masks
        
        print(f"After filtering: {len(self.image_paths)} samples remain")
    
    def _load_binary_mask(self, mask_path: str) -> np.ndarray:
        """Load RGB mask and convert to binary."""
        # Check cache first
        if self.cache_masks and mask_path in self.cached_masks:
            return self.cached_masks[mask_path]
        
        # Load RGB mask
        mask = cv2.imread(mask_path)
        if mask is None:
            raise ValueError(f"Could not load mask: {mask_path}")
        
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        
        # Convert to binary (slum vs non-slum)
        slum_mask = np.all(mask == self.slum_rgb, axis=-1).astype(np.uint8)
        
        # Cache if enabled
        if self.cache_masks:
            self.cached_masks[mask_path] = slum_mask
        
        return slum_mask
    
    def _load_image(self, image_path: str) -> np.ndarray:
        """Load and preprocess image."""
        # Try different loading methods
        try:
            # Try PIL first
            image = Image.open(image_path).convert('RGB')
            image = np.array(image)
        except:
            # Fallback to OpenCV
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Ensure correct size
        if image.shape[:2] != self.image_size:
            image = cv2.resize(image, (self.image_size[1], self.image_size[0]))
        
        return image
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get image and mask pair."""
        # Load image and mask
        image = self._load_image(self.image_paths[idx])
        mask = self._load_binary_mask(self.mask_paths[idx])
        
        # Ensure mask is correct size
        if mask.shape != self.image_size:
            mask = cv2.resize(mask, (self.image_size[1], self.image_size[0]))
          # Apply transforms if provided
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        else:
            # Convert to tensors manually
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            mask = torch.from_numpy(mask).float()
        
        # Ensure mask is float and in correct range [0, 1]
        mask = mask.float()

        return image, mask
    
    def _print_dataset_stats(self):
        """Print dataset statistics."""
        if not self.slum_stats:
            return
        
        stats = list(self.slum_stats.values())
        slum_percentages = [s['slum_percentage'] for s in stats]
        
        print("\n📊 Dataset Statistics:")
        print(f"  Total samples: {len(self.image_paths)}")
        print(f"  Images with slums: {sum(1 for p in slum_percentages if p > 0)}")
        print(f"  Average slum percentage: {np.mean(slum_percentages):.2%}")
        print(f"  Min slum percentage: {np.min(slum_percentages):.2%}")
        print(f"  Max slum percentage: {np.max(slum_percentages):.2%}")
        print()
    
    def get_class_weights(self) -> Dict[str, float]:
        """Calculate class weights for handling imbalance."""
        if not self.slum_stats:
            return {'pos_weight': 1.0}
        
        total_pixels = sum(s['total_pixels'] for s in self.slum_stats.values())
        total_slum_pixels = sum(s['slum_pixels'] for s in self.slum_stats.values())
        total_non_slum_pixels = total_pixels - total_slum_pixels
        
        if total_slum_pixels == 0:
            pos_weight = 1.0
        else:
            pos_weight = total_non_slum_pixels / total_slum_pixels
        
        return {'pos_weight': pos_weight}
    
    def get_sample_weights(self) -> List[float]:
        """Get sample weights for weighted sampling."""
        weights = []
        for img_path in self.image_paths:
            img_name = Path(img_path).stem
            if img_name in self.slum_stats:
                slum_percentage = self.slum_stats[img_name]['slum_percentage']
                # Higher weight for images with more slums
                weight = 1.0 + slum_percentage
            else:
                weight = 1.0
            weights.append(weight)
        return weights


def create_data_loaders(
    train_dataset: SlumDataset,
    val_dataset: SlumDataset,
    test_dataset: Optional[SlumDataset] = None,
    batch_size: int = 16,
    num_workers: int = 4,
    pin_memory: bool = True,
    use_weighted_sampling: bool = False
) -> Dict[str, DataLoader]:
    """
    Create data loaders for training, validation, and testing.
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        test_dataset: Test dataset (optional)
        batch_size: Batch size for all loaders
        num_workers: Number of worker processes
        pin_memory: Pin memory for faster GPU transfer
        use_weighted_sampling: Use weighted sampling for class balance
    
    Returns:
        Dictionary containing data loaders
    """
    loaders = {}
    
    # Training loader with optional weighted sampling
    if use_weighted_sampling:
        from torch.utils.data import WeightedRandomSampler
        weights = train_dataset.get_sample_weights()
        sampler = WeightedRandomSampler(weights, len(weights))
        shuffle = False
    else:
        sampler = None
        shuffle = True
    
    loaders['train'] = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )
    
    # Validation loader
    loaders['val'] = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    
    # Test loader
    if test_dataset:
        loaders['test'] = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False
        )
    
    print(f"Data loaders created:")
    print(f"  Train: {len(loaders['train'])} batches")
    print(f"  Val: {len(loaders['val'])} batches")
    if 'test' in loaders:
        print(f"  Test: {len(loaders['test'])} batches")
    
    return loaders


def verify_dataset_setup(data_config) -> bool:
    """
    Verify that dataset directories exist and contain data.
    
    Args:
        data_config: DataConfig instance
    
    Returns:
        True if setup is valid, False otherwise
    """
    paths = data_config.get_paths()
    
    print("🔍 Verifying dataset setup...")
    
    all_valid = True
    for split, path in paths.items():
        if not os.path.exists(path):
            print(f"❌ Missing directory: {path}")
            all_valid = False
        else:
            # Count files
            if 'images' in split:
                extensions = ['*.tif', '*.png', '*.jpg', '*.jpeg']
            else:
                extensions = ['*.png', '*.tif']
            
            file_count = 0
            for ext in extensions:
                file_count += len(glob.glob(os.path.join(path, ext)))
            
            if file_count == 0:
                print(f"⚠️  Empty directory: {path}")
                all_valid = False
            else:
                print(f"✅ {split}: {file_count} files")
    
    if all_valid:
        print("✅ Dataset setup verified successfully!")
    else:
        print("❌ Dataset setup has issues!")
    
    return all_valid


if __name__ == "__main__":
    # Test dataset loading
    from config.data_config import get_data_config
    
    config = get_data_config("standard")
    
    # Test dataset creation
    print("Testing SlumDataset...")
    
    paths = config.get_paths()
    dataset = SlumDataset(
        images_dir=paths['train_images'],
        masks_dir=paths['train_masks'],
        slum_rgb=config.slum_rgb,
        image_size=config.image_size,
        use_tile_masks_only=config.use_tile_masks_only,
        min_slum_pixels=config.min_slum_pixels
    )
    
    if len(dataset) > 0:
        # Test data loading
        image, mask = dataset[0]
        print(f"Sample shapes - Image: {image.shape}, Mask: {mask.shape}")
        
        # Test class weights
        weights = dataset.get_class_weights()
        print(f"Class weights: {weights}")
    
    # Test dataset verification
    verify_dataset_setup(config)
