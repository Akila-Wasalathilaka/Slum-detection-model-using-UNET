"""
Dataset and augmentation logic for production slum detection.
"""
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from config import config

class ProductionDataset(Dataset):
    def __init__(self, image_dir: Path, mask_dir: Path, transform=None, image_size: int = 160):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.transform = transform
        self.image_size = image_size
        self.image_files = sorted(list(self.image_dir.glob("*.tif")))
        self.mask_files = sorted(list(self.mask_dir.glob("*.png")))
        assert len(self.image_files) == len(self.mask_files), f"Mismatch: {len(self.image_files)} images vs {len(self.mask_files)} masks"
    def __len__(self):
        return len(self.image_files)
    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        mask_path = self.mask_files[idx]
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        binary_mask = (mask == config.SLUM_CLASS_ID).astype(np.float32)
        if image.shape[:2] != (self.image_size, self.image_size):
            image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
            binary_mask = cv2.resize(binary_mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        if self.transform:
            augmented = self.transform(image=image, mask=binary_mask)
            image = augmented['image']
            binary_mask = augmented['mask']
        if len(binary_mask.shape) == 3:
            binary_mask = binary_mask[:, :, 0]
        return image, binary_mask.unsqueeze(0)

def get_production_augmentations(image_size: int, phase: str = 'train'):
    if phase == 'train':
        return A.Compose([
            A.Resize(height=image_size, width=image_size, interpolation=cv2.INTER_LINEAR),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=30, p=0.6),
            A.ElasticTransform(alpha=1, sigma=50, p=0.3),
            A.OneOf([
                A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=1.0),
                A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1.0),
                A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1.0)
            ], p=0.7),
            A.OneOf([
                A.GaussNoise(p=1.0),
                A.ISONoise(p=1.0),
                A.MultiplicativeNoise(multiplier=(0.9, 1.1), p=1.0)
            ], p=0.4),
            A.OneOf([
                A.Blur(blur_limit=3, p=1.0),
                A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                A.MotionBlur(blur_limit=3, p=1.0)
            ], p=0.3),
            A.CoarseDropout(max_holes=8, max_height=16, max_width=16, p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(height=image_size, width=image_size, interpolation=cv2.INTER_LINEAR),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

def get_ultra_data_loaders():
    """Get ultra-accurate data loaders for training, validation, and testing."""
    
    # Get transforms
    train_transform = get_production_augmentations(config.PRIMARY_SIZE, 'train')
    val_transform = get_production_augmentations(config.PRIMARY_SIZE, 'val')
    test_transform = get_production_augmentations(config.PRIMARY_SIZE, 'val')
    
    # Create datasets
    train_dataset = ProductionDataset(
        image_dir=config.TRAIN_IMG_DIR,
        mask_dir=config.TRAIN_MASK_DIR,
        transform=train_transform,
        image_size=config.PRIMARY_SIZE
    )
    
    val_dataset = ProductionDataset(
        image_dir=config.VAL_IMG_DIR,
        mask_dir=config.VAL_MASK_DIR,
        transform=val_transform,
        image_size=config.PRIMARY_SIZE
    )
    
    test_dataset = ProductionDataset(
        image_dir=config.TEST_IMG_DIR,
        mask_dir=config.TEST_MASK_DIR,
        transform=test_transform,
        image_size=config.PRIMARY_SIZE
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )
    
    return train_loader, val_loader, test_loader
