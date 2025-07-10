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
    """Enhanced dataset class for global slum detection with context awareness"""
    def __init__(self, image_dir: Path, mask_dir: Path, transform=None, image_size: int = 160,
                 context_size: int = 320):  # Larger context window
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.transform = transform
        self.image_size = image_size
        self.context_size = context_size  # Larger window to capture settlement patterns
        self.image_files = sorted(list(self.image_dir.glob("*.tif")))
        self.mask_files = sorted(list(self.mask_dir.glob("*.png")))
        
        # Ensure data integrity
        assert len(self.image_files) == len(self.mask_files), \
            f"Mismatch: {len(self.image_files)} images vs {len(self.mask_files)} masks"
            
        # Calculate texture statistics for normalization
        self.calculate_settlement_statistics()
    def __len__(self):
        return len(self.image_files)
    def calculate_settlement_statistics(self):
        """Advanced settlement pattern analysis for global detection"""
        self.density_stats = []
        self.texture_stats = []
        self.pattern_stats = []
        self.regularity_stats = []
        self.morphology_stats = []
        self.connectivity_stats = []
        self.road_pattern_stats = []
        self.building_orientation_stats = []
        self.settlement_boundary_stats = []
        
        for image_path in self.image_files[:min(100, len(self.image_files))]:
            img = cv2.imread(str(image_path))
            if img is None:
                continue
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # 1. Building Density Analysis
            edges = cv2.Canny(gray, 50, 150)
            density = np.mean(edges > 0)
            self.density_stats.append(density)
            
            # 2. Settlement Pattern Analysis
            # Calculate FFT for pattern regularity
            f_transform = np.fft.fft2(gray)
            f_shift = np.fft.fftshift(f_transform)
            magnitude_spectrum = np.abs(f_shift)
            pattern_regularity = np.std(magnitude_spectrum)  # Lower = more regular pattern
            self.regularity_stats.append(pattern_regularity)
            
            # 3. Regional Style Detection
            # Create patches to analyze local patterns
            patches = self.create_patches(gray, patch_size=32)
            for patch in patches:
                if patch.size == 0:
                    continue
                # Texture analysis using GLCM (Gray-Level Co-occurrence Matrix)
                glcm = self.calculate_glcm(patch)
                contrast = self.glcm_contrast(glcm)
                homogeneity = self.glcm_homogeneity(glcm)
                self.pattern_stats.append((contrast, homogeneity))
            
            # 4. Advanced Morphological Analysis
            contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) > 0:
                # Building size and shape analysis
                areas = [cv2.contourArea(c) for c in contours]
                perimeters = [cv2.arcLength(c, True) for c in contours]
                self.building_size_variation = np.std(areas) / np.mean(areas) if np.mean(areas) > 0 else 0
                
                # Shape complexity (circularity)
                circularities = [4 * np.pi * area / (perim * perim) if perim > 0 else 0 
                               for area, perim in zip(areas, perimeters)]
                self.morphology_stats.append(np.mean(circularities))
                
                # Building orientation analysis
                orientations = []
                for c in contours:
                    if len(c) >= 5:  # Minimum points for ellipse fitting
                        try:
                            (_, _), (MA, ma), angle = cv2.fitEllipse(c)
                            orientations.append(angle)
                        except:
                            continue
                if orientations:
                    # Calculate orientation consistency
                    orientation_hist = np.histogram(orientations, bins=36)[0]
                    self.building_orientation_stats.append(np.std(orientation_hist))
            
            # 5. Road Network Analysis
            # Detect potential roads using morphological operations
            road_kernel = np.ones((5,5), np.uint8)
            dilated = cv2.dilate(edges, road_kernel, iterations=2)
            eroded = cv2.erode(dilated, road_kernel, iterations=1)
            road_pattern = cv2.subtract(dilated, eroded)
            
            # Analyze road network properties
            road_stats = cv2.connectedComponentsWithStats(road_pattern, 8, cv2.CV_32S)
            if road_stats[0] > 1:  # If roads detected
                road_areas = road_stats[2][1:, cv2.CC_STAT_AREA]
                road_lengths = road_stats[2][1:, cv2.CC_STAT_WIDTH] + road_stats[2][1:, cv2.CC_STAT_HEIGHT]
                self.road_pattern_stats.append(np.mean(road_lengths) / np.mean(road_areas) if len(road_areas) > 0 else 0)
            
            # 6. Settlement Boundary Analysis
            # Use distance transform to analyze settlement boundaries
            dist_transform = cv2.distanceTransform(255 - edges, cv2.DIST_L2, 5)
            boundary_strength = np.mean(dist_transform[dist_transform < 20])  # Analysis of edge transitions
            self.settlement_boundary_stats.append(boundary_strength)
            
            # 7. Enhanced Texture Analysis
            texture = cv2.cornerHarris(gray, 2, 3, 0.04)
            self.texture_stats.append(np.mean(texture))
            
            # 8. Connectivity Analysis
            # Analyze settlement connectivity using skeleton
            skeleton = cv2.ximgproc.thinning(edges)
            num_endpoints = len(np.where(cv2.filter2D(skeleton, -1, np.ones((3,3))) == 1)[0])
            num_junctions = len(np.where(cv2.filter2D(skeleton, -1, np.ones((3,3))) > 2)[0])
            self.connectivity_stats.append(num_junctions / (num_endpoints + 1))
        
        # Calculate final statistics
        self.avg_density = np.mean(self.density_stats) if self.density_stats else 0
        self.avg_texture = np.mean(self.texture_stats) if self.texture_stats else 0
        self.avg_regularity = np.mean(self.regularity_stats) if self.regularity_stats else 0
        
        if self.pattern_stats:
            contrasts, homogeneities = zip(*self.pattern_stats)
            self.avg_contrast = np.mean(contrasts)
            self.avg_homogeneity = np.mean(homogeneities)
        else:
            self.avg_contrast = 0
            self.avg_homogeneity = 0

    def create_patches(self, image, patch_size=32):
        """Create patches for local pattern analysis"""
        patches = []
        for i in range(0, image.shape[0] - patch_size, patch_size):
            for j in range(0, image.shape[1] - patch_size, patch_size):
                patch = image[i:i + patch_size, j:j + patch_size]
                if patch.size == patch_size * patch_size:
                    patches.append(patch)
        return patches

    def calculate_glcm(self, patch, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4]):
        """Calculate Gray-Level Co-occurrence Matrix"""
        glcm = np.zeros((256, 256))
        for distance in distances:
            for angle in angles:
                dx = int(round(distance * np.cos(angle)))
                dy = int(round(distance * np.sin(angle)))
                for i in range(patch.shape[0]):
                    for j in range(patch.shape[1]):
                        if 0 <= i + dy < patch.shape[0] and 0 <= j + dx < patch.shape[1]:
                            i2 = i + dy
                            j2 = j + dx
                            glcm[patch[i, j], patch[i2, j2]] += 1
        return glcm / glcm.sum() if glcm.sum() > 0 else glcm

    def glcm_contrast(self, glcm):
        """Calculate GLCM contrast feature"""
        contrast = 0
        for i in range(glcm.shape[0]):
            for j in range(glcm.shape[1]):
                contrast += glcm[i,j] * (i-j)**2
        return contrast

    def glcm_homogeneity(self, glcm):
        """Calculate GLCM homogeneity feature"""
        homogeneity = 0
        for i in range(glcm.shape[0]):
            for j in range(glcm.shape[1]):
                homogeneity += glcm[i,j] / (1 + abs(i-j))
        return homogeneity

    def __getitem__(self, idx):
        """Enhanced data loading with global settlement pattern recognition"""
        image_path = self.image_files[idx]
        mask_path = self.mask_files[idx]
        
        # Load image with extended context
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        binary_mask = (mask == config.SLUM_CLASS_ID).astype(np.float32)
        
        # 1. Settlement Pattern Analysis
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Calculate local density
        edges = cv2.Canny(gray, 50, 150)
        local_density = np.mean(edges > 0)
        
        # Calculate pattern regularity
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.abs(f_shift)
        local_regularity = np.std(magnitude_spectrum)
        
        # 2. Regional Style Detection
        patches = self.create_patches(gray, patch_size=32)
        pattern_features = []
        for patch in patches:
            if patch.size == 0:
                continue
            glcm = self.calculate_glcm(patch)
            contrast = self.glcm_contrast(glcm)
            homogeneity = self.glcm_homogeneity(glcm)
            pattern_features.append((contrast, homogeneity))
        
        # 3. Settlement Type Classification
        if pattern_features:
            contrasts, homogeneities = zip(*pattern_features)
            avg_contrast = np.mean(contrasts)
            avg_homogeneity = np.mean(homogeneities)
            
            # Advanced Settlement Type Classification
            road_pattern = np.mean(self.road_pattern_stats) if self.road_pattern_stats else 0
            connectivity = np.mean(self.connectivity_stats) if self.connectivity_stats else 0
            boundary_strength = np.mean(self.settlement_boundary_stats) if self.settlement_boundary_stats else 0
            morphology = np.mean(self.morphology_stats) if self.morphology_stats else 0
            
            # 1. Dense Asian Urban Slums
            if local_density > self.avg_density * 1.5 and local_regularity < self.avg_regularity * 0.8:
                if connectivity > 0.7:  # High internal connectivity
                    # Very dense, organic pattern with high connectivity
                    clahe = cv2.createCLAHE(clipLimit=3.8, tileGridSize=(6,6))
                    gray = clahe.apply(gray)
                    # Enhance edge definition for dense building patterns
                    edges = cv2.Canny(gray, 50, 150)
                    gray = cv2.addWeighted(gray, 0.8, edges, 0.2, 0)
                    image = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
            
            # 2. Latin American Hillside Settlements
            elif local_density > self.avg_density * 1.2 and road_pattern > 0.5:
                if avg_contrast > self.avg_contrast * 1.1:
                    # Terraced pattern with strong elevation changes
                    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(12,12))
                    gray = clahe.apply(gray)
                    # Enhance vertical structures
                    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
                    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
                    gradient_mag = np.sqrt(sobelx**2 + sobely**2)
                    gradient_mag = cv2.normalize(gradient_mag, None, 0, 255, cv2.NORM_MINMAX)
                    gray = cv2.addWeighted(gray, 0.7, gradient_mag.astype(np.uint8), 0.3, 0)
                    image = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
            
            # 3. African Informal Settlements
            elif morphology < 0.6 and boundary_strength > self.avg_texture * 1.2:
                # Organic growth pattern with strong community boundaries
                clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(16,16))
                gray = clahe.apply(gray)
                # Enhance settlement boundaries
                edges = cv2.Canny(gray, 30, 150)
                kernel = np.ones((3,3), np.uint8)
                edges = cv2.dilate(edges, kernel, iterations=1)
                image = cv2.addWeighted(image, 0.85, cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB), 0.15, 0)
            
            # 4. Temporary or Transit Settlements
            elif avg_homogeneity > self.avg_homogeneity * 1.3 and local_density < self.avg_density * 0.8:
                # Regular, sparse pattern typical of planned temporary settlements
                image = cv2.convertScaleAbs(image, alpha=1.2, beta=5)
                # Enhance regular patterns
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
                pattern = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, np.ones((5,5)))
                image = cv2.addWeighted(image, 0.9, cv2.cvtColor(pattern, cv2.COLOR_GRAY2RGB), 0.1, 0)
            
            # 5. Peri-Urban Expansions
            elif connectivity < 0.5 and boundary_strength < self.avg_texture * 0.8:
                # Scattered development pattern with weak boundaries
                # Enhance settlement edges and connectivity
                edges = cv2.Canny(gray, 40, 160)
                kernel = np.ones((3,3), np.uint8)
                edges = cv2.dilate(edges, kernel, iterations=2)
                edges = cv2.erode(edges, kernel, iterations=1)
                image = cv2.addWeighted(image, 0.8, cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB), 0.2, 0)
                
        # 4. Context-Aware Enhancement
        if local_density < self.avg_density * 0.5:
            # Enhance edges for sparse settlements
            edges = cv2.Canny(gray, 30, 150)
            kernel = np.ones((3,3), np.uint8)
            edges = cv2.dilate(edges, kernel, iterations=1)
            image = cv2.addWeighted(image, 0.8, cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB), 0.2, 0)
            
        # Resize while preserving settlement patterns
        if image.shape[:2] != (self.image_size, self.image_size):
            image = cv2.resize(image, (self.image_size, self.image_size), 
                             interpolation=cv2.INTER_LINEAR)
            binary_mask = cv2.resize(binary_mask, (self.image_size, self.image_size), 
                                   interpolation=cv2.INTER_NEAREST)
        if self.transform:
            augmented = self.transform(image=image, mask=binary_mask)
            image = augmented['image']
            binary_mask = augmented['mask']
        if len(binary_mask.shape) == 3:
            binary_mask = binary_mask[:, :, 0]
        return image, binary_mask.unsqueeze(0)

def get_production_augmentations(image_size: int, phase: str = 'train'):
    """Enhanced augmentations for global slum detection"""
    if phase == 'train':
        return A.Compose([
            # Multi-scale processing for different settlement densities
            A.OneOf([
                A.Resize(height=image_size, width=image_size),
                A.SmallestMaxSize(max_size=image_size, p=1.0),
                A.LongestMaxSize(max_size=image_size, p=1.0)
            ], p=1.0),
            
            # Pattern-preserving spatial augmentations
            A.RandomRotate90(p=0.5),
            A.OneOf([
                A.HorizontalFlip(p=1.0),
                A.VerticalFlip(p=1.0),
                A.Transpose(p=1.0)  # Handles different settlement orientations
            ], p=0.7),
            
            # Careful geometric transforms to maintain settlement density patterns
            A.ShiftScaleRotate(
                shift_limit=0.1,        # Reduced shift to preserve settlement boundaries
                scale_limit=0.2,        # Careful scaling to maintain building density
                rotate_limit=30,        # Limited rotation to preserve orientation patterns
                p=0.7
            ),
            
            # Texture and pattern preserving transforms
            A.GridDistortion(
                num_steps=5,            # Controlled distortion to maintain settlement patterns
                distort_limit=0.2,      # Limited distortion
                p=0.3
            ),
            
            # Enhanced contrast and texture transforms for building detection
            A.OneOf([
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.3,
                    p=1.0
                ),
                A.CLAHE(clip_limit=3, p=1.0),  # Improve local contrast for building detection
            ], p=0.7),
            A.OneOf([
                A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2, p=1.0),
                A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=1.0),
                A.HueSaturationValue(hue_shift_limit=30, sat_shift_limit=40, val_shift_limit=30, p=1.0),
                A.RandomGamma(p=1.0),  # Added gamma adjustment
            ], p=0.8),
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
            A.CoarseDropout(p=0.3),  # Use default parameters to avoid warnings
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
