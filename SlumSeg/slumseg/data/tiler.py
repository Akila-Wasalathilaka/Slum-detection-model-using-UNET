"""Tiling utilities for large satellite images."""

import os
import cv2
import numpy as np
import rasterio
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import logging
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

logger = logging.getLogger(__name__)


class ImageTiler:
    """Tile large images with overlap and filtering."""
    
    def __init__(
        self,
        tile_size: int = 512,
        overlap: int = 64,
        min_slum_pixels: int = 512,
        save_format: str = "png"
    ):
        self.tile_size = tile_size
        self.overlap = overlap
        self.min_slum_pixels = min_slum_pixels
        self.save_format = save_format
        
    def tile_image_pair(
        self, 
        image_path: str, 
        mask_path: str, 
        output_dir: str,
        prefix: str = ""
    ) -> List[Dict[str, str]]:
        """Tile an image-mask pair."""
        
        # Load image and mask
        image = self._load_image(image_path)
        mask = self._load_mask(mask_path)
        
        if image.shape[:2] != mask.shape[:2]:
            logger.warning(f"Size mismatch: {image.shape[:2]} vs {mask.shape[:2]}")
            # Resize mask to match image
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        # Generate tiles
        tiles_info = []
        tile_idx = 0
        
        h, w = image.shape[:2]
        step = self.tile_size - self.overlap
        
        for y in range(0, h - self.tile_size + 1, step):
            for x in range(0, w - self.tile_size + 1, step):
                # Extract tile
                img_tile = image[y:y+self.tile_size, x:x+self.tile_size]
                mask_tile = mask[y:y+self.tile_size, x:x+self.tile_size]
                
                # Filter by slum content
                slum_pixels = np.sum(mask_tile > 0)
                if slum_pixels < self.min_slum_pixels:
                    continue
                
                # Save tiles
                base_name = f"{prefix}_{Path(image_path).stem}_{tile_idx:04d}"
                img_save_path = Path(output_dir) / "images" / f"{base_name}.{self.save_format}"
                mask_save_path = Path(output_dir) / "masks" / f"{base_name}.{self.save_format}"
                
                # Create directories
                img_save_path.parent.mkdir(parents=True, exist_ok=True)
                mask_save_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Save tiles
                cv2.imwrite(str(img_save_path), cv2.cvtColor(img_tile, cv2.COLOR_RGB2BGR))
                cv2.imwrite(str(mask_save_path), mask_tile)
                
                tiles_info.append({
                    "image_path": str(img_save_path),
                    "mask_path": str(mask_save_path),
                    "original_image": image_path,
                    "original_mask": mask_path,
                    "tile_x": x,
                    "tile_y": y,
                    "slum_pixels": slum_pixels,
                    "slum_ratio": slum_pixels / (self.tile_size * self.tile_size)
                })
                
                tile_idx += 1
        
        return tiles_info
    
    def tile_dataset(
        self,
        image_dir: str,
        mask_dir: str,
        output_dir: str,
        split_name: str = "train",
        max_workers: int = 4
    ) -> List[Dict[str, str]]:
        """Tile entire dataset."""
        
        # Find all image-mask pairs
        image_paths = sorted(list(Path(image_dir).glob("*.tif")) + list(Path(image_dir).glob("*.png")))
        mask_paths = sorted(list(Path(mask_dir).glob("*.tif")) + list(Path(mask_dir).glob("*.png")))
        
        if len(image_paths) != len(mask_paths):
            raise ValueError(f"Mismatch: {len(image_paths)} images vs {len(mask_paths)} masks")
        
        logger.info(f"Tiling {len(image_paths)} image-mask pairs for {split_name}")
        
        all_tiles = []
        
        def process_pair(args):
            img_path, mask_path, idx = args
            try:
                tiles = self.tile_image_pair(
                    str(img_path), 
                    str(mask_path), 
                    output_dir,
                    prefix=f"{split_name}_{idx:04d}"
                )
                return tiles
            except Exception as e:
                logger.error(f"Error processing {img_path}: {e}")
                return []
        
        # Process in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            args_list = [(img_path, mask_paths[i], i) for i, img_path in enumerate(image_paths)]
            results = list(tqdm(
                executor.map(process_pair, args_list),
                total=len(args_list),
                desc=f"Tiling {split_name}"
            ))
        
        # Flatten results
        for result in results:
            all_tiles.extend(result)
        
        logger.info(f"Generated {len(all_tiles)} tiles for {split_name}")
        
        return all_tiles
    
    def _load_image(self, path: str) -> np.ndarray:
        """Load image with rasterio/opencv fallback."""
        try:
            with rasterio.open(path) as src:
                image = src.read()  # (C, H, W)
                if image.shape[0] > 3:
                    image = image[:3]  # RGB only
                image = np.transpose(image, (1, 2, 0))  # (H, W, C)
        except Exception:
            image = cv2.imread(path)
            if image is None:
                raise ValueError(f"Could not load image: {path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        return image.astype(np.uint8)
    
    def _load_mask(self, path: str) -> np.ndarray:
        """Load mask."""
        try:
            with rasterio.open(path) as src:
                mask = src.read(1)  # Single channel
        except Exception:
            mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise ValueError(f"Could not load mask: {path}")
        
        # Binarize
        mask = (mask > 0).astype(np.uint8)
        return mask


def create_tiles_from_config(config: Dict, output_dir: str) -> Dict[str, List[Dict]]:
    """Create tiles from configuration."""
    
    data_config = config["data"]
    root = data_config["root"]
    
    tiler = ImageTiler(
        tile_size=data_config.get("tile_size", 512),
        overlap=data_config.get("tile_overlap", 64),
        min_slum_pixels=data_config.get("min_slum_px", 512)
    )
    
    all_tiles = {}
    
    # Process each split
    for split in ["train", "val", "test"]:
        split_img_dir = Path(root) / split / "images"
        split_mask_dir = Path(root) / split / "masks"
        
        if not split_img_dir.exists():
            logger.warning(f"Skipping {split} - directory not found: {split_img_dir}")
            continue
            
        if not split_mask_dir.exists():
            logger.warning(f"Skipping {split} - mask directory not found: {split_mask_dir}")
            continue
        
        split_output = Path(output_dir) / split
        tiles = tiler.tile_dataset(
            str(split_img_dir),
            str(split_mask_dir),
            str(split_output),
            split_name=split
        )
        
        all_tiles[split] = tiles
    
    return all_tiles


def filter_tiles_by_content(
    tiles: List[Dict[str, str]], 
    min_slum_ratio: float = 0.01,
    max_background_ratio: float = 0.95
) -> List[Dict[str, str]]:
    """Filter tiles by slum content."""
    
    filtered = []
    for tile in tiles:
        slum_ratio = tile.get("slum_ratio", 0)
        
        if min_slum_ratio <= slum_ratio <= (1 - max_background_ratio):
            filtered.append(tile)
    
    logger.info(f"Filtered {len(tiles)} -> {len(filtered)} tiles")
    return filtered
