#!/usr/bin/env python3
"""
Colombo Satellite Image Processing Script
=========================================

Processes satellite images of Colombo for slum detection:
1. Loads high-resolution satellite images
2. Splits them into 120x120 pixel tiles for model compatibility
3. Applies preprocessing according to model requirements
4. Saves tiles for inference and analysis
"""

import os
import sys
import numpy as np
from PIL import Image
import cv2
from pathlib import Path
import json
import argparse
from datetime import datetime
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

class ColomboImageProcessor:
    """Process Colombo satellite images for slum detection."""
    
    def __init__(self, output_dir="colombo", tile_size=(120, 120), overlap=0):
        """
        Initialize the processor.
        
        Args:
            output_dir: Directory to save processed tiles
            tile_size: Size of output tiles (width, height)
            overlap: Overlap between tiles in pixels
        """
        self.output_dir = Path(output_dir)
        self.tile_size = tile_size
        self.overlap = overlap
        self.tiles_dir = self.output_dir / "tiles"
        self.metadata_dir = self.output_dir / "metadata"
        
        # Create directories
        self.tiles_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"🏙️ Colombo Image Processor initialized")
        print(f"   Output directory: {self.output_dir}")
        print(f"   Tile size: {tile_size}")
        print(f"   Overlap: {overlap} pixels")
    
    def load_image(self, image_path):
        """Load and validate input image."""
        print(f"📂 Loading image: {image_path}")
        
        try:
            # Try PIL first
            image = Image.open(image_path).convert('RGB')
            image = np.array(image)
            print(f"   ✅ Loaded with PIL: {image.shape}")
        except Exception as e:
            print(f"   ⚠️ PIL failed: {e}")
            try:
                # Fallback to OpenCV
                image = cv2.imread(str(image_path))
                if image is None:
                    raise ValueError(f"Could not load image: {image_path}")
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                print(f"   ✅ Loaded with OpenCV: {image.shape}")
            except Exception as e2:
                raise ValueError(f"Failed to load image with both PIL and OpenCV: {e2}")
        
        # Validate image
        if len(image.shape) != 3 or image.shape[2] != 3:
            raise ValueError(f"Image must be 3-channel RGB, got shape: {image.shape}")
        
        print(f"   📏 Image dimensions: {image.shape[1]}x{image.shape[0]} pixels")
        return image
    
    def split_into_tiles(self, image):
        """Split image into 120x120 pixel tiles."""
        height, width = image.shape[:2]
        tile_width, tile_height = self.tile_size
        
        print(f"🔪 Splitting image into {tile_width}x{tile_height} tiles...")
        
        tiles = []
        tile_metadata = []
        
        # Calculate step size considering overlap
        step_x = tile_width - self.overlap
        step_y = tile_height - self.overlap
        
        # Calculate number of tiles
        num_tiles_x = (width - self.overlap) // step_x
        num_tiles_y = (height - self.overlap) // step_y
        
        print(f"   📊 Will create {num_tiles_x} x {num_tiles_y} = {num_tiles_x * num_tiles_y} tiles")
        
        tile_count = 0
        for y in range(0, height - tile_height + 1, step_y):
            for x in range(0, width - tile_width + 1, step_x):
                # Extract tile
                tile = image[y:y+tile_height, x:x+tile_width]
                
                # Ensure exact tile size
                if tile.shape[:2] != (tile_height, tile_width):
                    tile = cv2.resize(tile, (tile_width, tile_height))
                
                tiles.append(tile)
                
                # Store metadata
                metadata = {
                    'tile_id': tile_count,
                    'x_start': int(x),
                    'y_start': int(y),
                    'x_end': int(x + tile_width),
                    'y_end': int(y + tile_height),
                    'tile_shape': tile.shape
                }
                tile_metadata.append(metadata)
                
                tile_count += 1
        
        print(f"   ✅ Created {len(tiles)} tiles")
        return tiles, tile_metadata
    
    def preprocess_tiles(self, tiles):
        """Apply model-specific preprocessing to tiles."""
        print("🔧 Preprocessing tiles for model compatibility...")
        
        processed_tiles = []
        
        for i, tile in enumerate(tiles):
            # Ensure RGB format
            if len(tile.shape) != 3 or tile.shape[2] != 3:
                print(f"   ⚠️ Tile {i} has incorrect shape: {tile.shape}")
                continue
            
            # Ensure correct size
            if tile.shape[:2] != self.tile_size:
                tile = cv2.resize(tile, self.tile_size)
            
            # Normalize to [0, 1] range (model expects this)
            tile_normalized = tile.astype(np.float32) / 255.0
            
            # Validate pixel range
            if tile_normalized.min() < 0 or tile_normalized.max() > 1:
                print(f"   ⚠️ Tile {i} has invalid pixel range: [{tile_normalized.min()}, {tile_normalized.max()}]")
            
            processed_tiles.append(tile_normalized)
        
        print(f"   ✅ Preprocessed {len(processed_tiles)} tiles")
        return processed_tiles
    
    def save_tiles(self, tiles, tile_metadata, prefix="colombo_tile"):
        """Save tiles and metadata to disk."""
        print(f"💾 Saving tiles to {self.tiles_dir}...")
        
        saved_files = []
        
        for i, (tile, metadata) in enumerate(zip(tiles, tile_metadata)):
            # Convert back to uint8 for saving
            tile_uint8 = (tile * 255).astype(np.uint8)
            
            # Create filename
            filename = f"{prefix}_{i:04d}.png"
            filepath = self.tiles_dir / filename
            
            # Save tile
            try:
                Image.fromarray(tile_uint8).save(filepath)
                saved_files.append(str(filepath))
                
                # Update metadata with file path
                metadata['file_path'] = str(filepath)
                metadata['filename'] = filename
                
            except Exception as e:
                print(f"   ❌ Failed to save tile {i}: {e}")
        
        # Save metadata
        metadata_file = self.metadata_dir / f"{prefix}_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump({
                'processing_timestamp': datetime.now().isoformat(),
                'total_tiles': len(tile_metadata),
                'tile_size': self.tile_size,
                'overlap': self.overlap,
                'tiles': tile_metadata
            }, f, indent=2)
        
        print(f"   ✅ Saved {len(saved_files)} tiles")
        print(f"   📄 Metadata saved: {metadata_file}")
        
        return saved_files, str(metadata_file)
    
    def create_tile_overview(self, tiles, tile_metadata, max_display=50):
        """Create an overview visualization of the tiles."""
        print("🎨 Creating tile overview visualization...")
        
        # Limit number of tiles to display
        display_tiles = tiles[:max_display]
        
        # Calculate grid size
        grid_size = int(np.ceil(np.sqrt(len(display_tiles))))
        
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
        fig.suptitle(f'Colombo Satellite Image Tiles (showing {len(display_tiles)}/{len(tiles)})', 
                     fontsize=16, fontweight='bold')
        
        for i in range(grid_size * grid_size):
            row = i // grid_size
            col = i % grid_size
            
            if grid_size == 1:
                ax = axes
            elif len(axes.shape) == 1:
                ax = axes[i]
            else:
                ax = axes[row, col]
            
            if i < len(display_tiles):
                # Display tile
                tile = display_tiles[i]
                ax.imshow(tile)
                ax.set_title(f'Tile {i}', fontsize=8)
                ax.axis('off')
            else:
                ax.axis('off')
        
        plt.tight_layout()
        overview_path = self.output_dir / "tile_overview.png"
        plt.savefig(overview_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ Overview saved: {overview_path}")
        return str(overview_path)
    
    def process_image(self, image_path, prefix="colombo_tile"):
        """Complete processing pipeline for a Colombo satellite image."""
        print(f"\n🚀 PROCESSING COLOMBO SATELLITE IMAGE")
        print("=" * 50)
        
        # Load image
        image = self.load_image(image_path)
        
        # Split into tiles
        tiles, tile_metadata = self.split_into_tiles(image)
        
        # Preprocess tiles
        processed_tiles = self.preprocess_tiles(tiles)
        
        # Save tiles
        saved_files, metadata_file = self.save_tiles(processed_tiles, tile_metadata, prefix)
        
        # Create overview
        overview_path = self.create_tile_overview(processed_tiles, tile_metadata)
        
        # Summary
        print(f"\n✅ PROCESSING COMPLETE!")
        print("=" * 50)
        print(f"📊 Summary:")
        print(f"   Input image: {image_path}")
        print(f"   Original size: {image.shape[1]}x{image.shape[0]} pixels")
        print(f"   Tiles created: {len(processed_tiles)}")
        print(f"   Tile size: {self.tile_size[0]}x{self.tile_size[1]} pixels")
        print(f"   Output directory: {self.output_dir}")
        print(f"   Tiles saved to: {self.tiles_dir}")
        print(f"   Metadata: {metadata_file}")
        print(f"   Overview: {overview_path}")
        
        return {
            'tiles_saved': len(saved_files),
            'metadata_file': metadata_file,
            'overview_path': overview_path,
            'tile_files': saved_files
        }


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description='Process Colombo satellite images for slum detection')
    parser.add_argument('--image', required=True, help='Path to input satellite image')
    parser.add_argument('--output', default='colombo', help='Output directory')
    parser.add_argument('--tile-size', type=int, default=120, help='Tile size (will create square tiles)')
    parser.add_argument('--overlap', type=int, default=0, help='Overlap between tiles in pixels')
    parser.add_argument('--prefix', default='colombo_tile', help='Prefix for tile filenames')
    
    args = parser.parse_args()
    
    # Create processor
    processor = ColomboImageProcessor(
        output_dir=args.output,
        tile_size=(args.tile_size, args.tile_size),
        overlap=args.overlap
    )
    
    # Process image
    try:
        results = processor.process_image(args.image, args.prefix)
        print(f"\n🎉 Successfully processed {results['tiles_saved']} tiles!")
    except Exception as e:
        print(f"\n❌ Processing failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
