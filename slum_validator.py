#!/usr/bin/env python3
"""
Slum Validator - Check specific slum tiles and find patterns
"""

import cv2
import numpy as np
from pathlib import Path
import json

def validate_slum_tiles():
    """Check the specific slum tiles mentioned by user"""
    
    # Known slum tiles from user
    slum_tiles = ["tile_1.16_34.tif", "tile_1.16_32.tif", "tile_1.16_33.tif"]
    
    print("VALIDATING KNOWN SLUM TILES")
    print("=" * 40)
    
    slum_class_data = {}
    
    # Check all splits for these tiles
    for split in ['train', 'val', 'test']:
        images_dir = Path(f"data/{split}/images")
        masks_dir = Path(f"data/{split}/masks")
        
        if not images_dir.exists():
            continue
            
        print(f"\nChecking {split} split...")
        
        for tile_name in slum_tiles:
            img_path = images_dir / tile_name
            mask_path = masks_dir / (tile_name.replace('.tif', '.png'))
            
            if img_path.exists() and mask_path.exists():
                print(f"  Found: {tile_name}")
                
                # Load and analyze
                image = cv2.imread(str(img_path))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                
                unique_classes = np.unique(mask)
                print(f"    Classes in this slum: {unique_classes}")
                
                # Analyze each class in this confirmed slum
                for cls in unique_classes:
                    class_pixels = image[mask == cls]
                    if len(class_pixels) > 100:
                        mean_rgb = np.mean(class_pixels, axis=0)
                        pixel_count = len(class_pixels)
                        percentage = (pixel_count / mask.size) * 100
                        
                        if cls not in slum_class_data:
                            slum_class_data[cls] = {'rgb_samples': [], 'percentages': []}
                        
                        slum_class_data[cls]['rgb_samples'].append(mean_rgb)
                        slum_class_data[cls]['percentages'].append(percentage)
                        
                        print(f"      Class {cls}: RGB({mean_rgb[0]:.0f},{mean_rgb[1]:.0f},{mean_rgb[2]:.0f}) - {percentage:.1f}% of image")
    
    # Analyze slum characteristics
    print(f"\nSLUM CLASS ANALYSIS FROM CONFIRMED TILES:")
    print("-" * 50)
    
    slum_classes = []
    for cls, data in slum_class_data.items():
        avg_rgb = np.mean(data['rgb_samples'], axis=0)
        avg_percentage = np.mean(data['percentages'])
        sample_count = len(data['rgb_samples'])
        
        print(f"Class {cls}:")
        print(f"  Avg RGB: ({avg_rgb[0]:.1f}, {avg_rgb[1]:.1f}, {avg_rgb[2]:.1f})")
        print(f"  Avg coverage: {avg_percentage:.1f}%")
        print(f"  Samples: {sample_count}")
        
        # Determine if this is likely a slum class
        if avg_percentage > 5:  # Significant coverage in slum images
            slum_classes.append(cls)
            print(f"  -> IDENTIFIED AS SLUM CLASS")
        else:
            print(f"  -> Minor class (background/other)")
        print()
    
    # Save results
    results = {
        'confirmed_slum_tiles': slum_tiles,
        'slum_classes_identified': [int(x) for x in slum_classes],
        'class_analysis': {
            str(cls): {
                'avg_rgb': [float(x) for x in np.mean(data['rgb_samples'], axis=0)],
                'avg_coverage_percent': float(np.mean(data['percentages'])),
                'sample_count': len(data['rgb_samples'])
            } for cls, data in slum_class_data.items()
        }
    }
    
    with open('slum_validation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"CONFIRMED SLUM CLASSES: {slum_classes}")
    print(f"Results saved to: slum_validation_results.json")
    
    return slum_classes

if __name__ == "__main__":
    validate_slum_tiles()