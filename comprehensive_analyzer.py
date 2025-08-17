#!/usr/bin/env python3
"""
Comprehensive Dataset Analysis - Find ALL class meanings
"""

import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
import matplotlib.pyplot as plt

def comprehensive_analysis():
    """Complete analysis of all classes across entire dataset"""
    
    print("COMPREHENSIVE DATASET ANALYSIS")
    print("=" * 60)
    
    # Analyze all splits
    all_class_data = {}
    file_analysis = {}
    
    for split in ['train', 'val', 'test']:
        images_dir = Path(f"data/{split}/images")
        masks_dir = Path(f"data/{split}/masks")
        
        if not masks_dir.exists():
            continue
            
        print(f"\nAnalyzing {split} split...")
        mask_files = list(masks_dir.glob("*.png"))
        
        split_stats = {}
        
        for mask_path in tqdm(mask_files[:500], desc=f"{split}"):  # Sample 500 per split
            img_path = images_dir / (mask_path.stem + ".tif")
            
            if img_path.exists():
                image = cv2.imread(str(img_path))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                
                unique_classes = np.unique(mask)
                file_analysis[mask_path.name] = {
                    'classes': unique_classes.tolist(),
                    'split': split
                }
                
                # Analyze each class
                for cls in unique_classes:
                    class_pixels = image[mask == cls]
                    if len(class_pixels) > 50:
                        
                        if cls not in all_class_data:
                            all_class_data[cls] = {
                                'rgb_values': [],
                                'pixel_counts': [],
                                'files': [],
                                'splits': []
                            }
                        
                        mean_rgb = np.mean(class_pixels, axis=0)
                        pixel_count = len(class_pixels)
                        
                        all_class_data[cls]['rgb_values'].append(mean_rgb)
                        all_class_data[cls]['pixel_counts'].append(pixel_count)
                        all_class_data[cls]['files'].append(mask_path.name)
                        all_class_data[cls]['splits'].append(split)
    
    # Comprehensive analysis
    print(f"\nCOMPREHENSIVE CLASS ANALYSIS:")
    print("=" * 50)
    
    class_interpretations = {}
    
    for cls in sorted(all_class_data.keys()):
        rgb_values = np.array(all_class_data[cls]['rgb_values'])
        pixel_counts = all_class_data[cls]['pixel_counts']
        
        avg_rgb = np.mean(rgb_values, axis=0)
        std_rgb = np.std(rgb_values, axis=0)
        total_samples = len(rgb_values)
        total_pixels = sum(pixel_counts)
        
        r, g, b = avg_rgb
        brightness = np.mean(avg_rgb)
        color_variance = np.mean(std_rgb)
        
        print(f"\nClass {cls}:")
        print(f"  Samples: {total_samples}")
        print(f"  Total pixels: {total_pixels:,}")
        print(f"  Avg RGB: ({r:.1f}, {g:.1f}, {b:.1f})")
        print(f"  Brightness: {brightness:.1f}")
        print(f"  Color variance: {color_variance:.1f}")
        
        # Advanced classification logic
        interpretation = classify_land_use(avg_rgb, std_rgb, brightness, color_variance, cls)
        class_interpretations[cls] = interpretation
        
        print(f"  -> {interpretation}")
        
        # Show sample files
        sample_files = all_class_data[cls]['files'][:3]
        print(f"  Sample files: {sample_files}")
    
    # Find slum patterns
    print(f"\nSLUM IDENTIFICATION:")
    print("-" * 30)
    
    slum_classes = []
    water_classes = []
    vegetation_classes = []
    building_classes = []
    
    for cls, interpretation in class_interpretations.items():
        if 'SLUM' in interpretation or 'INFORMAL' in interpretation:
            slum_classes.append(cls)
        elif 'WATER' in interpretation:
            water_classes.append(cls)
        elif 'VEGETATION' in interpretation or 'GREEN' in interpretation:
            vegetation_classes.append(cls)
        elif 'BUILDING' in interpretation or 'CONCRETE' in interpretation:
            building_classes.append(cls)
    
    print(f"Slum classes: {slum_classes}")
    print(f"Water classes: {water_classes}")
    print(f"Vegetation classes: {vegetation_classes}")
    print(f"Building classes: {building_classes}")
    
    # Save comprehensive results
    results = {
        'class_interpretations': {str(k): v for k, v in class_interpretations.items()},
        'class_statistics': {
            str(cls): {
                'avg_rgb': [float(x) for x in np.mean(all_class_data[cls]['rgb_values'], axis=0)],
                'std_rgb': [float(x) for x in np.std(all_class_data[cls]['rgb_values'], axis=0)],
                'sample_count': len(all_class_data[cls]['rgb_values']),
                'total_pixels': int(sum(all_class_data[cls]['pixel_counts'])),
                'brightness': float(np.mean(np.mean(all_class_data[cls]['rgb_values'], axis=0))),
                'sample_files': all_class_data[cls]['files'][:5]
            } for cls in all_class_data.keys()
        },
        'land_use_categories': {
            'slums': [int(x) for x in slum_classes],
            'water': [int(x) for x in water_classes],
            'vegetation': [int(x) for x in vegetation_classes],
            'buildings': [int(x) for x in building_classes]
        },
        'file_analysis': file_analysis
    }
    
    with open('comprehensive_analysis.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nComprehensive analysis saved to: comprehensive_analysis.json")
    return results

def classify_land_use(avg_rgb, std_rgb, brightness, color_variance, cls):
    """Advanced land use classification"""
    
    r, g, b = avg_rgb
    
    # Known patterns from user confirmation
    if cls == 233:
        return "PRIMARY SLUMS (confirmed)"
    elif cls == 111:
        return "SECONDARY SLUMS (confirmed)"
    
    # Color-based classification
    if cls == 0:
        return "BACKGROUND/VOID"
    elif b > r + 25 and b > g + 20 and b > 100:
        return "WATER BODIES"
    elif g > r + 25 and g > b + 20 and g > 100:
        return "VEGETATION/FORESTS"
    elif brightness > 160 and color_variance < 15:
        return "CONCRETE/FORMAL BUILDINGS"
    elif brightness < 70:
        return "SHADOWS/ROADS/DARK SURFACES"
    elif 120 <= brightness <= 160 and color_variance > 20:
        return "MIXED URBAN/RESIDENTIAL"
    elif 80 <= brightness <= 130 and 10 <= color_variance <= 25:
        return "INFORMAL SETTLEMENTS (potential slums)"
    elif brightness > 130:
        return "BRIGHT URBAN/COMMERCIAL"
    else:
        return "MIXED DEVELOPMENT"

if __name__ == "__main__":
    comprehensive_analysis()