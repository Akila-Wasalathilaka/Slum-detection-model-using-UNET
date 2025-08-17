#!/usr/bin/env python3
"""
Simple Class Checker - What do these classes actually represent?
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def check_classes():
    """Comprehensive analysis of what each class represents"""
    
    images_dir = Path("data/train/images")
    masks_dir = Path("data/train/masks")
    
    if not (images_dir.exists() and masks_dir.exists()):
        print("Data directories not found!")
        return
    
    # Analyze ALL samples for complete understanding
    mask_files = list(masks_dir.glob("*.png"))
    print(f"Analyzing {len(mask_files)} samples...")
    
    print("COMPREHENSIVE URBAN LAND USE ANALYSIS")
    print("=" * 50)
    
    # Collect statistics for each class
    class_stats = {}
    
    from tqdm import tqdm
    
    for mask_path in tqdm(mask_files, desc="Processing"):
        img_path = images_dir / (mask_path.stem + ".tif")
        
        if img_path.exists():
            image = cv2.imread(str(img_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            
            unique_classes = np.unique(mask)
            
            for cls in unique_classes:
                class_pixels = image[mask == cls]
                if len(class_pixels) > 50:
                    if cls not in class_stats:
                        class_stats[cls] = {'colors': [], 'pixel_counts': [], 'files': []}
                    
                    mean_rgb = np.mean(class_pixels, axis=0)
                    class_stats[cls]['colors'].append(mean_rgb)
                    class_stats[cls]['pixel_counts'].append(len(class_pixels))
                    class_stats[cls]['files'].append(mask_path.name)
    
    # Analyze and classify each class
    print("\nCLASS INTERPRETATION RESULTS:")
    print("-" * 40)
    
    class_names = {}
    
    for cls in sorted(class_stats.keys()):
        colors = np.array(class_stats[cls]['colors'])
        avg_color = np.mean(colors, axis=0)
        std_color = np.std(colors, axis=0)
        total_pixels = sum(class_stats[cls]['pixel_counts'])
        
        print(f"\nClass {cls}:")
        print(f"  Avg RGB: ({avg_color[0]:.1f}, {avg_color[1]:.1f}, {avg_color[2]:.1f})")
        print(f"  Std RGB: ({std_color[0]:.1f}, {std_color[1]:.1f}, {std_color[2]:.1f})")
        print(f"  Files: {len(colors)}, Total pixels: {total_pixels:,}")
        print(f"  Frequency: {len(colors)/len(mask_files)*100:.1f}% of images")
        
        # Advanced classification
        r, g, b = avg_color
        brightness = np.mean(avg_color)
        
        if cls == 0:
            class_type = "BACKGROUND/VOID"
        elif b > r + 15 and b > g + 10 and b > 100:
            class_type = "WATER BODIES"
        elif g > r + 15 and g > b + 10 and g > 90:
            class_type = "VEGETATION/PARKS"
        elif brightness > 140 and std_color.mean() < 20:
            class_type = "CONCRETE/FORMAL BUILDINGS"
        elif brightness < 90:
            class_type = "SHADOWS/DARK AREAS"
        elif 90 <= brightness <= 120 and std_color.mean() > 15:
            class_type = "INFORMAL SETTLEMENTS/SLUMS"
        elif brightness > 120:
            class_type = "MIXED URBAN/RESIDENTIAL"
        else:
            class_type = "MIXED DEVELOPMENT"
        
        class_names[cls] = class_type
        print(f"  -> INTERPRETED AS: {class_type}")
    
    return class_names
            
    # Save comprehensive results
    results = {
        'class_interpretations': {str(k): v for k, v in class_names.items()},
        'class_statistics': {
            str(cls): {
                'avg_rgb': [float(x) for x in np.mean(class_stats[cls]['colors'], axis=0)],
                'total_pixels': int(sum(class_stats[cls]['pixel_counts'])),
                'file_count': len(class_stats[cls]['colors']),
                'frequency_percent': round(len(class_stats[cls]['colors'])/len(mask_files)*100, 2)
            } for cls in class_stats.keys()
        },
        'dataset_info': {
            'total_files_analyzed': len(mask_files),
            'classes_found': sorted([int(x) for x in class_stats.keys()])
        }
    }
    
    import json
    with open('complete_urban_analysis.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nComplete analysis saved to: complete_urban_analysis.json")
    return class_names, results

if __name__ == "__main__":
    check_classes()