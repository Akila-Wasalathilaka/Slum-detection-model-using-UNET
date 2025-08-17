#!/usr/bin/env python3
"""
Dataset Analyzer - Find actual class meanings
"""

import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json

def analyze_dataset():
    """Analyze dataset to find what each class actually represents"""
    
    # Check all splits to find diverse classes
    all_samples = []
    for split in ['train', 'val', 'test']:
        images_dir = Path(f"data/{split}/images")
        masks_dir = Path(f"data/{split}/masks")
        if masks_dir.exists():
            all_samples.extend([(images_dir, masks_dir, f) for f in masks_dir.glob("*.png")])
    
    # Find samples containing each class across all splits
    target_classes = [0, 105, 109, 111, 158, 200, 233]
    class_samples = {cls: [] for cls in target_classes}
    
    print(f"Searching {len(all_samples)} files across all splits...")
    for images_dir, masks_dir, mask_file in tqdm(all_samples[:2000]):
        mask_path = masks_dir / mask_file.name
        img_path = images_dir / (mask_file.stem + ".tif")
        
        if mask_path.exists():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            unique_classes = np.unique(mask)
            
            for cls in target_classes:
                if cls in unique_classes and len(class_samples[cls]) < 15:
                    class_samples[cls].append((img_path, mask_path))
    
    # Use found samples
    sample_files = []
    for cls, samples in class_samples.items():
        sample_files.extend(samples)
        print(f"Class {cls}: Found {len(samples)} samples")
    
    print(f"\nANALYZING {len(sample_files)} DIVERSE SAMPLES TO IDENTIFY CLASSES")
    print("=" * 60)
    
    class_analysis = {}
    
    for img_path, mask_path in tqdm(sample_files):
        if img_path.exists():
            image = cv2.imread(str(img_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            
            for cls in np.unique(mask):
                if cls not in class_analysis:
                    class_analysis[cls] = {'rgb_values': [], 'pixel_counts': []}
                
                class_pixels = image[mask == cls]
                if len(class_pixels) > 100:
                    mean_rgb = np.mean(class_pixels, axis=0)
                    class_analysis[cls]['rgb_values'].append(mean_rgb)
                    class_analysis[cls]['pixel_counts'].append(len(class_pixels))
    
    # Analyze results
    print("\nCLASS IDENTIFICATION RESULTS:")
    print("-" * 40)
    
    class_meanings = {}
    
    for cls in sorted(class_analysis.keys()):
        rgb_values = np.array(class_analysis[cls]['rgb_values'])
        avg_rgb = np.mean(rgb_values, axis=0)
        std_rgb = np.std(rgb_values, axis=0)
        total_samples = len(rgb_values)
        
        r, g, b = avg_rgb
        brightness = np.mean(avg_rgb)
        
        print(f"\nClass {cls}:")
        print(f"  RGB: ({r:.1f}, {g:.1f}, {b:.1f})")
        print(f"  Brightness: {brightness:.1f}")
        print(f"  Samples: {total_samples}")
        
        # Classify based on color characteristics
        if cls == 0:
            meaning = "BACKGROUND/VOID"
        elif b > r + 20 and b > g + 15:  # Strong blue
            meaning = "WATER"
        elif g > r + 20 and g > b + 15:  # Strong green
            meaning = "VEGETATION"
        elif brightness > 150:  # Very bright
            meaning = "CONCRETE/FORMAL_BUILDINGS"
        elif brightness < 80:   # Very dark
            meaning = "SHADOWS/ROADS"
        elif 80 <= brightness <= 120:  # Medium brightness, mixed colors
            meaning = "INFORMAL_SETTLEMENTS/SLUMS"
        else:  # Medium-high brightness
            meaning = "MIXED_URBAN/RESIDENTIAL"
        
        class_meanings[cls] = meaning
        print(f"  -> {meaning}")
    
    # Save results
    results = {
        'class_meanings': {str(k): v for k, v in class_meanings.items()},
        'detailed_stats': {
            str(cls): {
                'avg_rgb': [float(x) for x in np.mean(class_analysis[cls]['rgb_values'], axis=0)],
                'sample_count': len(class_analysis[cls]['rgb_values'])
            } for cls in class_analysis.keys()
        }
    }
    
    with open('dataset_class_meanings.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: dataset_class_meanings.json")
    
    # Identify slum classes
    slum_classes = [cls for cls, meaning in class_meanings.items() 
                   if 'SLUM' in meaning or 'INFORMAL' in meaning]
    
    print(f"\nSLUM CLASSES IDENTIFIED: {slum_classes}")
    
    return class_meanings

if __name__ == "__main__":
    analyze_dataset()