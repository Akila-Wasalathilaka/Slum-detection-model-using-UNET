#!/usr/bin/env python3
"""
Dataset Class Analysis - Identify Slum vs Non-Slum Classes
Analyzes the mask files to understand class distribution and characteristics
"""

import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import json
from collections import defaultdict
import cv2

def analyze_mask_values(mask_path):
    """Analyze unique values in a mask to understand classes"""
    try:
        mask = Image.open(mask_path)
        mask_array = np.array(mask)
        
        # Get unique values
        unique_values = np.unique(mask_array)
        value_counts = {}
        
        for val in unique_values:
            count = np.sum(mask_array == val)
            percentage = (count / mask_array.size) * 100
            value_counts[int(val)] = {
                'count': int(count),
                'percentage': float(percentage)
            }
        
        return value_counts, mask_array.shape
    except Exception as e:
        print(f"Error analyzing {mask_path}: {e}")
        return None, None

def analyze_dataset_classes():
    """Comprehensive analysis of dataset classes"""
    
    print("🔍 DATASET CLASS ANALYSIS")
    print("=" * 60)
    
    # Directories to analyze
    mask_dirs = [
        "data/train/masks",
        "data/val/masks", 
        "data/test/masks"
    ]
    
    all_class_info = {}
    global_value_counts = defaultdict(int)
    total_pixels = 0
    
    for mask_dir in mask_dirs:
        if not os.path.exists(mask_dir):
            print(f"❌ Directory not found: {mask_dir}")
            continue
            
        print(f"\n📁 Analyzing: {mask_dir}")
        mask_files = [f for f in os.listdir(mask_dir) if f.endswith(('.png', '.jpg', '.tif'))]
        
        if not mask_files:
            print(f"   ⚠️ No mask files found")
            continue
            
        print(f"   📊 Found {len(mask_files)} mask files")
        
        dir_value_counts = defaultdict(int)
        dir_total_pixels = 0
        sample_shapes = []
        
        # Analyze first 10 masks in detail
        print(f"   🔍 Analyzing first 10 masks in detail...")
        
        for i, mask_file in enumerate(mask_files[:10]):
            mask_path = os.path.join(mask_dir, mask_file)
            value_counts, shape = analyze_mask_values(mask_path)
            
            if value_counts:
                print(f"      {mask_file}: Shape {shape}")
                for val, info in value_counts.items():
                    print(f"         Value {val}: {info['count']} pixels ({info['percentage']:.2f}%)")
                    dir_value_counts[val] += info['count']
                    global_value_counts[val] += info['count']
                
                dir_total_pixels += shape[0] * shape[1]
                total_pixels += shape[0] * shape[1]
                sample_shapes.append(shape)
        
        # Quick analysis of all masks for value distribution
        print(f"   📈 Quick analysis of all {len(mask_files)} masks...")
        all_values = set()
        
        for mask_file in mask_files:
            mask_path = os.path.join(mask_dir, mask_file)
            try:
                mask = Image.open(mask_path)
                mask_array = np.array(mask)
                unique_vals = np.unique(mask_array)
                all_values.update(unique_vals)
            except:
                continue
        
        print(f"   🎯 Unique values found: {sorted(all_values)}")
        
        all_class_info[mask_dir] = {
            'file_count': len(mask_files),
            'unique_values': sorted(all_values),
            'sample_shapes': sample_shapes,
            'total_pixels_analyzed': dir_total_pixels
        }
    
    # Global analysis
    print(f"\n🌍 GLOBAL DATASET ANALYSIS")
    print("=" * 40)
    
    all_global_values = sorted(global_value_counts.keys())
    print(f"📊 All unique values across dataset: {all_global_values}")
    
    # Determine class mapping
    print(f"\n🏷️ CLASS IDENTIFICATION:")
    
    if len(all_global_values) == 2:
        if 0 in all_global_values and 255 in all_global_values:
            class_mapping = {0: "Non-Slum (Background)", 255: "Slum"}
            print("   ✅ Binary classification detected!")
            print("   🏠 Class 0 (Value 0): Non-Slum Areas (Background)")
            print("   🏘️ Class 1 (Value 255): Slum Areas")
        elif 0 in all_global_values and 1 in all_global_values:
            class_mapping = {0: "Non-Slum (Background)", 1: "Slum"}
            print("   ✅ Binary classification detected!")
            print("   🏠 Class 0 (Value 0): Non-Slum Areas (Background)")
            print("   🏘️ Class 1 (Value 1): Slum Areas")
        else:
            class_mapping = {}
            for i, val in enumerate(all_global_values):
                class_mapping[val] = f"Class_{i}"
            print(f"   ⚠️ Unusual value range: {all_global_values}")
    else:
        class_mapping = {}
        for i, val in enumerate(all_global_values):
            if val == 0:
                class_mapping[val] = "Background"
            else:
                class_mapping[val] = f"Class_{i}"
        print(f"   🔍 Multi-class or unusual classification: {len(all_global_values)} classes")
    
    # Calculate class distribution
    print(f"\n📊 CLASS DISTRIBUTION:")
    total_analyzed_pixels = sum(global_value_counts.values())
    
    for val in sorted(global_value_counts.keys()):
        count = global_value_counts[val]
        percentage = (count / total_analyzed_pixels) * 100
        class_name = class_mapping.get(val, f"Unknown_{val}")
        print(f"   {class_name} (Value {val}): {count:,} pixels ({percentage:.2f}%)")
    
    # Class imbalance analysis
    print(f"\n⚖️ CLASS IMBALANCE ANALYSIS:")
    if len(all_global_values) == 2:
        val1, val2 = sorted(global_value_counts.keys())
        count1, count2 = global_value_counts[val1], global_value_counts[val2]
        ratio = max(count1, count2) / min(count1, count2)
        
        minority_class = val1 if count1 < count2 else val2
        majority_class = val2 if count1 < count2 else val1
        
        print(f"   📈 Majority Class: {class_mapping.get(majority_class)} ({max(count1, count2):,} pixels)")
        print(f"   📉 Minority Class: {class_mapping.get(minority_class)} ({min(count1, count2):,} pixels)")
        print(f"   ⚖️ Imbalance Ratio: {ratio:.2f}:1")
        
        if ratio > 10:
            print("   🚨 SEVERE class imbalance detected! Consider:")
            print("      - Weighted loss functions")
            print("      - Data augmentation for minority class")
            print("      - Focal loss or similar techniques")
        elif ratio > 3:
            print("   ⚠️ Moderate class imbalance. Consider weighted training.")
        else:
            print("   ✅ Relatively balanced classes.")
    
    return {
        'class_mapping': class_mapping,
        'global_distribution': dict(global_value_counts),
        'directory_info': all_class_info,
        'total_pixels_analyzed': total_analyzed_pixels,
        'unique_values': all_global_values
    }

def visualize_sample_masks():
    """Visualize sample masks to understand classes"""
    
    print(f"\n🖼️ VISUALIZING SAMPLE MASKS")
    print("=" * 40)
    
    # Find some sample masks
    sample_masks = []
    for mask_dir in ["data/train/masks", "data/val/masks"]:
        if os.path.exists(mask_dir):
            mask_files = [f for f in os.listdir(mask_dir) if f.endswith(('.png', '.jpg', '.tif'))]
            if mask_files:
                # Get first 3 masks
                for mask_file in mask_files[:3]:
                    mask_path = os.path.join(mask_dir, mask_file)
                    sample_masks.append((mask_path, mask_file))
    
    if not sample_masks:
        print("❌ No sample masks found for visualization")
        return
    
    # Create visualization
    fig, axes = plt.subplots(2, len(sample_masks), figsize=(15, 8))
    if len(sample_masks) == 1:
        axes = axes.reshape(-1, 1)
    
    for i, (mask_path, mask_name) in enumerate(sample_masks):
        try:
            # Load corresponding image if exists
            img_path = mask_path.replace('/masks/', '/images/').replace('_mask', '').replace('.png', '.tif')
            if not os.path.exists(img_path):
                img_path = mask_path.replace('/masks/', '/images/')
            
            if os.path.exists(img_path):
                image = Image.open(img_path)
                axes[0, i].imshow(image)
                axes[0, i].set_title(f'Image: {mask_name}')
                axes[0, i].axis('off')
            else:
                axes[0, i].text(0.5, 0.5, 'Image\nNot Found', ha='center', va='center', transform=axes[0, i].transAxes)
                axes[0, i].set_title(f'Image: {mask_name}')
            
            # Load mask
            mask = Image.open(mask_path)
            mask_array = np.array(mask)
            
            # Show mask with colormap
            im = axes[1, i].imshow(mask_array, cmap='hot')
            axes[1, i].set_title(f'Mask: {mask_name}\nValues: {np.unique(mask_array)}')
            axes[1, i].axis('off')
            
            # Add colorbar
            plt.colorbar(im, ax=axes[1, i], fraction=0.046, pad=0.04)
            
        except Exception as e:
            print(f"Error visualizing {mask_path}: {e}")
    
    plt.tight_layout()
    plt.savefig('sample_masks_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("✅ Sample visualization saved as 'sample_masks_analysis.png'")

if __name__ == "__main__":
    print("🚀 Starting comprehensive dataset class analysis...")
    
    # Analyze classes
    analysis_results = analyze_dataset_classes()
    
    # Visualize samples
    visualize_sample_masks()
    
    # Save results
    with open('class_analysis_results.json', 'w') as f:
        json.dump(analysis_results, f, indent=2)
    
    print(f"\n💾 Analysis results saved to 'class_analysis_results.json'")
    print("🎉 Class analysis complete!")
