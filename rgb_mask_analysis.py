#!/usr/bin/env python3
"""
Dataset Class Analysis - CORRECTED
Key Discovery: This is actually RGB mask data, not grayscale!
"""

import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import json

def analyze_rgb_masks():
    """Analyze RGB masks to understand the true class structure"""
    
    print("🚨 IMPORTANT DISCOVERY: RGB MASKS DETECTED!")
    print("=" * 60)
    print("Your masks are RGB (120x120x3) not grayscale!")
    print("This means the classes are encoded as RGB colors, not single values.")
    print()
    
    # Sample some masks to understand RGB patterns
    mask_dir = "data/train/masks"
    mask_files = [f for f in os.listdir(mask_dir) if f.endswith('.png')][:5]
    
    print("🔍 ANALYZING RGB COLOR PATTERNS:")
    print("-" * 40)
    
    unique_colors = set()
    
    for i, mask_file in enumerate(mask_files):
        mask_path = os.path.join(mask_dir, mask_file)
        mask = Image.open(mask_path)
        mask_array = np.array(mask)
        
        print(f"\n📄 {mask_file}:")
        print(f"   Shape: {mask_array.shape}")
        
        # Get unique RGB combinations
        reshaped = mask_array.reshape(-1, 3)
        unique_rgb = np.unique(reshaped, axis=0)
        
        print(f"   Unique RGB colors found:")
        for rgb in unique_rgb:
            r, g, b = rgb
            count = np.sum(np.all(reshaped == rgb, axis=1))
            percentage = (count / len(reshaped)) * 100
            print(f"     RGB({r:3d}, {g:3d}, {b:3d}): {count:6d} pixels ({percentage:5.1f}%)")
            unique_colors.add(tuple(rgb))
    
    print(f"\n🎨 ALL UNIQUE COLORS ACROSS SAMPLES:")
    print("-" * 40)
    for color in sorted(unique_colors):
        r, g, b = color
        print(f"RGB({r:3d}, {g:3d}, {b:3d})")
    
    return list(unique_colors)

def detailed_color_analysis():
    """More detailed analysis of what each color represents"""
    
    print(f"\n🔍 DETAILED COLOR ANALYSIS")
    print("=" * 40)
    
    # Analyze more masks to get better statistics
    mask_dir = "data/train/masks"
    mask_files = [f for f in os.listdir(mask_dir) if f.endswith('.png')]
    
    # Sample 20 masks for analysis
    sample_files = mask_files[::len(mask_files)//20][:20]
    
    color_stats = {}
    total_pixels = 0
    
    for mask_file in sample_files:
        mask_path = os.path.join(mask_dir, mask_file)
        mask = Image.open(mask_path)
        mask_array = np.array(mask)
        
        reshaped = mask_array.reshape(-1, 3)
        unique_rgb, counts = np.unique(reshaped, axis=0, return_counts=True)
        
        for rgb, count in zip(unique_rgb, counts):
            color_key = tuple(rgb)
            if color_key not in color_stats:
                color_stats[color_key] = 0
            color_stats[color_key] += count
            total_pixels += count
    
    print(f"📊 COLOR DISTRIBUTION (from {len(sample_files)} sample masks):")
    print("-" * 50)
    
    # Sort by frequency
    sorted_colors = sorted(color_stats.items(), key=lambda x: x[1], reverse=True)
    
    class_mapping = {}
    for i, (color, count) in enumerate(sorted_colors):
        r, g, b = color
        percentage = (count / total_pixels) * 100
        
        # Try to interpret the colors
        if r == g == b:  # Grayscale
            if r == 0:
                class_name = "🌑 Background/No-Slum"
            elif r == 255:
                class_name = "🏘️ Slum Areas" 
            else:
                class_name = f"🔘 Class_{i} (Gray)"
        else:
            # Different RGB values - could be specific classes
            if r > g and r > b:
                class_name = f"🔴 Class_{i} (Red-dominant)"
            elif g > r and g > b:
                class_name = f"🟢 Class_{i} (Green-dominant)"
            elif b > r and b > g:
                class_name = f"🔵 Class_{i} (Blue-dominant)"
            else:
                class_name = f"🎨 Class_{i} (Mixed)"
        
        class_mapping[color] = class_name
        
        print(f"RGB({r:3d}, {g:3d}, {b:3d}): {count:8,} pixels ({percentage:5.2f}%) - {class_name}")
    
    # Analyze class balance
    print(f"\n⚖️ CLASS BALANCE ANALYSIS:")
    print("-" * 30)
    
    if len(sorted_colors) == 2:
        color1, count1 = sorted_colors[0]
        color2, count2 = sorted_colors[1]
        ratio = max(count1, count2) / min(count1, count2)
        
        print(f"✅ Binary classification detected!")
        print(f"   Majority class: {class_mapping[color1]} ({count1:,} pixels)")
        print(f"   Minority class: {class_mapping[color2]} ({count2:,} pixels)")
        print(f"   Imbalance ratio: {ratio:.2f}:1")
        
        if ratio > 10:
            print("   🚨 SEVERE imbalance - need special handling!")
        elif ratio > 3:
            print("   ⚠️ Moderate imbalance - consider weighted loss")
        else:
            print("   ✅ Well balanced classes")
            
    elif len(sorted_colors) == 3:
        print(f"✅ Multi-class (3 classes) detected!")
        for i, (color, count) in enumerate(sorted_colors):
            print(f"   Class {i+1}: {class_mapping[color]} ({count:,} pixels)")
    
    return class_mapping, sorted_colors

def create_proper_visualization():
    """Create proper visualization of RGB masks"""
    
    print(f"\n🖼️ CREATING PROPER MASK VISUALIZATION")
    print("=" * 40)
    
    # Get sample images and masks
    img_dir = "data/train/images"
    mask_dir = "data/train/masks"
    
    # Find matching image-mask pairs
    img_files = [f for f in os.listdir(img_dir) if f.endswith(('.tif', '.png', '.jpg'))]
    mask_files = [f for f in os.listdir(mask_dir) if f.endswith('.png')]
    
    # Match first 4 pairs
    sample_pairs = []
    for img_file in img_files[:4]:
        # Try to find corresponding mask
        base_name = img_file.split('.')[0]
        for mask_file in mask_files:
            if base_name in mask_file:
                sample_pairs.append((
                    os.path.join(img_dir, img_file),
                    os.path.join(mask_dir, mask_file)
                ))
                break
    
    if not sample_pairs:
        print("❌ Could not find matching image-mask pairs")
        return
    
    fig, axes = plt.subplots(3, len(sample_pairs), figsize=(16, 12))
    
    for i, (img_path, mask_path) in enumerate(sample_pairs):
        try:
            # Load and display original image
            image = Image.open(img_path)
            axes[0, i].imshow(image)
            axes[0, i].set_title(f'Original Image {i+1}')
            axes[0, i].axis('off')
            
            # Load and display RGB mask as-is
            rgb_mask = Image.open(mask_path)
            axes[1, i].imshow(rgb_mask)
            axes[1, i].set_title(f'RGB Mask {i+1}')
            axes[1, i].axis('off')
            
            # Convert to grayscale for class visualization
            mask_array = np.array(rgb_mask)
            # Simple conversion: if all channels are same, use that value
            # Otherwise, convert to grayscale
            if len(mask_array.shape) == 3:
                # Check if it's actually grayscale stored as RGB
                r, g, b = mask_array[:,:,0], mask_array[:,:,1], mask_array[:,:,2]
                if np.array_equal(r, g) and np.array_equal(g, b):
                    gray_mask = r  # It's grayscale stored as RGB
                else:
                    # True RGB - convert differently
                    gray_mask = np.mean(mask_array, axis=2)
            else:
                gray_mask = mask_array
            
            im = axes[2, i].imshow(gray_mask, cmap='jet')
            axes[2, i].set_title(f'Class Map {i+1}')
            axes[2, i].axis('off')
            
            # Add colorbar
            plt.colorbar(im, ax=axes[2, i], fraction=0.046, pad=0.04)
            
        except Exception as e:
            print(f"Error processing pair {i}: {e}")
    
    plt.tight_layout()
    plt.savefig('rgb_mask_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("✅ RGB mask visualization saved as 'rgb_mask_analysis.png'")

if __name__ == "__main__":
    print("🚀 CORRECTED DATASET CLASS ANALYSIS")
    print("🎯 Focus: Understanding RGB mask structure")
    print()
    
    # Step 1: Understand RGB patterns
    unique_colors = analyze_rgb_masks()
    
    # Step 2: Detailed color analysis
    class_mapping, color_distribution = detailed_color_analysis()
    
    # Step 3: Proper visualization
    create_proper_visualization()
    
    # Summary
    print(f"\n🎊 SUMMARY FINDINGS:")
    print("=" * 50)
    print(f"📊 Dataset contains {len(unique_colors)} unique RGB colors")
    print(f"📏 Mask size: 120x120 pixels")
    print(f"🎨 Mask format: RGB (3 channels)")
    print(f"📁 Training masks: {len([f for f in os.listdir('data/train/masks') if f.endswith('.png')])}")
    print(f"📁 Validation masks: {len([f for f in os.listdir('data/val/masks') if f.endswith('.png')])}")
    print(f"📁 Test masks: {len([f for f in os.listdir('data/test/masks') if f.endswith('.png')])}")
    
    if len(unique_colors) <= 3:
        print(f"\n💡 RECOMMENDATION:")
        print(f"   This appears to be a multi-class segmentation problem")
        print(f"   with {len(unique_colors)} classes encoded as RGB colors.")
        print(f"   For training, convert RGB masks to class indices (0, 1, 2, etc.)")
    
    print(f"\n✅ Analysis complete!")
