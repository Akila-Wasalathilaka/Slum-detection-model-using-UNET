#!/usr/bin/env python3
"""
COMPREHENSIVE RGB MASK CLASS IDENTIFICATION
This script will identify ALL unique RGB colors across the entire dataset
and map them to semantic classes for slum detection
"""

import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import json
from collections import defaultdict, Counter

def scan_all_masks_for_colors():
    """Scan ALL masks to find every unique RGB color"""
    
    print("🔍 SCANNING ALL MASKS FOR UNIQUE COLORS")
    print("=" * 60)
    
    all_colors = Counter()
    mask_dirs = ["data/train/masks", "data/val/masks", "data/test/masks"]
    
    total_masks = 0
    total_pixels = 0
    
    for mask_dir in mask_dirs:
        if not os.path.exists(mask_dir):
            continue
            
        mask_files = [f for f in os.listdir(mask_dir) if f.endswith('.png')]
        print(f"📁 Scanning {len(mask_files)} masks in {mask_dir}...")
        
        for i, mask_file in enumerate(mask_files):
            if i % 1000 == 0:
                print(f"   Progress: {i}/{len(mask_files)} masks processed...")
                
            mask_path = os.path.join(mask_dir, mask_file)
            try:
                mask = Image.open(mask_path)
                mask_array = np.array(mask)
                
                # Get unique colors in this mask
                reshaped = mask_array.reshape(-1, 3)
                unique_colors, counts = np.unique(reshaped, axis=0, return_counts=True)
                
                for color, count in zip(unique_colors, counts):
                    all_colors[tuple(color)] += count
                    total_pixels += count
                
                total_masks += 1
                
            except Exception as e:
                print(f"   Error processing {mask_file}: {e}")
    
    print(f"✅ Completed scanning {total_masks} masks ({total_pixels:,} total pixels)")
    return all_colors, total_masks, total_pixels

def analyze_color_semantics(color_counts, total_pixels):
    """Analyze what each color might represent semantically"""
    
    print(f"\n🎨 COMPLETE COLOR ANALYSIS")
    print("=" * 50)
    
    # Sort colors by frequency
    sorted_colors = sorted(color_counts.items(), key=lambda x: x[1], reverse=True)
    
    print(f"📊 Found {len(sorted_colors)} unique RGB colors:")
    print("-" * 70)
    
    class_mapping = {}
    
    for i, (color, count) in enumerate(sorted_colors):
        r, g, b = color
        percentage = (count / total_pixels) * 100
        
        # Semantic interpretation based on RGB values
        if r == g == b:  # Grayscale colors
            if r == 0:
                semantic = "🌑 BACKGROUND/WATER (Black)"
                class_name = "background"
            elif r == 255:
                semantic = "☁️ CLOUDS/BRIGHT (White)"
                class_name = "clouds"
            elif 100 <= r <= 150:
                semantic = "🏢 URBAN/CONCRETE (Gray)"
                class_name = "urban"
            elif 50 <= r <= 99:
                semantic = "🏗️ INDUSTRIAL (Dark Gray)"
                class_name = "industrial"
            else:
                semantic = f"🔘 GRAY_CLASS_{i}"
                class_name = f"gray_class_{i}"
        else:
            # Color analysis
            if r > g and r > b and r > 150:  # Red dominant
                if g > 100:  # Reddish-brown
                    semantic = "🧱 SLUM/INFORMAL (Reddish-Brown)"
                    class_name = "slum"
                else:
                    semantic = "🔴 RED_STRUCTURES (Red)"
                    class_name = "red_structures"
            elif g > r and g > b and g > 100:  # Green dominant
                if r < 100 and b < 100:
                    semantic = "🌿 VEGETATION (Green)"
                    class_name = "vegetation"
                else:
                    semantic = "🟢 GREEN_AREAS"
                    class_name = "green_areas"
            elif b > r and b > g and b > 100:  # Blue dominant
                if r < 100 and g < 150:
                    semantic = "💧 WATER_BODIES (Blue)"
                    class_name = "water"
                else:
                    semantic = "🔵 BLUE_STRUCTURES"
                    class_name = "blue_structures"
            elif r > 150 and g > 150 and b < 100:  # Yellow-ish
                semantic = "🟡 ROADS/SAND (Yellow-ish)"
                class_name = "roads"
            elif r > 100 and g > 100 and b > 100:  # Mixed bright
                semantic = "🏠 FORMAL_HOUSING (Mixed)"
                class_name = "formal_housing"
            else:
                semantic = f"🎨 MIXED_CLASS_{i}"
                class_name = f"mixed_class_{i}"
        
        class_mapping[color] = {
            'index': i,
            'semantic': semantic,
            'class_name': class_name,
            'count': count,
            'percentage': percentage
        }
        
        print(f"Class {i:2d}: RGB({r:3d}, {g:3d}, {b:3d}) | {count:10,} pixels ({percentage:6.2f}%) | {semantic}")
    
    return class_mapping, sorted_colors

def create_class_distribution_chart(class_mapping):
    """Create visualization of class distribution"""
    
    print(f"\n📊 CREATING CLASS DISTRIBUTION CHART")
    print("=" * 40)
    
    # Prepare data for plotting
    colors = list(class_mapping.keys())
    percentages = [class_mapping[color]['percentage'] for color in colors]
    labels = [class_mapping[color]['semantic'].split('(')[0].strip() for color in colors]
    rgb_colors = [tuple(np.array(color)/255.0) for color in colors]
    
    # Create pie chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Pie chart
    wedges, texts, autotexts = ax1.pie(percentages, labels=labels, colors=rgb_colors, 
                                       autopct='%1.1f%%', startangle=90)
    ax1.set_title('Class Distribution by Pixel Count', fontsize=14, fontweight='bold')
    
    # Bar chart
    bars = ax2.bar(range(len(colors)), percentages, color=rgb_colors)
    ax2.set_xlabel('Class Index')
    ax2.set_ylabel('Percentage (%)')
    ax2.set_title('Class Distribution (Bar Chart)', fontsize=14, fontweight='bold')
    ax2.set_xticks(range(len(colors)))
    ax2.set_xticklabels([f'C{i}' for i in range(len(colors))])
    
    # Add percentage labels on bars
    for i, (bar, pct) in enumerate(zip(bars, percentages)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{pct:.1f}%', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('class_distribution_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("✅ Class distribution chart saved as 'class_distribution_analysis.png'")

def create_sample_visualizations(class_mapping):
    """Create sample visualizations showing each class"""
    
    print(f"\n🖼️ CREATING SAMPLE CLASS VISUALIZATIONS")
    print("=" * 40)
    
    # Find masks that contain different classes
    mask_dir = "data/train/masks"
    mask_files = [f for f in os.listdir(mask_dir) if f.endswith('.png')]
    
    # Sample some masks to show different classes
    sample_masks = []
    colors_found = set()
    
    for mask_file in mask_files[:50]:  # Check first 50 masks
        mask_path = os.path.join(mask_dir, mask_file)
        try:
            mask = Image.open(mask_path)
            mask_array = np.array(mask)
            reshaped = mask_array.reshape(-1, 3)
            unique_colors = set(tuple(color) for color in np.unique(reshaped, axis=0))
            
            # If this mask has colors we haven't shown yet, include it
            new_colors = unique_colors - colors_found
            if new_colors:
                sample_masks.append((mask_path, mask_file, unique_colors))
                colors_found.update(unique_colors)
                
                if len(sample_masks) >= 6:  # Show 6 diverse examples
                    break
        except:
            continue
    
    if not sample_masks:
        print("❌ Could not find diverse mask samples")
        return
    
    fig, axes = plt.subplots(2, len(sample_masks), figsize=(20, 8))
    
    for i, (mask_path, mask_file, mask_colors) in enumerate(sample_masks):
        try:
            # Load RGB mask
            rgb_mask = Image.open(mask_path)
            rgb_array = np.array(rgb_mask)
            
            # Show original RGB mask
            axes[0, i].imshow(rgb_array)
            axes[0, i].set_title(f'RGB Mask {i+1}\n{mask_file[:15]}...', fontsize=10)
            axes[0, i].axis('off')
            
            # Create class index mask
            class_mask = np.zeros(rgb_array.shape[:2], dtype=np.uint8)
            
            for color in mask_colors:
                if color in class_mapping:
                    class_idx = class_mapping[color]['index']
                    mask = np.all(rgb_array == color, axis=2)
                    class_mask[mask] = class_idx
            
            # Show class mask with colormap
            im = axes[1, i].imshow(class_mask, cmap='tab10', vmin=0, vmax=len(class_mapping)-1)
            
            # Add legend text
            legend_text = []
            for color in mask_colors:
                if color in class_mapping:
                    idx = class_mapping[color]['index']
                    name = class_mapping[color]['class_name']
                    legend_text.append(f'C{idx}:{name[:8]}')
            
            axes[1, i].set_title(f'Classes: {", ".join(legend_text[:3])}', fontsize=9)
            axes[1, i].axis('off')
            
        except Exception as e:
            print(f"Error visualizing {mask_file}: {e}")
    
    plt.tight_layout()
    plt.savefig('sample_class_visualizations.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("✅ Sample visualizations saved as 'sample_class_visualizations.png'")

def generate_training_recommendations(class_mapping):
    """Generate recommendations for model training"""
    
    print(f"\n💡 TRAINING RECOMMENDATIONS")
    print("=" * 50)
    
    num_classes = len(class_mapping)
    
    print(f"📊 Dataset Summary:")
    print(f"   🎯 Number of classes: {num_classes}")
    print(f"   📏 Image size: 120x120 pixels")
    print(f"   🎨 Original format: RGB masks")
    print(f"   🔢 Required format: Class indices (0-{num_classes-1})")
    
    # Analyze class balance
    percentages = [info['percentage'] for info in class_mapping.values()]
    max_pct, min_pct = max(percentages), min(percentages)
    imbalance_ratio = max_pct / min_pct
    
    print(f"\n⚖️ Class Balance:")
    print(f"   📈 Largest class: {max_pct:.2f}%")
    print(f"   📉 Smallest class: {min_pct:.2f}%")
    print(f"   ⚖️ Imbalance ratio: {imbalance_ratio:.2f}:1")
    
    if imbalance_ratio > 20:
        print(f"   🚨 SEVERE class imbalance!")
        print(f"   💡 Use: Focal Loss, Class Weights, Data Augmentation")
    elif imbalance_ratio > 5:
        print(f"   ⚠️ Moderate class imbalance")
        print(f"   💡 Use: Weighted Cross Entropy Loss")
    else:
        print(f"   ✅ Relatively balanced classes")
    
    # Model recommendations
    print(f"\n🧠 Model Architecture Recommendations:")
    print(f"   🏗️ UNET with {num_classes} output channels")
    print(f"   📊 Loss: CrossEntropyLoss or Focal Loss")
    print(f"   📈 Metrics: IoU per class, Overall Accuracy, F1-Score")
    print(f"   🎛️ Data preprocessing: RGB mask → Class indices")
    
    # Generate class weights for training
    total_pixels = sum(info['count'] for info in class_mapping.values())
    class_weights = []
    
    print(f"\n🎛️ Suggested Class Weights for Training:")
    for i, (color, info) in enumerate(class_mapping.items()):
        weight = total_pixels / (num_classes * info['count'])
        class_weights.append(weight)
        print(f"   Class {i} ({info['class_name']}): {weight:.3f}")
    
    return {
        'num_classes': num_classes,
        'class_weights': class_weights,
        'imbalance_ratio': imbalance_ratio,
        'recommendations': {
            'architecture': 'UNET',
            'loss_function': 'CrossEntropyLoss' if imbalance_ratio < 5 else 'FocalLoss',
            'use_class_weights': imbalance_ratio > 3,
            'data_augmentation': imbalance_ratio > 5
        }
    }

if __name__ == "__main__":
    print("🚀 COMPREHENSIVE RGB MASK CLASS IDENTIFICATION")
    print("🎯 Goal: Identify ALL classes for slum detection model")
    print()
    
    # Step 1: Scan all masks for unique colors
    color_counts, total_masks, total_pixels = scan_all_masks_for_colors()
    
    # Step 2: Analyze semantic meaning of colors
    class_mapping, sorted_colors = analyze_color_semantics(color_counts, total_pixels)
    
    # Step 3: Create visualizations
    create_class_distribution_chart(class_mapping)
    create_sample_visualizations(class_mapping)
    
    # Step 4: Generate training recommendations
    training_info = generate_training_recommendations(class_mapping)
    
    # Step 5: Save complete analysis
    complete_analysis = {
        'dataset_info': {
            'total_masks': total_masks,
            'total_pixels': total_pixels,
            'num_classes': len(class_mapping)
        },
        'class_mapping': {str(k): v for k, v in class_mapping.items()},  # Convert tuples to strings for JSON
        'training_recommendations': training_info
    }
    
    with open('complete_class_analysis.json', 'w') as f:
        json.dump(complete_analysis, f, indent=2)
    
    print(f"\n💾 Complete analysis saved to 'complete_class_analysis.json'")
    print(f"🎉 Comprehensive class identification complete!")
    print(f"\n📋 QUICK SUMMARY:")
    print(f"   📊 {len(class_mapping)} classes identified")
    print(f"   📁 {total_masks:,} masks analyzed")
    print(f"   🎯 Ready for multi-class segmentation training!")
