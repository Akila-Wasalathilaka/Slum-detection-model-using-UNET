#!/usr/bin/env python3
"""
Class Inspector - Visual Analysis of What Each Class Actually Represents
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import glob

def inspect_classes():
    """Visually inspect what each class actually represents"""
    
    # Find sample images and masks
    images_dir = Path("data/train/images")
    masks_dir = Path("data/train/masks")
    
    if not (images_dir.exists() and masks_dir.exists()):
        print("❌ Data directories not found!")
        return
    
    # Get all classes
    classes_found = {0, 105, 109, 111, 158, 200, 233}
    
    print("VISUAL CLASS INSPECTION")
    print("=" * 50)
    
    # Find images that contain each class
    class_samples = {}
    
    for mask_path in list(masks_dir.glob("*.png"))[:100]:  # Check first 100
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        unique_classes = set(np.unique(mask))
        
        for cls in classes_found:
            if cls in unique_classes and cls not in class_samples:
                img_path = images_dir / (mask_path.stem + ".tif")
                if img_path.exists():
                    class_samples[cls] = (str(img_path), str(mask_path))
                    print(f"Found sample for Class {cls}")
    
    # Create visualization
    fig, axes = plt.subplots(len(class_samples), 3, figsize=(15, 4*len(class_samples)))
    fig.suptitle('Class Visual Inspection - What Each Class Actually Represents', fontsize=16)
    
    for idx, (cls, (img_path, mask_path)) in enumerate(sorted(class_samples.items())):
        # Load image and mask
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # Create class-specific mask
        class_mask = (mask == cls).astype(np.uint8) * 255
        
        # Plot original image
        axes[idx, 0].imshow(image)
        axes[idx, 0].set_title(f'Original Image (Class {cls})')
        axes[idx, 0].axis('off')
        
        # Plot full mask
        axes[idx, 1].imshow(mask, cmap='tab10')
        axes[idx, 1].set_title(f'Full Mask (Classes: {sorted(np.unique(mask))})')
        axes[idx, 1].axis('off')
        
        # Plot class-specific mask
        axes[idx, 2].imshow(class_mask, cmap='Reds')
        axes[idx, 2].set_title(f'Class {cls} Only')
        axes[idx, 2].axis('off')
        
        # Analyze what this class represents
        class_pixels = np.sum(mask == cls)
        total_pixels = mask.size
        percentage = (class_pixels / total_pixels) * 100
        
        print(f"Class {cls}: {class_pixels:,} pixels ({percentage:.1f}% of this image)")
    
    plt.tight_layout()
    plt.savefig('class_visual_inspection.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Additional analysis - color analysis for each class
    print("\nCOLOR ANALYSIS BY CLASS")
    print("=" * 30)
    
    for cls, (img_path, mask_path) in sorted(class_samples.items()):
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # Get pixels for this class
        class_pixels = image[mask == cls]
        
        if len(class_pixels) > 0:
            mean_color = np.mean(class_pixels, axis=0)
            std_color = np.std(class_pixels, axis=0)
            
            print(f"Class {cls}:")
            print(f"  Mean RGB: ({mean_color[0]:.1f}, {mean_color[1]:.1f}, {mean_color[2]:.1f})")
            print(f"  Std RGB:  ({std_color[0]:.1f}, {std_color[1]:.1f}, {std_color[2]:.1f})")
            
            # Interpret colors
            if mean_color[2] > mean_color[0] and mean_color[2] > mean_color[1]:
                print(f"  -> Likely: WATER/BLUE areas")
            elif mean_color[1] > mean_color[0] and mean_color[1] > mean_color[2]:
                print(f"  -> Likely: VEGETATION/GREEN areas")
            elif np.all(mean_color > 150):
                print(f"  -> Likely: CONCRETE/BRIGHT buildings")
            elif np.all(mean_color < 100):
                print(f"  -> Likely: SHADOWS/DARK areas")
            else:
                print(f"  -> Likely: MIXED/RESIDENTIAL areas")
            print()

if __name__ == "__main__":
    inspect_classes()