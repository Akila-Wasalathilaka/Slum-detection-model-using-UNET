import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from collections import Counter

def analyze_mask_variations():
    """Analyze the actual variations in mask content"""
    
    print("=" * 60)
    print("ANALYZING MASK VARIATIONS")
    print("=" * 60)
    
    # Check masks that were identified as different
    train_mask_dir = "data/train/masks"
    mask_files = [f for f in os.listdir(train_mask_dir) if f.endswith('.png')]
    
    # Let's sample more masks to find different patterns
    sample_size = min(100, len(mask_files))
    step = len(mask_files) // sample_size
    sample_files = mask_files[::step]
    
    print(f"Analyzing {len(sample_files)} masks for variations...")
    
    # Group masks by their content
    mask_groups = {}
    
    for mask_file in sample_files:
        mask_path = os.path.join(train_mask_dir, mask_file)
        mask = np.array(Image.open(mask_path))
        
        # Create a signature for this mask
        unique_rgb = tuple(sorted([tuple(rgb) for rgb in np.unique(mask.reshape(-1, 3), axis=0)]))
        
        if unique_rgb not in mask_groups:
            mask_groups[unique_rgb] = []
        mask_groups[unique_rgb].append((mask_file, mask))
    
    print(f"\nFound {len(mask_groups)} different mask patterns:")
    
    for i, (rgb_signature, files_and_masks) in enumerate(mask_groups.items()):
        print(f"\nPattern {i+1}: {len(files_and_masks)} masks")
        print(f"  RGB combinations: {rgb_signature}")
        print(f"  Example files: {[f[0] for f in files_and_masks[:3]]}")
        
        # Analyze the first mask in this group
        sample_mask = files_and_masks[0][1]
        for ch in range(3):
            channel_data = sample_mask[:, :, ch]
            unique_vals = np.unique(channel_data)
            value_counts = [(val, np.sum(channel_data == val)) for val in unique_vals]
            print(f"  Channel {ch} ({'RGB'[ch]}): {value_counts}")
    
    return mask_groups

def create_pattern_visualization(mask_groups):
    """Visualize different mask patterns found"""
    
    print(f"\n" + "=" * 60)
    print("CREATING PATTERN VISUALIZATIONS")
    print("=" * 60)
    
    num_patterns = len(mask_groups)
    fig, axes = plt.subplots(num_patterns, 5, figsize=(25, 5*num_patterns))
    
    if num_patterns == 1:
        axes = axes.reshape(1, -1)
    
    for i, (rgb_signature, files_and_masks) in enumerate(mask_groups.items()):
        # Take the first file from this pattern
        mask_file, mask = files_and_masks[0]
        
        # Load corresponding image
        img_file = mask_file.replace('.png', '.tif')
        img_path = f"data/train/images/{img_file}"
        
        if os.path.exists(img_path):
            img = np.array(Image.open(img_path))
            
            # Show original image
            axes[i, 0].imshow(img)
            axes[i, 0].set_title(f"Pattern {i+1}: Image\n{mask_file}")
            axes[i, 0].axis('off')
            
            # Show RGB mask
            axes[i, 1].imshow(mask)
            axes[i, 1].set_title(f"RGB Mask\n{rgb_signature}")
            axes[i, 1].axis('off')
            
            # Show individual channels
            for ch in range(3):
                channel_data = mask[:, :, ch]
                im = axes[i, 2+ch].imshow(channel_data, cmap='gray', vmin=0, vmax=255)
                unique_vals = np.unique(channel_data)
                axes[i, 2+ch].set_title(f"{'RGB'[ch]} Channel\nValues: {unique_vals}")
                axes[i, 2+ch].axis('off')
                plt.colorbar(im, ax=axes[i, 2+ch], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig('mask_patterns.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("Pattern visualization saved as 'mask_patterns.png'")

def determine_class_mapping(mask_groups):
    """Determine the class mapping based on patterns found"""
    
    print(f"\n" + "=" * 60)
    print("DETERMINING CLASS MAPPING")
    print("=" * 60)
    
    patterns = list(mask_groups.keys())
    
    print(f"Found {len(patterns)} distinct patterns:")
    
    # Analyze each pattern to understand what it might represent
    for i, pattern in enumerate(patterns):
        print(f"\nPattern {i+1}: {pattern}")
        
        # Get a sample mask from this pattern
        sample_mask = mask_groups[pattern][0][1]
        
        # If it's a single RGB combination repeated everywhere
        if len(pattern) == 1:
            rgb = pattern[0]
            print(f"  Uniform pattern: RGB{rgb}")
            
            # Try to interpret based on RGB values
            r, g, b = rgb
            if r == g == b:  # Grayscale
                if r < 100:
                    print(f"  Likely: Background/Water (dark)")
                elif r > 200:
                    print(f"  Likely: Built-up/Impervious (bright)")
                else:
                    print(f"  Likely: Vegetation/Other (medium)")
            else:  # Color
                if r > g and r > b:
                    print(f"  Likely: Informal settlements/Slums (red dominant)")
                elif g > r and g > b:
                    print(f"  Likely: Vegetation (green dominant)")
                elif b > r and b > g:
                    print(f"  Likely: Water (blue dominant)")
                else:
                    print(f"  Likely: Mixed/Other class")
        
        # If it has multiple RGB combinations, it's a segmentation mask
        else:
            print(f"  Multi-class pattern with {len(pattern)} classes")
            for j, rgb in enumerate(pattern):
                print(f"    Class {j}: RGB{rgb}")
    
    # Based on your original description, let's create a mapping
    print(f"\n" + "=" * 40)
    print("PROPOSED CLASS MAPPING")
    print("=" * 40)
    
    # Create mapping based on the patterns we found
    class_mapping = {}
    
    if len(patterns) == 1 and len(patterns[0]) == 1:
        # All masks have the same single color
        rgb = patterns[0][0]
        print(f"All masks are uniform with RGB{rgb}")
        print(f"This could indicate:")
        print(f"1. All tiles belong to the same class")
        print(f"2. Masks are not properly annotated")
        print(f"3. This RGB represents a multi-class encoding")
        
        # Check if this might be a multi-class encoding in a single RGB
        r, g, b = rgb
        print(f"\nTrying to decode RGB{rgb} as multi-class:")
        print(f"  Red channel ({r}): Class index {r // 40}")
        print(f"  Green channel ({g}): Class index {g // 40}")
        print(f"  Blue channel ({b}): Class index {b // 40}")
        
        # For your specific case with (40, 120, 240):
        if rgb == (40, 120, 240):
            print(f"\nSpecial case RGB(40, 120, 240) interpretation:")
            print(f"  This might represent 3 different classes in one pixel:")
            print(f"  - Channel R=40: Background/Barren")
            print(f"  - Channel G=120: Vegetation/Built-up")
            print(f"  - Channel B=240: Informal settlements/Slums")
            
            # Create a class mapping where we extract the blue channel as slum class
            class_mapping = {
                "type": "channel_based",
                "slum_channel": 2,  # Blue channel
                "slum_threshold": 200,  # Values above this are slum
                "description": "Blue channel represents slum probability/presence"
            }
    
    else:
        # Multiple patterns - different tiles have different classes
        for i, pattern in enumerate(patterns):
            if len(pattern) == 1:
                rgb = pattern[0]
                # Map based on color characteristics
                r, g, b = rgb
                if b > 200:  # High blue
                    class_mapping[pattern] = 1  # Slum
                else:
                    class_mapping[pattern] = 0  # Non-slum
    
    print(f"\nFinal class mapping: {class_mapping}")
    return class_mapping

if __name__ == "__main__":
    mask_groups = analyze_mask_variations()
    create_pattern_visualization(mask_groups)
    class_mapping = determine_class_mapping(mask_groups)
