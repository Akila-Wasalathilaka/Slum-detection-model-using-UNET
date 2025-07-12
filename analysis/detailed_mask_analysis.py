import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from collections import Counter, defaultdict

def comprehensive_mask_analysis():
    """Comprehensive analysis of all masks to understand class encoding"""
    
    print("=" * 60)
    print("COMPREHENSIVE MASK ANALYSIS")
    print("=" * 60)
    
    splits = ['train', 'val', 'test']
    all_rgb_combinations = set()
    split_rgb_stats = {}
    
    # Analyze each split
    for split in splits:
        print(f"\n--- {split.upper()} SET ---")
        mask_dir = f"data/{split}/masks"
        mask_files = [f for f in os.listdir(mask_dir) if f.endswith('.png')]
        
        split_rgb_counter = Counter()
        split_unique_combos = set()
        
        # Analyze first 50 files for speed, then sample more if needed
        sample_files = mask_files[:50]
        print(f"Analyzing {len(sample_files)} mask files...")
        
        for mask_file in sample_files:
            mask_path = os.path.join(mask_dir, mask_file)
            try:
                mask = np.array(Image.open(mask_path))
                
                # Get RGB combinations
                rgb_pixels = mask.reshape(-1, 3)
                unique_rgb = np.unique(rgb_pixels, axis=0)
                
                for rgb in unique_rgb:
                    rgb_tuple = tuple(rgb)
                    split_unique_combos.add(rgb_tuple)
                    all_rgb_combinations.add(rgb_tuple)
                    
                    # Count pixels for this RGB combination
                    pixel_count = np.sum(np.all(rgb_pixels == rgb, axis=1))
                    split_rgb_counter[rgb_tuple] += pixel_count
                    
            except Exception as e:
                print(f"Error processing {mask_file}: {e}")
        
        split_rgb_stats[split] = {
            'counter': split_rgb_counter,
            'unique_combos': split_unique_combos,
            'total_pixels': sum(split_rgb_counter.values())
        }
        
        print(f"Found {len(split_unique_combos)} unique RGB combinations:")
        for rgb in sorted(split_unique_combos):
            count = split_rgb_counter[rgb]
            percentage = (count / split_rgb_stats[split]['total_pixels']) * 100
            print(f"  RGB{rgb}: {count:,} pixels ({percentage:.2f}%)")
    
    print(f"\n" + "=" * 60)
    print("OVERALL DATASET ANALYSIS")
    print("=" * 60)
    
    print(f"Total unique RGB combinations across all splits: {len(all_rgb_combinations)}")
    
    # Combine stats from all splits
    overall_counter = Counter()
    for split_stats in split_rgb_stats.values():
        overall_counter.update(split_stats['counter'])
    
    total_pixels = sum(overall_counter.values())
    
    print(f"\nOverall RGB distribution:")
    for rgb in sorted(all_rgb_combinations):
        count = overall_counter[rgb]
        percentage = (count / total_pixels) * 100
        print(f"  RGB{rgb}: {count:,} pixels ({percentage:.2f}%)")
    
    return all_rgb_combinations, split_rgb_stats

def analyze_individual_channels():
    """Analyze each RGB channel separately to understand encoding"""
    
    print(f"\n" + "=" * 60)
    print("INDIVIDUAL CHANNEL ANALYSIS")
    print("=" * 60)
    
    # Analyze a few sample masks
    sample_masks = [
        "data/train/masks/jp22_1.1_1.png",
        "data/train/masks/jp22_1.1_2.png", 
        "data/train/masks/jp22_1.1_10.png",
        "data/val/masks/jp22_1.10_1.png",
        "data/test/masks/jp22_1.0_1.png"
    ]
    
    channel_values = {0: set(), 1: set(), 2: set()}  # R, G, B
    
    for i, mask_path in enumerate(sample_masks):
        if os.path.exists(mask_path):
            print(f"\nSample {i+1}: {os.path.basename(mask_path)}")
            mask = np.array(Image.open(mask_path))
            
            for channel in range(3):
                unique_vals = np.unique(mask[:, :, channel])
                channel_values[channel].update(unique_vals)
                print(f"  Channel {channel} ({'RGB'[channel]}): {unique_vals}")
    
    print(f"\nOverall channel value ranges:")
    for channel in range(3):
        values = sorted(channel_values[channel])
        print(f"  Channel {channel} ({'RGB'[channel]}): {values}")
    
    return channel_values

def create_visualizations(rgb_combinations):
    """Create visualizations to help identify the slum class"""
    
    print(f"\n" + "=" * 60)
    print("CREATING VISUALIZATIONS")
    print("=" * 60)
    
    # Load several samples for visualization
    sample_data = []
    
    sample_paths = [
        ("data/train/images/jp22_1.1_1.tif", "data/train/masks/jp22_1.1_1.png"),
        ("data/train/images/jp22_1.1_2.tif", "data/train/masks/jp22_1.1_2.png"),
        ("data/train/images/jp22_1.1_5.tif", "data/train/masks/jp22_1.1_5.png"),
        ("data/val/images/jp22_1.10_1.tif", "data/val/masks/jp22_1.10_1.png"),
        ("data/test/images/jp22_1.0_1.tif", "data/test/masks/jp22_1.0_1.png"),
    ]
    
    fig, axes = plt.subplots(len(sample_paths), 4, figsize=(20, 4*len(sample_paths)))
    if len(sample_paths) == 1:
        axes = axes.reshape(1, -1)
    
    for i, (img_path, mask_path) in enumerate(sample_paths):
        if os.path.exists(img_path) and os.path.exists(mask_path):
            # Load image and mask
            img = Image.open(img_path)
            mask = np.array(Image.open(mask_path))
            
            # Original image
            axes[i, 0].imshow(img)
            axes[i, 0].set_title(f"Image {i+1}")
            axes[i, 0].axis('off')
            
            # RGB mask
            axes[i, 1].imshow(mask)
            axes[i, 1].set_title(f"RGB Mask {i+1}")
            axes[i, 1].axis('off')
            
            # Individual channels
            for ch in range(3):
                channel_name = ['Red', 'Green', 'Blue'][ch]
                im = axes[i, 2].imshow(mask[:, :, ch], cmap='gray') if ch == 0 else None
                if ch == 0:
                    axes[i, 2].set_title(f"R Channel")
                    axes[i, 2].axis('off')
            
            # Create a composite view showing unique regions
            unique_rgb = np.unique(mask.reshape(-1, 3), axis=0)
            composite = np.zeros_like(mask[:, :, 0])
            
            for idx, rgb in enumerate(unique_rgb):
                mask_region = np.all(mask == rgb, axis=2)
                composite[mask_region] = idx
            
            im = axes[i, 3].imshow(composite, cmap='viridis')
            axes[i, 3].set_title(f"Class Regions")
            axes[i, 3].axis('off')
            
            # Add text showing RGB values
            rgb_text = "\n".join([f"Class {idx}: RGB{tuple(rgb)}" for idx, rgb in enumerate(unique_rgb)])
            axes[i, 3].text(1.05, 0.5, rgb_text, transform=axes[i, 3].transAxes, 
                          verticalalignment='center', fontsize=8,
                          bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('mask_analysis_detailed.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("Detailed visualization saved as 'mask_analysis_detailed.png'")

def suggest_slum_mapping(rgb_combinations, channel_values):
    """Suggest which RGB combination likely represents slums"""
    
    print(f"\n" + "=" * 60)
    print("SLUM CLASS IDENTIFICATION")
    print("=" * 60)
    
    print("Based on the analysis, here are the RGB combinations found:")
    for i, rgb in enumerate(sorted(rgb_combinations)):
        print(f"  Class {i}: RGB{rgb}")
    
    print(f"\nChannel analysis:")
    for channel in range(3):
        values = sorted(channel_values[channel])
        print(f"  Channel {'RGB'[channel]}: {values}")
    
    # Common patterns in annotation tools:
    # 1. Each channel might represent a different class (one-hot encoding)
    # 2. Different RGB combinations for different classes
    # 3. Grayscale values in one channel
    
    print(f"\nPossible encoding patterns:")
    print(f"1. One-hot encoding: Each channel represents a different class")
    print(f"2. RGB color coding: Different RGB combinations for different classes") 
    print(f"3. Grayscale encoding: Values in one channel represent different classes")
    
    # Make educated guesses based on common patterns
    rgb_list = sorted(rgb_combinations)
    
    print(f"\n🤔 EDUCATED GUESSES for slum class:")
    
    if len(rgb_list) >= 3:
        print(f"If using RGB color coding:")
        for i, rgb in enumerate(rgb_list):
            r, g, b = rgb
            # Slums often represented by:
            # - Red/orange colors (urban, informal)
            # - Specific contrasting colors
            if r > g and r > b:  # Red dominant
                print(f"  🔴 RGB{rgb} (Class {i}) - RED dominant, could be slums")
            elif g > r and g > b:  # Green dominant  
                print(f"  🟢 RGB{rgb} (Class {i}) - GREEN dominant, likely vegetation")
            elif b > r and b > g:  # Blue dominant
                print(f"  🔵 RGB{rgb} (Class {i}) - BLUE dominant, could be water")
            else:
                print(f"  ⚫ RGB{rgb} (Class {i}) - Balanced/dark, likely background")
    
    # Check if it's one-hot encoding
    if len(set(sum(rgb) for rgb in rgb_list)) == len(rgb_list):
        print(f"\nIf using one-hot encoding (each channel = one class):")
        print(f"  Red channel (255,0,0): Class A")
        print(f"  Green channel (0,255,0): Class B") 
        print(f"  Blue channel (0,0,255): Class C")
    
    print(f"\n⚠️  MANUAL VERIFICATION NEEDED:")
    print(f"   Please visually inspect the generated visualization to identify")
    print(f"   which regions in the satellite images correspond to slums/informal settlements.")
    print(f"   Then match those regions to the RGB values shown.")

if __name__ == "__main__":
    # Run comprehensive analysis
    rgb_combinations, split_stats = comprehensive_mask_analysis()
    channel_values = analyze_individual_channels()
    create_visualizations(rgb_combinations)
    suggest_slum_mapping(rgb_combinations, channel_values)
