import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def load_image(img_path):
    """Load RGB image"""
    try:
        # For TIFF files, use OpenCV for better support
        if img_path.endswith('.tif'):
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = np.array(Image.open(img_path))
        return img
    except Exception as e:
        print(f"Error loading {img_path}: {e}")
        return None

def load_mask(mask_path):
    """Load mask as RGB"""
    try:
        mask = np.array(Image.open(mask_path))
        return mask
    except Exception as e:
        print(f"Error loading {mask_path}: {e}")
        return None

def create_binary_slum_mask(rgb_mask):
    """Convert RGB mask to binary slum mask"""
    slum_color = np.array([250, 235, 185])  # RGB for informal settlements
    
    # Create binary mask where slum pixels are white (255), others are black (0)
    slum_pixels = np.all(rgb_mask == slum_color, axis=-1)
    binary_mask = np.zeros(slum_pixels.shape, dtype=np.uint8)
    binary_mask[slum_pixels] = 255
    
    return binary_mask

def analyze_class_distribution(rgb_mask):
    """Analyze class distribution in mask"""
    # Reshape to get unique colors
    pixels = rgb_mask.reshape(-1, 3)
    unique_colors, counts = np.unique(pixels, axis=0, return_counts=True)
    
    class_info = []
    total_pixels = len(pixels)
    
    for color, count in zip(unique_colors, counts):
        percentage = (count / total_pixels) * 100
        class_info.append({
            'color': tuple(color),
            'count': count,
            'percentage': percentage
        })
    
    return sorted(class_info, key=lambda x: x['count'], reverse=True)

# Define class color mapping (based on our analysis)
CLASS_COLORS = {
    (0, 0, 0): 'No Data',
    (40, 120, 240): 'Water',
    (80, 140, 50): 'Trees/Forest',
    (100, 100, 150): 'Cropland',
    (200, 160, 40): 'Shrubland',
    (200, 200, 200): 'Built Area',
    (250, 235, 185): 'Informal Settlements (Slums)'
}

def visualize_samples():
    """Visualize sample images with their masks"""
    
    # Sample files with different slum percentages
    sample_files = [
        'tile_1.0_1',     # 69.4% slums
        'tile_1.0_17',    # 100% slums
        'tile_1.0_28',    # 6.0% slums
        'tile_1.0_46',    # 23.7% slums
        'tile_6.4_21',    # 100% slums
        'tile_6.4_42',    # 0.3% slums
    ]
    
    base_path = 'data/train'
    
    # Create visualization
    fig, axes = plt.subplots(len(sample_files), 4, figsize=(16, len(sample_files) * 3))
    
    for i, file_base in enumerate(sample_files):
        img_path = os.path.join(base_path, 'images', f'{file_base}.tif')
        mask_path = os.path.join(base_path, 'masks', f'{file_base}.png')
        
        # Load image and mask
        img = load_image(img_path)
        rgb_mask = load_mask(mask_path)
        
        if img is None or rgb_mask is None:
            print(f"Could not load {file_base}")
            continue
        
        # Create binary slum mask
        binary_mask = create_binary_slum_mask(rgb_mask)
        
        # Get class distribution
        class_dist = analyze_class_distribution(rgb_mask)
        slum_percentage = 0
        for class_info in class_dist:
            if class_info['color'] == (250, 235, 185):
                slum_percentage = class_info['percentage']
                break
        
        # Plot original image
        if len(sample_files) == 1:
            axes = [axes]
        axes[i][0].imshow(img)
        axes[i][0].set_title(f'Original Image\n{file_base}')
        axes[i][0].axis('off')
        
        # Plot RGB mask
        axes[i][1].imshow(rgb_mask)
        axes[i][1].set_title(f'RGB Mask\n{slum_percentage:.1f}% slums')
        axes[i][1].axis('off')
        
        # Plot binary slum mask
        axes[i][2].imshow(binary_mask, cmap='gray')
        axes[i][2].set_title('Binary Slum Mask\n(White = Slums)')
        axes[i][2].axis('off')
        
        # Plot overlay
        overlay = img.copy()
        # Add red tint where slums are present
        slum_pixels = binary_mask > 0
        if np.any(slum_pixels):
            overlay[slum_pixels] = overlay[slum_pixels] * 0.7 + np.array([255, 0, 0]) * 0.3
        
        axes[i][3].imshow(overlay.astype(np.uint8))
        axes[i][3].set_title('Image + Slum Overlay\n(Red = Slums)')
        axes[i][3].axis('off')
    
    plt.tight_layout()
    plt.savefig('slum_examples_visualization.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Print class mapping for reference
    print("\n" + "="*60)
    print("CLASS COLOR MAPPING:")
    print("="*60)
    for color, label in CLASS_COLORS.items():
        print(f"RGB {color} -> {label}")
    
    # Analyze some specific examples
    print("\n" + "="*60)
    print("DETAILED ANALYSIS OF SAMPLE MASKS:")
    print("="*60)
    
    for file_base in sample_files[:3]:  # Analyze first 3 examples
        mask_path = os.path.join(base_path, 'masks', f'{file_base}.png')
        rgb_mask = load_mask(mask_path)
        
        if rgb_mask is not None:
            print(f"\n{file_base}.png:")
            class_dist = analyze_class_distribution(rgb_mask)
            
            for class_info in class_dist:
                color = class_info['color']
                class_name = CLASS_COLORS.get(color, 'Unknown')
                print(f"  RGB {color}: {class_name} - {class_info['count']:,} pixels ({class_info['percentage']:.1f}%)")

if __name__ == "__main__":
    print("Visualizing slum detection examples...")
    visualize_samples()
    print("Visualization saved as 'slum_examples_visualization.png'")
