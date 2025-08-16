import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import json
from collections import defaultdict
import pandas as pd

def analyze_dataset(data_dir):
    """Comprehensive dataset analysis"""
    results = {
        "dataset_structure": {},
        "data_distribution": {},
        "image_properties": {},
        "mask_properties": {},
        "slum_distribution": {},
        "recommendations": []
    }
    
    # 1. Dataset Structure Analysis
    for split in ['train', 'val', 'test']:
        split_path = os.path.join(data_dir, split)
        if os.path.exists(split_path):
            images_path = os.path.join(split_path, 'images')
            masks_path = os.path.join(split_path, 'masks')
            
            num_images = len([f for f in os.listdir(images_path) if f.endswith('.tif')])
            num_masks = len([f for f in os.listdir(masks_path) if f.endswith('.png')])
            
            results["dataset_structure"][split] = {
                "num_images": num_images,
                "num_masks": num_masks,
                "matched": num_images == num_masks
            }
    
    # 2. Detailed Image Analysis
    train_images_path = os.path.join(data_dir, 'train', 'images')
    train_masks_path = os.path.join(data_dir, 'train', 'masks')
    
    # Sample analysis on first 100 images
    image_files = [f for f in os.listdir(train_images_path) if f.endswith('.tif')][:100]
    
    image_sizes = []
    mask_stats = []
    grid_distribution = defaultdict(int)
    
    for img_file in image_files:
        # Parse grid coordinates from filename
        # Format: tile_X.Y_Z.tif
        parts = img_file.replace('.tif', '').split('_')
        if len(parts) >= 3:
            grid_coord = f"{parts[1]}_{parts[2]}"
            grid_distribution[grid_coord] += 1
        
        # Analyze image properties
        img_path = os.path.join(train_images_path, img_file)
        img = cv2.imread(img_path)
        if img is not None:
            image_sizes.append(img.shape)
        
        # Analyze corresponding mask
        mask_file = img_file.replace('.tif', '.png')
        mask_path = os.path.join(train_masks_path, mask_file)
        
        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                unique_values = np.unique(mask)
                slum_pixels = np.sum(mask > 0)
                total_pixels = mask.shape[0] * mask.shape[1]
                slum_percentage = (slum_pixels / total_pixels) * 100
                
                mask_stats.append({
                    'file': mask_file,
                    'unique_values': unique_values.tolist(),
                    'slum_percentage': slum_percentage,
                    'has_slum': slum_pixels > 0
                })
    
    # Compile results
    if image_sizes:
        unique_sizes = list(set(image_sizes))
        results["image_properties"] = {
            "sample_size": len(image_sizes),
            "unique_dimensions": unique_sizes,
            "consistent_size": len(unique_sizes) == 1,
            "common_size": unique_sizes[0] if len(unique_sizes) == 1 else "varied"
        }
    
    if mask_stats:
        slum_files = [m for m in mask_stats if m['has_slum']]
        no_slum_files = [m for m in mask_stats if not m['has_slum']]
        
        results["mask_properties"] = {
            "sample_size": len(mask_stats),
            "files_with_slums": len(slum_files),
            "files_without_slums": len(no_slum_files),
            "slum_percentage_dist": [m['slum_percentage'] for m in mask_stats],
            "avg_slum_percentage": np.mean([m['slum_percentage'] for m in mask_stats]),
            "unique_mask_values": list(set([val for m in mask_stats for val in m['unique_values']]))
        }
    
    results["slum_distribution"] = dict(grid_distribution)
    
    # 3. Generate Recommendations
    recommendations = []
    
    # Data augmentation recommendations
    total_samples = results["dataset_structure"].get("train", {}).get("num_images", 0)
    if total_samples < 10000:
        recommendations.append(f"Consider data augmentation - current training set has {total_samples} samples")
    
    # Class imbalance check
    if mask_stats:
        imbalance_ratio = len(no_slum_files) / max(len(slum_files), 1)
        if imbalance_ratio > 3:
            recommendations.append(f"Class imbalance detected - {len(no_slum_files)} no-slum vs {len(slum_files)} slum tiles")
    
    # Model architecture recommendations
    if results["image_properties"].get("common_size"):
        size = results["image_properties"]["common_size"]
        if len(size) >= 2:
            h, w = size[:2]
            if h >= 512 and w >= 512:
                recommendations.append("Image size suitable for deep U-Net with multiple encoder levels")
            else:
                recommendations.append("Consider using lighter model or resizing images for better performance")
    
    results["recommendations"] = recommendations
    
    return results

def visualize_analysis(results, save_dir="analysis_plots"):
    """Create visualizations for the analysis"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Dataset distribution plot
    structure = results["dataset_structure"]
    splits = list(structure.keys())
    counts = [structure[split]["num_images"] for split in splits]
    
    plt.figure(figsize=(10, 6))
    plt.bar(splits, counts)
    plt.title('Dataset Distribution Across Splits')
    plt.ylabel('Number of Images')
    plt.savefig(os.path.join(save_dir, 'dataset_distribution.png'))
    plt.close()
    
    # 2. Slum percentage distribution
    if "slum_percentage_dist" in results["mask_properties"]:
        plt.figure(figsize=(12, 6))
        slum_percentages = results["mask_properties"]["slum_percentage_dist"]
        
        plt.subplot(1, 2, 1)
        plt.hist(slum_percentages, bins=20, alpha=0.7)
        plt.title('Distribution of Slum Percentage per Tile')
        plt.xlabel('Slum Percentage (%)')
        plt.ylabel('Number of Tiles')
        
        plt.subplot(1, 2, 2)
        slum_binary = [1 if p > 0 else 0 for p in slum_percentages]
        labels = ['No Slum', 'Has Slum']
        counts = [slum_binary.count(0), slum_binary.count(1)]
        plt.pie(counts, labels=labels, autopct='%1.1f%%')
        plt.title('Slum vs No-Slum Tiles')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'slum_distribution.png'))
        plt.close()
    
    print(f"Analysis plots saved to: {save_dir}")

def save_analysis_report(results, filename="dataset_analysis_report.json"):
    """Save comprehensive analysis report"""
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Also create a readable text report
    text_filename = filename.replace('.json', '.txt')
    with open(text_filename, 'w') as f:
        f.write("=== COMPREHENSIVE DATASET ANALYSIS REPORT ===\n\n")
        
        f.write("1. DATASET STRUCTURE:\n")
        for split, info in results["dataset_structure"].items():
            f.write(f"   {split.upper()}: {info['num_images']} images, {info['num_masks']} masks, "
                   f"Matched: {info['matched']}\n")
        
        f.write(f"\n2. IMAGE PROPERTIES:\n")
        img_props = results["image_properties"]
        f.write(f"   Sample Size: {img_props.get('sample_size', 'N/A')}\n")
        f.write(f"   Consistent Size: {img_props.get('consistent_size', 'N/A')}\n")
        f.write(f"   Common Size: {img_props.get('common_size', 'N/A')}\n")
        
        f.write(f"\n3. MASK PROPERTIES:\n")
        mask_props = results["mask_properties"]
        f.write(f"   Sample Size: {mask_props.get('sample_size', 'N/A')}\n")
        f.write(f"   Files with Slums: {mask_props.get('files_with_slums', 'N/A')}\n")
        f.write(f"   Files without Slums: {mask_props.get('files_without_slums', 'N/A')}\n")
        f.write(f"   Average Slum Percentage: {mask_props.get('avg_slum_percentage', 'N/A'):.2f}%\n")
        f.write(f"   Unique Mask Values: {mask_props.get('unique_mask_values', 'N/A')}\n")
        
        f.write(f"\n4. RECOMMENDATIONS:\n")
        for i, rec in enumerate(results["recommendations"], 1):
            f.write(f"   {i}. {rec}\n")
    
    print(f"Analysis report saved to: {filename} and {text_filename}")

if __name__ == "__main__":
    data_dir = "data"
    
    print("Starting comprehensive dataset analysis...")
    results = analyze_dataset(data_dir)
    
    print("Creating visualizations...")
    visualize_analysis(results)
    
    print("Saving analysis report...")
    save_analysis_report(results, "comprehensive_dataset_analysis.json")
    
    print("\n=== QUICK SUMMARY ===")
    print(f"Training samples: {results['dataset_structure'].get('train', {}).get('num_images', 'N/A')}")
    print(f"Validation samples: {results['dataset_structure'].get('val', {}).get('num_images', 'N/A')}")
    print(f"Test samples: {results['dataset_structure'].get('test', {}).get('num_images', 'N/A')}")
    
    if "mask_properties" in results:
        mask_props = results["mask_properties"]
        print(f"Slum tiles: {mask_props.get('files_with_slums', 'N/A')}")
        print(f"No-slum tiles: {mask_props.get('files_without_slums', 'N/A')}")
        print(f"Average slum coverage: {mask_props.get('avg_slum_percentage', 0):.2f}%")
    
    print("\nRecommendations:")
    for rec in results["recommendations"]:
        print(f"- {rec}")
