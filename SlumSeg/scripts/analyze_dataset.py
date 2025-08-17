#!/usr/bin/env python3
"""
Dataset analysis script for slum segmentation.
Generates comprehensive analysis charts and statistics.
"""

import os
import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings

import yaml
import matplotlib.pyplot as plt
import seaborn as sns
import rasterio
from rasterio.plot import show

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from slumseg.data.dataset import get_image_ids_from_dir, calculate_class_weights
from slumseg.utils.visualize import create_sample_grid, plot_class_distribution

warnings.filterwarnings("ignore")
plt.style.use('seaborn-v0_8')


def analyze_dataset_structure(data_root: str) -> dict:
    """Analyze the overall dataset structure."""
    data_path = Path(data_root)
    
    analysis = {
        'splits': {},
        'total_images': 0,
        'total_masks': 0
    }
    
    # Check each split
    for split in ['train', 'val', 'test']:
        split_path = data_path / split
        if split_path.exists():
            images_dir = split_path / 'images'
            masks_dir = split_path / 'masks'
            
            image_ids = get_image_ids_from_dir(str(images_dir)) if images_dir.exists() else []
            mask_ids = get_image_ids_from_dir(str(masks_dir)) if masks_dir.exists() else []
            
            regions = extract_regions(image_ids)
            analysis['splits'][split] = {
                'images': len(image_ids),
                'masks': len(mask_ids),
                'image_ids': image_ids[:10],  # Sample for display
                'regions': regions
            }
            
            analysis['total_images'] += len(image_ids)
            analysis['total_masks'] += len(mask_ids)
    
    return analysis


def extract_regions(image_ids: list) -> dict:
    """Extract region information from image IDs."""
    regions = {}
    for img_id in image_ids:
        # Extract region from filename (e.g., jp22_1.0_1 -> jp22)
        parts = img_id.split('_')
        if len(parts) >= 1:
            region = parts[0]
            if region not in regions:
                regions[region] = 0
            regions[region] += 1
    return regions


def analyze_image_properties(images_dir: str, sample_size: int = 50) -> dict:
    """Analyze image properties like size, channels, data type."""
    images_path = Path(images_dir)
    image_files = list(images_path.glob("*.tif"))[:sample_size]
    
    properties = {
        'shapes': [],
        'dtypes': [],
        'channels': [],
        'pixel_stats': []
    }
    
    for img_path in tqdm(image_files, desc="Analyzing images"):
        try:
            with rasterio.open(img_path) as src:
                shape = (src.height, src.width)
                dtype = str(src.dtypes[0])
                n_channels = src.count
                
                # Read a sample for pixel stats
                data = src.read()
                
                properties['shapes'].append(shape)
                properties['dtypes'].append(dtype)
                properties['channels'].append(n_channels)
                properties['pixel_stats'].append({
                    'mean': float(np.mean(data)),
                    'std': float(np.std(data)),
                    'min': float(np.min(data)),
                    'max': float(np.max(data))
                })
        except Exception as e:
            print(f"Error analyzing {img_path}: {e}")
            continue
    
    return properties


def analyze_mask_properties(masks_dir: str, image_ids: list) -> dict:
    """Analyze mask properties and class distribution."""
    masks_path = Path(masks_dir)
    
    properties = {
        'class_counts': {'background': 0, 'slum': 0},
        'slum_coverage': [],
        'mask_shapes': [],
        'images_with_slums': 0,
        'images_without_slums': 0
    }
    
    for img_id in tqdm(image_ids, desc="Analyzing masks"):
        # Sanitize filename to prevent path traversal
        clean_id = os.path.basename(str(img_id).replace('..', ''))
        mask_path = masks_path / f"{clean_id}.tif"
        
        if mask_path.exists():
            try:
                with rasterio.open(mask_path) as src:
                    mask = src.read(1)
                    
                    properties['mask_shapes'].append(mask.shape)
                    
                    # Count pixels
                    total_pixels = mask.size
                    slum_pixels = (mask > 0).sum()
                    bg_pixels = total_pixels - slum_pixels
                    
                    properties['class_counts']['background'] += bg_pixels
                    properties['class_counts']['slum'] += slum_pixels
                    
                    # Coverage percentage
                    coverage = (slum_pixels / total_pixels) * 100
                    properties['slum_coverage'].append(coverage)
                    
                    if slum_pixels > 0:
                        properties['images_with_slums'] += 1
                    else:
                        properties['images_without_slums'] += 1
                        
            except Exception as e:
                print(f"Error analyzing mask {mask_path}: {e}")
                continue
        else:
            properties['images_without_slums'] += 1
            properties['slum_coverage'].append(0.0)
    
    return properties


def create_analysis_charts(analysis_data: dict, output_dir: str):
    """Create comprehensive analysis charts."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 1. Dataset structure overview
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Split counts
    splits = list(analysis_data['structure']['splits'].keys())
    image_counts = [analysis_data['structure']['splits'][s]['images'] for s in splits]
    mask_counts = [analysis_data['structure']['splits'][s]['masks'] for s in splits]
    
    x = np.arange(len(splits))
    width = 0.35
    
    axes[0, 0].bar(x - width/2, image_counts, width, label='Images', alpha=0.8)
    axes[0, 0].bar(x + width/2, mask_counts, width, label='Masks', alpha=0.8)
    axes[0, 0].set_xlabel('Dataset Split')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('Images and Masks per Split')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(splits)
    axes[0, 0].legend()
    
    # Region distribution
    all_regions = {}
    for split_data in analysis_data['structure']['splits'].values():
        for region, count in split_data['regions'].items():
            all_regions[region] = all_regions.get(region, 0) + count
    
    if all_regions:
        regions = list(all_regions.keys())
        counts = list(all_regions.values())
        axes[0, 1].pie(counts, labels=regions, autopct='%1.1f%%')
        axes[0, 1].set_title('Distribution by Region')
    
    # Class distribution
    if 'masks' in analysis_data:
        class_counts = analysis_data['masks']['class_counts']
        classes = list(class_counts.keys())
        counts = list(class_counts.values())
        
        axes[1, 0].bar(classes, counts, alpha=0.8)
        axes[1, 0].set_xlabel('Class')
        axes[1, 0].set_ylabel('Pixel Count')
        axes[1, 0].set_title('Class Distribution (Pixel-wise)')
        axes[1, 0].set_yscale('log')
    
    # Slum coverage histogram
    if 'masks' in analysis_data:
        coverage_data = analysis_data['masks']['slum_coverage']
        axes[1, 1].hist(coverage_data, bins=50, alpha=0.7, edgecolor='black')
        axes[1, 1].set_xlabel('Slum Coverage (%)')
        axes[1, 1].set_ylabel('Number of Images')
        axes[1, 1].set_title('Slum Coverage Distribution')
    
    plt.tight_layout()
    plt.savefig(output_path / '01_dataset_overview.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Image properties
    if 'images' in analysis_data:
        img_props = analysis_data['images']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Image shapes
        shapes = img_props['shapes']
        heights = [s[0] for s in shapes]
        widths = [s[1] for s in shapes]
        
        axes[0, 0].scatter(widths, heights, alpha=0.6)
        axes[0, 0].set_xlabel('Width')
        axes[0, 0].set_ylabel('Height')
        axes[0, 0].set_title('Image Dimensions')
        
        # Channel distribution
        channels = img_props['channels']
        unique_channels, counts = np.unique(channels, return_counts=True)
        axes[0, 1].bar(unique_channels, counts)
        axes[0, 1].set_xlabel('Number of Channels')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].set_title('Channel Distribution')
        
        # Pixel value statistics
        stats = img_props['pixel_stats']
        means = [s['mean'] for s in stats]
        stds = [s['std'] for s in stats]
        
        axes[1, 0].hist(means, bins=30, alpha=0.7, label='Mean')
        axes[1, 0].set_xlabel('Pixel Value')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Pixel Value Means')
        
        axes[1, 1].hist(stds, bins=30, alpha=0.7, label='Std Dev', color='orange')
        axes[1, 1].set_xlabel('Standard Deviation')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Pixel Value Standard Deviations')
        
        plt.tight_layout()
        plt.savefig(output_path / '02_image_properties.png', dpi=300, bbox_inches='tight')
        plt.close()


def generate_statistics_report(analysis_data: dict, output_dir: str):
    """Generate a comprehensive statistics report."""
    output_path = Path(output_dir)
    
    report = []
    report.append("# SlumSeg Dataset Analysis Report\n")
    report.append(f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    # Dataset structure
    report.append("## Dataset Structure\n")
    structure = analysis_data['structure']
    report.append(f"- Total Images: {structure['total_images']}")
    report.append(f"- Total Masks: {structure['total_masks']}\n")
    
    for split, data in structure['splits'].items():
        report.append(f"### {split.capitalize()} Split")
        report.append(f"- Images: {data['images']}")
        report.append(f"- Masks: {data['masks']}")
        if data['regions']:
            report.append("- Regions:")
            for region, count in data['regions'].items():
                report.append(f"  - {region}: {count} images")
        report.append("")
    
    # Class distribution
    if 'masks' in analysis_data:
        masks = analysis_data['masks']
        report.append("## Class Distribution\n")
        total_pixels = sum(masks['class_counts'].values())
        for class_name, count in masks['class_counts'].items():
            percentage = (count / total_pixels) * 100
            report.append(f"- {class_name}: {count:,} pixels ({percentage:.2f}%)")
        
        report.append(f"\n- Images with slums: {masks['images_with_slums']}")
        report.append(f"- Images without slums: {masks['images_without_slums']}")
        
        if masks['slum_coverage']:
            coverage = np.array(masks['slum_coverage'])
            report.append(f"\n### Slum Coverage Statistics")
            report.append(f"- Mean coverage: {np.mean(coverage):.2f}%")
            report.append(f"- Median coverage: {np.median(coverage):.2f}%")
            report.append(f"- Max coverage: {np.max(coverage):.2f}%")
            report.append(f"- Images with >1% slums: {np.sum(coverage > 1)}")
            report.append(f"- Images with >10% slums: {np.sum(coverage > 10)}")
    
    # Image properties
    if 'images' in analysis_data:
        images = analysis_data['images']
        report.append(f"\n## Image Properties\n")
        
        if images['shapes']:
            shapes = np.array(images['shapes'])
            report.append(f"- Typical image size: {np.mean(shapes, axis=0).astype(int)}")
            report.append(f"- Size range: {np.min(shapes, axis=0)} to {np.max(shapes, axis=0)}")
        
        if images['pixel_stats']:
            stats = images['pixel_stats']
            means = [s['mean'] for s in stats]
            report.append(f"- Average pixel mean: {np.mean(means):.2f}")
            report.append(f"- Pixel value range: 0-255 (typical)")
    
    # Write report
    with open(output_path / 'analysis_report.md', 'w') as f:
        f.write('\n'.join(report))


def main():
    parser = argparse.ArgumentParser(description='Analyze slum segmentation dataset')
    parser.add_argument('--config', type=str, required=True, help='Config file path')
    parser.add_argument('--out', type=str, required=True, help='Output directory for charts')
    
    args = parser.parse_args()
    
    # Load config with error handling
    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    except (FileNotFoundError, yaml.YAMLError, PermissionError) as e:
        print(f"Error loading config: {e}")
        sys.exit(1)
    
    data_root = config['data']['root']
    print(f"Analyzing dataset at: {data_root}")
    
    # Create output directory
    os.makedirs(args.out, exist_ok=True)
    
    # Run analysis
    analysis_data = {}
    
    print("1. Analyzing dataset structure...")
    analysis_data['structure'] = analyze_dataset_structure(data_root)
    
    # Analyze train split in detail
    train_images_dir = os.path.join(data_root, 'train', 'images')
    train_masks_dir = os.path.join(data_root, 'train', 'masks')
    
    if os.path.exists(train_images_dir):
        print("2. Analyzing image properties...")
        analysis_data['images'] = analyze_image_properties(train_images_dir)
        
        print("3. Analyzing mask properties...")
        train_image_ids = get_image_ids_from_dir(train_images_dir)
        analysis_data['masks'] = analyze_mask_properties(train_masks_dir, train_image_ids)
    
    print("4. Creating analysis charts...")
    create_analysis_charts(analysis_data, args.out)
    
    print("5. Generating statistics report...")
    generate_statistics_report(analysis_data, args.out)
    
    print(f"Analysis complete! Results saved to: {args.out}")
    
    # Print summary
    if 'masks' in analysis_data:
        class_counts = analysis_data['masks']['class_counts']
        total_pixels = sum(class_counts.values())
        slum_ratio = class_counts['slum'] / total_pixels
        print(f"\nKey findings:")
        print(f"- Total images: {analysis_data['structure']['total_images']}")
        print(f"- Slum pixel ratio: {slum_ratio:.4f} ({slum_ratio*100:.2f}%)")
        print(f"- Suggested pos_weight: {(1-slum_ratio)/slum_ratio:.2f}")


if __name__ == "__main__":
    main()
