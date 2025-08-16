#!/usr/bin/env python3
"""
Quick Dataset Analysis for Slum Detection
========================================
Identifies all classes and creates essential plots.
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from collections import defaultdict, Counter
import glob
from tqdm import tqdm

class QuickAnalyzer:
    def __init__(self, data_root="data", output_dir="analysis"):
        self.data_root = Path(data_root)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def analyze_classes(self):
        """Analyze all mask classes across dataset"""
        print("ANALYZING MASK CLASSES")
        print("=" * 50)
        
        all_unique_values = set()
        class_distribution = defaultdict(int)
        
        for split in ['train', 'val', 'test']:
            masks_dir = self.data_root / split / "masks"
            if not masks_dir.exists():
                continue
                
            mask_files = list(masks_dir.glob("*.png"))
            print(f"Processing {len(mask_files)} {split} masks...")
            
            for mask_path in tqdm(mask_files, desc=f"{split} masks"):
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    unique_vals, counts = np.unique(mask, return_counts=True)
                    all_unique_values.update(unique_vals)
                    
                    for val, count in zip(unique_vals, counts):
                        class_distribution[int(val)] += int(count)
        
        # Print results
        print(f"\nFOUND {len(all_unique_values)} CLASSES:")
        print(f"Class values: {sorted(all_unique_values)}")
        
        total_pixels = sum(class_distribution.values())
        print(f"\nCLASS DISTRIBUTION:")
        for class_val in sorted(all_unique_values):
            pixels = class_distribution[int(class_val)]
            percentage = (pixels/total_pixels)*100
            print(f"  Class {class_val}: {pixels:,} pixels ({percentage:.2f}%)")
        
        return all_unique_values, class_distribution
    
    def create_class_plot(self, class_distribution):
        """Create class distribution plot"""
        classes = sorted(class_distribution.keys())
        counts = [class_distribution[c] for c in classes]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Bar plot
        bars = ax1.bar(classes, counts, color='skyblue', edgecolor='navy', alpha=0.7)
        ax1.set_title('Class Distribution', fontweight='bold', fontsize=14)
        ax1.set_xlabel('Class Value')
        ax1.set_ylabel('Pixel Count')
        
        # Add value labels
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + max(counts)*0.01,
                    f'{count:,}', ha='center', va='bottom', fontsize=10)
        
        # Pie chart
        total_pixels = sum(counts)
        percentages = [(c/total_pixels)*100 for c in counts]
        
        # Create labels with class names
        labels = []
        for cls in classes:
            if cls == 0:
                labels.append(f'Background ({cls})')
            elif cls == 109:
                labels.append(f'Slum Main ({cls})')
            elif cls == 105:
                labels.append(f'Slum Type A ({cls})')
            elif cls == 111:
                labels.append(f'Slum Type B ({cls})')
            elif cls == 158:
                labels.append(f'Slum Type C ({cls})')
            elif cls == 200:
                labels.append(f'Slum Type D ({cls})')
            elif cls == 233:
                labels.append(f'Slum Type E ({cls})')
            else:
                labels.append(f'Class {cls}')
        
        ax2.pie(percentages, labels=labels, autopct='%1.1f%%', startangle=90)
        ax2.set_title('Class Distribution (%)', fontweight='bold', fontsize=14)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'class_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def create_sample_plot(self):
        """Create sample visualization"""
        fig, axes = plt.subplots(2, 6, figsize=(18, 8))
        fig.suptitle('Sample Images and Masks', fontsize=16, fontweight='bold')
        
        sample_count = 0
        for split_idx, split in enumerate(['train', 'val']):
            images_dir = self.data_root / split / "images"
            masks_dir = self.data_root / split / "masks"
            
            if not (images_dir.exists() and masks_dir.exists()):
                continue
            
            image_files = list(images_dir.glob("*.tif"))[:3]
            
            for i, img_path in enumerate(image_files):
                mask_path = masks_dir / (img_path.stem + ".png")
                
                if mask_path.exists():
                    # Load files
                    image = cv2.imread(str(img_path))
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                    
                    # Plot image
                    axes[split_idx, i*2].imshow(image)
                    axes[split_idx, i*2].set_title(f'{split.upper()} Image {i+1}')
                    axes[split_idx, i*2].axis('off')
                    
                    # Plot mask with custom colormap
                    im = axes[split_idx, i*2+1].imshow(mask, cmap='tab10')
                    axes[split_idx, i*2+1].set_title(f'Mask (Classes: {len(np.unique(mask))})')
                    axes[split_idx, i*2+1].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'sample_images.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_results(self, unique_values, class_distribution):
        """Save analysis results"""
        results = {
            "total_classes": len(unique_values),
            "class_values": sorted([int(x) for x in unique_values]),
            "class_distribution": {str(k): int(v) for k, v in class_distribution.items()},
            "class_interpretation": {
                "0": "Background",
                "105": "Slum Type A", 
                "109": "Slum Main Type",
                "111": "Slum Type B",
                "158": "Slum Type C", 
                "200": "Slum Type D",
                "233": "Slum Type E"
            }
        }
        
        with open(self.output_dir / 'class_analysis_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: {self.output_dir}")
        return results
    
    def run_analysis(self):
        """Run complete analysis"""
        print("STARTING DATASET ANALYSIS")
        print("=" * 40)
        
        # Analyze classes
        unique_values, class_distribution = self.analyze_classes()
        
        # Create plots
        print("\nCREATING VISUALIZATIONS...")
        self.create_class_plot(class_distribution)
        self.create_sample_plot()
        
        # Save results
        results = self.save_results(unique_values, class_distribution)
        
        print("\nANALYSIS COMPLETE!")
        print(f"Found {len(unique_values)} classes: {sorted(unique_values)}")
        
        return results

if __name__ == "__main__":
    analyzer = QuickAnalyzer()
    analyzer.run_analysis()