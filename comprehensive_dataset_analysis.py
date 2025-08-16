#!/usr/bin/env python3
"""
Comprehensive Dataset Analysis for Slum Detection
================================================
Analyzes the complete dataset structure, identifies all classes,
and generates detailed visualization plots.
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from collections import defaultdict, Counter
import glob
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class ComprehensiveDatasetAnalyzer:
    def __init__(self, data_root="data", output_dir="analysis"):
        self.data_root = Path(data_root)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.splits = ['train', 'val', 'test']
        self.results = {}
        
    def analyze_dataset_structure(self):
        """Analyze basic dataset structure"""
        print("ANALYZING DATASET STRUCTURE")
        print("=" * 50)
        
        structure = {}
        for split in self.splits:
            images_dir = self.data_root / split / "images"
            masks_dir = self.data_root / split / "masks"
            
            # Count files
            image_files = list(images_dir.glob("*.tif")) if images_dir.exists() else []
            mask_files = list(masks_dir.glob("*.png")) if masks_dir.exists() else []
            
            structure[split] = {
                'images': len(image_files),
                'masks': len(mask_files),
                'image_files': [f.name for f in image_files[:5]],  # Sample files
                'mask_files': [f.name for f in mask_files[:5]]
            }
            
            print(f"{split.upper()}: {len(image_files)} images, {len(mask_files)} masks")
        
        self.results['structure'] = structure
        return structure
    
    def analyze_image_properties(self):
        """Analyze image dimensions, channels, and data types"""
        print("\nANALYZING IMAGE PROPERTIES")
        print("=" * 50)
        
        image_stats = defaultdict(list)
        
        for split in self.splits:
            images_dir = self.data_root / split / "images"
            if not images_dir.exists():
                continue
                
            image_files = list(images_dir.glob("*.tif"))[:100]  # Sample 100 images
            
            for img_path in tqdm(image_files, desc=f"Analyzing {split} images"):
                img = cv2.imread(str(img_path))
                if img is not None:
                    image_stats[f'{split}_shapes'].append(img.shape)
                    image_stats[f'{split}_dtypes'].append(str(img.dtype))
                    image_stats[f'{split}_sizes'].append(img.nbytes)
        
        # Analyze dimensions
        all_shapes = []
        for split in self.splits:
            all_shapes.extend(image_stats[f'{split}_shapes'])
        
        unique_shapes = list(set(all_shapes))
        shape_counts = Counter(all_shapes)
        
        print(f"Unique image shapes: {len(unique_shapes)}")
        print(f"Most common shape: {shape_counts.most_common(1)[0] if shape_counts else 'None'}")
        
        self.results['image_properties'] = {
            'unique_shapes': unique_shapes,
            'shape_distribution': dict(shape_counts),
            'total_analyzed': len(all_shapes)
        }
        
        return image_stats
    
    def analyze_mask_classes(self):
        """Comprehensive mask analysis to identify all classes"""
        print("\nANALYZING MASK CLASSES")
        print("=" * 50)
        
        all_unique_values = set()
        class_distribution = defaultdict(int)
        mask_stats = defaultdict(list)
        
        for split in self.splits:
            masks_dir = self.data_root / split / "masks"
            if not masks_dir.exists():
                continue
                
            mask_files = list(masks_dir.glob("*.png"))
            print(f"Analyzing {len(mask_files)} {split} masks...")
            
            split_unique_values = set()
            
            for mask_path in tqdm(mask_files, desc=f"Processing {split} masks"):
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    unique_vals, counts = np.unique(mask, return_counts=True)
                    
                    # Track all unique values
                    all_unique_values.update(unique_vals)
                    split_unique_values.update(unique_vals)
                    
                    # Count class occurrences
                    for val, count in zip(unique_vals, counts):
                        class_distribution[val] += count
                    
                    # Store mask statistics
                    mask_stats[f'{split}_unique_per_mask'].append(len(unique_vals))
                    mask_stats[f'{split}_mask_shapes'].append(mask.shape)
                    mask_stats[f'{split}_has_slums'].append(len(unique_vals) > 1)
            
            print(f"{split} unique values: {sorted(split_unique_values)}")
        
        # Analyze class distribution
        total_pixels = sum(class_distribution.values())
        class_percentages = {k: (v/total_pixels)*100 for k, v in class_distribution.items()}
        
        print(f"\nALL UNIQUE CLASS VALUES: {sorted(all_unique_values)}")
        print(f"TOTAL CLASSES FOUND: {len(all_unique_values)}")
        print("\nCLASS DISTRIBUTION:")
        for class_val in sorted(all_unique_values):
            pixels = class_distribution[class_val]
            percentage = class_percentages[class_val]
            print(f"  Class {class_val}: {pixels:,} pixels ({percentage:.2f}%)")
        
        self.results['mask_analysis'] = {
            'all_unique_values': sorted(all_unique_values),
            'total_classes': len(all_unique_values),
            'class_distribution': dict(class_distribution),
            'class_percentages': class_percentages,
            'mask_stats': dict(mask_stats)
        }
        
        return all_unique_values, class_distribution, mask_stats
    
    def create_class_distribution_plot(self, class_distribution):
        """Create detailed class distribution visualization"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Comprehensive Class Distribution Analysis', fontsize=16, fontweight='bold')
        
        # 1. Bar plot of class distribution
        classes = sorted(class_distribution.keys())
        counts = [class_distribution[c] for c in classes]
        
        axes[0, 0].bar(classes, counts, color='skyblue', edgecolor='navy', alpha=0.7)
        axes[0, 0].set_title('Pixel Count per Class', fontweight='bold')
        axes[0, 0].set_xlabel('Class Value')
        axes[0, 0].set_ylabel('Pixel Count')
        axes[0, 0].tick_params(axis='y', rotation=45)
        
        # Add value labels on bars
        for i, (cls, count) in enumerate(zip(classes, counts)):
            axes[0, 0].text(cls, count + max(counts)*0.01, f'{count:,}', 
                           ha='center', va='bottom', fontsize=8)
        
        # 2. Pie chart of class percentages
        total_pixels = sum(counts)
        percentages = [(c/total_pixels)*100 for c in counts]
        
        # Only show classes with >0.1% for readability
        significant_classes = [(cls, pct) for cls, pct in zip(classes, percentages) if pct > 0.1]
        if len(significant_classes) < len(classes):
            other_pct = sum(pct for cls, pct in zip(classes, percentages) if pct <= 0.1)
            significant_classes.append(('Others', other_pct))
        
        labels, sizes = zip(*significant_classes) if significant_classes else ([], [])
        
        if sizes:
            axes[0, 1].pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
            axes[0, 1].set_title('Class Distribution (%)', fontweight='bold')
        
        # 3. Log scale bar plot for better visibility
        axes[1, 0].bar(classes, counts, color='lightcoral', edgecolor='darkred', alpha=0.7)
        axes[1, 0].set_yscale('log')
        axes[1, 0].set_title('Pixel Count per Class (Log Scale)', fontweight='bold')
        axes[1, 0].set_xlabel('Class Value')
        axes[1, 0].set_ylabel('Pixel Count (Log Scale)')
        
        # 4. Class frequency analysis
        class_names = []
        for cls in classes:
            if cls == 0:
                class_names.append('Background')
            elif cls == 255:
                class_names.append('Slum (White)')
            elif cls == 109:
                class_names.append('Slum (Gray)')
            else:
                class_names.append(f'Class {cls}')
        
        y_pos = np.arange(len(classes))
        axes[1, 1].barh(y_pos, counts, color='lightgreen', edgecolor='darkgreen', alpha=0.7)
        axes[1, 1].set_yticks(y_pos)
        axes[1, 1].set_yticklabels(class_names)
        axes[1, 1].set_xlabel('Pixel Count')
        axes[1, 1].set_title('Class Distribution (Horizontal)', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'class_distribution_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_dataset_overview_plot(self, structure, mask_stats):
        """Create dataset overview visualization"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Dataset Overview and Statistics', fontsize=16, fontweight='bold')
        
        # 1. Dataset size comparison
        splits = list(structure.keys())
        image_counts = [structure[split]['images'] for split in splits]
        mask_counts = [structure[split]['masks'] for split in splits]
        
        x = np.arange(len(splits))
        width = 0.35
        
        axes[0, 0].bar(x - width/2, image_counts, width, label='Images', color='skyblue', alpha=0.8)
        axes[0, 0].bar(x + width/2, mask_counts, width, label='Masks', color='lightcoral', alpha=0.8)
        axes[0, 0].set_xlabel('Dataset Split')
        axes[0, 0].set_ylabel('File Count')
        axes[0, 0].set_title('Dataset Size by Split')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels([s.upper() for s in splits])
        axes[0, 0].legend()
        
        # Add value labels
        for i, (img, mask) in enumerate(zip(image_counts, mask_counts)):
            axes[0, 0].text(i - width/2, img + max(image_counts)*0.01, str(img), 
                           ha='center', va='bottom', fontweight='bold')
            axes[0, 0].text(i + width/2, mask + max(mask_counts)*0.01, str(mask), 
                           ha='center', va='bottom', fontweight='bold')
        
        # 2. Total dataset composition
        total_images = sum(image_counts)
        total_masks = sum(mask_counts)
        
        axes[0, 1].pie([total_images, total_masks], 
                      labels=['Images', 'Masks'], 
                      autopct='%1.0f%%',
                      colors=['skyblue', 'lightcoral'],
                      startangle=90)
        axes[0, 1].set_title(f'Total Files: {total_images + total_masks}')
        
        # 3. Split distribution
        axes[0, 2].pie(image_counts, 
                      labels=[f'{s.upper()}\n({c})' for s, c in zip(splits, image_counts)], 
                      autopct='%1.1f%%',
                      startangle=90)
        axes[0, 2].set_title('Images per Split')
        
        # 4. Masks with slums analysis
        if mask_stats:
            slum_stats = {}
            for split in splits:
                has_slums_key = f'{split}_has_slums'
                if has_slums_key in mask_stats:
                    has_slums = mask_stats[has_slums_key]
                    slum_stats[split] = {
                        'with_slums': sum(has_slums),
                        'without_slums': len(has_slums) - sum(has_slums),
                        'total': len(has_slums)
                    }
            
            if slum_stats:
                splits_with_data = list(slum_stats.keys())
                with_slums = [slum_stats[s]['with_slums'] for s in splits_with_data]
                without_slums = [slum_stats[s]['without_slums'] for s in splits_with_data]
                
                x = np.arange(len(splits_with_data))
                axes[1, 0].bar(x, with_slums, label='With Slums', color='red', alpha=0.7)
                axes[1, 0].bar(x, without_slums, bottom=with_slums, label='Background Only', color='green', alpha=0.7)
                axes[1, 0].set_xlabel('Dataset Split')
                axes[1, 0].set_ylabel('Number of Masks')
                axes[1, 0].set_title('Masks: Slum vs Background')
                axes[1, 0].set_xticks(x)
                axes[1, 0].set_xticklabels([s.upper() for s in splits_with_data])
                axes[1, 0].legend()
        
        # 5. Unique classes per mask distribution
        if mask_stats:
            all_unique_counts = []
            for split in splits:
                unique_key = f'{split}_unique_per_mask'
                if unique_key in mask_stats:
                    all_unique_counts.extend(mask_stats[unique_key])
            
            if all_unique_counts:
                axes[1, 1].hist(all_unique_counts, bins=max(all_unique_counts), 
                               color='purple', alpha=0.7, edgecolor='black')
                axes[1, 1].set_xlabel('Number of Unique Classes per Mask')
                axes[1, 1].set_ylabel('Frequency')
                axes[1, 1].set_title('Distribution of Classes per Mask')
        
        # 6. Summary statistics
        axes[1, 2].axis('off')
        summary_text = f"""
DATASET SUMMARY
{'='*20}

Total Images: {total_images:,}
Total Masks: {total_masks:,}

Split Distribution:
"""
        for split, count in zip(splits, image_counts):
            percentage = (count/total_images)*100 if total_images > 0 else 0
            summary_text += f"• {split.upper()}: {count:,} ({percentage:.1f}%)\n"
        
        if 'mask_analysis' in self.results:
            mask_analysis = self.results['mask_analysis']
            summary_text += f"""
Classes Found: {mask_analysis['total_classes']}
Unique Values: {mask_analysis['all_unique_values']}
"""
        
        axes[1, 2].text(0.1, 0.9, summary_text, transform=axes[1, 2].transAxes, 
                       fontsize=12, verticalalignment='top', fontfamily='monospace',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'dataset_overview.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_sample_visualization(self):
        """Create sample image and mask visualizations"""
        fig, axes = plt.subplots(3, 6, figsize=(20, 12))
        fig.suptitle('Sample Images and Masks from Each Split', fontsize=16, fontweight='bold')
        
        for split_idx, split in enumerate(self.splits):
            images_dir = self.data_root / split / "images"
            masks_dir = self.data_root / split / "masks"
            
            if not (images_dir.exists() and masks_dir.exists()):
                continue
            
            # Get sample files
            image_files = list(images_dir.glob("*.tif"))[:2]
            
            for sample_idx, img_path in enumerate(image_files):
                if sample_idx >= 2:  # Only show 2 samples per split
                    break
                
                # Find corresponding mask
                mask_path = masks_dir / (img_path.stem + ".png")
                
                if mask_path.exists():
                    # Load image and mask
                    image = cv2.imread(str(img_path))
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                    
                    # Plot image
                    axes[split_idx, sample_idx*3].imshow(image)
                    axes[split_idx, sample_idx*3].set_title(f'{split.upper()} - Image {sample_idx+1}')
                    axes[split_idx, sample_idx*3].axis('off')
                    
                    # Plot mask
                    axes[split_idx, sample_idx*3+1].imshow(mask, cmap='viridis')
                    axes[split_idx, sample_idx*3+1].set_title(f'Mask (Classes: {len(np.unique(mask))})')
                    axes[split_idx, sample_idx*3+1].axis('off')
                    
                    # Plot overlay
                    overlay = image.copy()
                    slum_pixels = mask > 0
                    if np.any(slum_pixels):
                        overlay[slum_pixels] = [255, 0, 0]  # Red for slums
                        blended = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)
                    else:
                        blended = image
                    
                    axes[split_idx, sample_idx*3+2].imshow(blended)
                    axes[split_idx, sample_idx*3+2].set_title('Overlay')
                    axes[split_idx, sample_idx*3+2].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'sample_visualizations.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_analysis_results(self):
        """Save comprehensive analysis results to JSON"""
        output_file = self.output_dir / 'comprehensive_analysis_results.json'
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            return obj
        
        results_serializable = convert_numpy_types(self.results)
        
        with open(output_file, 'w') as f:
            json.dump(results_serializable, f, indent=2)
        
        print(f"\nAnalysis results saved to: {output_file}")
    
    def run_complete_analysis(self):
        """Run the complete dataset analysis pipeline"""
        print("STARTING COMPREHENSIVE DATASET ANALYSIS")
        print("=" * 60)
        
        # 1. Analyze dataset structure
        structure = self.analyze_dataset_structure()
        
        # 2. Analyze image properties
        image_stats = self.analyze_image_properties()
        
        # 3. Analyze mask classes
        unique_values, class_distribution, mask_stats = self.analyze_mask_classes()
        
        # 4. Create visualizations
        print("\nCREATING VISUALIZATIONS")
        print("=" * 50)
        
        self.create_class_distribution_plot(class_distribution)
        self.create_dataset_overview_plot(structure, mask_stats)
        self.create_sample_visualization()
        
        # 5. Save results
        self.save_analysis_results()
        
        # 6. Print summary
        print("\nANALYSIS COMPLETE!")
        print("=" * 50)
        print(f"Results saved in: {self.output_dir}")
        print(f"Classes found: {len(unique_values)}")
        print(f"Unique values: {sorted(unique_values)}")
        print(f"Total files analyzed: {sum(structure[split]['images'] + structure[split]['masks'] for split in structure)}")
        
        return self.results

def main():
    """Main function to run the analysis"""
    analyzer = ComprehensiveDatasetAnalyzer()
    results = analyzer.run_complete_analysis()
    return results

if __name__ == "__main__":
    main()