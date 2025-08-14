"""
Fixed Dataset Analysis for Slum Detection
========================================
"""

import os
import cv2
import numpy as np
from pathlib import Path
from collections import Counter
import json

class FixedSlumDatasetAnalyzer:
    def __init__(self, data_root="data"):
        self.data_root = Path(data_root)
        self.analysis_results = {}
        
    def analyze_complete_dataset(self):
        """Complete analysis of dataset classes and patterns."""
        print("🔍 FIXED DATASET ANALYSIS")
        print("=" * 50)
        
        # 1. Analyze mask classes
        self.analyze_mask_classes()
        
        # 2. Generate report
        self.generate_analysis_report()
        
        return self.analysis_results
    
    def analyze_mask_classes(self):
        """Analyze all unique classes in mask files."""
        print("\n📊 Analyzing mask classes...")
        
        all_classes = []
        class_counts = {}
        slum_patterns = []
        
        # Check all mask directories
        mask_dirs = [
            self.data_root / "train" / "masks",
            self.data_root / "val" / "masks", 
            self.data_root / "test" / "masks"
        ]
        
        for mask_dir in mask_dirs:
            if not mask_dir.exists():
                continue
                
            print(f"  Analyzing: {mask_dir}")
            
            for mask_file in mask_dir.glob("*.png"):
                mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    continue
                
                # Get unique values
                unique_values = np.unique(mask)
                
                for value in unique_values:
                    value_str = str(int(value))
                    all_classes.append(value_str)
                    class_counts[value_str] = class_counts.get(value_str, 0) + 1
                    
                    # Check if this is a slum value (non-zero)
                    if value > 0:
                        slum_patterns.append({
                            'file': mask_file.name,
                            'value': int(value),
                            'pixels': int(np.sum(mask == value))
                        })
        
        self.analysis_results['classes'] = {
            'all_classes': list(set(all_classes)),
            'class_counts': class_counts,
            'slum_patterns': slum_patterns,
            'total_classes': len(set(all_classes))
        }
        
        print(f"  Found {len(set(all_classes))} unique classes")
        print(f"  Identified {len(slum_patterns)} slum patterns")
        
    def generate_analysis_report(self):
        """Generate analysis report."""
        print("\n📋 Generating analysis report...")
        
        report = {
            'dataset_summary': {
                'total_classes': self.analysis_results.get('classes', {}).get('total_classes', 0),
                'slum_patterns_found': len(self.analysis_results.get('classes', {}).get('slum_patterns', [])),
                'class_distribution': self.analysis_results.get('classes', {}).get('class_counts', {})
            }
        }
        
        # Save analysis
        with open('fixed_dataset_analysis.json', 'w') as f:
            json.dump(self.analysis_results, f, indent=2)
        
        self.print_analysis_summary(report)
        
        return report
    
    def print_analysis_summary(self, report):
        """Print analysis summary."""
        print("\n" + "="*60)
        print("🎯 FIXED SLUM DETECTION DATASET ANALYSIS")
        print("="*60)
        
        summary = report['dataset_summary']
        print(f"📊 Total Classes Found: {summary['total_classes']}")
        print(f"🏘️ Slum Patterns Detected: {summary['slum_patterns_found']}")
        print(f"📋 Class Distribution: {summary['class_distribution']}")
        
        print("\n✅ Fixed analysis complete!")

if __name__ == "__main__":
    analyzer = FixedSlumDatasetAnalyzer()
    results = analyzer.analyze_complete_dataset()