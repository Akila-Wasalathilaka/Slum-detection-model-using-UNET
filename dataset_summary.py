#!/usr/bin/env python3
"""
DATASET CLASS SUMMARY
Based on comprehensive analysis of all 8,910 masks
"""

import json

def create_dataset_summary():
    """Create a clean summary of the dataset findings"""
    
    print("🎊 DATASET CLASS IDENTIFICATION COMPLETE!")
    print("=" * 60)
    
    # The classes we discovered
    classes = {
        0: {
            'rgb': (40, 120, 240),
            'name': 'Water Bodies',
            'semantic': 'Rivers, lakes, coastal water',
            'percentage': 31.89,
            'pixels': 40922200,
            'color_description': 'Blue'
        },
        1: {
            'rgb': (80, 140, 50),
            'name': 'Vegetation',
            'semantic': 'Trees, parks, green spaces',
            'percentage': 18.75,
            'pixels': 24059668,
            'color_description': 'Green'
        },
        2: {
            'rgb': (200, 160, 40),
            'name': 'Slum/Informal Settlements (Type 1)',
            'semantic': 'Dense informal housing, slum areas',
            'percentage': 15.84,
            'pixels': 20327313,
            'color_description': 'Reddish-Brown'
        },
        3: {
            'rgb': (100, 100, 150),
            'name': 'Formal Structures',
            'semantic': 'Buildings, infrastructure',
            'percentage': 12.44,
            'pixels': 15958561,
            'color_description': 'Blue-Gray'
        },
        4: {
            'rgb': (250, 235, 185),
            'name': 'Slum/Informal Settlements (Type 2)',
            'semantic': 'Another type of informal housing',
            'percentage': 11.84,
            'pixels': 15196108,
            'color_description': 'Light Brown/Beige'
        },
        5: {
            'rgb': (200, 200, 200),
            'name': 'Urban/Concrete',
            'semantic': 'Roads, concrete areas, urban',
            'percentage': 9.15,
            'pixels': 11734194,
            'color_description': 'Gray'
        },
        6: {
            'rgb': (0, 0, 0),
            'name': 'Background',
            'semantic': 'No data, shadow, background',
            'percentage': 0.08,
            'pixels': 105956,
            'color_description': 'Black'
        }
    }
    
    print("📊 IDENTIFIED CLASSES:")
    print("-" * 60)
    
    slum_percentage = 0
    for class_id, info in classes.items():
        r, g, b = info['rgb']
        print(f"Class {class_id}: RGB({r:3d}, {g:3d}, {b:3d}) | {info['percentage']:6.2f}% | {info['name']}")
        print(f"         {info['semantic']}")
        
        if 'Slum' in info['name']:
            slum_percentage += info['percentage']
        print()
    
    print("🏘️ SLUM DETECTION SUMMARY:")
    print("-" * 30)
    print(f"✅ This is a MULTI-CLASS segmentation problem")
    print(f"🎯 TWO classes represent slum areas:")
    print(f"   • Class 2: Slum Type 1 (15.84%)")
    print(f"   • Class 4: Slum Type 2 (11.84%)")
    print(f"📊 Total slum coverage: {slum_percentage:.2f}%")
    print(f"📊 Non-slum coverage: {100-slum_percentage:.2f}%")
    
    print(f"\n⚖️ CLASS IMBALANCE ISSUES:")
    print("-" * 30)
    print(f"🚨 SEVERE class imbalance detected!")
    print(f"   • Largest class (Water): 31.89%")
    print(f"   • Smallest class (Background): 0.08%")
    print(f"   • Imbalance ratio: 386:1")
    print(f"💡 REQUIRED: Weighted loss functions!")
    
    print(f"\n🧠 MODEL ARCHITECTURE REQUIREMENTS:")
    print("-" * 40)
    print(f"📐 Input: RGB images (120×120×3)")
    print(f"🎯 Output: 7 classes (120×120×7)")
    print(f"🏗️ Architecture: UNET with 7 output channels")
    print(f"📊 Loss Function: Focal Loss (handles imbalance)")
    print(f"🎛️ Class Weights: Essential due to severe imbalance")
    
    # Calculate proper class weights
    total_pixels = sum(info['pixels'] for info in classes.values())
    
    print(f"\n🎛️ CALCULATED CLASS WEIGHTS:")
    print("-" * 30)
    for class_id, info in classes.items():
        weight = total_pixels / (7 * info['pixels'])
        print(f"Class {class_id} ({info['name'][:15]}): {weight:.3f}")
    
    print(f"\n🎯 TRAINING STRATEGY:")
    print("-" * 20)
    print(f"1. 📊 Use Focal Loss with α=0.25, γ=2.0")
    print(f"2. ⚖️ Apply calculated class weights")
    print(f"3. 🔄 Data augmentation for minority classes")
    print(f"4. 📈 Monitor IoU per class (especially slum classes)")
    print(f"5. 🎲 Random sampling to balance training batches")
    
    # Save clean summary
    dataset_summary = {
        'total_masks': 8910,
        'total_pixels': int(total_pixels),
        'num_classes': 7,
        'image_size': [120, 120],
        'format': 'RGB masks → Class indices',
        'classes': {}
    }
    
    for class_id, info in classes.items():
        dataset_summary['classes'][str(class_id)] = {
            'rgb': list(info['rgb']),
            'name': info['name'],
            'semantic': info['semantic'],
            'percentage': float(info['percentage']),
            'pixels': int(info['pixels']),
            'weight': float(total_pixels / (7 * info['pixels']))
        }
    
    with open('dataset_class_summary.json', 'w') as f:
        json.dump(dataset_summary, f, indent=2)
    
    print(f"\n💾 Summary saved to 'dataset_class_summary.json'")
    return dataset_summary

if __name__ == "__main__":
    summary = create_dataset_summary()
    
    print(f"\n🚀 READY FOR MODEL DEVELOPMENT!")
    print(f"✅ Class structure identified")
    print(f"✅ Training strategy defined")
    print(f"✅ Class weights calculated")
    print(f"🎯 Next: Build comprehensive UNET model!")
