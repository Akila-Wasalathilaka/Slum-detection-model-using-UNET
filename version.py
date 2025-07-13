"""
Version information for the Slum Detection Model
"""

__version__ = "1.0.0"
__title__ = "Advanced Slum Detection Using Deep Learning"
__description__ = "State-of-the-art UNet-based semantic segmentation for detecting informal settlements in satellite imagery"
__url__ = "https://github.com/Akila-Wasalathilaka/Slum-detection-model-using-UNET"
__author__ = "Akila Wasalathilaka"
__author_email__ = "akila.wasalathilaka@example.com"  # Update with actual email
__license__ = "MIT"
__copyright__ = "Copyright 2025 Akila Wasalathilaka"

# Version components
VERSION_MAJOR = 1
VERSION_MINOR = 0
VERSION_PATCH = 0
VERSION_INFO = (VERSION_MAJOR, VERSION_MINOR, VERSION_PATCH)

# Release information
RELEASE_DATE = "2025-07-13"
RELEASE_NAME = "Genesis"
RELEASE_CODENAME = "Initial Release"

# Performance metrics for this version
PERFORMANCE_METRICS = {
    "auc_roc": 0.9967,
    "accuracy": 0.9889,
    "f1_score": 0.9567,
    "precision": 0.9423,
    "recall": 0.9715,
    "specificity": 0.9914
}

# Model information
MODEL_INFO = {
    "architecture": "unet",
    "encoder": "resnet34",
    "input_size": (120, 120),
    "output_classes": 2,  # slum, non-slum
    "loss_function": "combined_bce_dice_focal",
    "optimizer": "adamw",
    "scheduler": "cosine_annealing"
}

# System requirements
REQUIREMENTS = {
    "python": ">=3.8",
    "pytorch": ">=2.0.0",
    "cuda": ">=11.0",  # Optional but recommended
    "memory": "8GB RAM",
    "storage": "10GB free space"
}

def get_version():
    """Get the current version string."""
    return __version__

def get_version_info():
    """Get detailed version information."""
    return {
        "version": __version__,
        "title": __title__,
        "description": __description__,
        "author": __author__,
        "release_date": RELEASE_DATE,
        "release_name": RELEASE_NAME,
        "performance": PERFORMANCE_METRICS,
        "model_info": MODEL_INFO
    }

def print_version_info():
    """Print formatted version information."""
    print(f"🏘️  {__title__}")
    print(f"📦 Version: {__version__} ({RELEASE_NAME})")
    print(f"👤 Author: {__author__}")
    print(f"📅 Release Date: {RELEASE_DATE}")
    print(f"🏆 AUC-ROC: {PERFORMANCE_METRICS['auc_roc']:.4f}")
    print(f"🎯 Accuracy: {PERFORMANCE_METRICS['accuracy']:.4f}")
    print(f"🔗 URL: {__url__}")

if __name__ == "__main__":
    print_version_info()
