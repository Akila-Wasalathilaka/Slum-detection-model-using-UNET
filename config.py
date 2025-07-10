"""
Configuration and hyperparameters for production slum detection pipeline.
"""
from pathlib import Path
import os

class UltraAccurateSlumConfig:
    """Ultra-accurate slum detection configuration - Best of the best!"""
    BASE_DIR = Path(os.getcwd())
    DATA_DIR = BASE_DIR / "data_preprocessed"
    DATA_ROOT = DATA_DIR  # Add this for compatibility
    RESULTS_DIR = BASE_DIR / "results_production"
    MODEL_DIR = BASE_DIR / "models_production"
    MODEL_SAVE_DIR = MODEL_DIR  # Add this for compatibility
    CHECKPOINT_DIR = MODEL_DIR / "checkpoints"
    TRAIN_IMG_DIR = DATA_DIR / "train" / "images"
    TRAIN_MASK_DIR = DATA_DIR / "train" / "masks"
    VAL_IMG_DIR = DATA_DIR / "val" / "images"
    VAL_MASK_DIR = DATA_DIR / "val" / "masks"
    TEST_IMG_DIR = DATA_DIR / "test" / "images"
    TEST_MASK_DIR = DATA_DIR / "test" / "masks"
    
    # Optimized resolution for 4GB GPU memory
    IMAGE_SIZES = [160, 224, 256, 320]  # Reduced for memory efficiency
    PRIMARY_SIZE = 224  # Reduced from 384 to fit in 4GB GPU
    INPUT_CHANNELS = 3
    OUTPUT_CHANNELS = 1
    
    # Slum class identification - Ultra precise
    SLUM_CLASS_ID = 2
    BINARY_THRESHOLDS = {
        'ultra_conservative': 0.8,  # Highest precision
        'conservative': 0.65,       # High precision  
        'balanced': 0.45,           # Balanced precision/recall
        'sensitive': 0.3,           # High recall
        'ultra_sensitive': 0.15     # Maximum recall
    }
    DEFAULT_THRESHOLD = 'balanced'
    
    # Re-optimized training parameters for better accuracy
    BATCH_SIZE = 8              # Increased for better gradient estimates
    EPOCHS = 50                 # Increased training time
    NUM_EPOCHS = 50             # Add this for compatibility
    LEARNING_RATE = 3e-4        # Increased learning rate with warm-up
    WEIGHT_DECAY = 2e-4         # Slightly increased regularization
    
    # Advanced training settings for high accuracy
    WARMUP_EPOCHS = 3           # Reduced warmup
    PATIENCE = 8                # Reduced patience for faster training
    MIN_DELTA = 1e-4            # Larger minimum delta
    GRAD_CLIP_VALUE = 1.0       # Higher gradient clipping for stability
    
    # Hardware optimization for 4GB GPU
    NUM_WORKERS = 2 if os.cpu_count() >= 4 else 1  # Reduced workers
    PIN_MEMORY = False  # Disable pin_memory for memory efficiency
    
    # Reproducibility
    SEED = 42
    
    # Re-optimized loss function weights for slum detection
    LOSS_WEIGHTS = {
        'dice': 0.35,           # Balanced Dice loss
        'focal': 0.25,          # Focal loss for hard examples
        'bce': 0.15,            # BCE for basic segmentation
        'tversky': 0.15,        # Tversky for class imbalance
        'boundary': 0.10        # Re-enabled boundary loss for better edge detection
    }
    
    # Optimized focal loss parameters for slum detection
    FOCAL_ALPHA = 0.25          # Keep balanced
    FOCAL_GAMMA = 2.0           # Reduced for easier training
    
    # Tversky loss parameters - Optimized for slum recall
    TVERSKY_ALPHA = 0.7         # Favor recall for slum detection
    TVERSKY_BETA = 0.3
    
    # Memory-efficient encoder for 4GB GPU
    ENCODERS = [
        'timm-efficientnet-b2',     # Smaller than b5 for memory efficiency
        'timm-efficientnet-b1',     # Even smaller option
        'resnet50',                 # Lighter than resnet101
        'timm-regnety_016',         # Smaller RegNet
    ]
    ENCODER_WEIGHTS = 'imagenet'
    
    # Enhanced Test Time Augmentation
    TTA_TRANSFORMS = 12         # More transforms for accuracy
    
    # Ultra-precise post-processing parameters
    MIN_OBJECT_SIZE = 15        # Smaller minimum for detailed detection
    MORPHOLOGY_KERNEL = 3       # Smaller kernel for precision
    
    # Ensemble parameters for ultra-accuracy
    ENSEMBLE_MODELS = 5         # More models for better ensemble
    UNCERTAINTY_THRESHOLD = 0.05 # Stricter uncertainty filtering
    
    # Enable multi-scale inference for better accuracy
    MULTI_SCALE_INFERENCE = True   # Enabled for better predictions
    SCALE_FACTORS = [0.8, 1.0, 1.2]  # Multiple scales for inference
    
    # Advanced slum detection features
    ENABLE_BOUNDARY_REFINEMENT = True
    ENABLE_CONTEXT_ANALYSIS = True
    ENABLE_DENSITY_ANALYSIS = True
    
    # Mapping and visualization parameters
    SLUM_MARKER_COLOR = (255, 0, 0)      # Red markers like reference
    SLUM_MARKER_SIZE = 8
    CONFIDENCE_THRESHOLD_FOR_MARKING = 0.7
    
    # Geographic analysis parameters
    ENABLE_GEOGRAPHIC_CLUSTERING = True
    CLUSTER_MIN_SAMPLES = 3
    CLUSTER_EPS = 50  # meters

config = UltraAccurateSlumConfig()
