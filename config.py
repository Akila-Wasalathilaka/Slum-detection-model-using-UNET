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
    
    # Ultra-high resolution for maximum accuracy
    IMAGE_SIZES = [256, 320, 384, 512]  # Multi-scale ultra-high resolution
    PRIMARY_SIZE = 384  # Increased from 160 for maximum detail capture
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
    
    # Ultra-optimized training parameters
    BATCH_SIZE = 8              # Reduced for higher resolution
    EPOCHS = 120                # Increased for better convergence
    NUM_EPOCHS = 120            # Add this for compatibility
    LEARNING_RATE = 5e-5        # More conservative for stability
    WEIGHT_DECAY = 2e-4         # Increased regularization
    
    # Advanced training settings
    WARMUP_EPOCHS = 10
    PATIENCE = 25               # Increased patience for ultra-accuracy
    MIN_DELTA = 5e-5
    GRAD_CLIP_VALUE = 0.5       # Tighter gradient clipping
    
    # Hardware optimization for high resolution
    NUM_WORKERS = 6 if os.cpu_count() >= 6 else 4
    PIN_MEMORY = False  # Disable pin_memory for CPU training
    
    # Reproducibility
    SEED = 42
    
    # Ultra-optimized loss function weights for slum detection
    LOSS_WEIGHTS = {
        'dice': 0.35,
        'focal': 0.25, 
        'bce': 0.15,
        'tversky': 0.15,
        'boundary': 0.1     # New boundary loss for precise edges
    }
    
    # Focal loss parameters - Optimized for informal settlements
    FOCAL_ALPHA = 0.25          # Better for minority class
    FOCAL_GAMMA = 3.0           # Stronger focus on hard examples
    
    # Tversky loss parameters - Optimized for slum recall
    TVERSKY_ALPHA = 0.7         # Favor recall for slum detection
    TVERSKY_BETA = 0.3
    
    # Ultra-high performance encoder options
    ENCODERS = [
        'timm-efficientnet-b5',     # Best performance for high-res
        'timm-efficientnet-b4',     # Balanced performance
        'timm-regnety_032',         # Fast and accurate
        'resnet101',                # Robust baseline
        'timm-convnext_base'        # State-of-the-art CNN
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
    
    # Multi-scale inference for maximum accuracy
    MULTI_SCALE_INFERENCE = True
    SCALE_FACTORS = [0.8, 1.0, 1.2, 1.4]
    
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
