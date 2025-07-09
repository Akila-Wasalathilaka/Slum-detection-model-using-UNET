"""
Test script to check memory usage and validate fixes.
"""
import torch
import os
from config import config
from model import UltraAccurateUNet

def test_memory_usage():
    """Test if the model fits in GPU memory."""
    print("🧪 Testing Memory Usage...")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        # Clear memory
        torch.cuda.empty_cache()
        
        # Set memory fraction
        torch.cuda.set_per_process_memory_fraction(0.8)
        
        print(f"GPU: {torch.cuda.get_device_name()}")
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"Total GPU Memory: {total_memory:.1f} GB")
        
        # Check initial memory
        initial_memory = torch.cuda.memory_allocated() / 1e9
        print(f"Initial memory usage: {initial_memory:.2f} GB")
    
    try:
        # Create model
        print(f"\n📦 Creating model with size {config.PRIMARY_SIZE}x{config.PRIMARY_SIZE}")
        model = UltraAccurateUNet().to(device)
        
        if torch.cuda.is_available():
            model_memory = torch.cuda.memory_allocated() / 1e9
            print(f"Memory after model loading: {model_memory:.2f} GB")
        
        # Test forward pass with different batch sizes
        batch_sizes = [1, 2, 4]
        
        for batch_size in batch_sizes:
            try:
                print(f"\n🧮 Testing batch size {batch_size}...")
                
                # Create dummy input
                dummy_input = torch.randn(batch_size, 3, config.PRIMARY_SIZE, config.PRIMARY_SIZE).to(device)
                
                if torch.cuda.is_available():
                    input_memory = torch.cuda.memory_allocated() / 1e9
                    print(f"Memory after input creation: {input_memory:.2f} GB")
                
                # Forward pass
                with torch.no_grad():
                    output = model(dummy_input)
                
                if torch.cuda.is_available():
                    peak_memory = torch.cuda.max_memory_allocated() / 1e9
                    print(f"Peak memory usage: {peak_memory:.2f} GB")
                    
                    # Clear memory
                    del dummy_input, output
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                
                print(f"✅ Batch size {batch_size} works!")
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"❌ Batch size {batch_size} failed: Out of memory")
                else:
                    print(f"❌ Batch size {batch_size} failed: {e}")
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
        
        print(f"\n✅ Memory test completed!")
        print(f"📊 Recommended batch size: {config.BATCH_SIZE}")
        print(f"📊 Image size: {config.PRIMARY_SIZE}x{config.PRIMARY_SIZE}")
        
    except Exception as e:
        print(f"❌ Error during memory test: {e}")

if __name__ == "__main__":
    test_memory_usage()
