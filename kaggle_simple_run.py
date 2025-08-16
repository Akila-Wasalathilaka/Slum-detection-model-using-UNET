# 🏘️ SIMPLE KAGGLE SCRIPT - Use Akila's Existing Pipeline
# Copy and paste this into Kaggle cell

import os, sys, subprocess
import matplotlib.pyplot as plt
import numpy as np

print("🚀 Cloning and running Akila's UNET pipeline...")

# Clone and setup
os.chdir('/kaggle/working')
subprocess.run("git clone https://github.com/Akila-Wasalathilaka/Slum-detection-model-using-UNET.git", shell=True)
os.chdir('/kaggle/working/Slum-detection-model-using-UNET')

# Install requirements if they exist
if os.path.exists('requirements.txt'):
    print("📦 Installing requirements...")
    subprocess.run("pip install -r requirements.txt", shell=True)
else:
    print("📦 Installing common dependencies...")
    subprocess.run("pip install torch torchvision opencv-python pillow matplotlib numpy scikit-learn", shell=True)

print("📂 Repository structure:")
for item in os.listdir('.'):
    if os.path.isdir(item):
        print(f"  📁 {item}/")
    else:
        print(f"  📄 {item}")

# Look for training scripts
training_files = []
for root, dirs, files in os.walk('.'):
    for file in files:
        if 'train' in file.lower() and file.endswith('.py'):
            training_files.append(os.path.join(root, file))

print(f"\n🏋️ Found training scripts:")
for tf in training_files:
    print(f"  {tf}")

# Look for main scripts
main_files = []
for file in os.listdir('.'):
    if file.endswith('.py') and any(name in file.lower() for name in ['main', 'run', 'train', 'model']):
        main_files.append(file)

print(f"\n🎯 Main scripts found:")
for mf in main_files:
    print(f"  {mf}")

# Try to run the main training script
if main_files:
    main_script = main_files[0]
    print(f"\n🚀 Running {main_script}...")
    try:
        result = subprocess.run(f"python {main_script}", shell=True, capture_output=True, text=True, timeout=600)
        print("✅ Training output:")
        print(result.stdout)
        if result.stderr:
            print("⚠️ Warnings/Errors:")
            print(result.stderr)
    except subprocess.TimeoutExpired:
        print("⏰ Training taking longer than 10 minutes - continuing...")
    except Exception as e:
        print(f"❌ Error running script: {e}")

# Look for results/outputs
print("\n📊 Looking for results...")
result_dirs = ['results', 'output', 'outputs', 'models', 'checkpoints']
for dir_name in result_dirs:
    if os.path.exists(dir_name):
        print(f"📁 Found {dir_name}:")
        for item in os.listdir(dir_name)[:10]:  # Show first 10 items
            print(f"  {item}")

# Look for images/plots
image_files = []
for root, dirs, files in os.walk('.'):
    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_files.append(os.path.join(root, file))

if image_files:
    print(f"\n🖼️ Found {len(image_files)} images:")
    for img in image_files[:5]:  # Show first 5
        print(f"  {img}")

# Try to run prediction/inference if available
inference_files = []
for file in os.listdir('.'):
    if any(name in file.lower() for name in ['predict', 'inference', 'test']) and file.endswith('.py'):
        inference_files.append(file)

if inference_files:
    print(f"\n🔍 Running inference with {inference_files[0]}...")
    try:
        result = subprocess.run(f"python {inference_files[0]}", shell=True, capture_output=True, text=True, timeout=300)
        print("✅ Inference output:")
        print(result.stdout[-1000:])  # Last 1000 chars
    except Exception as e:
        print(f"⚠️ Inference failed: {e}")

# Display any generated images
print("\n📊 Displaying results...")
for img_path in image_files[:3]:  # Display first 3 images
    try:
        from PIL import Image
        img = Image.open(img_path)
        plt.figure(figsize=(10, 6))
        plt.imshow(img)
        plt.title(f"Result: {os.path.basename(img_path)}")
        plt.axis('off')
        plt.show()
    except Exception as e:
        print(f"Could not display {img_path}: {e}")

# Final summary
print("\n" + "="*60)
print("🏆 PIPELINE EXECUTION SUMMARY")
print("="*60)
print(f"📁 Repository: /kaggle/working/Slum-detection-model-using-UNET")
print(f"📄 Training scripts found: {len(training_files)}")
print(f"🖼️ Images generated: {len(image_files)}")
print(f"📊 Result directories: {[d for d in result_dirs if os.path.exists(d)]}")

# Quick demo if no results
if not image_files:
    print("\n📊 Creating quick visualization demo...")
    
    # Simple demo plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Demo satellite image
    np.random.seed(42)
    demo_img = np.random.uniform(0.3, 0.7, (100, 100, 3))
    demo_img[30:60, 40:70] = [0.8, 0.5, 0.3]  # Slum area
    
    axes[0].imshow(demo_img)
    axes[0].set_title('Demo Satellite Image')
    axes[0].axis('off')
    
    # Demo prediction
    axes[1].imshow(demo_img)
    overlay = np.zeros((100, 100, 3))
    overlay[30:60, 40:70] = [1, 0, 0]
    axes[1].imshow(overlay, alpha=0.5)
    axes[1].set_title('Demo Slum Detection')
    axes[1].axis('off')
    
    # Demo metrics
    epochs = range(1, 11)
    accuracy = [0.65 + 0.03*i + np.random.uniform(-0.02, 0.02) for i in epochs]
    axes[2].plot(epochs, accuracy, 'b-', linewidth=2)
    axes[2].set_title('Demo Training Accuracy')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Accuracy')
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.show()

print("\n🎉 COMPLETE! Check the outputs above for results.")
print("💡 If training scripts didn't run automatically, check the file list and run them manually.")
