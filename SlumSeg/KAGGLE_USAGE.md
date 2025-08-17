# SlumSeg Kaggle Pipelines

## 🚂 Pipeline 1: Training Only

```python
# 1. Clone repo
import os, subprocess
os.chdir('/kaggle/working')
subprocess.run('git clone https://github.com/YOUR-ORG/SlumSeg.git', shell=True)
os.chdir('SlumSeg')

# 2. Install dependencies
subprocess.run('pip install -r requirements.txt --no-input', shell=True)

# 3. Update config for Kaggle
import yaml
with open('configs/default.yaml', 'r') as f:
    config = yaml.safe_load(f)
config['data']['root'] = '/kaggle/input/YOUR-DATASET-NAME'
config['train']['batch_size'] = 8
config['train']['epochs'] = 15
with open('configs/kaggle.yaml', 'w') as f:
    yaml.dump(config, f)

# 4. Train model
subprocess.run('python scripts/train.py --config configs/kaggle.yaml', shell=True)

# 5. Package results
import zipfile
with zipfile.ZipFile('trained_model.zip', 'w') as z:
    for f in os.listdir('outputs/checkpoints'):
        z.write(f'outputs/checkpoints/{f}', f'checkpoints/{f}')
print("✅ Download trained_model.zip from Output panel")
```

## 📊 Pipeline 2: Analysis & Evaluation

```python
# 1. Clone repo
import os, subprocess
os.chdir('/kaggle/working')
subprocess.run('git clone https://github.com/YOUR-ORG/SlumSeg.git', shell=True)
os.chdir('SlumSeg')

# 2. Install dependencies
subprocess.run('pip install -r requirements.txt --no-input', shell=True)

# 3. Update config
import yaml
with open('configs/default.yaml', 'r') as f:
    config = yaml.safe_load(f)
config['data']['root'] = '/kaggle/input/YOUR-DATASET-NAME'
with open('configs/kaggle.yaml', 'w') as f:
    yaml.dump(config, f)

# 4. Run analysis pipeline
MODEL_PATH = '/kaggle/input/your-model/best.ckpt'  # UPDATE THIS

# Dataset analysis
subprocess.run('python scripts/analyze_dataset.py --config configs/kaggle.yaml --out outputs/charts', shell=True)

# Model evaluation (if model exists)
if os.path.exists(MODEL_PATH):
    subprocess.run(f'python scripts/evaluate.py --config configs/kaggle.yaml --ckpt {MODEL_PATH} --tiles . --charts outputs/charts', shell=True)
    subprocess.run(f'python scripts/infer.py --config configs/kaggle.yaml --ckpt {MODEL_PATH} --images /kaggle/input/YOUR-DATASET-NAME/val/images --out outputs/predictions --num 20', shell=True)

# 5. Package results
import zipfile
with zipfile.ZipFile('analysis_results.zip', 'w') as z:
    for root, dirs, files in os.walk('outputs'):
        for file in files:
            file_path = os.path.join(root, file)
            z.write(file_path, file_path.replace('outputs/', ''))
print("✅ Download analysis_results.zip from Output panel")
```

## 🔧 Setup Instructions

1. **Update variables**:
   - Replace `YOUR-ORG` with your GitHub username
   - Replace `YOUR-DATASET-NAME` with your Kaggle dataset name
   - Update `MODEL_PATH` in Pipeline 2

2. **Dataset structure**:
   ```
   your-dataset/
   ├── train/images/*.tif
   ├── train/masks/*.tif
   ├── val/images/*.tif
   └── val/masks/*.tif
   ```

3. **Run**:
   - Copy Pipeline 1 code → Train model → Download `trained_model.zip`
   - Copy Pipeline 2 code → Generate analysis → Download `analysis_results.zip`