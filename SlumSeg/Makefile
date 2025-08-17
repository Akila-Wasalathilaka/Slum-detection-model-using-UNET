# Default pipeline for Kaggle T4
.PHONY: all install analyze prep train evaluate infer clean

# Main pipeline
all: install analyze prep train evaluate infer

# Install dependencies
install:
	pip install -r requirements.txt --no-input

# Dataset analysis (generates first batch of charts)
analyze:
	python scripts/analyze_dataset.py --config configs/default.yaml --out outputs/charts

# Prepare dataset (tiling + splits + weights)
prep:
	python scripts/prep_dataset.py --config configs/default.yaml --out /kaggle/working/tiles

# Train single model
train:
	python scripts/train.py --config configs/default.yaml --tiles /kaggle/working/tiles

# Train 20-model ensemble (optional, heavy)
train-ensemble:
	python scripts/kfold_train.py --config configs/default.yaml --folds configs/folds.yaml --tiles /kaggle/working/tiles

# Evaluate and generate 20 charts
evaluate:
	python scripts/evaluate.py --config configs/default.yaml --ckpt outputs/checkpoints/best.ckpt --tiles /kaggle/working/tiles --charts outputs/charts

# Generate 20 prediction overlays
infer:
	python scripts/infer.py --config configs/default.yaml --ckpt outputs/checkpoints/best.ckpt --images /kaggle/working/tiles/val/images --out outputs/predictions --num 20

# Export models
export:
	python scripts/export_onnx.py --config configs/default.yaml --ckpt outputs/checkpoints/best.ckpt --out outputs/models

# Benchmark performance
benchmark:
	python scripts/speed_bench.py --config configs/default.yaml --ckpt outputs/checkpoints/best.ckpt

# Package results
package:
	cd outputs && zip -r slumseg_artifacts.zip charts predictions checkpoints models

# Clean outputs
clean:
	rm -rf outputs/*
	rm -rf /kaggle/working/tiles

# Development helpers
lint:
	black slumseg scripts
	isort slumseg scripts
	flake8 slumseg scripts

test:
	pytest tests/ -v

# Quick local test (smaller dataset)
test-local:
	python scripts/train.py --config configs/local_test.yaml --tiles data/sample_tiles
