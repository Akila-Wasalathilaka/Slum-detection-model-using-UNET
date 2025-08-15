"""
Idempotent Colab setup and single-run training
=============================================
Pure Python version (no notebook magics) to avoid double clone or nested dirs.
Run with: `python colab_simple.py`
"""

import os
import subprocess
import sys


def ensure_repo():
	repo = "Slum-detection-model-using-UNET"
	cwd = os.getcwd()
	already_in_repo = (
		os.path.isfile(os.path.join(cwd, "requirements.txt"))
		and os.path.isdir(os.path.join(cwd, "scripts"))
		and os.path.isdir(os.path.join(cwd, "utils"))
	)
	if already_in_repo:
		return cwd

	base_dir = "/content" if os.path.isdir("/content") else cwd
	repo_dir = os.path.join(base_dir, repo)
	if not os.path.exists(repo_dir):
		subprocess.run(["git", "-C", base_dir, "clone", "https://github.com/Akila-Wasalathilaka/Slum-detection-model-using-UNET.git"], check=True)
	return repo_dir


def main():
	repo_dir = ensure_repo()
	os.chdir(repo_dir)
	# Install deps (safe to re-run)
	subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
	# Train once
	subprocess.run([sys.executable, "scripts/train_global.py", "--data_root", "data", "--batch_size", "8", "--epochs", "50", "--lr", "1e-4"], check=True)


if __name__ == "__main__":
	main()