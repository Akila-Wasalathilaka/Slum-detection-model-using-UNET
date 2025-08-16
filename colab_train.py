#!/usr/bin/env python3
"""
Backward-compat wrapper for legacy notebooks.
Delegates to the unified trainer: scripts/train.py

Usage:
    python colab_train.py --model upscale --training upscale --data upscale
    # Any args are forwarded to scripts/train.py
"""

import sys
import subprocess


def main():
        args = sys.argv[1:]
        cmd = [sys.executable, "scripts/train.py", *args]
        print("[compat] Delegating to:", " ".join(cmd))
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
        main()