#!/usr/bin/env python3
"""
DEPRECATED: Use the unified trainer instead:
    python scripts/train.py --data_root data --batch_size 16 --epochs 100 --lr 1e-4
"""

import sys

def main():
    print("[deprecated] scripts/train_global.py → use scripts/train.py")
    raise SystemExit(0)

if __name__ == "__main__":
    main()