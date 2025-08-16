"""
Scan mask images to list unique RGB colors and their pixel counts.
Helps configure DataConfig.class_rgb_map for multiclass training.

Usage (PowerShell):
  python scripts/inspect_mask_colors.py --masks data/train/masks --top 20
"""

import argparse
from pathlib import Path
import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--masks', required=True)
    ap.add_argument('--top', type=int, default=20)
    args = ap.parse_args()

    mask_dir = Path(args.masks)
    exts = ['*.png', '*.jpg', '*.jpeg', '*.tif']
    files = []
    for e in exts:
        files.extend(mask_dir.glob(e))
    if not files:
        print(f"No masks found in {mask_dir}")
        return

    counts = {}
    for p in files:
        img = cv2.imread(str(p))
        if img is None:
            continue
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        flat = rgb.reshape(-1, 3)
        # use structured array to count rows quickly
        view = np.ascontiguousarray(flat).view([('', flat.dtype)] * 3)
        uniques, idx, inv, c = np.unique(view, return_index=True, return_inverse=True, return_counts=True)
        for u, cnt in zip(uniques, c):
            color = tuple(flat[idx][0])  # representative value
            counts[color] = counts.get(color, 0) + int(cnt)

    sorted_items = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    print(f"Top {args.top} colors in {mask_dir}:")
    for (r, g, b), c in sorted_items[:args.top]:
        print(f"RGB({r},{g},{b}) -> {c} pixels")


if __name__ == '__main__':
    main()
