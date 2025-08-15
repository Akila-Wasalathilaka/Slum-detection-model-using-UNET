#!/usr/bin/env python3
"""
Batch Prediction Script
=======================
Run the trained model over a folder of images and save overlays, masks, and probabilities.
"""

import os
import sys
from pathlib import Path
import argparse
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from global_slum_detector import GlobalSlumDetector


def find_images(input_dir):
    exts = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")
    for p in sorted(Path(input_dir).glob("**/*")):
        if p.suffix.lower() in exts:
            yield p


ess = """
Usage example:
python scripts/batch_predict.py --checkpoint best_global_model.pth \
    --input data/test/images --output outputs --use-tta --use-tent --max 50
"""


def main():
    parser = argparse.ArgumentParser(description="Batch predict slum overlays and masks")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint .pth")
    parser.add_argument("--input", required=True, help="Input folder with images")
    parser.add_argument("--output", default="outputs", help="Output folder")
    parser.add_argument("--use-tta", action="store_true", help="Enable test-time augmentations")
    parser.add_argument("--use-tent", action="store_true", help="Enable TENT adaptation")
    parser.add_argument("--max", type=int, default=0, help="Max images to process (0 = all)")
    args = parser.parse_args()

    out_dir = Path(args.output)
    (out_dir / "overlays").mkdir(parents=True, exist_ok=True)
    (out_dir / "masks").mkdir(parents=True, exist_ok=True)
    (out_dir / "probs").mkdir(parents=True, exist_ok=True)

    detector = GlobalSlumDetector(args.checkpoint, device="auto")

    count = 0
    for img_path in find_images(args.input):
        if args.max and count >= args.max:
            break
        try:
            res = detector.predict_global(str(img_path), use_tta=args.use_tta, use_tent=args.use_tent)
            name = img_path.stem
            cv2.imwrite(str(out_dir / "overlays" / f"{name}_overlay.jpg"), cv2.cvtColor(res["overlay"], cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(out_dir / "masks" / f"{name}_mask.png"), (res["binary_mask"] * 255).astype(np.uint8))
            cv2.imwrite(str(out_dir / "probs" / f"{name}_prob.png"), (res["probability"] * 255).astype(np.uint8))
            count += 1
        except Exception as e:
            print(f"Failed: {img_path} -> {e}")

    print(f"Done. Saved results for {count} images to {out_dir}")


if __name__ == "__main__":
    main()
