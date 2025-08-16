"""
Predict multiclass segmentation (e.g., background/slum/water) and save overlays.
Usage example (PowerShell):
  python scripts/predict_multiclass.py --checkpoint experiments/.../checkpoints/best.pth \
      --images data/test/images --output outputs_multiclass --classes background,slum,water
"""

import argparse
from pathlib import Path
import cv2
import numpy as np
import torch

from models import create_model

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_image(path, size=None):
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    orig_hw = img.shape[:2]
    if size is not None and (orig_hw[0] != size[0] or orig_hw[1] != size[1]):
        img = cv2.resize(img, (size[1], size[0]))
    return img, orig_hw


essential_colors = {
    'background': (0, 0, 0),
    'slum': (255, 0, 0),      # red
    'water': (0, 153, 255),   # orange-blue
}


def to_tensor(img):
    x = img.astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    x = np.transpose(x, (2, 0, 1))
    return torch.from_numpy(x).unsqueeze(0)


def overlay_mask(rgb, mask_idx, classes):
    color = np.zeros_like(rgb)
    for i, name in enumerate(classes):
        if name in essential_colors and i != 0:  # skip background coloring by default
            c = essential_colors[name]
            m = (mask_idx == i)[..., None]
            color[m.repeat(3, axis=2)] = c
    out = (0.6 * color + 0.4 * rgb).astype(np.uint8)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--images', required=True)
    ap.add_argument('--output', default='outputs_multiclass')
    ap.add_argument('--arch', default='unet')
    ap.add_argument('--encoder', default='resnet34')
    ap.add_argument('--input-h', type=int, default=120)
    ap.add_argument('--input-w', type=int, default=120)
    ap.add_argument('--classes', type=str, default='background,slum,water')
    args = ap.parse_args()

    classes = [c.strip() for c in args.classes.split(',') if c.strip()]
    num_classes = len(classes)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_model(
        architecture=args.arch,
        encoder=args.encoder,
        pretrained=False,
        num_classes=num_classes,
        in_channels=3
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    state = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(state, strict=False)
    model.eval()

    in_size = (args.input_h, args.input_w)

    img_dir = Path(args.images)
    out_dir = Path(args.output)
    (out_dir / 'overlays').mkdir(parents=True, exist_ok=True)
    (out_dir / 'masks').mkdir(parents=True, exist_ok=True)

    exts = ['*.png', '*.jpg', '*.jpeg', '*.tif']
    img_paths = []
    for e in exts:
        img_paths.extend(img_dir.glob(e))

    for p in img_paths:
        rgb, orig_hw = load_image(p, in_size)
        x = to_tensor(rgb).to(device)
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()  # (C,H,W)
        pred_idx = np.argmax(probs, axis=0).astype(np.uint8)

        # Resize back if needed
        if orig_hw != in_size:
            pred_idx = cv2.resize(pred_idx, (orig_hw[1], orig_hw[0]), interpolation=cv2.INTER_NEAREST)
            rgb = cv2.resize(rgb, (orig_hw[1], orig_hw[0]))

        overlay = overlay_mask(rgb, pred_idx, classes)

        cv2.imwrite(str(out_dir / 'overlays' / (p.stem + '_overlay.jpg')), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        # Save per-class mask indices as PNG
        cv2.imwrite(str(out_dir / 'masks' / (p.stem + '_mask.png')), pred_idx)

    print(f"Saved predictions to: {out_dir}")


if __name__ == '__main__':
    main()
