#!/usr/bin/env python3
"""
Evaluation and Charts Generator
===============================
Generates per-image prediction panels and aggregate charts from a trained checkpoint.
- Saves at least N per-image panels (default 10)
- Produces 10+ charts: ROC, PR, metric bars/hists, threshold sweep, calibration, uncertainty, etc.
"""

import os
import sys
import argparse
from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import Optional, List, Tuple

from sklearn.metrics import (
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    roc_auc_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from global_slum_detector import GlobalSlumDetector


def find_images(input_dir: str) -> List[Path]:
    exts = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")
    return [p for p in sorted(Path(input_dir).glob("**/*")) if p.suffix.lower() in exts]


def load_mask(mask_path: Path) -> Optional[np.ndarray]:
    if not mask_path.exists():
        return None
    m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        return None
    m = (m > 127).astype(np.uint8)
    return m


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def per_image_panel(out_dir: Path, name: str, rgb: np.ndarray, overlay: np.ndarray, prob: np.ndarray,
                    mask: Optional[np.ndarray], pred_mask: np.ndarray):
    plt.figure(figsize=(12, 9))
    # 2x2 grid
    plt.subplot(2, 2, 1)
    plt.imshow(rgb)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(2, 2, 2)
    plt.imshow(overlay)
    plt.title("Overlay")
    plt.axis("off")

    plt.subplot(2, 2, 3)
    plt.imshow(prob, cmap="viridis")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.title("Probability")
    plt.axis("off")

    plt.subplot(2, 2, 4)
    if mask is not None:
        # TP(green), FP(red), FN(blue)
        tp = (pred_mask == 1) & (mask == 1)
        fp = (pred_mask == 1) & (mask == 0)
        fn = (pred_mask == 0) & (mask == 1)
        vis = np.zeros((*pred_mask.shape, 3), dtype=np.uint8)
        vis[tp] = (0, 255, 0)
        vis[fp] = (255, 0, 0)
        vis[fn] = (0, 0, 255)
        plt.imshow(vis)
        plt.title("TP/FP/FN")
    else:
        plt.imshow(pred_mask, cmap="gray")
        plt.title("Binary Prediction")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(str(out_dir / f"panel_{name}.png"), dpi=150)
    plt.close()


def plot_roc_pr(out_dir: Path, y_true: np.ndarray, y_score: np.ndarray):
    # ROC
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = roc_auc_score(y_true, y_score)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC Curve"); plt.legend(); plt.tight_layout()
    plt.savefig(str(out_dir / "roc_curve.png"), dpi=150)
    plt.close()

    # PR
    p, r, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    plt.figure(figsize=(6, 5))
    plt.plot(r, p, label=f"AP={ap:.3f}")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("PR Curve"); plt.legend(); plt.tight_layout()
    plt.savefig(str(out_dir / "pr_curve.png"), dpi=150)
    plt.close()


def plot_metric_bars(out_dir: Path, names: List[str], iou: List[float], f1: List[float], acc: List[float]):
    x = np.arange(len(names))
    width = 0.25
    plt.figure(figsize=(max(8, len(names) * 0.8), 5))
    plt.bar(x - width, iou, width, label="IoU")
    plt.bar(x, f1, width, label="F1")
    plt.bar(x + width, acc, width, label="Acc")
    plt.xticks(x, names, rotation=45, ha="right")
    plt.ylim(0, 1)
    plt.legend(); plt.title("Per-image Metrics"); plt.tight_layout()
    plt.savefig(str(out_dir / "metrics_bars.png"), dpi=150)
    plt.close()


def plot_hists(out_dir: Path, values: List[np.ndarray], labels: List[str], title: str, fname: str):
    plt.figure(figsize=(6, 5))
    for v, lbl in zip(values, labels):
        plt.hist(v, bins=30, alpha=0.5, label=lbl)
    plt.title(title)
    plt.legend(); plt.tight_layout()
    plt.savefig(str(out_dir / fname), dpi=150)
    plt.close()


def threshold_sweep(out_dir: Path, y_true: np.ndarray, y_score: np.ndarray):
    ths = np.linspace(0, 1, 51)
    ious, f1s = [], []
    for t in ths:
        pred = (y_score >= t).astype(np.uint8)
        inter = np.logical_and(y_true == 1, pred == 1).sum()
        union = np.logical_or(y_true == 1, pred == 1).sum()
        iou = inter / (union + 1e-8)
        tp = inter
        fp = ((y_true == 0) & (pred == 1)).sum()
        fn = ((y_true == 1) & (pred == 0)).sum()
        f1 = 2 * tp / (2 * tp + fp + fn + 1e-8)
        ious.append(iou)
        f1s.append(f1)
    plt.figure(figsize=(6, 5))
    plt.plot(ths, ious, label="IoU")
    plt.plot(ths, f1s, label="F1")
    plt.xlabel("Threshold"); plt.ylabel("Score"); plt.title("Threshold Sweep"); plt.legend(); plt.tight_layout()
    plt.savefig(str(out_dir / "threshold_sweep.png"), dpi=150)
    plt.close()


def calibration_curve_plot(out_dir: Path, y_true: np.ndarray, y_score: np.ndarray, bins: int = 10):
    bin_edges = np.linspace(0, 1, bins + 1)
    inds = np.digitize(y_score, bin_edges) - 1
    avg_conf, avg_acc = [], []
    for b in range(bins):
        mask = inds == b
        if mask.sum() == 0:
            continue
        avg_conf.append(y_score[mask].mean())
        avg_acc.append((y_true[mask] == 1).mean())
    plt.figure(figsize=(6, 5))
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.plot(avg_conf, avg_acc, marker="o")
    plt.xlabel("Confidence"); plt.ylabel("Empirical Accuracy"); plt.title("Calibration Curve"); plt.tight_layout()
    plt.savefig(str(out_dir / "calibration_curve.png"), dpi=150)
    plt.close()


def conf_matrix_plot(out_dir: Path, y_true: np.ndarray, y_pred: np.ndarray):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xticks([0, 1], ["Neg", "Pos"]) ; plt.yticks([0, 1], ["Neg", "Pos"])
    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, str(v), ha='center', va='center')
    plt.tight_layout()
    plt.savefig(str(out_dir / "confusion_matrix.png"), dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Evaluate model and generate charts")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint .pth")
    parser.add_argument("--images", required=True, help="Folder of images")
    parser.add_argument("--masks", default="", help="Folder of GT masks (optional)")
    parser.add_argument("--output", default="charts", help="Output directory for charts")
    parser.add_argument("--num", type=int, default=10, help="Number of images to evaluate")
    parser.add_argument("--use-tta", action="store_true", help="Enable TTA")
    args = parser.parse_args()

    out_dir = Path(args.output)
    ensure_dir(out_dir)
    ensure_dir(out_dir / "panels")

    det = GlobalSlumDetector(args.checkpoint, device="auto")

    images = find_images(args.images)
    if len(images) == 0:
        print("No images found.")
        return
    images = images[: args.num]

    y_true_all: List[np.ndarray] = []
    y_score_all: List[np.ndarray] = []
    names: List[str] = []
    ious: List[float] = []
    f1s: List[float] = []
    accs: List[float] = []
    uncerts: List[float] = []

    masks_dir = Path(args.masks) if args.masks else None

    for img_path in images:
        name = img_path.stem
        img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            print(f"Skip unreadable: {img_path}")
            continue
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        res = det.predict_global(str(img_path), use_tta=args.use_tta, use_tent=False)
        overlay = res["overlay"]
        prob = res["probability"].astype(np.float32)
        pred_mask = res["binary_mask"].astype(np.uint8)
        uncert = res.get("uncertainty")
        mean_uncert = float(np.mean(uncert)) if isinstance(uncert, np.ndarray) else 0.0

        mask = None
        if masks_dir:
            cand = masks_dir / f"{name}.png"
            if not cand.exists():
                cand = masks_dir / f"{name}.jpg"
            if not cand.exists():
                cand = masks_dir / f"{name}.tif"
            mask = load_mask(cand) if cand.exists() else None

        # Per-image panel
        per_image_panel(out_dir / "panels", name, rgb, overlay, prob, mask, pred_mask)

        # Metrics if GT present
        if mask is not None and mask.shape == pred_mask.shape:
            y_true = mask.flatten()
            y_score = prob.flatten()
            y_pred = pred_mask.flatten()
            inter = np.logical_and(mask == 1, pred_mask == 1).sum()
            union = np.logical_or(mask == 1, pred_mask == 1).sum()
            iou = float(inter / (union + 1e-8))
            f1 = f1_score(y_true, y_pred, zero_division=0)
            acc = accuracy_score(y_true, y_pred)

            y_true_all.append(y_true)
            y_score_all.append(y_score)
            names.append(name)
            ious.append(iou)
            f1s.append(float(f1))
            accs.append(float(acc))
            uncerts.append(mean_uncert)

    # Aggregate charts only if we have GT
    if len(y_true_all) > 0:
        y_true_cat = np.concatenate(y_true_all)
        y_score_cat = np.concatenate(y_score_all)

        plot_roc_pr(out_dir, y_true_cat, y_score_cat)
        plot_metric_bars(out_dir, names, ious, f1s, accs)
        plot_hists(out_dir, [y_score_cat], ["score"], "Probability Distribution", "probs_hist.png")
        threshold_sweep(out_dir, y_true_cat, y_score_cat)
        calibration_curve_plot(out_dir, y_true_cat, y_score_cat)
        y_pred_cat = (y_score_cat >= 0.5).astype(np.uint8)
        conf_matrix_plot(out_dir, y_true_cat, y_pred_cat)
        if len(uncerts) > 0:
            plot_hists(out_dir, [np.array(uncerts)], ["uncertainty"], "Uncertainty (mean per image)", "uncertainty_hist.png")

        # Save CSV
        try:
            import csv
            with open(out_dir / "per_image_metrics.csv", "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["image", "iou", "f1", "acc", "mean_uncertainty"])
                for n, i, f, a, u in zip(names, ious, f1s, accs, uncerts):
                    w.writerow([n, i, f, a, u])
        except Exception as e:
            print(f"Could not write metrics CSV: {e}")

    print(f"Done. Panels: { (out_dir / 'panels').resolve() } | Charts: { out_dir.resolve() }")


if __name__ == "__main__":
    main()
