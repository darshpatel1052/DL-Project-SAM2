"""
Baseline Model Evaluation Script

Runs pure SAM 2 tracking on specified datasets (GOT-10k val or LaSOT),
evaluating failures and saving visualization clips.
"""

import argparse
import json
import cv2
import numpy as np
from pathlib import Path
from typing import List
import sys
import time

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from got10k.datasets import GOT10k
from sam2.build_sam import build_sam2_video_predictor
from models.baseline import SAM2Tracker, DEVICE

# Config paths
CONFIG = {
    "got10k_root": PROJECT_ROOT / "datasets" / "got10k",
    "lasot_root": PROJECT_ROOT / "datasets" / "lasot_small" / "small_LaSOT",
    "sam2_checkpoint": PROJECT_ROOT / "models" / "sam2.1_hiera_small.pt",
    "sam2_config": "configs/sam2.1/sam2.1_hiera_s.yaml",
}

VIDEO_FPS = 15

# ============================================================================
# IoU & Video Creation
# ============================================================================
def compute_iou(b1: List[int], b2: List[int]) -> float:
    """Compute IoU between two [x, y, w, h] boxes."""
    x1, y1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    x2 = min(b1[0] + b1[2], b2[0] + b2[2])
    y2 = min(b1[1] + b1[3], b2[1] + b2[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    union = b1[2] * b1[3] + b2[2] * b2[3] - inter
    return inter / union if union > 0 else 0.0


def compute_avg_iou(gt_bboxes, pred_bboxes):
    total, count = 0.0, 0
    for i in range(min(len(gt_bboxes), len(pred_bboxes))):
        if gt_bboxes[i] is None or pred_bboxes[i] is None:
            continue
        total += compute_iou(gt_bboxes[i], pred_bboxes[i])
        count += 1
    return total / count if count > 0 else 0.0

from evaluation.metrics import compute_f_score

def _save_results(path, results, args, start_time):
    j_scores = [r["j_metric"] for r in results]
    with open(path, 'w') as f:
        json.dump({
            "dataset": f"{args.dataset}",
            "model": "SAM 2.1 Hiera Small Baseline",
            "total_sequences": len(results),
            "overall_j_metric": round(float(np.mean(j_scores)), 4) if j_scores else 0,
            "elapsed_seconds": round(time.time() - start_time, 1),
            "sequences": results,
        }, f, indent=2)

# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="Baseline Evaluation - Native Metrics")
    parser.add_argument("--dataset", choices=["got10k_val", "lasot"], required=True,
                       help="Dataset to evaluate.")
    parser.add_argument("--max-sequences", type=int, default=0,
                       help="Max sequences to evaluate (0=all)")
    parser.add_argument("--chunk-size", type=int, default=100,
                       help="Frames per SAM 2 chunk (default: 100)")
    parser.add_argument("--resume", action="store_true",
                       help="Skip sequences already in results JSON")
    args = parser.parse_args()

    RESULTS_DIR = PROJECT_ROOT / "results"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"{args.dataset.upper()} BASELINE EVALUATION")
    print("=" * 60)
    print(f"Device: {DEVICE}")

    # Load dataset
    if args.dataset == "got10k_val":
        from datasets.dataset_loaders import GOT10kDataset
        dataset = GOT10kDataset(root_dir=str(CONFIG["got10k_root"]), split='val')
    elif args.dataset == "lasot":
        from datasets.dataset_loaders import LaSOTDataset
        dataset = LaSOTDataset(root_dir=str(CONFIG["lasot_root"]), split='all')

    total_seqs = len(dataset)
    n_eval = total_seqs if args.max_sequences == 0 else min(args.max_sequences, total_seqs)
    print(f"Sequences: {n_eval}/{total_seqs}")

    results_file = RESULTS_DIR / f"{args.dataset}_baseline_evaluation.json"
    existing = {}
    if args.resume and results_file.exists():
        with open(results_file) as f:
            data = json.load(f)
            existing = {r["sequence"]: r for r in data.get("sequences", [])}
        print(f"Resuming: {len(existing)} sequences already done")

    tracker = SAM2Tracker(
        sam2_config=CONFIG["sam2_config"],
        sam2_checkpoint=str(CONFIG["sam2_checkpoint"])
    )

    results = []
    skipped = 0

    start_time = time.time()

    for s in range(n_eval):
        seq_dict = dataset[s]
        seq_name = seq_dict["name"]

        if seq_name in existing:
            results.append(existing[seq_name])
            skipped += 1
            print(f"\n[{s+1}/{n_eval}] {seq_name}: Skipped (Cached)")
            continue

        img_files = seq_dict["frames"]
        anno = seq_dict["groundtruth"]
        gt_bboxes = anno.astype(int).tolist()  # [x, y, w, h]
        img_dir = str(Path(img_files[0]).parent)
        num_frames = len(img_files)

        print(f"\n[{s+1}/{n_eval}] {seq_name}: {num_frames} frames", end="", flush=True)

        pred_bboxes, occ_scores = tracker.track(img_dir, gt_bboxes[0], num_frames,
                                                chunk_size=args.chunk_size)

        # Compute metric arrays
        j_scores = []
        f_scores = []
        for i in range(min(num_frames, len(pred_bboxes))):
            gt = gt_bboxes[i] if i < len(gt_bboxes) else None
            pred = pred_bboxes[i] if i < len(pred_bboxes) else None
            if gt and pred:
                j_scores.append(compute_iou(gt, pred))
                f_scores.append(compute_f_score(gt, pred))

        avg_j = float(np.mean(j_scores)) if j_scores else 0.0
        avg_f = float(np.mean(f_scores)) if f_scores else 0.0
        avg_occ = float(np.mean(occ_scores)) if occ_scores else 0.0

        result = {
            "sequence": seq_name,
            "num_frames": num_frames,
            "j_metric": round(avg_j, 4),
            "f_metric": float(round(avg_f, 4)),
            "occlusion_score": float(round(avg_occ, 4)),
            "baseline_prediction": pred_bboxes,
        }

        print(f" -> J(IoU)={avg_j:.3f} | F={avg_f:.3f} | Occ={avg_occ:.3f}")

        results.append(result)

        if DEVICE == "cuda":
            torch.cuda.empty_cache()

        if (s + 1) % 1 == 0 or s == n_eval - 1:
            _save_results(results_file, results, args, start_time)

    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
    if results:
        js = [r["j_metric"] for r in results]
        fs = [r["f_metric"] for r in results]
        print(f"\nOverall avg J (IoU): {np.mean(js):.3f}")
        print(f"Overall avg F (Boundary): {np.mean(fs):.3f}")

if __name__ == "__main__":
    main()
