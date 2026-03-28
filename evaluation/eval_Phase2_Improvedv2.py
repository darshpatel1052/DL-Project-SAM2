"""
Phase 2 Improved v2 Evaluation Script

Runs the advanced motion-aware SAM 2 tracker on LaSOT/GOT-10k.
"""

import argparse
import json
import numpy as np
from pathlib import Path
from typing import List, Tuple
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from models.Phase2_Improvedv2 import SAM2KalmanTrackerV2

CONFIG = {
    "got10k_root": PROJECT_ROOT / "datasets" / "got10k",
    "lasot_root": PROJECT_ROOT / "datasets" / "lasot_small" / "small_LaSOT",
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class Phase2TrackerV2Wrapper:
    """Wrapper for SAM2KalmanTrackerV2 with configurable parameters."""

    def __init__(self, confidence_threshold: float = 0.7,
                 occlusion_threshold: float = 0.3,
                 lost_threshold: float = 0.15,
                 recovery_frames: int = 5,
                 use_appearance: bool = True,
                 use_smoothing: bool = True,
                 spatial_consistency_weight: float = 0.3):
        self.tracker = SAM2KalmanTrackerV2(
            mode="kalman",
            confidence_threshold=confidence_threshold,
            occlusion_threshold=occlusion_threshold,
            lost_threshold=lost_threshold,
            recovery_frames=recovery_frames,
            use_appearance=use_appearance,
            use_smoothing=use_smoothing,
            spatial_consistency_weight=spatial_consistency_weight,
        )

    def track(self, img_dir: str, init_bbox: List[int], num_frames: int,
              chunk_size: int = 200) -> Tuple[List[List[int]], List[float]]:
        return self.tracker.track(img_dir, init_bbox, num_frames, chunk_size)


def compute_iou(b1: List[int], b2: List[int]) -> float:
    x1, y1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    x2 = min(b1[0] + b1[2], b2[0] + b2[2])
    y2 = min(b1[1] + b1[3], b2[1] + b2[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    union = b1[2] * b1[3] + b2[2] * b2[3] - inter
    return inter / union if union > 0 else 0.0


def _convert_to_json_serializable(pred_bboxes):
    """Convert numpy types to native Python types for JSON serialization."""
    result = []
    for bbox in pred_bboxes:
        if bbox is None:
            result.append(None)
        else:
            result.append([int(x) for x in bbox])
    return result


def _save_results(path, results, args, start_time):
    j_scores = [r["j_metric"] for r in results]
    with open(path, 'w') as f:
        json.dump({
            "dataset": args.dataset,
            "model": "Phase 2 Improved v2 (Advanced Motion-Aware)",
            "parameters": {
                "confidence_threshold": args.confidence_threshold,
                "occlusion_threshold": args.occlusion_threshold,
                "lost_threshold": args.lost_threshold,
                "recovery_frames": args.recovery_frames,
                "chunk_size": args.chunk_size,
                "use_appearance": args.use_appearance,
                "use_smoothing": args.use_smoothing,
                "spatial_consistency_weight": args.spatial_consistency_weight,
            },
            "total_sequences": len(results),
            "overall_j_metric": round(float(np.mean(j_scores)), 4) if j_scores else 0,
            "elapsed_seconds": round(time.time() - start_time, 1),
            "sequences": results,
        }, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Phase 2 Improved v2 Evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Dataset args
    parser.add_argument("--dataset", choices=["got10k_val", "lasot"], required=True)
    parser.add_argument("--max-sequences", type=int, default=0)
    parser.add_argument("--resume", action="store_true")

    # Tracker tuning args
    parser.add_argument("--chunk-size", type=int, default=200)
    parser.add_argument("--confidence-threshold", type=float, default=0.7)
    parser.add_argument("--occlusion-threshold", type=float, default=0.3)
    parser.add_argument("--lost-threshold", type=float, default=0.15)
    parser.add_argument("--recovery-frames", type=int, default=5)

    # Advanced v2 parameters
    parser.add_argument("--use-appearance", type=bool, default=True,
                        help="Use appearance model for detection validation")
    parser.add_argument("--use-smoothing", type=bool, default=True,
                        help="Use temporal smoothing for stable predictions")
    parser.add_argument("--spatial-consistency-weight", type=float, default=0.3,
                        help="Weight for spatial consistency penalty")

    args = parser.parse_args()

    RESULTS_DIR = PROJECT_ROOT / "results"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"PHASE 2 IMPROVED v2 EVALUATION - {args.dataset.upper()}")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"\nCore Parameters:")
    print(f"  chunk_size: {args.chunk_size}")
    print(f"  confidence_threshold: {args.confidence_threshold}")
    print(f"  occlusion_threshold: {args.occlusion_threshold}")
    print(f"  lost_threshold: {args.lost_threshold}")
    print(f"  recovery_frames: {args.recovery_frames}")
    print(f"\nAdvanced v2 Parameters:")
    print(f"  use_appearance: {args.use_appearance}")
    print(f"  use_smoothing: {args.use_smoothing}")
    print(f"  spatial_consistency_weight: {args.spatial_consistency_weight}")
    print("=" * 70)

    # Load dataset
    if args.dataset == "got10k_val":
        from datasets.dataset_loaders import GOT10kDataset
        dataset = GOT10kDataset(root_dir=str(CONFIG["got10k_root"]), split='val')
    elif args.dataset == "lasot":
        from datasets.dataset_loaders import LaSOTDataset
        dataset = LaSOTDataset(root_dir=str(CONFIG["lasot_root"]), split='all')

    total_seqs = len(dataset)
    n_eval = total_seqs if args.max_sequences == 0 else min(args.max_sequences, total_seqs)
    print(f"\nEvaluating {n_eval}/{total_seqs} sequences")

    results_file = RESULTS_DIR / f"{args.dataset}_Phase2_Improvedv2_evaluation.json"
    existing = {}
    if args.resume and results_file.exists():
        with open(results_file) as f:
            data = json.load(f)
            existing = {r["sequence"]: r for r in data.get("sequences", [])}
        print(f"Resuming: {len(existing)} sequences already done")

    # Initialize tracker
    tracker = Phase2TrackerV2Wrapper(
        confidence_threshold=args.confidence_threshold,
        occlusion_threshold=args.occlusion_threshold,
        lost_threshold=args.lost_threshold,
        recovery_frames=args.recovery_frames,
        use_appearance=args.use_appearance,
        use_smoothing=args.use_smoothing,
        spatial_consistency_weight=args.spatial_consistency_weight,
    )

    results = []
    start_time = time.time()

    for s in range(n_eval):
        seq_dict = dataset[s]
        seq_name = seq_dict["name"]

        if seq_name in existing:
            results.append(existing[seq_name])
            print(f"\n[{s+1}/{n_eval}] {seq_name}: SKIPPED (cached)")
            continue

        img_files = seq_dict["frames"]
        anno = seq_dict["groundtruth"]
        gt_bboxes = anno.astype(int).tolist()
        img_dir = str(Path(img_files[0]).parent)
        num_frames = len(img_files)

        print(f"\n[{s+1}/{n_eval}] {seq_name}: {num_frames} frames")

        pred_bboxes, occ_scores = tracker.track(
            img_dir, gt_bboxes[0], num_frames, chunk_size=args.chunk_size
        )

        # Compute metrics
        j_scores = []
        iou_per_frame = []
        visible_frames = 0
        occluded_frames = 0

        for i in range(min(num_frames, len(pred_bboxes))):
            gt = gt_bboxes[i] if i < len(gt_bboxes) else None
            pred = pred_bboxes[i] if i < len(pred_bboxes) else None

            if pred is None:
                iou_per_frame.append(None)
                occluded_frames += 1
            elif gt:
                iou = compute_iou(gt, pred)
                j_scores.append(iou)
                iou_per_frame.append(round(iou, 4))
                visible_frames += 1
            else:
                iou_per_frame.append(None)

        avg_j = float(np.mean(j_scores)) if j_scores else 0.0
        avg_occ = float(np.mean(occ_scores)) if occ_scores else 0.0

        result = {
            "sequence": seq_name,
            "num_frames": num_frames,
            "visible_frames": visible_frames,
            "occluded_frames": occluded_frames,
            "j_metric": round(avg_j, 4),
            "occlusion_score": round(avg_occ, 4),
            "predictions": _convert_to_json_serializable(pred_bboxes),
            "occlusion_scores_per_frame": [round(float(o), 4) for o in occ_scores],
            "iou_per_frame": iou_per_frame,
        }

        print(f"  -> J(IoU)={avg_j:.3f} | Visible={visible_frames} | Occluded={occluded_frames}")

        results.append(result)

        if DEVICE == "cuda":
            torch.cuda.empty_cache()

        _save_results(results_file, results, args, start_time)

    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)

    if results:
        js = [r["j_metric"] for r in results]
        print(f"\nOverall avg J (IoU): {np.mean(js):.4f}")
        print(f"Total time: {elapsed:.1f}s")
        print(f"Results saved to: {results_file}")


if __name__ == "__main__":
    main()
