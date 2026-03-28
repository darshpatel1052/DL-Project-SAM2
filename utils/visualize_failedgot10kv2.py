"""
Visualize Failed GOT-10k Sequences with Phase 2 Improved v2 Tracker

This script re-runs and visualizes GOT-10k sequences that previously failed
with the baseline tracker, using the advanced HiM2SAM-style v2 tracker.
"""

import argparse
import json
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import sys
import re

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from models.Phase2_Improvedv2 import SAM2KalmanTrackerV2

CONFIG = {
    "got10k_val_root": PROJECT_ROOT / "datasets" / "got10k" / "val",
    "sam2_checkpoint": PROJECT_ROOT / "models" / "sam2.1_hiera_small.pt",
    "sam2_config": "configs/sam2.1/sam2.1_hiera_s.yaml",
}

VIDEO_FPS = 15
VIDEO_CODEC = "mp4v"
RESULTS_DIR = PROJECT_ROOT / "results"
VIDEO_OUTPUT_DIR = RESULTS_DIR / "videos"
FAILED_VIDEO_DIR = VIDEO_OUTPUT_DIR / "failed_got10k"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_failed_sequences() -> List[str]:
    """Extract sequence names from failed video filenames."""
    if not FAILED_VIDEO_DIR.exists():
        print(f"Warning: {FAILED_VIDEO_DIR} does not exist")
        return []

    sequences = []
    for f in sorted(FAILED_VIDEO_DIR.glob("*.mp4")):
        match = re.match(r"(GOT-10k_Val_\d+)_failed\.mp4", f.name)
        if match:
            sequences.append(match.group(1))

    return sequences


def load_got10k_val_sequence(sequence_name: str) -> Tuple[str, List[List[int]], int]:
    """Load a GOT-10k validation sequence."""
    seq_path = CONFIG["got10k_val_root"] / sequence_name
    if not seq_path.exists():
        raise ValueError(f"Sequence {sequence_name} not found at {seq_path}")

    frame_files = sorted(seq_path.glob("*.jpg"))
    if not frame_files:
        frame_files = sorted(seq_path.glob("*.png"))
    num_frames = len(frame_files)

    gt_file = seq_path / "groundtruth.txt"
    gt_bboxes = []
    if gt_file.exists():
        with open(gt_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.replace(',', ' ').split()
                    if len(parts) >= 4:
                        x, y, w, h = map(float, parts[:4])
                        gt_bboxes.append([int(x), int(y), int(w), int(h)])

    return str(seq_path), gt_bboxes, num_frames


def compute_iou(bbox1: List[int], bbox2: List[int]) -> float:
    """Compute IoU between two bboxes [x, y, w, h]."""
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    box1, box2 = [x1, y1, x1 + w1, y1 + h1], [x2, y2, x2 + w2, y2 + h2]

    inter_x1 = max(box1[0], box2[0])
    inter_y1 = max(box1[1], box2[1])
    inter_x2 = min(box1[2], box2[2])
    inter_y2 = min(box1[3], box2[3])

    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0

    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    union_area = (w1 * h1) + (w2 * h2) - inter_area
    return inter_area / union_area if union_area > 0 else 0.0


def draw_bbox(frame: np.ndarray, bbox: List[int], color: Tuple, label: str = "", thickness: int = 2) -> np.ndarray:
    """Draw a bounding box on a frame."""
    x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
    if label:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale, font_thickness = 0.6, 2
        (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, font_thickness)
        cv2.rectangle(frame, (x, y - text_h - 10), (x + text_w + 10, y), color, -1)
        cv2.putText(frame, label, (x + 5, y - 5), font, font_scale, (255, 255, 255), font_thickness)
    return frame


def create_tracking_video(
    img_dir: str,
    gt_bboxes: List[List[int]],
    pred_bboxes: List,
    occ_scores: List[float],
    output_path: Path,
    sequence_name: str,
    max_frames: int = 0,
    fps: int = VIDEO_FPS
) -> float:
    """Create a tracking visualization video."""
    img_dir_path = Path(img_dir)
    frame_files = sorted(img_dir_path.glob("*.jpg")) or sorted(img_dir_path.glob("*.png"))
    if not frame_files:
        return 0.0

    num_frames = len(frame_files) if max_frames <= 0 else min(max_frames, len(frame_files))
    height, width = cv2.imread(str(frame_files[0])).shape[:2]

    out = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*VIDEO_CODEC), fps, (width, height))
    total_iou, valid_frames = 0.0, 0
    occluded_frames = 0

    for i in range(num_frames):
        frame = cv2.imread(str(frame_files[i]))
        if frame is None:
            continue
        vis_frame = frame.copy()

        gt_bbox = gt_bboxes[i] if i < len(gt_bboxes) else None
        pred_bbox = pred_bboxes[i] if i < len(pred_bboxes) else None
        occ_score = occ_scores[i] if i < len(occ_scores) else 0.0

        iou = 0.0
        is_occluded = pred_bbox is None

        if is_occluded:
            occluded_frames += 1

        if gt_bbox and pred_bbox:
            iou = compute_iou(gt_bbox, pred_bbox)
            total_iou += iou
            valid_frames += 1

        # Draw GT bbox (green)
        if gt_bbox:
            vis_frame = draw_bbox(vis_frame, gt_bbox, (0, 255, 0), "GT", 2)

        # Draw predicted bbox (red for visible)
        if pred_bbox:
            vis_frame = draw_bbox(vis_frame, pred_bbox, (0, 0, 255), "v2", 2)

        # Info overlay
        info_text = [
            f"Sequence: {sequence_name}",
            f"Frame: {i+1}/{num_frames}",
            f"IoU: {iou:.3f}" if not is_occluded else "IoU: N/A (Occluded)",
            f"Conf: {1.0 - occ_score:.2f}",
        ]

        if is_occluded:
            info_text.append("Status: OCCLUDED")
        else:
            info_text.append("Status: VISIBLE")

        cv2.rectangle(vis_frame, (0, 0), (350, 130), (0, 0, 0), -1)
        font = cv2.FONT_HERSHEY_SIMPLEX
        for j, text in enumerate(info_text):
            color = (255, 255, 255)
            if "OCCLUDED" in text:
                color = (0, 255, 255)
            cv2.putText(vis_frame, text, (10, 25 + j * 22), font, 0.55, color, 2)

        # Legend
        legend_y = height - 80
        cv2.rectangle(vis_frame, (10, legend_y), (280, height - 10), (0, 0, 0), -1)
        cv2.rectangle(vis_frame, (20, legend_y + 10), (40, legend_y + 25), (0, 255, 0), -1)
        cv2.putText(vis_frame, "Ground Truth", (50, legend_y + 22), font, 0.5, (255, 255, 255), 1)
        cv2.rectangle(vis_frame, (20, legend_y + 30), (40, legend_y + 45), (0, 0, 255), -1)
        cv2.putText(vis_frame, "Phase2 Improved v2", (50, legend_y + 42), font, 0.5, (255, 255, 255), 1)
        cv2.rectangle(vis_frame, (20, legend_y + 50), (40, legend_y + 65), (0, 255, 255), -1)
        cv2.putText(vis_frame, "Occluded Frame", (50, legend_y + 62), font, 0.5, (255, 255, 255), 1)

        out.write(vis_frame)

    out.release()
    avg_iou = total_iou / valid_frames if valid_frames > 0 else 0
    print(f"  Video saved: {output_path}")
    print(f"  Visible frames: {valid_frames}, Occluded: {occluded_frames}, Avg IoU: {avg_iou:.3f}")
    return avg_iou


def process_sequence(
    sequence_name: str,
    tracker: SAM2KalmanTrackerV2,
    output_dir: Path,
    max_frames: int = 0,
    chunk_size: int = 200
) -> Optional[Dict]:
    """Process a single sequence."""
    print(f"\n{'='*60}")
    print(f"Processing: {sequence_name}")
    print('='*60)

    try:
        img_dir, gt_bboxes, num_frames = load_got10k_val_sequence(sequence_name)
    except Exception as e:
        print(f"  Error loading sequence: {e}")
        return None

    if num_frames == 0 or not gt_bboxes:
        print(f"  No frames or GT found")
        return None

    print(f"  Running Phase 2 Improved v2 Tracking ({num_frames} frames)...")
    num_frames = min(max_frames, num_frames) if max_frames > 0 else num_frames
    pred_bboxes, occ_scores = tracker.track(img_dir, gt_bboxes[0], num_frames, chunk_size=chunk_size)

    output_path = output_dir / f"{sequence_name}_phase2improvedv2.mp4"
    avg_iou = create_tracking_video(
        img_dir, gt_bboxes, pred_bboxes, occ_scores,
        output_path, sequence_name, max_frames=max_frames
    )

    occluded_count = sum(1 for p in pred_bboxes if p is None)
    visible_count = len(pred_bboxes) - occluded_count

    return {
        "sequence": sequence_name,
        "num_frames": len(pred_bboxes),
        "visible_frames": visible_count,
        "occluded_frames": occluded_count,
        "avg_iou": round(avg_iou, 4),
        "video_path": str(output_path)
    }


def main():
    parser = argparse.ArgumentParser(
        description="Visualize Failed GOT-10k Sequences with Phase 2 Improved v2 Tracker",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--sequence", type=str, default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--max-sequences", type=int, default=5)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--chunk-size", type=int, default=200)
    parser.add_argument("--list", action="store_true")

    # Tracker tuning args
    parser.add_argument("--confidence-threshold", type=float, default=0.7)
    parser.add_argument("--occlusion-threshold", type=float, default=0.3)
    parser.add_argument("--lost-threshold", type=float, default=0.15)
    parser.add_argument("--recovery-frames", type=int, default=5)

    # v2 advanced parameters
    parser.add_argument("--use-appearance", type=bool, default=True)
    parser.add_argument("--use-smoothing", type=bool, default=True)
    parser.add_argument("--spatial-consistency-weight", type=float, default=0.3)

    parser.add_argument("--output-dir", type=str, default=None)

    args = parser.parse_args()

    failed_sequences = get_failed_sequences()

    if args.list:
        print(f"\nFailed GOT-10k sequences ({len(failed_sequences)}):")
        for seq in failed_sequences:
            print(f"  - {seq}")
        return

    if not failed_sequences:
        print("No failed sequences found in", FAILED_VIDEO_DIR)
        return

    print("="*70)
    print("PHASE 2 IMPROVED v2 - FAILED GOT-10K VISUALIZATION")
    print("="*70)
    print(f"Device: {DEVICE}")
    print(f"Failed sequences found: {len(failed_sequences)}")
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
    print("="*70)

    tracker = SAM2KalmanTrackerV2(
        mode="kalman",
        confidence_threshold=args.confidence_threshold,
        occlusion_threshold=args.occlusion_threshold,
        lost_threshold=args.lost_threshold,
        recovery_frames=args.recovery_frames,
        use_appearance=args.use_appearance,
        use_smoothing=args.use_smoothing,
        spatial_consistency_weight=args.spatial_consistency_weight,
    )

    output_dir = Path(args.output_dir) if args.output_dir else VIDEO_OUTPUT_DIR / "failed_got10k_phase2improvedv2"
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.sequence:
        if args.sequence in failed_sequences:
            sequences = [args.sequence]
        else:
            print(f"Sequence {args.sequence} not found in failed list")
            print("Available:", failed_sequences[:5], "...")
            return
    elif args.all:
        sequences = failed_sequences[:args.max_sequences] if args.max_sequences > 0 else failed_sequences
    else:
        sequences = failed_sequences[:min(3, len(failed_sequences))]

    print(f"\nProcessing {len(sequences)} sequences...")

    results = []
    for seq_name in sequences:
        res = process_sequence(
            seq_name, tracker, output_dir,
            max_frames=args.max_frames,
            chunk_size=args.chunk_size
        )
        if res:
            results.append(res)

        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    if results:
        total_iou = sum(r['avg_iou'] for r in results)
        avg_iou = total_iou / len(results)

        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        print(f"Processed: {len(results)} sequences")
        print(f"Overall Average IoU: {avg_iou:.4f}")

        summary_path = RESULTS_DIR / "failed_got10k_phase2improvedv2_results.json"
        with open(summary_path, 'w') as f:
            json.dump({
                "dataset": "got10k_val_failed",
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
                "sequences": results,
                "overall_avg_iou": round(avg_iou, 4)
            }, f, indent=2)

        print(f"Results saved to: {summary_path}")
        print(f"Videos saved to: {output_dir}")


if __name__ == "__main__":
    main()
