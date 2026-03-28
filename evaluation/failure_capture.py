"""
Failure Capture Visualizer

Trigger script that evaluates generated tracking JSONs against their dataset grounds truths 
and renders specifically annotated clipping output for sequences yielding a failing J-metric (IoU < threshold).
"""

import argparse
import json
import cv2
import numpy as np
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Reuse visualizer loaders
from evaluation.visualize_baseline import (
    load_lasot_sequence,
    load_got10k_sequence,
    load_got10k_val_sequence,
    load_otb_sequence
)
from evaluation.metrics import compute_iou

VIDEO_FPS = 15

def save_failure_video(img_dir: str, gt_bboxes: list, pred_bboxes: list,
                       output_path: Path, seq_name: str, avg_j: float, model_name: str):
    """Create video visualization for a failed sequence comparing GT and Pred."""
    img_path = Path(img_dir)
    frame_files = sorted(img_path.glob("*.jpg"))
    if not frame_files:
        frame_files = sorted(img_path.glob("*.png"))

    num_frames = min(len(frame_files), len(pred_bboxes))
    if num_frames == 0: return
    
    first = cv2.imread(str(frame_files[0]))
    if first is None: return
    h, w = first.shape[:2]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), VIDEO_FPS, (w, h))

    for i in range(num_frames):
        frame = cv2.imread(str(frame_files[i]))
        if frame is None: continue

        gt = gt_bboxes[i] if i < len(gt_bboxes) else None
        pred = pred_bboxes[i] if i < len(pred_bboxes) else None

        j_score = compute_iou(gt, pred) if gt and pred else 0.0

        if gt:
            cv2.rectangle(frame, (int(gt[0]), int(gt[1])),
                          (int(gt[0]+gt[2]), int(gt[1]+gt[3])), (0, 255, 0), 2)
        if pred:
            cv2.rectangle(frame, (int(pred[0]), int(pred[1])),
                          (int(pred[0]+pred[2]), int(pred[1]+pred[3])), (0, 0, 255), 2)

        cv2.rectangle(frame, (0, 0), (450, 80), (0, 0, 0), -1)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, f"{seq_name} (FAILED J-METRIC)", (10, 25), font, 0.6, (0, 0, 255), 2)
        cv2.putText(frame, f"Frame: {i+1}/{num_frames}  J-Score: {j_score:.3f}", (10, 50), font, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Avg J-Score: {avg_j:.3f}", (10, 72), font, 0.5, (100, 100, 255), 1)

        ly = h - 50
        cv2.rectangle(frame, (10, ly), (200, h - 5), (0, 0, 0), -1)
        cv2.rectangle(frame, (15, ly+5), (30, ly+18), (0, 255, 0), -1)
        cv2.putText(frame, "Ground Truth", (35, ly+16), font, 0.4, (255, 255, 255), 1)
        cv2.rectangle(frame, (15, ly+22), (30, ly+35), (0, 0, 255), -1)
        cv2.putText(frame, model_name, (35, ly+33), font, 0.4, (255, 255, 255), 1)

        out.write(frame)

    out.release()

def load_dataset_sequence(dataset: str, sequence_name: str):
    if dataset == "lasot":
        return load_lasot_sequence(sequence_name)
    elif dataset in ["got10k_val", "got10k-val"]:
        return load_got10k_val_sequence(sequence_name)
    elif dataset in ["otb", "otb2015"]:
        return load_otb_sequence(sequence_name)
    else:
        return load_got10k_sequence(sequence_name)

def main():
    parser = argparse.ArgumentParser(description="Failure Video Generation triggered from JSON outputs.")
    parser.add_argument("--json", required=True, type=str, help="Path to `_evaluation.json` output file.")
    parser.add_argument("--j-threshold", type=float, default=0.6, help="J-Metric (IoU) failure threshold.")
    args = parser.parse_args()

    json_path = Path(args.json)
    if not json_path.exists():
        print(f"Error: JSON file {json_path} does not exist.")
        return

    with open(json_path, 'r') as f:
        data = json.load(f)

    dataset_name = data.get("dataset", "unknown").lower()
    model_name = data.get("model", "Tracker")
    sequences = data.get("sequences", [])

    print(f"Loaded {len(sequences)} sequences from {json_path.name}")
    print(f"Dataset: {dataset_name} | Model: {model_name}")
    print(f"Failure Threshold: J < {args.j_threshold}")
    
    # Check what key houses predictions
    pred_key = "baseline_prediction" if "baseline" in json_path.name.lower() else "Phase2_Improved_prediction"
    
    output_dir = PROJECT_ROOT / "results" / "videos" / "failures" / json_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    failed_count = 0

    for seq_data in sequences:
        seq_name = seq_data.get("sequence")
        j_metric = seq_data.get("j_metric", 1.0) # alias for iou fallback
        if "avg_iou" in seq_data: 
            j_metric = seq_data["avg_iou"]

        if j_metric < args.j_threshold:
            print(f"  [TRIGGER] {seq_name} J={j_metric:.3f}")
            pred_bboxes = seq_data.get(pred_key, [])
            if not pred_bboxes:
                pred_bboxes = seq_data.get("predictions", []) # Legacy fallback
            
            if not pred_bboxes:
                print(f"    ERROR: No predictions found for {seq_name} in JSON under keys: {pred_key}")
                continue
            
            try:
                img_dir, gt_bboxes, _ = load_dataset_sequence(dataset_name, seq_name)
            except Exception as e:
                print(f"    ERROR loading sequence ground truth: {e}")
                continue

            output_vid_path = output_dir / f"{seq_name}_failed.mp4"
            print(f"    -> Saving failure video to {output_vid_path.name}")
            
            save_failure_video(
                img_dir=img_dir,
                gt_bboxes=gt_bboxes, 
                pred_bboxes=pred_bboxes,
                output_path=output_vid_path,
                seq_name=seq_name,
                avg_j=j_metric,
                model_name=model_name
            )
            failed_count += 1
            
    print(f"\nCompleted! Generated {failed_count} failure analysis videos.")
    if failed_count > 0:
        print(f"Videos saved securely in: {output_dir}")

if __name__ == "__main__":
    main()
