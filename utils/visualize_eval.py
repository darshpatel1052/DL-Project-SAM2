"""
Visualize Evaluation JSON

Reads an evaluation JSON file and generates side-by-side tracking comparison videos
(Ground Truth vs Predicted) for every sequence in the JSON.
"""

import argparse
import json
import cv2
import numpy as np
from pathlib import Path
import sys
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Use the robust dataloaders we just built
from datasets.dataset_loaders import GOT10kDataset, LaSOTDataset

VIDEO_FPS = 15

def get_dataset_loader(dataset_name: str):
    CONFIG = {
        "lasot_root": PROJECT_ROOT / "datasets" / "lasot_small" / "small_LaSOT",
        "got10k_val_root": PROJECT_ROOT / "datasets" / "got10k",
        "got10k_root": PROJECT_ROOT / "datasets" / "got10k" / "test",
    }
    
    if dataset_name.lower() == "lasot":
        return LaSOTDataset(root_dir=str(CONFIG["lasot_root"]), split='all')
    elif dataset_name.lower() in ["got10k_val", "got10k-val"]:
        return GOT10kDataset(root_dir=str(CONFIG["got10k_val_root"]), split='val')
    else:
        # Default GOT-10k test
        return GOT10kDataset(root_dir=str(CONFIG["got10k_root"]), split='test')

def compute_iou(b1, b2):
    if b1 is None or len(b1) == 0 or b2 is None or len(b2) == 0: return 0.0
    x1, y1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    x2 = min(b1[0] + b1[2], b2[0] + b2[2])
    y2 = min(b1[1] + b1[3], b2[1] + b2[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    union = b1[2] * b1[3] + b2[2] * b2[3] - inter
    return inter / union if union > 0 else 0.0

def save_comparison_video(img_files: list, gt_bboxes: list, pred_bboxes: list,
                          output_path: Path, seq_name: str, avg_j: float, model_name: str):
    """Create video visualization for a sequence comparing GT and Pred."""
    num_frames = min(len(img_files), len(pred_bboxes))
    if num_frames == 0: return
    
    first = cv2.imread(str(img_files[0]))
    if first is None: return
    h, w = first.shape[:2]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), VIDEO_FPS, (w, h))

    for i in range(num_frames):
        frame = cv2.imread(str(img_files[i]))
        if frame is None: continue

        gt = gt_bboxes[i] if gt_bboxes is not None and i < len(gt_bboxes) else None
        pred = pred_bboxes[i] if pred_bboxes is not None and i < len(pred_bboxes) else None

        j_score = compute_iou(gt, pred)

        if gt is not None and len(gt) >= 4:
            cv2.rectangle(frame, (int(gt[0]), int(gt[1])),
                          (int(gt[0]+gt[2]), int(gt[1]+gt[3])), (0, 255, 0), 2)
        if pred is not None and len(pred) >= 4:
            cv2.rectangle(frame, (int(pred[0]), int(pred[1])),
                          (int(pred[0]+pred[2]), int(pred[1]+pred[3])), (0, 0, 255), 2)

        cv2.rectangle(frame, (0, 0), (450, 80), (0, 0, 0), -1)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, f"{seq_name}", (10, 25), font, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Frame: {i+1}/{num_frames}  J-Score: {j_score:.3f}", (10, 50), font, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Avg J-Score: {avg_j:.3f}", (10, 72), font, 0.5, (100, 255, 100) if avg_j > 0.5 else (100, 100, 255), 1)

        ly = h - 50
        cv2.rectangle(frame, (10, ly), (200, h - 5), (0, 0, 0), -1)
        cv2.rectangle(frame, (15, ly+5), (30, ly+18), (0, 255, 0), -1)
        cv2.putText(frame, "Ground Truth", (35, ly+16), font, 0.4, (255, 255, 255), 1)
        cv2.rectangle(frame, (15, ly+22), (30, ly+35), (0, 0, 255), -1)
        cv2.putText(frame, model_name, (35, ly+33), font, 0.4, (255, 255, 255), 1)

        out.write(frame)

    out.release()

def main():
    parser = argparse.ArgumentParser(description="Full JSON Evaluation Video Generation.")
    parser.add_argument("--json", required=True, type=str, help="Path to `_evaluation.json` output file.")
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
    
    pred_key = "baseline_prediction" if "baseline" in json_path.name.lower() else "Phase2_Improved_prediction"
    
    output_dir = PROJECT_ROOT / "results" / "videos" / "eval" / json_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        dataset_loader = get_dataset_loader(dataset_name)
    except Exception as e:
        print(f"Failed to load dataset {dataset_name}: {e}")
        return

    print(f"\nGenerating videos in {output_dir}")
    for seq_data in tqdm(sequences, desc="Processing Sequences"):
        seq_name = seq_data.get("sequence")
        j_metric = seq_data.get("j_metric", 1.0)
        if "avg_iou" in seq_data: 
            j_metric = seq_data["avg_iou"]

        pred_bboxes = seq_data.get(pred_key, [])
        if not pred_bboxes:
            pred_bboxes = seq_data.get("predictions", []) 
        
        if not pred_bboxes:
            continue
        
        try:
            seq_dict = dataset_loader.get_sequence(seq_name)
            img_files = seq_dict["frames"]
            gt_bboxes = seq_dict["groundtruth"]
        except Exception as e:
            print(f"\n  [ERROR] loading GT for {seq_name}: {e}")
            continue

        output_vid_path = output_dir / f"{seq_name}_eval.mp4"
        
        save_comparison_video(
            img_files=img_files,
            gt_bboxes=gt_bboxes, 
            pred_bboxes=pred_bboxes,
            output_path=output_vid_path,
            seq_name=seq_name,
            avg_j=j_metric,
            model_name=model_name
        )
            
    print(f"\nCompleted! Generated videos for {len(sequences)} sequences.")
    print(f"Videos saved securely in: {output_dir}")

if __name__ == "__main__":
    main()
