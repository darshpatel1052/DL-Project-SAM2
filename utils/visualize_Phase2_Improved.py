"""
Phase 2 Improved Video Visualization Script

This script creates video visualizations of Phase 2 logic using SAM 2
augmented with Kalman tracking and the Extended Memory Bank.
"""

import argparse
import json
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import sys
import os

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from evaluation.eval_Phase2_Improved import Phase2TrackerWrapper

CONFIG = {
    "lasot_root": PROJECT_ROOT / "datasets" / "lasot_small" / "small_LaSOT",
    "got10k_root": PROJECT_ROOT / "datasets" / "got10k" / "test",
    "got10k_val_root": PROJECT_ROOT / "datasets" / "got10k",
    "otb_root": PROJECT_ROOT / "datasets" / "otb",
}

# Video settings
VIDEO_FPS = 15
VIDEO_CODEC = "mp4v"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_DIR = PROJECT_ROOT / "results"
VIDEO_OUTPUT_DIR = RESULTS_DIR / "videos"
VIDEO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# Dataset Loaders (matched from visualizers standard wrapper)
# ============================================================================
def load_lasot_sequence(sequence_name: str) -> Tuple[str, List[List[int]], int]:
    lasot_root = CONFIG["lasot_root"]
    seq_path = None
    for category_dir in lasot_root.iterdir():
        if not category_dir.is_dir(): continue
        for seq_dir in category_dir.iterdir():
            if seq_dir.name == sequence_name:
                seq_path = seq_dir
                break
        if seq_path: break

    if seq_path is None:
        for category_dir in lasot_root.iterdir():
            if category_dir.is_dir() and category_dir.name == sequence_name:
                seq_path = category_dir
                break

    if seq_path is None: raise ValueError(f"Sequence {sequence_name} not found in {lasot_root}")

    img_dir = seq_path / "img"
    if not img_dir.exists(): img_dir = seq_path

    frame_files = sorted(img_dir.glob("*.jpg"))
    if not frame_files: frame_files = sorted(img_dir.glob("*.png"))
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

    return str(img_dir), gt_bboxes, num_frames

def load_got10k_sequence(sequence_name: str) -> Tuple[str, List[List[int]], int]:
    got10k_root = CONFIG["got10k_root"]
    seq_path = got10k_root / sequence_name
    if not seq_path.exists(): raise ValueError(f"Sequence {sequence_name} not found")

    frame_files = sorted(seq_path.glob("*.jpg"))
    if not frame_files: frame_files = sorted(seq_path.glob("*.png"))
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
                        vals = list(map(float, parts[:4]))
                        x1, y1, x2, y2 = vals
                        w, h = x2 - x1, y2 - y1
                        gt_bboxes.append([int(x1), int(y1), int(w), int(h)])

    return str(seq_path), gt_bboxes, num_frames

def load_got10k_val_sequence(sequence_name: str) -> Tuple[str, List[List[int]], int]:
    val_root = CONFIG["got10k_val_root"] / "val"
    seq_path = val_root / sequence_name
    if not seq_path.exists(): raise ValueError(f"Sequence {sequence_name} not found")

    frame_files = sorted(seq_path.glob("*.jpg"))
    if not frame_files: frame_files = sorted(seq_path.glob("*.png"))
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

def load_otb_sequence(sequence_name: str) -> Tuple[str, List[List[int]], int]:
    from got10k.datasets import OTB
    otb_root = str(CONFIG["otb_root"])
    dataset = OTB(root_dir=otb_root, version=2015, download=False)

    if sequence_name not in dataset.seq_names:
        raise ValueError(f"Sequence {sequence_name} not found in OTB-2015.")

    idx = dataset.seq_names.index(sequence_name)
    img_files, anno = dataset[idx]

    gt_bboxes = anno.astype(int).tolist()
    num_frames = len(img_files)
    img_dir = str(Path(img_files[0]).parent)

    return img_dir, gt_bboxes, num_frames

def get_all_sequences(dataset: str) -> List[str]:
    sequences = []
    if dataset == "lasot":
        lasot_root = CONFIG["lasot_root"]
        if lasot_root.exists():
            for category_dir in sorted(lasot_root.iterdir()):
                if category_dir.is_dir():
                    for seq_dir in sorted(category_dir.iterdir()):
                        if seq_dir.is_dir() and (seq_dir / "img").exists():
                            sequences.append(seq_dir.name)
    elif dataset == "got10k":
        got10k_root = CONFIG["got10k_root"]
        if got10k_root.exists():
            for seq_dir in sorted(got10k_root.iterdir()):
                if seq_dir.is_dir() and seq_dir.name.startswith("GOT-10k"):
                    sequences.append(seq_dir.name)
    elif dataset == "got10k_val":
        val_root = CONFIG["got10k_val_root"] / "val"
        if val_root.exists():
            list_file = val_root / "list.txt"
            if list_file.exists():
                with open(list_file) as f:
                    sequences = f.read().strip().split('\n')
            else:
                for seq_dir in sorted(val_root.iterdir()):
                    if seq_dir.is_dir() and seq_dir.name.startswith("GOT-10k"):
                        sequences.append(seq_dir.name)
    elif dataset == "otb":
        otb_root = CONFIG["otb_root"]
        if otb_root.exists() and any(otb_root.iterdir()):
            from got10k.datasets import OTB
            ds = OTB(root_dir=str(otb_root), version=2015, download=False)
            sequences = list(ds.seq_names)
    return sequences

# ============================================================================
# Video Visualization
# ============================================================================
def compute_iou(bbox1: List[int], bbox2: List[int]) -> float:
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    box1, box2 = [x1, y1, x1 + w1, y1 + h1], [x2, y2, x2 + w2, y2 + h2]

    inter_x1 = max(box1[0], box2[0])
    inter_y1 = max(box1[1], box2[1])
    inter_x2 = min(box1[2], box2[2])
    inter_y2 = min(box1[3], box2[3])

    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1: return 0.0

    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    union_area = (w1 * h1) + (w2 * h2) - inter_area
    return inter_area / union_area if union_area > 0 else 0.0

def draw_bbox(frame: np.ndarray, bbox: List[int], color: Tuple, label: str = "", thickness: int = 2) -> np.ndarray:
    x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
    if label:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale, font_thickness = 0.6, 2
        (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, font_thickness)
        cv2.rectangle(frame, (x, y - text_h - 10), (x + text_w + 10, y), color, -1)
        cv2.putText(frame, label, (x + 5, y - 5), font, font_scale, (255, 255, 255), font_thickness)
    return frame

def create_tracking_video(img_dir: str, gt_bboxes: List[List[int]], pred_bboxes: List[List[int]],
                          output_path: Path, sequence_name: str, max_frames: int = 0, fps: int = VIDEO_FPS):
    img_dir_path = Path(img_dir)
    frame_files = sorted(img_dir_path.glob("*.jpg")) or sorted(img_dir_path.glob("*.png"))
    if not frame_files: return 0.0

    num_frames = len(frame_files) if max_frames <= 0 else min(max_frames, len(frame_files))
    height, width = cv2.imread(str(frame_files[0])).shape[:2]

    out = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*VIDEO_CODEC), fps, (width, height))
    total_iou, valid_frames = 0.0, 0

    for i in range(num_frames):
        frame = cv2.imread(str(frame_files[i]))
        if frame is None: continue
        vis_frame = frame.copy()

        gt_bbox = gt_bboxes[i] if i < len(gt_bboxes) else gt_bboxes[0] if len(gt_bboxes) == 1 else None
        pred_bbox = pred_bboxes[i] if i < len(pred_bboxes) else None

        iou = 0.0
        if gt_bbox and pred_bbox:
            iou = compute_iou(gt_bbox, pred_bbox)
            total_iou += iou
            valid_frames += 1

        if gt_bbox: vis_frame = draw_bbox(vis_frame, gt_bbox, (0, 255, 0), "GT", 2)
        if pred_bbox: vis_frame = draw_bbox(vis_frame, pred_bbox, (0, 0, 255), "Pred", 2)

        info_text = [f"Sequence: {sequence_name}", f"Frame: {i+1}/{num_frames}", f"IoU: {iou:.3f}"]
        cv2.rectangle(vis_frame, (0, 0), (350, 80), (0, 0, 0), -1)
        font = cv2.FONT_HERSHEY_SIMPLEX
        for j, text in enumerate(info_text):
            cv2.putText(vis_frame, text, (10, 25 + j * 25), font, 0.6, (255, 255, 255), 2)

        legend_y = height - 60
        cv2.rectangle(vis_frame, (10, legend_y), (300, height - 10), (0, 0, 0), -1)
        cv2.rectangle(vis_frame, (20, legend_y + 10), (40, legend_y + 25), (0, 255, 0), -1)
        cv2.putText(vis_frame, "Ground Truth", (50, legend_y + 22), font, 0.5, (255, 255, 255), 1)
        cv2.rectangle(vis_frame, (20, legend_y + 30), (40, legend_y + 45), (0, 0, 255), -1)
        cv2.putText(vis_frame, "Phase2 Improved", (50, legend_y + 42), font, 0.5, (255, 255, 255), 1)

        out.write(vis_frame)

    out.release()
    avg_iou = total_iou / valid_frames if valid_frames > 0 else 0
    return avg_iou

# ============================================================================
# Main Processing
# ============================================================================
def process_sequence(dataset: str, sequence_name: str, tracker: Phase2TrackerWrapper,
                     max_frames: int = 0, chunk_size: int = 100) -> Optional[Dict]:
    print(f"\n{'='*60}\nProcessing: {dataset}/{sequence_name}\n{'='*60}")

    try:
        if dataset == "lasot":
            img_dir, gt_bboxes, num_frames = load_lasot_sequence(sequence_name)
        elif dataset == "got10k_val":
            img_dir, gt_bboxes, num_frames = load_got10k_val_sequence(sequence_name)
        elif dataset == "otb":
            img_dir, gt_bboxes, num_frames = load_otb_sequence(sequence_name)
        else:
            img_dir, gt_bboxes, num_frames = load_got10k_sequence(sequence_name)
    except Exception as e:
        print(f"  Error loading sequence: {e}")
        return None

    if num_frames == 0 or not gt_bboxes: return None

    print(f"  Running Phase 2 Tracking...")
    num_frames = min(max_frames, num_frames) if max_frames > 0 else num_frames
    
    # Phase2TrackerWrapper track has (img_dir, init_bbox, num_frames)
    pred_bboxes, _ = tracker.track(img_dir, gt_bboxes[0], num_frames)

    # Render
    output_path = VIDEO_OUTPUT_DIR / f"{dataset}_{sequence_name}_Phase2_Improved.mp4"
    avg_iou = create_tracking_video(img_dir, gt_bboxes, pred_bboxes, output_path, sequence_name, max_frames=max_frames)

    return {
        "sequence": sequence_name,
        "num_frames": len(pred_bboxes),
        "avg_iou": avg_iou,
        "video_path": str(output_path)
    }

def main():
    parser = argparse.ArgumentParser(description="Phase2 Tracker Video Visualization")
    parser.add_argument("--dataset", choices=["lasot", "got10k", "got10k_val", "otb"], default="lasot")
    parser.add_argument("--sequence", type=str, default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--max-sequences", type=int, default=5)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--chunk-size", type=int, default=100)
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    print(f"Dataset: {args.dataset}")
    all_sequences = get_all_sequences(args.dataset)

    if args.list:
        print(f"\nAvailable sequences ({len(all_sequences)}):")
        for seq in all_sequences: print(f"  - {seq}")
        return

    tracker = Phase2TrackerWrapper()

    if args.sequence:
        sequences = [args.sequence]
    elif args.all:
        sequences = all_sequences[:args.max_sequences]
    else:
        sequences = all_sequences[:3]

    if not sequences: return

    results = []
    for seq_name in sequences:
        res = process_sequence(args.dataset, seq_name, tracker, max_frames=args.max_frames, chunk_size=args.chunk_size)
        if res: results.append(res)
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    if results:
        total_iou = sum(r['avg_iou'] for r in results)
        avg_iou = total_iou / len(results)
        print(f"\nOverall Average IoU: {avg_iou:.3f}")
        summary_path = RESULTS_DIR / f"{args.dataset}_Phase2_Improved_video_results.json"
        with open(summary_path, 'w') as f:
            json.dump({"dataset": args.dataset, "model": "Phase 2 Improved", "sequences": results, "overall_avg_iou": avg_iou}, f, indent=2)

if __name__ == "__main__":
    main()
