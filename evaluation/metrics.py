"""
Evaluation Metrics for Object Tracking

GOT-10k Metrics:
- AO (Average Overlap): Mean IoU across all frames
- SR_0.5 (Success Rate @ 0.5): Percentage of frames with IoU > 0.5
- SR_0.75 (Success Rate @ 0.75): Percentage of frames with IoU > 0.75

LaSOT Metrics:
- AUC (Area Under Curve): Area under success plot
- Precision: Percentage of frames within distance threshold
- Normalized Precision: Precision normalized by target size
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import json
from pathlib import Path


def compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    Compute IoU between two boxes.

    Args:
        box1: [x1, y1, w, h] or [x1, y1, x2, y2]
        box2: [x1, y1, w, h] or [x1, y1, x2, y2]

    Returns:
        iou: Intersection over Union (0 to 1)
    """
    # Convert to [x1, y1, x2, y2] if needed
    if len(box1) == 4:
        if box1[2] < box1[0] or box1[3] < box1[1]:
            # Already in x1,y1,x2,y2 format
            x1_1, y1_1, x2_1, y2_1 = box1
        else:
            # In x,y,w,h format
            x1_1 = box1[0]
            y1_1 = box1[1]
            x2_1 = box1[0] + box1[2]
            y2_1 = box1[1] + box1[3]
    else:
        return 0.0

    if len(box2) == 4:
        if box2[2] < box2[0] or box2[3] < box2[1]:
            x1_2, y1_2, x2_2, y2_2 = box2
        else:
            x1_2 = box2[0]
            y1_2 = box2[1]
            x2_2 = box2[0] + box2[2]
            y2_2 = box2[1] + box2[3]
    else:
        return 0.0

    # Intersection
    inter_x1 = max(x1_1, x1_2)
    inter_y1 = max(y1_1, y1_2)
    inter_x2 = min(x2_1, x2_2)
    inter_y2 = min(y2_1, y2_2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    # Union
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = area1 + area2 - inter_area

    return inter_area / union_area

def compute_f_score(box1: np.ndarray, box2: np.ndarray, bound_th: int = 2) -> float:
    """
    Compute pseudo F-measure (Boundary accuracy) for bounding boxes.
    Draws the rectangular perimeters on a localized mask and compares 
    precision and recall (like standard DAVIS contour evaluation).
    """
    import cv2
    
    # If using [x1, y1, w, h], just keep it, but ensure we have valid shapes
    if len(box1) == 4 and len(box2) == 4:
        # Normalize into [x, y, w, h] integers
        b1 = [int(box1[0]), int(box1[1]), int(box1[2]), int(box1[3])]
        b2 = [int(box2[0]), int(box2[1]), int(box2[2]), int(box2[3])]
        if box1[2] < box1[0] or box1[3] < box1[1]:
            b1 = [int(box1[0]), int(box1[1]), int(box1[2] - box1[0]), int(box1[3] - box1[1])]
        if box2[2] < box2[0] or box2[3] < box2[1]:
            b2 = [int(box2[0]), int(box2[1]), int(box2[2] - box2[0]), int(box2[3] - box2[1])]
    else:
        return 0.0

    if b1[2] <= 0 or b1[3] <= 0 or b2[2] <= 0 or b2[3] <= 0:
        return 0.0

    # Determine canvas bounds based on maximum extents
    max_x = max(b1[0] + b1[2], b2[0] + b2[2]) + 20
    max_y = max(b1[1] + b1[3], b2[1] + b2[3]) + 20
    
    if max_x <= 20 or max_y <= 20: 
        return 0.0
        
    # Translate coordinates to 0,0 locally to save memory if boxes are at massive offsets
    min_x = min(max(0, b1[0] - 10), max(0, b2[0] - 10))
    min_y = min(max(0, b1[1] - 10), max(0, b2[1] - 10))
    
    s1 = [b1[0] - min_x, b1[1] - min_y, b1[2], b1[3]]
    s2 = [b2[0] - min_x, b2[1] - min_y, b2[2], b2[3]]
    
    canvas_w = (max_x - min_x)
    canvas_h = (max_y - min_y)
    
    canvas1 = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    canvas2 = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    
    cv2.rectangle(canvas1, (s1[0], s1[1]), (s1[0] + s1[2], s1[1] + s1[3]), 1, bound_th)
    cv2.rectangle(canvas2, (s2[0], s2[1]), (s2[0] + s2[2], s2[1] + s2[3]), 1, bound_th)
    
    intersect = np.logical_and(canvas1, canvas2).sum()
    sum1 = canvas1.sum()
    sum2 = canvas2.sum()
    
    if sum1 == 0 and sum2 == 0: return 1.0
    if sum1 == 0 or sum2 == 0: return 0.0
    
    precision = intersect / sum1
    recall = intersect / sum2
    
    if precision + recall == 0: return 0.0
    return float(2 * (precision * recall) / (precision + recall))




def compute_center_distance(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    Compute distance between box centers.

    Args:
        box1: [x1, y1, w, h]
        box2: [x1, y1, w, h]

    Returns:
        distance: Euclidean distance between centers
    """
    cx1 = box1[0] + box1[2] / 2
    cy1 = box1[1] + box1[3] / 2
    cx2 = box2[0] + box2[2] / 2
    cy2 = box2[1] + box2[3] / 2

    return np.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)


class GOT10kMetrics:
    """
    GOT-10k evaluation metrics.

    Computes:
    - AO (Average Overlap)
    - SR_0.5 (Success Rate at IoU threshold 0.5)
    - SR_0.75 (Success Rate at IoU threshold 0.75)
    """

    def __init__(self, iou_thresholds: List[float] = [0.5, 0.75]):
        """
        Initialize GOT-10k metrics.

        Args:
            iou_thresholds: List of IoU thresholds for success rate
        """
        self.iou_thresholds = iou_thresholds
        self.results = []

    def add_sequence(
        self,
        predictions: np.ndarray,
        groundtruth: np.ndarray,
        seq_name: str = None
    ):
        """
        Add evaluation results for a sequence.

        Args:
            predictions: (N, 4) array of predicted boxes [x1, y1, w, h]
            groundtruth: (N, 4) array of ground truth boxes [x1, y1, w, h]
            seq_name: Optional sequence name for reporting
        """
        # Compute IoU for each frame
        assert len(predictions) == len(groundtruth), \
            f"Length mismatch: {len(predictions)} vs {len(groundtruth)}"

        ious = []
        for pred, gt in zip(predictions, groundtruth):
            # Skip if ground truth is invalid (marked as [0,0,0,0] or NaN)
            if np.any(np.isnan(gt)) or (gt[2] <= 0 and gt[3] <= 0):
                continue
            iou = compute_iou(pred, gt)
            ious.append(iou)

        if len(ious) == 0:
            return

        ious = np.array(ious)

        # Compute metrics
        ao = np.mean(ious)
        success_rates = {}
        for thresh in self.iou_thresholds:
            sr = np.mean(ious >= thresh)
            success_rates[thresh] = sr

        self.results.append({
            "name": seq_name,
            "ao": ao,
            "success_rates": success_rates,
            "ious": ious,
            "n_frames": len(ious),
        })

    def compute_overall(self) -> Dict:
        """
        Compute overall metrics across all sequences.

        Returns:
            metrics: Dictionary with AO, SR_0.5, SR_0.75, etc.
        """
        if len(self.results) == 0:
            return {"ao": 0.0, "sr_0.5": 0.0, "sr_0.75": 0.0}

        # Aggregate IoUs from all sequences
        all_ious = np.concatenate([r["ious"] for r in self.results])

        metrics = {
            "ao": np.mean(all_ious),
            "n_sequences": len(self.results),
            "n_frames": len(all_ious),
        }

        for thresh in self.iou_thresholds:
            key = f"sr_{thresh}"
            metrics[key] = np.mean(all_ious >= thresh)

        # Also compute per-sequence average
        metrics["ao_per_seq"] = np.mean([r["ao"] for r in self.results])

        return metrics

    def get_per_sequence_results(self) -> List[Dict]:
        """Get per-sequence results."""
        return self.results

    def reset(self):
        """Reset stored results."""
        self.results = []


class LaSOTMetrics:
    """
    LaSOT evaluation metrics.

    Computes:
    - AUC (Area Under Curve of success plot)
    - Precision (center location error threshold)
    - Normalized Precision
    """

    def __init__(
        self,
        n_bins: int = 100,
        precision_threshold: float = 20.0,  # pixels
    ):
        """
        Initialize LaSOT metrics.

        Args:
            n_bins: Number of bins for AUC computation
            precision_threshold: Distance threshold for precision (pixels)
        """
        self.n_bins = n_bins
        self.precision_threshold = precision_threshold
        self.results = []

    def add_sequence(
        self,
        predictions: np.ndarray,
        groundtruth: np.ndarray,
        seq_name: str = None,
        full_occlusion: np.ndarray = None,
        out_of_view: np.ndarray = None,
    ):
        """
        Add evaluation results for a sequence.

        Args:
            predictions: (N, 4) array of predicted boxes [x1, y1, w, h]
            groundtruth: (N, 4) array of ground truth boxes [x1, y1, w, h]
            seq_name: Optional sequence name
            full_occlusion: Optional (N,) binary array
            out_of_view: Optional (N,) binary array
        """
        assert len(predictions) == len(groundtruth), \
            f"Length mismatch: {len(predictions)} vs {len(groundtruth)}"

        ious = []
        distances = []
        norm_distances = []

        for i, (pred, gt) in enumerate(zip(predictions, groundtruth)):
            # Skip invalid frames
            if np.any(np.isnan(gt)) or (gt[2] <= 0 and gt[3] <= 0):
                continue

            # Skip fully occluded or out of view frames
            if full_occlusion is not None and full_occlusion[i] == 1:
                continue
            if out_of_view is not None and out_of_view[i] == 1:
                continue

            iou = compute_iou(pred, gt)
            dist = compute_center_distance(pred, gt)

            # Normalized distance (by target size)
            target_size = np.sqrt(gt[2] * gt[3])
            norm_dist = dist / target_size if target_size > 0 else float('inf')

            ious.append(iou)
            distances.append(dist)
            norm_distances.append(norm_dist)

        if len(ious) == 0:
            return

        ious = np.array(ious)
        distances = np.array(distances)
        norm_distances = np.array(norm_distances)

        # Compute success plot and AUC
        thresholds = np.linspace(0, 1, self.n_bins + 1)
        success_curve = np.array([np.mean(ious >= t) for t in thresholds])
        auc = np.trapezoid(success_curve, thresholds)

        # Compute precision (center error < threshold)
        precision = np.mean(distances < self.precision_threshold)

        # Compute normalized precision (center error < 0.5 * target_size)
        norm_precision = np.mean(norm_distances < 0.5)

        self.results.append({
            "name": seq_name,
            "auc": auc,
            "precision": precision,
            "norm_precision": norm_precision,
            "ious": ious,
            "distances": distances,
            "n_frames": len(ious),
            "success_curve": success_curve,
        })

    def compute_overall(self) -> Dict:
        """
        Compute overall metrics across all sequences.

        Returns:
            metrics: Dictionary with AUC, Precision, Normalized Precision
        """
        if len(self.results) == 0:
            return {"auc": 0.0, "precision": 0.0, "norm_precision": 0.0}

        # Aggregate
        all_ious = np.concatenate([r["ious"] for r in self.results])
        all_distances = np.concatenate([r["distances"] for r in self.results])

        # Compute overall success curve and AUC
        thresholds = np.linspace(0, 1, self.n_bins + 1)
        success_curve = np.array([np.mean(all_ious >= t) for t in thresholds])
        auc = np.trapezoid(success_curve, thresholds)

        metrics = {
            "auc": auc,
            "precision": np.mean(all_distances < self.precision_threshold),
            "n_sequences": len(self.results),
            "n_frames": len(all_ious),
        }

        # Per-sequence averages
        metrics["auc_per_seq"] = np.mean([r["auc"] for r in self.results])
        metrics["precision_per_seq"] = np.mean([r["precision"] for r in self.results])
        metrics["norm_precision_per_seq"] = np.mean([r["norm_precision"] for r in self.results])

        return metrics

    def get_per_sequence_results(self) -> List[Dict]:
        """Get per-sequence results."""
        return self.results

    def reset(self):
        """Reset stored results."""
        self.results = []


class UnifiedEvaluator:
    """
    Unified evaluator for both GOT-10k and LaSOT.

    Provides a single interface for evaluation on both datasets.
    """

    def __init__(self):
        self.got10k_metrics = GOT10kMetrics()
        self.lasot_metrics = LaSOTMetrics()

    def evaluate_sequence(
        self,
        predictions: np.ndarray,
        groundtruth: np.ndarray,
        dataset: str,  # "got10k" or "lasot"
        seq_name: str = None,
        **kwargs
    ):
        """
        Evaluate a single sequence.

        Args:
            predictions: (N, 4) predicted boxes
            groundtruth: (N, 4) ground truth boxes
            dataset: "got10k" or "lasot"
            seq_name: Sequence name
            **kwargs: Additional arguments (full_occlusion, out_of_view for LaSOT)
        """
        if dataset.lower() == "got10k":
            self.got10k_metrics.add_sequence(predictions, groundtruth, seq_name)
        elif dataset.lower() == "lasot":
            self.lasot_metrics.add_sequence(
                predictions, groundtruth, seq_name,
                full_occlusion=kwargs.get("full_occlusion"),
                out_of_view=kwargs.get("out_of_view"),
            )
        else:
            raise ValueError(f"Unknown dataset: {dataset}")

    def get_results(self, dataset: str = None) -> Dict:
        """
        Get evaluation results.

        Args:
            dataset: "got10k", "lasot", or None for both

        Returns:
            results: Dictionary of metrics
        """
        results = {}

        if dataset is None or dataset.lower() == "got10k":
            results["got10k"] = self.got10k_metrics.compute_overall()

        if dataset is None or dataset.lower() == "lasot":
            results["lasot"] = self.lasot_metrics.compute_overall()

        return results

    def print_results(self, dataset: str = None):
        """Print formatted results."""
        results = self.get_results(dataset)

        print("\n" + "=" * 60)
        print("EVALUATION RESULTS")
        print("=" * 60)

        if "got10k" in results:
            r = results["got10k"]
            print("\nGOT-10k:")
            print(f"  AO:      {r['ao']:.4f}")
            print(f"  SR_0.5:  {r.get('sr_0.5', 0):.4f}")
            print(f"  SR_0.75: {r.get('sr_0.75', 0):.4f}")
            print(f"  Sequences: {r.get('n_sequences', 0)}")

        if "lasot" in results:
            r = results["lasot"]
            print("\nLaSOT:")
            print(f"  AUC:            {r['auc']:.4f}")
            print(f"  Precision:      {r.get('precision', 0):.4f}")
            print(f"  Norm Precision: {r.get('norm_precision_per_seq', 0):.4f}")
            print(f"  Sequences: {r.get('n_sequences', 0)}")

        print("=" * 60)

    def save_results(self, output_path: str):
        """Save results to JSON file."""
        results = self.get_results()

        # Convert numpy arrays to lists for JSON serialization
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(v) for v in obj]
            return obj

        results = convert(results)

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"Results saved to {output_path}")

    def reset(self):
        """Reset all metrics."""
        self.got10k_metrics.reset()
        self.lasot_metrics.reset()


def compare_methods(
    results_baseline: Dict,
    results_kalman: Dict,
    dataset: str,
) -> str:
    """
    Generate comparison table between baseline and Kalman-enhanced results.

    Args:
        results_baseline: Baseline SAM 2 results
        results_kalman: Kalman-enhanced results
        dataset: "got10k" or "lasot"

    Returns:
        table: Formatted comparison table string
    """
    lines = []
    lines.append("\n" + "=" * 70)
    lines.append(f"COMPARISON: {dataset.upper()}")
    lines.append("=" * 70)

    if dataset.lower() == "got10k":
        lines.append(f"{'Metric':<20} {'SAM 2 (Baseline)':<20} {'SAM 2 + Kalman':<20} {'Δ':<10}")
        lines.append("-" * 70)

        for metric in ["ao", "sr_0.5", "sr_0.75"]:
            baseline = results_baseline.get(metric, 0)
            kalman = results_kalman.get(metric, 0)
            delta = kalman - baseline
            sign = "+" if delta >= 0 else ""
            lines.append(f"{metric.upper():<20} {baseline:<20.4f} {kalman:<20.4f} {sign}{delta:.4f}")

    elif dataset.lower() == "lasot":
        lines.append(f"{'Metric':<20} {'SAM 2 (Baseline)':<20} {'SAM 2 + Kalman':<20} {'Δ':<10}")
        lines.append("-" * 70)

        for metric in ["auc", "precision", "norm_precision_per_seq"]:
            baseline = results_baseline.get(metric, 0)
            kalman = results_kalman.get(metric, 0)
            delta = kalman - baseline
            sign = "+" if delta >= 0 else ""
            name = metric.replace("_per_seq", "").replace("_", " ").title()
            lines.append(f"{name:<20} {baseline:<20.4f} {kalman:<20.4f} {sign}{delta:.4f}")

    lines.append("=" * 70)

    return "\n".join(lines)


if __name__ == "__main__":
    # Test metrics computation
    np.random.seed(42)

    # Create dummy data
    n_frames = 100
    gt_boxes = np.random.rand(n_frames, 4) * 100 + 50
    gt_boxes[:, 2:] = 50  # Fixed size

    # Predictions with some noise
    pred_boxes = gt_boxes + np.random.randn(n_frames, 4) * 10

    # Test GOT-10k metrics
    print("Testing GOT-10k metrics:")
    got10k = GOT10kMetrics()
    got10k.add_sequence(pred_boxes, gt_boxes, "test_seq")
    print(got10k.compute_overall())

    # Test LaSOT metrics
    print("\nTesting LaSOT metrics:")
    lasot = LaSOTMetrics()
    lasot.add_sequence(pred_boxes, gt_boxes, "test_seq")
    print(lasot.compute_overall())

    # Test unified evaluator
    print("\nTesting unified evaluator:")
    evaluator = UnifiedEvaluator()
    evaluator.evaluate_sequence(pred_boxes, gt_boxes, "got10k", "seq1")
    evaluator.evaluate_sequence(pred_boxes, gt_boxes, "lasot", "seq1")
    evaluator.print_results()
