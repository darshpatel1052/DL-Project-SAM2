# Motion-Aware SAM 2: Kalman Filter Enhanced Video Object Segmentation

A training-free enhancement to SAM 2 (Segment Anything Model 2) that integrates Kalman filtering for robust video object tracking with occlusion handling.

## Highlights

- **Training-free**: No GPU training required - works out of the box with pretrained SAM 2
- **Motion-aware tracking**: Kalman filter predicts object motion to improve mask selection
- **Occlusion handling**: State machine (VISIBLE/UNCERTAIN/OCCLUDED/LOST) for robust tracking
- **Quality-gated memory**: Prevents error accumulation by filtering low-quality frames

## Results

### GOT-10k Validation Set (180 sequences)

| Method | J (IoU) | Improvement |
|--------|---------|-------------|
| SAM 2.1 Baseline | 84.36% | - |
| Motion-Aware SAM 2 (Ours) | 87.55% | +3.19% |

### LaSOT Dataset

| Method | J (IoU) | Improvement |
|--------|---------|-------------|
| SAM 2.1 Baseline | 51.55% | - |
| Motion-Aware SAM 2 (Ours) | 57.63% | +11.61% |

## Project Structure

```
motion_aware_sam2/
├── configs/
│   └── config.py                 # Configuration settings
├── datasets/
│   ├── dataset_loaders.py        # GOT-10k and LaSOT data loaders
│   └── setup_datasets.py         # Dataset download utilities
├── models/
│   ├── kalman_filter.py          # Kalman filter implementation
│   ├── sam2_tracker.py           # SAM 2 video tracker wrapper
│   ├── baseline.py               # Pure SAM 2 baseline tracker
│   └── Phase2_Improved.py        # Motion-aware tracker with occlusion handling
├── evaluation/
│   ├── metrics.py                # GOT-10k and LaSOT evaluation metrics
│   ├── eval_baseline.py          # Baseline evaluation script
│   ├── eval_Phase2_Improved.py   # Improved model evaluation script
│   └── failure_capture.py        # Failure case analysis
├── utils/
│   └── visualization.py          # Plotting and visualization
├── results/                      # Evaluation outputs (JSON + videos)
├── requirements.txt
└── README.md
```

## Installation

### 1. Clone and setup environment

```bash
git clone https://github.com/yourusername/motion_aware_sam2.git
cd motion_aware_sam2
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Install SAM 2

```bash
# From PyPI
pip install sam2

# Or from source (recommended)
git clone https://github.com/facebookresearch/sam2.git
cd sam2 && pip install -e . && cd ..
```

### 4. Download SAM 2 checkpoint

Download `sam2.1_hiera_small.pt` from [SAM 2 Model Zoo](https://github.com/facebookresearch/sam2#model-checkpoints) and place in `models/`.

## Dataset Setup

### GOT-10k

1. Register at http://got-10k.aitestunion.com/
2. Download validation split
3. Extract to `datasets/got10k/val/`

### LaSOT

1. Download from http://vision.cs.stonybrook.edu/~lasot/
2. Extract to `datasets/lasot_small/`

## Usage

### Run Baseline Evaluation

```bash
python evaluation/eval_baseline.py --dataset got10k_val --max-sequences 10
```

### Run Motion-Aware Evaluation

```bash
python evaluation/eval_Phase2_Improved.py \
    --dataset got10k_val \
    --confidence-threshold 0.7 \
    --occlusion-threshold 0.3 \
    --lost-threshold 0.15
```

### Generate Visualizations

```bash
python utils/visualization.py
```

## Approach

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    Motion-Aware SAM 2 Pipeline                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Frame t ──► SAM 2 Encoder ──► Memory Attention ──► Decoder    │
│                                       ▲                  │      │
│                                       │                  ▼      │
│                              Memory Bank          Mask Candidates│
│                           (quality-filtered)           │        │
│                                  ▲                     ▼        │
│                                  │           ┌─────────────┐    │
│                                  └───────────│   Kalman    │    │
│                                              │   Filter    │    │
│                                              │  + Scoring  │    │
│                                              └─────────────┘    │
│                                                     │           │
│                                                     ▼           │
│                                              Best Mask + BBox   │
└─────────────────────────────────────────────────────────────────┘
```

### Kalman Filter State Model

The Kalman filter maintains an 8-dimensional state vector:

```
State: [cx, cy, w, h, vx, vy, vw, vh]
        └─position─┘  └──velocity──┘
```

**Prediction step** (constant velocity model):
```
x̂_t = F · x_{t-1}
```

**Update step** (when mask is reliable):
```
x_t = x̂_t + K · (z_t - H · x̂_t)
```

### Tracking State Machine

```
    ┌──────────────────────────────────────────────────┐
    │                                                  │
    ▼                                                  │
┌───────┐  conf > 0.7   ┌──────────┐  conf > 0.7  ┌────────┐
│VISIBLE│◄──────────────│UNCERTAIN │◄─────────────│OCCLUDED│
└───┬───┘               └────┬─────┘              └────┬───┘
    │                        │                         │
    │ conf < 0.7             │ conf < 0.3              │ conf < 0.15
    │                        │                         │
    └────────►───────────────┴─────────►───────────────┘
                                              │
                                              ▼
                                          ┌──────┐
                                          │ LOST │
                                          └──────┘
```

### Mask Selection Scoring

```
M* = argmax(α · motion_score + (1-α) · appearance_score)

where:
  motion_score = IoU(kalman_predicted_bbox, candidate_bbox)
  appearance_score = SAM 2 confidence score
  α = 0.15 (motion weight)
```

### Quality-Gated Memory

Frames are added to memory only if:
```
motion_score > τ_motion (0.7)  AND
mask_iou > τ_mask (0.5)        AND
occlusion_score < τ_occlusion (0.5)
```

## Evaluation Metrics

### GOT-10k
- **AO** (Average Overlap): Mean IoU across all frames
- **SR₀.₅**: Success rate at IoU threshold 0.5
- **SR₀.₇₅**: Success rate at IoU threshold 0.75

### LaSOT
- **AUC**: Area under success curve (IoU thresholds 0-1)
- **P**: Precision (center error < 20px)
- **P_norm**: Normalized precision

## Configuration

Key parameters in `configs/config.py`:

```python
KALMAN_CONFIG = {
    "alpha_motion": 0.15,      # Motion score weight
    "tau_mask_iou": 0.5,       # Min mask confidence
    "tau_motion": 0.7,         # Min motion score
    "tau_occlusion": 0.5,      # Max occlusion score
}
```

## References

```bibtex
@article{ravi2024sam2,
  title={SAM 2: Segment Anything in Images and Videos},
  author={Ravi, Nikhila and others},
  journal={arXiv preprint arXiv:2408.00714},
  year={2024}
}

@article{yang2024samurai,
  title={SAMURAI: Adapting Segment Anything Model for Zero-Shot Visual Tracking},
  author={Yang, Cheng-Yen and others},
  journal={arXiv preprint arXiv:2411.11922},
  year={2024}
}

@article{huang2019got10k,
  title={GOT-10k: A Large High-Diversity Benchmark for Generic Object Tracking},
  author={Huang, Lianghua and Zhao, Xin and Huang, Kaiqi},
  journal={IEEE TPAMI},
  year={2019}
}

@inproceedings{fan2019lasot,
  title={LaSOT: A High-quality Benchmark for Large-scale Single Object Tracking},
  author={Fan, Heng and others},
  booktitle={CVPR},
  year={2019}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- [SAM 2](https://github.com/facebookresearch/sam2) by Meta AI
- [SAMURAI](https://github.com/yangchris11/samurai) for Kalman filter inspiration
- GOT-10k and LaSOT benchmark teams
