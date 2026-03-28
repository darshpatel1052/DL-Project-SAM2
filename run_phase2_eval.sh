#!/bin/bash
# =============================================================================
# Phase 2 Improved Evaluation Script
# HiM2SAM-style SAM2 tracking with occlusion recovery
# =============================================================================

cd /home/predator-linux/Acads/Sem6/DL/Project/motion_aware_sam2

# Activate virtual environment
source venv/bin/activate

echo "=============================================="
echo "Phase 2 Improved Evaluation"
echo "=============================================="

# Default parameters - TUNE THESE AS NEEDED
DATASET="got10k_val"                    # Options: lasot, got10k_val
MAX_SEQUENCES=1                    # 0 = all sequences, or set a number
CHUNK_SIZE=200                     # Frames per chunk (memory efficiency)
CONFIDENCE_THRESHOLD=0.6           # Above = VISIBLE (trust SAM2)
OCCLUSION_THRESHOLD=0.4            # Above = UNCERTAIN (blend SAM2+Kalman)
LOST_THRESHOLD=0.10               # Below = LOST (re-detect with init box)
RECOVERY_FRAMES=10                  # Consecutive low-conf frames before state change

# Run evaluation
python evaluation/eval_Phase2_Improved.py \
    --dataset "$DATASET" \
    --max-sequences "$MAX_SEQUENCES" \
    --chunk-size "$CHUNK_SIZE" \
    --confidence-threshold "$CONFIDENCE_THRESHOLD" \
    --occlusion-threshold "$OCCLUSION_THRESHOLD" \
    --lost-threshold "$LOST_THRESHOLD" \
    --recovery-frames "$RECOVERY_FRAMES"

echo ""
echo "=============================================="
echo "Evaluation Complete!"
echo "Results saved to: results/${DATASET}_Phase2_Improved_evaluation.json"
echo "=============================================="
