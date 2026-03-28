#!/usr/bin/env python3
"""
Dataset Setup Script for Motion-Aware SAM 2 Project

Downloads and configures tracking benchmark datasets for evaluation.
Uses the got10k toolkit where available.

Supported datasets:
  - OTB-2015: Auto-download (~2.5 GB), 100 sequences with full GT
  - GOT-10k val: Manual download (~1.0 GB), 180 sequences with full GT
  - LaSOT: Already present (small subset), 60 sequences with full GT

Usage:
    # Download OTB-2015 (auto-download)
    python setup_datasets.py --dataset otb

    # Setup GOT-10k val split (after manual download)
    python setup_datasets.py --dataset got10k_val --zip-path /path/to/got10k_val.zip

    # Check all datasets
    python setup_datasets.py --check

    # List sequences in a dataset
    python setup_datasets.py --dataset otb --list
"""

import argparse
import os
import sys
import json
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATASET_ROOT = PROJECT_ROOT / "datasets"
DATASET_ROOT.mkdir(exist_ok=True)

# Dataset paths
DATASET_PATHS = {
    "otb": DATASET_ROOT / "otb",
    "got10k_val": DATASET_ROOT / "got10k",  # val split alongside existing test
    "got10k_test": DATASET_ROOT / "got10k" / "test",
    "lasot": DATASET_ROOT / "lasot_small" / "small_LaSOT",
}


def check_all_datasets():
    """Check status of all datasets."""
    print("=" * 60)
    print("DATASET STATUS CHECK")
    print("=" * 60)

    datasets = {
        "LaSOT (small)": {
            "path": DATASET_PATHS["lasot"],
            "desc": "60 sequences, full GT annotations",
            "check": lambda p: p.exists() and any(p.iterdir()),
        },
        "GOT-10k (test)": {
            "path": DATASET_PATHS["got10k_test"],
            "desc": "180 sequences, GT only for frame 0 (NOT for evaluation)",
            "check": lambda p: p.exists() and any(p.iterdir()),
        },
        "GOT-10k (val)": {
            "path": DATASET_ROOT / "got10k" / "val",
            "desc": "180 sequences, FULL GT annotations",
            "check": lambda p: p.exists() and (p / "list.txt").exists(),
        },
        "OTB-2015": {
            "path": DATASET_PATHS["otb"],
            "desc": "100 sequences, full GT annotations, auto-download",
            "check": lambda p: p.exists() and any(p.iterdir()),
        },
    }

    for name, info in datasets.items():
        path = info["path"]
        exists = info["check"](path)
        status = "✓ READY" if exists else "✗ NOT FOUND"

        if exists:
            # Count sequences
            if name == "GOT-10k (val)":
                list_file = path / "list.txt"
                if list_file.exists():
                    with open(list_file) as f:
                        n_seqs = len(f.read().strip().split('\n'))
                else:
                    n_seqs = 0
            elif name == "OTB-2015":
                n_seqs = sum(1 for d in path.iterdir() if d.is_dir())
            else:
                n_seqs = sum(1 for d in path.rglob("groundtruth.txt"))

            # Get size
            import subprocess
            result = subprocess.run(
                ["du", "-sh", str(path)], capture_output=True, text=True
            )
            size = result.stdout.split()[0] if result.returncode == 0 else "?"
            print(f"\n  {status}  {name}")
            print(f"         Path: {path}")
            print(f"         Size: {size}, Sequences: {n_seqs}")
            print(f"         {info['desc']}")
        else:
            print(f"\n  {status}  {name}")
            print(f"         Path: {path}")
            print(f"         {info['desc']}")

    print("\n" + "=" * 60)
    print("DOWNLOAD INSTRUCTIONS")
    print("=" * 60)

    # Check what's missing
    if not (DATASET_ROOT / "got10k" / "val").exists():
        print("""
  GOT-10k Validation Set (1.0 GB):
    1. Go to: http://got-10k.aitestunion.com/downloads
    2. Click "Validation data only" and enter your email
    3. Download the zip file from the email link
    4. Run: python setup_datasets.py --dataset got10k_val --zip-path /path/to/val.zip
""")

    otb_path = DATASET_PATHS["otb"]
    if not otb_path.exists() or not any(otb_path.iterdir()):
        print("""
  OTB-2015 (2.5 GB):
    Auto-download available! Run:
    python setup_datasets.py --dataset otb
""")


def setup_otb():
    """Download and setup OTB-2015 dataset using got10k toolkit."""
    print("=" * 60)
    print("DOWNLOADING OTB-2015")
    print("=" * 60)

    otb_root = str(DATASET_PATHS["otb"])
    print(f"Target directory: {otb_root}")
    print("This will download ~2.5 GB of data...")
    print()

    from got10k.datasets import OTB
    dataset = OTB(root_dir=otb_root, version=2015, download=True)

    print(f"\n✓ OTB-2015 ready: {len(dataset)} sequences")
    print(f"  Path: {otb_root}")

    # Verify by loading first sequence
    img_files, anno = dataset[0]
    print(f"  First sequence: {dataset.seq_names[0]}")
    print(f"    Frames: {len(img_files)}")
    print(f"    GT boxes: {anno.shape}")
    print(f"    First GT: {anno[0].tolist()}")


def setup_got10k_val(zip_path: str):
    """Extract and setup GOT-10k validation set."""
    print("=" * 60)
    print("SETTING UP GOT-10k VALIDATION SET")
    print("=" * 60)

    if not os.path.exists(zip_path):
        print(f"Error: File not found: {zip_path}")
        sys.exit(1)

    got10k_root = DATASET_ROOT / "got10k"
    val_dir = got10k_root / "val"

    if val_dir.exists() and (val_dir / "list.txt").exists():
        print(f"GOT-10k val already exists at {val_dir}")
        print("Use --force to re-extract")
        return

    import zipfile

    print(f"Extracting {zip_path} to {got10k_root}...")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(str(got10k_root))

    # Verify
    if (val_dir / "list.txt").exists():
        with open(val_dir / "list.txt") as f:
            seqs = f.read().strip().split('\n')
        print(f"\n✓ GOT-10k val ready: {len(seqs)} sequences")
        print(f"  Path: {val_dir}")

        # Verify first sequence GT
        first_seq = val_dir / seqs[0]
        gt_file = first_seq / "groundtruth.txt"
        if gt_file.exists():
            with open(gt_file) as f:
                lines = f.readlines()
            print(f"  First sequence: {seqs[0]}")
            print(f"    GT lines: {len(lines)}")
            print(f"    First GT: {lines[0].strip()}")
            print(f"    Format: x, y, w, h (full annotations for all frames)")
    else:
        print(f"\nError: list.txt not found in {val_dir}")
        print("The zip may have a different structure. Check contents with:")
        print(f"  unzip -l {zip_path} | head -20")


def list_sequences(dataset_name: str):
    """List all sequences in a dataset."""
    print(f"Sequences in {dataset_name}:")

    if dataset_name == "otb":
        from got10k.datasets import OTB
        dataset = OTB(root_dir=str(DATASET_PATHS["otb"]), version=2015, download=False)
        for i, name in enumerate(dataset.seq_names):
            img_files, anno = dataset[i]
            print(f"  {i+1:3d}. {name:25s} | {len(img_files):5d} frames | GT: {anno.shape}")

    elif dataset_name == "got10k_val":
        from got10k.datasets import GOT10k
        got10k_root = str(DATASET_ROOT / "got10k")
        dataset = GOT10k(root_dir=got10k_root, subset='val')
        for i, name in enumerate(dataset.seq_names):
            img_files, anno = dataset[i]
            print(f"  {i+1:3d}. {name:30s} | {len(img_files):5d} frames | GT: {anno.shape}")

    elif dataset_name == "lasot":
        root = DATASET_PATHS["lasot"]
        for cat_dir in sorted(root.iterdir()):
            if cat_dir.is_dir():
                for seq_dir in sorted(cat_dir.iterdir()):
                    if seq_dir.is_dir() and (seq_dir / "img").exists():
                        n_frames = len(list((seq_dir / "img").glob("*.jpg")))
                        gt_file = seq_dir / "groundtruth.txt"
                        n_gt = 0
                        if gt_file.exists():
                            with open(gt_file) as f:
                                n_gt = len(f.readlines())
                        print(f"  {seq_dir.name:25s} | {n_frames:5d} frames | {n_gt:5d} GT boxes")


def main():
    parser = argparse.ArgumentParser(description="Dataset Setup for Motion-Aware SAM 2")
    parser.add_argument("--dataset", choices=["otb", "got10k_val", "lasot"],
                       help="Dataset to setup/download")
    parser.add_argument("--zip-path", type=str, default=None,
                       help="Path to downloaded zip file (for GOT-10k val)")
    parser.add_argument("--check", action="store_true",
                       help="Check status of all datasets")
    parser.add_argument("--list", action="store_true",
                       help="List sequences in a dataset")
    parser.add_argument("--force", action="store_true",
                       help="Force re-download/extract")

    args = parser.parse_args()

    if args.check or (not args.dataset and not args.list):
        check_all_datasets()
        return

    if args.list and args.dataset:
        list_sequences(args.dataset)
        return

    if args.dataset == "otb":
        setup_otb()
    elif args.dataset == "got10k_val":
        if not args.zip_path:
            print("Error: --zip-path required for GOT-10k val")
            print("Download from: http://got-10k.aitestunion.com/downloads")
            print("Then run: python setup_datasets.py --dataset got10k_val --zip-path /path/to/val.zip")
            sys.exit(1)
        setup_got10k_val(args.zip_path)
    elif args.dataset == "lasot":
        print("LaSOT small subset is already configured.")
        check_all_datasets()


if __name__ == "__main__":
    main()
