"""
Dataset Downloaders and Loaders for GOT-10k and LaSOT

GOT-10k: Generic Object Tracking benchmark (10K videos, 563 classes)
LaSOT: Large-scale Single Object Tracking (1,400 videos, 70 classes)
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Generator
import cv2
from tqdm import tqdm
# import requests
import zipfile
import shutil


class GOT10kDataset:
    """
    GOT-10k Dataset Loader

    Dataset structure:
    GOT-10k/
    ├── test/
    │   ├── GOT-10k_Test_000001/
    │   │   ├── 00000001.jpg
    │   │   ├── 00000002.jpg
    │   │   └── ...
    │   └── ...
    └── val/
        └── ...

    Ground truth format: [x1, y1, w, h] (top-left corner + size)
    """

    def __init__(self, root_dir: str, split: str = "test"):
        """
        Initialize GOT-10k dataset.

        Args:
            root_dir: Path to GOT-10k root directory
            split: "test" or "val"
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.split_dir = self.root_dir / split

        self.sequences = self._get_sequences()

    def _get_sequences(self) -> List[str]:
        """Get list of sequence names."""
        if not self.split_dir.exists():
            print(f"Warning: {self.split_dir} does not exist")
            return []

        sequences = sorted([
            d.name for d in self.split_dir.iterdir()
            if d.is_dir() and not d.name.startswith('.')
        ])
        return sequences

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict:
        """
        Get a sequence by index.

        Returns:
            dict with keys:
                - name: Sequence name
                - frames: List of frame paths
                - init_bbox: Initial bounding box [x1, y1, w, h]
                - groundtruth: List of bounding boxes (if available)
        """
        seq_name = self.sequences[idx]
        seq_dir = self.split_dir / seq_name

        # Get frame paths
        frame_files = sorted([
            f for f in seq_dir.iterdir()
            if f.suffix.lower() in ['.jpg', '.jpeg', '.png']
        ])
        frames = [str(f) for f in frame_files]

        # Load ground truth
        gt_file = seq_dir / "groundtruth.txt"
        if gt_file.exists():
            groundtruth = self._load_groundtruth(gt_file)
            init_bbox = groundtruth[0]
        else:
            # Test set may not have GT - use init file
            init_file = seq_dir / "init.txt"
            if init_file.exists():
                init_bbox = self._load_init_bbox(init_file)
            else:
                init_bbox = None
            groundtruth = None

        return {
            "name": seq_name,
            "frames": frames,
            "init_bbox": init_bbox,
            "groundtruth": groundtruth,
        }

    def _load_groundtruth(self, gt_file: Path) -> np.ndarray:
        """Load ground truth from file."""
        with open(gt_file, 'r') as f:
            lines = f.readlines()

        groundtruth = []
        for line in lines:
            line = line.strip()
            if line:
                # Handle both comma and space separated formats
                if ',' in line:
                    values = [float(x) for x in line.split(',')]
                else:
                    values = [float(x) for x in line.split()]
                groundtruth.append(values[:4])  # [x1, y1, w, h]

        return np.array(groundtruth, dtype=np.float32)

    def _load_init_bbox(self, init_file: Path) -> np.ndarray:
        """Load initial bounding box."""
        with open(init_file, 'r') as f:
            line = f.readline().strip()

        if ',' in line:
            values = [float(x) for x in line.split(',')]
        else:
            values = [float(x) for x in line.split()]

        return np.array(values[:4], dtype=np.float32)

    def get_sequence(self, name: str) -> Dict:
        """Get sequence by name."""
        if name in self.sequences:
            idx = self.sequences.index(name)
            return self[idx]
        else:
            raise ValueError(f"Sequence {name} not found")

    def iterate_sequences(self) -> Generator:
        """Iterate over all sequences."""
        for idx in range(len(self)):
            yield self[idx]


class LaSOTDataset:
    """
    LaSOT Dataset Loader

    Dataset structure:
    LaSOT/
    ├── airplane/
    │   ├── airplane-1/
    │   │   ├── img/
    │   │   │   ├── 00000001.jpg
    │   │   │   └── ...
    │   │   ├── groundtruth.txt
    │   │   ├── full_occlusion.txt
    │   │   └── out_of_view.txt
    │   └── ...
    └── ...

    Ground truth format: [x1, y1, w, h] (top-left corner + size)
    """

    def __init__(self, root_dir: str, split: str = "test"):
        """
        Initialize LaSOT dataset.

        Args:
            root_dir: Path to LaSOT root directory
            split: "test" or "train"
        """
        self.root_dir = Path(root_dir)
        self.split = split

        # LaSOT test split
        self.test_sequences = self._get_test_sequences()
        self.sequences = self._get_sequences()

    def _get_test_sequences(self) -> List[str]:
        """Get list of test sequence names (from LaSOT protocol)."""
        # LaSOT uses specific sequences for testing
        # Each category has some sequences designated for test
        test_seqs = []

        if not self.root_dir.exists():
            return test_seqs

        for category_dir in sorted(self.root_dir.iterdir()):
            if not category_dir.is_dir() or category_dir.name.startswith('.'):
                continue

            category = category_dir.name
            seq_dirs = sorted([d for d in category_dir.iterdir() if d.is_dir()])

            # LaSOT test split: sequences with IDs >= 16 are test
            for seq_dir in seq_dirs:
                seq_name = seq_dir.name
                try:
                    # Extract ID from name like "airplane-1"
                    seq_id = int(seq_name.split('-')[-1])
                    if self.split == "test" and seq_id >= 16:
                        test_seqs.append(seq_name)
                    elif self.split == "train" and seq_id < 16:
                        test_seqs.append(seq_name)
                except ValueError:
                    continue

        return test_seqs

    def _get_sequences(self) -> List[str]:
        """Get list of all sequences for current split."""
        if self.split == "test":
            return self.test_sequences

        sequences = []
        if not self.root_dir.exists():
            return sequences

        for category_dir in sorted(self.root_dir.iterdir()):
            if not category_dir.is_dir() or category_dir.name.startswith('.'):
                continue

            for seq_dir in sorted(category_dir.iterdir()):
                if seq_dir.is_dir():
                    sequences.append(seq_dir.name)

        return sequences

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict:
        """
        Get a sequence by index.

        Returns:
            dict with keys:
                - name: Sequence name
                - category: Object category
                - frames: List of frame paths
                - init_bbox: Initial bounding box [x1, y1, w, h]
                - groundtruth: Array of bounding boxes
                - full_occlusion: Binary array (1=occluded)
                - out_of_view: Binary array (1=out of view)
        """
        seq_name = self.sequences[idx]
        category = seq_name.rsplit('-', 1)[0]
        seq_dir = self.root_dir / category / seq_name

        # Get frame paths
        img_dir = seq_dir / "img"
        if img_dir.exists():
            frame_files = sorted([
                f for f in img_dir.iterdir()
                if f.suffix.lower() in ['.jpg', '.jpeg', '.png']
            ])
        else:
            frame_files = sorted([
                f for f in seq_dir.iterdir()
                if f.suffix.lower() in ['.jpg', '.jpeg', '.png']
            ])
        frames = [str(f) for f in frame_files]

        # Load ground truth
        gt_file = seq_dir / "groundtruth.txt"
        groundtruth = self._load_groundtruth(gt_file) if gt_file.exists() else None

        # Load occlusion annotations
        occ_file = seq_dir / "full_occlusion.txt"
        full_occlusion = self._load_attribute(occ_file) if occ_file.exists() else None

        oov_file = seq_dir / "out_of_view.txt"
        out_of_view = self._load_attribute(oov_file) if oov_file.exists() else None

        init_bbox = groundtruth[0] if groundtruth is not None else None

        return {
            "name": seq_name,
            "category": category,
            "frames": frames,
            "init_bbox": init_bbox,
            "groundtruth": groundtruth,
            "full_occlusion": full_occlusion,
            "out_of_view": out_of_view,
        }

    def _load_groundtruth(self, gt_file: Path) -> np.ndarray:
        """Load ground truth from file."""
        with open(gt_file, 'r') as f:
            lines = f.readlines()

        groundtruth = []
        for line in lines:
            line = line.strip()
            if line:
                if ',' in line:
                    values = [float(x) for x in line.split(',')]
                else:
                    values = [float(x) for x in line.split()]
                groundtruth.append(values[:4])

        return np.array(groundtruth, dtype=np.float32)

    def _load_attribute(self, attr_file: Path) -> np.ndarray:
        """Load binary attribute (occlusion/out-of-view)."""
        with open(attr_file, 'r') as f:
            line = f.readline().strip()

        if ',' in line:
            values = [int(x) for x in line.split(',')]
        else:
            values = [int(x) for x in line.split()]

        return np.array(values, dtype=np.int32)

    def get_sequence(self, name: str) -> Dict:
        """Get sequence by name."""
        if name in self.sequences:
            idx = self.sequences.index(name)
            return self[idx]
        else:
            raise ValueError(f"Sequence {name} not found")

    def iterate_sequences(self) -> Generator:
        """Iterate over all sequences."""
        for idx in range(len(self)):
            yield self[idx]


class FrameLoader:
    """Utility class for loading video frames."""

    @staticmethod
    def load_frame(path: str) -> np.ndarray:
        """
        Load a single frame as RGB numpy array.

        Args:
            path: Path to image file

        Returns:
            frame: (H, W, 3) RGB numpy array
        """
        frame = cv2.imread(path)
        if frame is None:
            raise ValueError(f"Could not load frame: {path}")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame

    @staticmethod
    def load_frames(paths: List[str], max_frames: int = None) -> List[np.ndarray]:
        """
        Load multiple frames.

        Args:
            paths: List of frame paths
            max_frames: Maximum number of frames to load

        Returns:
            frames: List of (H, W, 3) RGB numpy arrays
        """
        if max_frames is not None:
            paths = paths[:max_frames]

        frames = []
        for path in paths:
            frames.append(FrameLoader.load_frame(path))
        return frames


def download_got10k(root_dir: str, split: str = "test"):
    """
    Download GOT-10k dataset.

    Note: GOT-10k requires registration. This function provides
    instructions for manual download.
    """
    root_dir = Path(root_dir)
    root_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("GOT-10k Download Instructions")
    print("=" * 60)
    print("""
GOT-10k requires registration for download.

1. Go to: http://got-10k.aitestunion.com/
2. Register for an account
3. Download the following:
   - Test set: GOT-10k_Test_000.zip (and any additional parts)
   - Validation set: GOT-10k_Val_000.zip (optional)
4. Extract to: {root_dir}

Expected structure:
{root_dir}/
├── test/
│   ├── GOT-10k_Test_000001/
│   └── ...
└── val/
    └── ...

After downloading, the dataset loader will work automatically.
""".format(root_dir=root_dir))


def download_lasot(root_dir: str):
    """
    Download LaSOT dataset.

    Note: LaSOT is large (~240GB). This provides download instructions.
    """
    root_dir = Path(root_dir)
    root_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("LaSOT Download Instructions")
    print("=" * 60)
    print("""
LaSOT is a large dataset (~240GB total).

Option 1 - Download subset for testing:
1. Go to: http://vision.cs.stonybrook.edu/~lasot/download.html
2. Download individual categories you need
3. Extract to: {root_dir}

Option 2 - Download from Google Drive:
1. Go to: https://drive.google.com/drive/folders/1U9yvJNR5aWQU-sGRz2y1AqZA7pHKQ5jR
2. Download the zip files for categories you need
3. Extract to: {root_dir}

Option 3 - Download test split only (recommended for evaluation):
- Download only the test sequences (categories with ID >= 16)

Expected structure:
{root_dir}/
├── airplane/
│   ├── airplane-1/
│   │   ├── img/
│   │   │   ├── 00000001.jpg
│   │   │   └── ...
│   │   └── groundtruth.txt
│   └── ...
├── basketball/
└── ...

After downloading, the dataset loader will work automatically.
""".format(root_dir=root_dir))


def create_mini_dataset(root_dir: str, dataset_type: str = "got10k", num_sequences: int = 5):
    """
    Create a mini synthetic dataset for testing the pipeline.

    Args:
        root_dir: Where to create the dataset
        dataset_type: "got10k" or "lasot"
        num_sequences: Number of sequences to create
    """
    root_dir = Path(root_dir)

    if dataset_type == "got10k":
        test_dir = root_dir / "test"
        test_dir.mkdir(parents=True, exist_ok=True)

        for i in range(num_sequences):
            seq_name = f"GOT-10k_Test_{i+1:06d}"
            seq_dir = test_dir / seq_name
            seq_dir.mkdir(exist_ok=True)

            # Create dummy frames (colored rectangles moving)
            num_frames = 50
            h, w = 480, 640

            # Random initial bbox
            x1 = np.random.randint(50, w - 150)
            y1 = np.random.randint(50, h - 150)
            bw, bh = 100, 100

            groundtruth = []

            for j in range(num_frames):
                # Create frame
                frame = np.zeros((h, w, 3), dtype=np.uint8)
                frame[:, :] = [50, 50, 50]  # Gray background

                # Move bbox (with some velocity)
                vx, vy = np.random.randint(-5, 6), np.random.randint(-5, 6)
                x1 = np.clip(x1 + vx, 0, w - bw)
                y1 = np.clip(y1 + vy, 0, h - bh)

                # Draw object
                frame[int(y1):int(y1+bh), int(x1):int(x1+bw)] = [200, 100, 50]

                # Save frame
                frame_path = seq_dir / f"{j+1:08d}.jpg"
                cv2.imwrite(str(frame_path), frame)

                groundtruth.append([x1, y1, bw, bh])

            # Save ground truth
            gt_file = seq_dir / "groundtruth.txt"
            with open(gt_file, 'w') as f:
                for gt in groundtruth:
                    f.write(",".join([f"{x:.2f}" for x in gt]) + "\n")

        print(f"Created mini GOT-10k dataset with {num_sequences} sequences at {root_dir}")

    elif dataset_type == "lasot":
        # Create one category with sequences
        category = "object"
        cat_dir = root_dir / category
        cat_dir.mkdir(parents=True, exist_ok=True)

        for i in range(num_sequences):
            seq_name = f"{category}-{i+16}"  # Test split has ID >= 16
            seq_dir = cat_dir / seq_name
            img_dir = seq_dir / "img"
            img_dir.mkdir(parents=True, exist_ok=True)

            # Create dummy frames
            num_frames = 100
            h, w = 480, 640

            x1 = np.random.randint(50, w - 150)
            y1 = np.random.randint(50, h - 150)
            bw, bh = 100, 100

            groundtruth = []
            occlusion = []
            out_of_view = []

            for j in range(num_frames):
                frame = np.zeros((h, w, 3), dtype=np.uint8)
                frame[:, :] = [50, 50, 50]

                vx, vy = np.random.randint(-3, 4), np.random.randint(-3, 4)
                x1 = np.clip(x1 + vx, 0, w - bw)
                y1 = np.clip(y1 + vy, 0, h - bh)

                # Simulate occlusion occasionally
                is_occluded = 1 if (j > 30 and j < 40) else 0

                if not is_occluded:
                    frame[int(y1):int(y1+bh), int(x1):int(x1+bw)] = [200, 100, 50]

                frame_path = img_dir / f"{j+1:08d}.jpg"
                cv2.imwrite(str(frame_path), frame)

                groundtruth.append([x1, y1, bw, bh])
                occlusion.append(is_occluded)
                out_of_view.append(0)

            # Save annotations
            with open(seq_dir / "groundtruth.txt", 'w') as f:
                for gt in groundtruth:
                    f.write(",".join([f"{x:.2f}" for x in gt]) + "\n")

            with open(seq_dir / "full_occlusion.txt", 'w') as f:
                f.write(",".join([str(x) for x in occlusion]))

            with open(seq_dir / "out_of_view.txt", 'w') as f:
                f.write(",".join([str(x) for x in out_of_view]))

        print(f"Created mini LaSOT dataset with {num_sequences} sequences at {root_dir}")


if __name__ == "__main__":
    # Test dataset creation
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create mini datasets
        got10k_dir = Path(tmpdir) / "GOT-10k"
        lasot_dir = Path(tmpdir) / "LaSOT"

        create_mini_dataset(str(got10k_dir), "got10k", num_sequences=3)
        create_mini_dataset(str(lasot_dir), "lasot", num_sequences=3)

        # Test loaders
        print("\nTesting GOT-10k loader:")
        got10k = GOT10kDataset(str(got10k_dir), split="test")
        print(f"Found {len(got10k)} sequences")
        if len(got10k) > 0:
            seq = got10k[0]
            print(f"First sequence: {seq['name']}, {len(seq['frames'])} frames")

        print("\nTesting LaSOT loader:")
        lasot = LaSOTDataset(str(lasot_dir), split="test")
        print(f"Found {len(lasot)} sequences")
        if len(lasot) > 0:
            seq = lasot[0]
            print(f"First sequence: {seq['name']}, {len(seq['frames'])} frames")
