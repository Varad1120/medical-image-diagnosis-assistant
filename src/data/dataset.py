"""
PyTorch Dataset class for Chest X-Ray Pneumonia detection.
"""

import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class ChestXRayDataset(Dataset):
    """
    Custom PyTorch Dataset for Chest X-Ray images.

    Directory structure expected:
        root/
            NORMAL/
                img1.jpeg
                ...
            PNEUMONIA/
                img1.jpeg
                ...
    """

    CLASS_MAP = {"NORMAL": 0, "PNEUMONIA": 1}

    def __init__(
        self,
        root_dir: str,
        transform: Optional[transforms.Compose] = None,
        split: str = "train",
    ):
        """
        Args:
            root_dir: Path to dataset split directory (train/val/test).
            transform: Optional torchvision transforms to apply.
            split: One of 'train', 'val', 'test'.
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.split = split
        self.samples = []
        self.labels = []

        self._load_samples()

    def _load_samples(self):
        """Walk directory and collect image paths and labels."""
        for class_name, label in self.CLASS_MAP.items():
            class_dir = self.root_dir / class_name
            if not class_dir.exists():
                raise FileNotFoundError(f"Class directory not found: {class_dir}")
            for img_file in class_dir.iterdir():
                if img_file.suffix.lower() in (".jpg", ".jpeg", ".png"):
                    self.samples.append(img_file)
                    self.labels.append(label)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple:
        img_path = self.samples[idx]
        label = self.labels[idx]

        # Load image as RGB
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label

    def get_class_weights(self) -> np.ndarray:
        """Compute inverse frequency class weights for handling imbalance."""
        label_array = np.array(self.labels)
        class_counts = np.bincount(label_array)
        total = len(label_array)
        weights = total / (len(class_counts) * class_counts)
        return weights
