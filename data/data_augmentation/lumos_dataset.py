"""
lumos_dataset.py
================
Definition of the LumosDataset class and its image transforms.

CHANGE: stronger data augmentation, to test whether it helps reduce the
residual overfitting observed with the base configuration (dropout 0.3,
no weight decay). Compared to the original transforms:
    - Rotation: 10 -> 15 degrees
    - RandomResizedCrop scale: (0.85, 1.0) -> (0.75, 1.0)
    - ColorJitter: 0.15 -> 0.2 (brightness and contrast)
    - Added: small random translation (RandomAffine)
"""

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transforms(train: bool):
    """Augmentation for training; only formatting + normalization for val/test."""
    steps = [T.ToPILImage()]
    if train:
        steps += [
            T.RandomRotation(degrees=15),                        # was 10
            T.RandomResizedCrop(224, scale=(0.75, 1.0)),         # was (0.85, 1.0)
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.2, contrast=0.2),          # was 0.15
            T.RandomAffine(degrees=0, translate=(0.05, 0.05)),   # NEW: small shifts
        ]
    steps += [
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
    return T.Compose(steps)


class LumosDataset(Dataset):
    def __init__(self, images, meta: pd.DataFrame, train: bool, bmd_col: str = "bmd"):
        self.images = images
        self.meta = meta.reset_index(drop=True)
        self.transform = build_transforms(train)
        self.bmd_col = bmd_col

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, i):
        gray = self.images[i]
        rgb = np.stack([gray, gray, gray], axis=-1)
        image = self.transform(rgb)

        label = int(self.meta.at[i, "label"])
        bmd = float(self.meta.at[i, self.bmd_col])

        return image, torch.tensor(label, dtype=torch.long), torch.tensor(bmd, dtype=torch.float32)