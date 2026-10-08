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

NUEVO: soporte opcional para covariables clínicas (edad, sexo, IMC...),
devueltas como un cuarto elemento del batch cuando se especifica
`covariate_cols`. Si no se especifica, el comportamiento es idéntico al
anterior (compatible con el pipeline de referencia).
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
    def __init__(self, images, meta: pd.DataFrame, train: bool, bmd_col: str = "bmd",
                 covariate_cols: list[str] | None = None,
                 covariate_stats: dict | None = None):
        """
        covariate_cols: lista de columnas de `meta` a usar como covariables
            clínicas (p.ej. ["age", "gender_encoded", "BMI"]). Si es None
            (default), el Dataset se comporta EXACTAMENTE igual que antes:
            __getitem__ devuelve (image, label, bmd), sin cuarto elemento.

        covariate_stats: dict {col: (media, std)} para normalizar cada
            covariable numérica (z-score), calculado SIEMPRE sobre el split
            de train y reutilizado en val/test para evitar fuga de datos
            (data leakage). Si covariate_cols no es None y covariate_stats
            es None, se calculan aquí mismo (asume que este Dataset es el
            de train) -- ver nota en dataloaders.py sobre cómo se propaga
            train -> val/test.
        """
        self.images = images
        self.meta = meta.reset_index(drop=True)
        self.transform = build_transforms(train)
        self.bmd_col = bmd_col
        self.covariate_cols = covariate_cols

        if covariate_cols is not None:
            missing = [c for c in covariate_cols if c not in self.meta.columns]
            if missing:
                raise ValueError(f"Columnas de covariables no encontradas en meta: {missing}")

            if covariate_stats is None:
                covariate_stats = {
                    col: (self.meta[col].mean(), self.meta[col].std() or 1.0)
                    for col in covariate_cols
                }
            self.covariate_stats = covariate_stats

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, i):
        gray = self.images[i]
        rgb = np.stack([gray, gray, gray], axis=-1)
        image = self.transform(rgb)

        label = int(self.meta.at[i, "label"])
        bmd = float(self.meta.at[i, self.bmd_col])

        if self.covariate_cols is None:
            return image, torch.tensor(label, dtype=torch.long), torch.tensor(bmd, dtype=torch.float32)

        cov_values = []
        for col in self.covariate_cols:
            mean, std = self.covariate_stats[col]
            raw = float(self.meta.at[i, col])
            cov_values.append((raw - mean) / std)  # z-score, con stats del train

        covariates = torch.tensor(cov_values, dtype=torch.float32)

        return (
            image,
            torch.tensor(label, dtype=torch.long),
            torch.tensor(bmd, dtype=torch.float32),
            covariates,
        )