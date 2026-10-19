import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset

# Necesario al usar pesos preentrenados en ImageNet
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transforms(train: bool):
    steps = [T.ToPILImage()]
    if train:
        steps += [
            T.RandomRotation(degrees=15),
            T.RandomResizedCrop(224, scale=(0.75, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.2, contrast=0.2),
            T.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        ]
    steps += [
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
    return T.Compose(steps)


def gray_to_rgb(gray: np.ndarray) -> np.ndarray:
    """(H, W) -> (H, W, 3): ResNet espera 3 canales."""
    return np.stack([gray, gray, gray], axis=-1)


def compute_covariate_stats(meta: pd.DataFrame, cols: list[str]) -> dict:
    """{col: (media, std)} para z-score. Calcular SIEMPRE sobre train para evitar fuga de datos."""
    return {col: (meta[col].mean(), meta[col].std() or 1.0) for col in cols}


def normalize_covariates(row, cols: list[str], stats: dict) -> list[float]:
    return [(float(row[col]) - stats[col][0]) / stats[col][1] for col in cols]


class LumosDataset(Dataset):
    """Devuelve (image, label, bmd) o, si hay covariate_cols, (image, label, bmd, covariates)."""

    def __init__(self, images, meta: pd.DataFrame, train: bool, bmd_col: str = "bmd",
                 covariate_cols: list[str] | None = None,
                 covariate_stats: dict | None = None):
        self.images = images
        self.meta = meta.reset_index(drop=True)
        self.transform = build_transforms(train)
        self.bmd_col = bmd_col
        self.covariate_cols = covariate_cols

        if covariate_cols is not None:
            missing = [c for c in covariate_cols if c not in self.meta.columns]
            if missing:
                raise ValueError(f"Columnas de covariables no encontradas en meta: {missing}")
            # None solo para el split de train; val/test reciben las stats de train
            self.covariate_stats = covariate_stats or compute_covariate_stats(self.meta, covariate_cols)

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, i):
        image = self.transform(gray_to_rgb(self.images[i]))
        label = torch.tensor(int(self.meta.at[i, "label"]), dtype=torch.long)
        bmd = torch.tensor(float(self.meta.at[i, self.bmd_col]), dtype=torch.float32)

        if self.covariate_cols is None:
            return image, label, bmd

        row = self.meta.iloc[i]
        covariates = torch.tensor(
            normalize_covariates(row, self.covariate_cols, self.covariate_stats), dtype=torch.float32
        )
        return image, label, bmd, covariates
