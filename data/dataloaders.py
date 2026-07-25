from pathlib import Path

import numpy as np
import pandas as pd

from torch.utils.data import DataLoader
from lumos_dataset import LumosDataset

ROOT = Path(__file__).resolve().parent.parent
PROC_DIR = ROOT / "res" / "processed"


def make_dataloaders(batch_size: int = 32, num_workers: int = 0, bmd_col: str = "bmd"):
    images = np.load(PROC_DIR / "images.npy")
    meta = pd.read_csv(PROC_DIR / "metadata.csv")

    loaders = {}
    for split in ["train", "val", "test"]:
        mask = (meta["split"] == split).values
        subset_meta = meta[mask]
        subset_images = images[mask]

        ds = LumosDataset(
            subset_images,
            subset_meta,
            train=(split == "train"),
            bmd_col=bmd_col,
        )
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            drop_last=(split == "train"),
        )
        print(f"{split:5s}: {len(ds):4d} images -> {len(loaders[split])} batches")

    return loaders


if __name__ == "__main__":
    loaders = make_dataloaders(batch_size=8)

    # quick sanity check on one batch
    images, labels, bmd = next(iter(loaders["train"]))
    print("\nOne training batch:")
    print(f"  images: {tuple(images.shape)}  dtype {images.dtype}")
    print(f"  labels: {tuple(labels.shape)}  -> {labels.tolist()}")
    print(f"  bmd   : {tuple(bmd.shape)}  -> {[round(x, 3) for x in bmd.tolist()]}")
    print(f"  image value range: [{images.min():.2f}, {images.max():.2f}]  (normalized, can be negative)")