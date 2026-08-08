"""
dataloaders.py
==============
Builds train/val/test DataLoaders from the preprocessed files.
Supports both the full dataset (both views) and the ROI dataset (AP only,
cropped), selectable via the img/meta filenames.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from data.lumos_dataset import LumosDataset

ROOT = Path(__file__).resolve().parent.parent
PROC_DIR = ROOT / "res" / "processed"


def make_dataloaders(batch_size: int = 32, num_workers: int = 0, bmd_col: str = "bmd",
                      img_file: str = "images.npy", meta_file: str = "metadata.csv"):
    """
    img_file / meta_file let you switch datasets, e.g.:
        make_dataloaders()                                       # original (both views)
        make_dataloaders(img_file="images_roi.npy",
                          meta_file="metadata_roi.csv")           # AP + ROI
    """
    images = np.load(PROC_DIR / img_file)
    meta = pd.read_csv(PROC_DIR / meta_file)

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
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--roi", action="store_true", help="use the AP+ROI dataset")
    args = parser.parse_args()

    if args.roi:
        loaders = make_dataloaders(batch_size=8, img_file="images_roi.npy",
                                    meta_file="metadata_roi.csv")
    else:
        loaders = make_dataloaders(batch_size=8)

    images, labels, bmd = next(iter(loaders["train"]))
    print("\nOne training batch:")
    print(f"  images: {tuple(images.shape)}  dtype {images.dtype}")
    print(f"  labels: {tuple(labels.shape)}  -> {labels.tolist()}")
    print(f"  bmd   : {tuple(bmd.shape)}  -> {[round(x, 3) for x in bmd.tolist()]}")