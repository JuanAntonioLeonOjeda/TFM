"""DICOM -> images[_<size>].npy + metadata[_<size>].csv (ambas vistas, split por paciente).

Uso: python -m data.df_preprocess [--size 320]
"""

import argparse

import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from data.df_import import PROC_DIR, load_lumos, read_image

IMG_SIZE = 224
CLAHE_CLIP = 2.0
CLAHE_GRID = (8, 8)

VAL_FRACTION = 0.15
TEST_FRACTION = 0.15
SEED = 42

META_COLS = ["patient_id", "image_path", "view", "label",
             "bmd", "bmd_L1-L4", "t_value", "T_L1-L4",
             "age", "gender", "BMI", "split"]


def split_by_patient(df: pd.DataFrame) -> pd.DataFrame:
    """Añade la columna 'split' (train/val/test), estratificando por paciente."""
    patients = df.groupby("patient_id")["label"].first().reset_index()

    trainval_ids, test_ids = train_test_split(
        patients["patient_id"], test_size=TEST_FRACTION,
        stratify=patients["label"], random_state=SEED,
    )
    trainval = patients[patients["patient_id"].isin(trainval_ids)]
    # VAL_FRACTION es sobre el total, no sobre train+val
    val_ratio = VAL_FRACTION / (1.0 - TEST_FRACTION)
    train_ids, val_ids = train_test_split(
        trainval["patient_id"], test_size=val_ratio,
        stratify=trainval["label"], random_state=SEED,
    )

    split = pd.Series("train", index=df.index)
    split[df["patient_id"].isin(val_ids)] = "val"
    split[df["patient_id"].isin(test_ids)] = "test"
    df = df.copy()
    df["split"] = split.values
    return df


def process_image(path: str, crop=None, size: int = IMG_SIZE) -> np.ndarray:
    """DICOM -> uint8 -> crop opcional (en resolución original) -> resize -> CLAHE."""
    img = (read_image(path) * 255).astype(np.uint8)
    if crop is not None:
        img = crop(img)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_GRID)
    return clahe.apply(img)


def process_all(df: pd.DataFrame, process_row, size: int = IMG_SIZE) -> np.ndarray:
    images = np.zeros((len(df), size, size), dtype=np.uint8)
    failed = []
    print(f"\nProcessing {len(df)} images...")
    for i, row in enumerate(tqdm(df.itertuples(index=False), total=len(df))):
        try:
            images[i] = process_row(row)
        except Exception as e:
            failed.append((i, row.image_path, str(e)))

    if failed:
        print(f"\nWARNING: {len(failed)} images failed:")
        for i, path, err in failed[:5]:
            print(f"  [{i}] {path} -> {err}")
    return images


def print_split_summary(df: pd.DataFrame):
    print("\nSplit summary (images):")
    print(df["split"].value_counts().to_string())
    print("\nClass distribution per split:")
    print(pd.crosstab(df["split"], df["label"]).to_string())


def save_dataset(df: pd.DataFrame, images: np.ndarray, suffix: str = ""):
    """Guarda images{suffix}.npy y metadata{suffix}.csv en PROC_DIR."""
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    meta = df[[c for c in META_COLS if c in df.columns]].reset_index(drop=True)
    meta.insert(0, "idx", range(len(meta)))

    img_path = PROC_DIR / f"images{suffix}.npy"
    meta_path = PROC_DIR / f"metadata{suffix}.csv"
    np.save(img_path, images)
    meta.to_csv(meta_path, index=False)

    size_mb = img_path.stat().st_size / 1e6
    print(f"\nSaved:")
    print(f"  {img_path}   ({size_mb:.0f} MB, shape {images.shape})")
    print(f"  {meta_path} ({len(meta)} rows)")


def main(size: int = IMG_SIZE):
    print("Loading index + clinical data...")
    df = split_by_patient(load_lumos())
    print_split_summary(df)

    images = process_all(df, lambda row: process_image(row.image_path, size=size), size)
    save_dataset(df, images, suffix="" if size == IMG_SIZE else f"_{size}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=IMG_SIZE)
    main(parser.parse_args().size)
