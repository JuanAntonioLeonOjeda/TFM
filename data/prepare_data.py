from pathlib import Path
import sys

import numpy as np
import pandas as pd

# project imports (adjust the module paths to your folder layout)
from df_import import load_clinical, load_image_index
from df_preprocess import main as run_preprocess

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "res" / "dataset"
PROC_DIR = ROOT / "res" / "processed"


def step(msg):
    print("\n" + "=" * 62)
    print(f"  {msg}")
    print("=" * 62)


# ---------------------------------------------------------------------
# 1. CHECK RAW DATA
# ---------------------------------------------------------------------
def check_raw():
    step("1. CHECKING RAW DATA")

    if not RAW_DIR.exists():
        sys.exit(f"ERROR: raw dataset folder not found: {RAW_DIR}")

    # clinical excel
    try:
        clinical = load_clinical()
        print(f"Clinical rows: {len(clinical)}  (expected 803)")
    except Exception as e:
        sys.exit(f"ERROR reading clinical Excel: {e}")

    # image folders
    images = load_image_index()
    if images.empty:
        sys.exit("ERROR: no DICOM images found. Check the folders are unzipped.")
    print(f"Images found : {len(images)}  (expected ~1620)")
    print(f"Patients     : {images['patient_id'].nunique()}")

    return True


# ---------------------------------------------------------------------
# 2. RUN PREPROCESSING
# ---------------------------------------------------------------------
def run():
    step("2. PREPROCESSING (DICOM -> npy)")

    if (PROC_DIR / "images.npy").exists():
        resp = input("Processed files already exist. Regenerate? [y/N] ").strip().lower()
        if resp != "y":
            print("Skipping preprocessing, keeping existing files.")
            return
    run_preprocess()


# ---------------------------------------------------------------------
# 3. VERIFY OUTPUT
# ---------------------------------------------------------------------
def verify():
    step("3. VERIFYING OUTPUT")

    img_path = PROC_DIR / "images.npy"
    meta_path = PROC_DIR / "metadata.csv"

    if not img_path.exists() or not meta_path.exists():
        sys.exit("ERROR: processed files were not created.")

    images = np.load(img_path)
    meta = pd.read_csv(meta_path)

    print(f"images.npy : shape {images.shape}, dtype {images.dtype}, "
          f"{img_path.stat().st_size / 1e6:.0f} MB")
    print(f"metadata   : {len(meta)} rows")

    assert len(images) == len(meta), "images and metadata are misaligned!"

    print("\nSplit distribution:")
    print(meta["split"].value_counts().to_string())

    print("\nClass per split:")
    print(pd.crosstab(meta["split"], meta["label"]).to_string())

    # leakage check: no patient in more than one split
    per_patient_splits = meta.groupby("patient_id")["split"].nunique()
    leaks = (per_patient_splits > 1).sum()
    if leaks:
        print(f"\nWARNING: {leaks} patients appear in more than one split (data leakage!)")
    else:
        print("\nOK: no patient leakage between splits.")


if __name__ == "__main__":
    check_raw()
    run()
    verify()

    step("Dataset ready for training")