
from pathlib import Path

import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from db_data import load_lumos, read_image   # <-- adjust if your module has another name

# =====================================================================
# CONFIG
# =====================================================================
IMG_SIZE = 224                 # ResNet50 input size
USE_CLAHE = True               # local contrast enhancement
CLAHE_CLIP = 2.0
CLAHE_GRID = (8, 8)

VAL_FRACTION = 0.15            # of all patients
TEST_FRACTION = 0.15
SEED = 42

OUT_DIR = Path(__file__).resolve().parent.parent / "res" / "processed"


# =====================================================================
# SPLIT BY PATIENT
# =====================================================================
def split_by_patient(df: pd.DataFrame) -> pd.DataFrame:
    """Assigns each row a split ('train'/'val'/'test'), splitting by patient."""
    # one label per patient (all its images share the same label)
    patients = df.groupby("patient_id")["label"].first().reset_index()

    # train+val  vs  test
    trainval_ids, test_ids = train_test_split(
        patients["patient_id"],
        test_size=TEST_FRACTION,
        stratify=patients["label"],
        random_state=SEED,
    )

    # train  vs  val   (val_fraction relative to the whole -> adjust)
    trainval = patients[patients["patient_id"].isin(trainval_ids)]
    val_ratio = VAL_FRACTION / (1.0 - TEST_FRACTION)
    train_ids, val_ids = train_test_split(
        trainval["patient_id"],
        test_size=val_ratio,
        stratify=trainval["label"],
        random_state=SEED,
    )

    split = pd.Series("train", index=df.index)
    split[df["patient_id"].isin(val_ids)] = "val"
    split[df["patient_id"].isin(test_ids)] = "test"
    df = df.copy()
    df["split"] = split.values
    return df


# =====================================================================
# IMAGE PROCESSING
# =====================================================================
def process_image(path: str) -> np.ndarray:
    """DICOM -> uint8 grayscale, CLAHE, resized to IMG_SIZE x IMG_SIZE."""
    img = read_image(path)                       # float32 in [0, 1], MONOCHROME1 fixed
    img = (img * 255).astype(np.uint8)

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)

    if USE_CLAHE:
        clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_GRID)
        img = clahe.apply(img)

    return img


# =====================================================================
# MAIN
# =====================================================================
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading index + clinical data...")
    df = load_lumos()
    df = split_by_patient(df)

    print("\nSplit summary (images):")
    print(df["split"].value_counts().to_string())
    print("\nClass distribution per split:")
    print(pd.crosstab(df["split"], df["label"]).to_string())

    # process every image
    n = len(df)
    images = np.zeros((n, IMG_SIZE, IMG_SIZE), dtype=np.uint8)

    print(f"\nProcessing {n} images...")
    failed = []
    for i, path in enumerate(tqdm(df["image_path"].tolist())):
        try:
            images[i] = process_image(path)
        except Exception as e:
            failed.append((i, path, str(e)))

    if failed:
        print(f"\nWARNING: {len(failed)} images failed:")
        for i, path, err in failed[:5]:
            print(f"  [{i}] {path} -> {err}")

    # metadata
    meta_cols = [c for c in ["patient_id", "image_path", "label",
                             "bmd", "bmd_L1-L4", "t_value", "T_L1-L4",
                             "age", "gender", "BMI", "split"] if c in df.columns]
    meta = df[meta_cols].reset_index(drop=True)
    meta.insert(0, "idx", range(n))

    # save
    np.save(OUT_DIR / "images.npy", images)
    meta.to_csv(OUT_DIR / "metadata.csv", index=False)

    size_mb = (OUT_DIR / "images.npy").stat().st_size / 1e6
    print(f"\nSaved:")
    print(f"  {OUT_DIR / 'images.npy'}   ({size_mb:.0f} MB, shape {images.shape})")
    print(f"  {OUT_DIR / 'metadata.csv'} ({n} rows)")
    print("\nDone. Load these two files in your training code.")


if __name__ == "__main__":
    main()