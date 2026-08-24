"""
preprocess_roi.py
==================
Variante del preprocesamiento que:
  1. Usa SOLO la vista AP de cada paciente (res/processed/ap_only.csv).
  2. Aplica el recorte heurístico del ROI (roi_crop.py) sobre la imagen
     ORIGINAL, antes del resize.
  3. Redimensiona a 224x224 y aplica CLAHE, igual que antes.
  4. Divide en train/val/test por paciente (estratificado).

Genera, en res/processed/:
    images_roi.npy     array (N, 224, 224) uint8
    metadata_roi.csv    idx, patient_id, label, bmd, t_value, split, ...

Requiere haber ejecutado antes:
    python check_view_position2.py   (genera view_labels.csv)
    python filter_ap_view.py         (genera ap_only.csv)

Run:
    python preprocess_roi.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from data.df_import import load_lumos, read_image   # AJUSTA al nombre real de tu módulo
from roi_crop import crop_roi

# =====================================================================
# CONFIG
# =====================================================================
IMG_SIZE = 224
USE_CLAHE = True
CLAHE_CLIP = 2.0
CLAHE_GRID = (8, 8)

VAL_FRACTION = 0.15
TEST_FRACTION = 0.15
SEED = 42

ROOT = Path(__file__).resolve().parent
PROC_DIR = ROOT / "res" / "processed"


# =====================================================================
# SPLIT BY PATIENT (idéntico al preprocess.py original)
# =====================================================================
def split_by_patient(df: pd.DataFrame) -> pd.DataFrame:
    patients = df.groupby("patient_id")["label"].first().reset_index()

    trainval_ids, test_ids = train_test_split(
        patients["patient_id"], test_size=TEST_FRACTION,
        stratify=patients["label"], random_state=SEED,
    )
    trainval = patients[patients["patient_id"].isin(trainval_ids)]
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


# =====================================================================
# IMAGE PROCESSING (con recorte ROI antes del resize)
# =====================================================================
def process_image(path: str) -> np.ndarray:
    img = read_image(path)                       # float32 [0,1], MONOCHROME1 ya corregido
    img = (img * 255).astype(np.uint8)

    img = crop_roi(img)                          # <-- NUEVO: recorte ROI en alta resolución

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)

    if USE_CLAHE:
        clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_GRID)
        img = clahe.apply(img)

    return img


# =====================================================================
# MAIN
# =====================================================================
def main():
    PROC_DIR.mkdir(parents=True, exist_ok=True)

    ap_only_path = PROC_DIR / "ap_only.csv"
    if not ap_only_path.exists():
        raise FileNotFoundError(
            f"No existe {ap_only_path}. Ejecuta antes check_view_position2.py "
            "y filter_ap_view.py."
        )

    print("Cargando datos clínicos + índice de imágenes AP...")
    ap_only = pd.read_csv(ap_only_path)[["patient_id", "image_path"]]

    full = load_lumos(all_columns=False)
    clinical_cols = [c for c in full.columns if c not in ("image_path",)]
    clinical = full[clinical_cols].drop_duplicates(subset="patient_id")

    df = ap_only.merge(clinical, on="patient_id", how="inner")
    print(f"Pacientes con AP + datos clínicos: {len(df)}")

    df = split_by_patient(df)

    print("\nSplit summary (patients, 1 image each):")
    print(df["split"].value_counts().to_string())
    print("\nClass distribution per split:")
    print(pd.crosstab(df["split"], df["label"]).to_string())

    n = len(df)
    images = np.zeros((n, IMG_SIZE, IMG_SIZE), dtype=np.uint8)

    print(f"\nProcessing {n} images (AP + ROI crop)...")
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

    meta_cols = [c for c in ["patient_id", "image_path", "label",
                             "bmd", "bmd_L1-L4", "t_value", "T_L1-L4",
                             "age", "gender", "BMI", "split"] if c in df.columns]
    meta = df[meta_cols].reset_index(drop=True)
    meta.insert(0, "idx", range(n))

    np.save(PROC_DIR / "images_roi.npy", images)
    meta.to_csv(PROC_DIR / "metadata_roi.csv", index=False)

    size_mb = (PROC_DIR / "images_roi.npy").stat().st_size / 1e6
    print(f"\nSaved:")
    print(f"  {PROC_DIR / 'images_roi.npy'}   ({size_mb:.0f} MB, shape {images.shape})")
    print(f"  {PROC_DIR / 'metadata_roi.csv'} ({n} rows)")


if __name__ == "__main__":
    main()