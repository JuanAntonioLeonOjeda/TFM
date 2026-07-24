from pathlib import Path
import re

import numpy as np
import pandas as pd
import pydicom

PROJECT_ROOT = Path(__file__).resolve().parent.parent   # tfm/
DATA_DIR = PROJECT_ROOT / "res" / "dataset"
EXCEL_PATH = DATA_DIR / "lumos_clinical_data.xlsx"

_PATTERN = re.compile(r"^lumos_x_(\d+)$")

CLINICAL_COLUMNS = [
    "patient_id", "age", "gender", "ethnicity", "height", "weight", "BMI",
    "Osteoporosis", "bmd", "bmd_L1-L4", "bmc_L1-L4",
    "t_value", "T_L1-L4", "Z_L1-L4",
]


def load_clinical(path: Path = EXCEL_PATH) -> pd.DataFrame:
    """Reads the clinical Excel file."""
    df = pd.read_excel(path)
    df["patient_id"] = df["patient_id"].astype(str).str.extract(r"(\d+)")[0].astype(int)
    return df


def load_image_index(base: Path = DATA_DIR) -> pd.DataFrame:
    """Builds a DataFrame with patient_id and the path of every DICOM file."""
    rows = [
        {"patient_id": int(_PATTERN.match(folder.name).group(1)), "image_path": str(f)}
        for folder in Path(base).rglob("*")
        if folder.is_dir() and _PATTERN.match(folder.name)
        for f in sorted(folder.iterdir())
        if f.is_file() and f.suffix.lower() == ".dcm"
    ]
    return pd.DataFrame(rows)


def load_lumos(base: Path = DATA_DIR, all_columns: bool = False) -> pd.DataFrame:
    base = Path(base)
    clinical = load_clinical(base / EXCEL_PATH.name)
    images = load_image_index(base)

    if images.empty:
        raise FileNotFoundError(f"No lumos_x_XXX folders found in {base.resolve()}")

    if not all_columns:
        clinical = clinical[[c for c in CLINICAL_COLUMNS if c in clinical.columns]]

    df = images.merge(clinical, on="patient_id", how="inner")
    df["label"] = df["Osteoporosis"].astype(int)

    return df.sort_values(["patient_id", "image_path"]).reset_index(drop=True)


def read_image(path, normalize: bool = True) -> np.ndarray:
    ds = pydicom.dcmread(path)
    img = ds.pixel_array.astype(np.float32)

    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        img = img.max() - img

    if normalize:
        lo, hi = img.min(), img.max()
        img = (img - lo) / (hi - lo) if hi > lo else np.zeros_like(img)

    return img


if __name__ == "__main__":
    df = load_lumos()
    print(df.shape)
    print(df["label"].value_counts().sort_index())
    print(df.head())