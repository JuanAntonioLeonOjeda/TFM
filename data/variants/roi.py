"""Variante ROI: solo vista AP, recortada a la franja central (25-75% del ancho).

Requiere: python -m data.views (genera ap_only.csv)
Uso:      python -m data.variants.roi
"""

import numpy as np
import pandas as pd

from data.df_import import PROC_DIR, load_lumos
from data.df_preprocess import (
    print_split_summary, process_all, process_image, save_dataset, split_by_patient,
)

LEFT_FRACTION = 0.25
RIGHT_FRACTION = 0.75


def crop_roi(img: np.ndarray) -> np.ndarray:
    """Franja vertical central, sin recortar en altura. Puede cortar columnas con escoliosis marcada."""
    w = img.shape[1]
    return img[:, int(w * LEFT_FRACTION):int(w * RIGHT_FRACTION)]


def load_clinical_per_patient() -> pd.DataFrame:
    full = load_lumos(all_columns=False)
    return full.drop(columns="image_path").drop_duplicates(subset="patient_id")


def main():
    ap_only_path = PROC_DIR / "ap_only.csv"
    if not ap_only_path.exists():
        raise FileNotFoundError(f"No existe {ap_only_path}. Ejecuta antes: python -m data.views")

    ap_only = pd.read_csv(ap_only_path)[["patient_id", "image_path"]]
    df = ap_only.merge(load_clinical_per_patient(), on="patient_id", how="inner")
    print(f"Pacientes con AP + datos clínicos: {len(df)}")

    df = split_by_patient(df)
    print_split_summary(df)

    images = process_all(df, lambda row: process_image(row.image_path, crop=crop_roi))
    save_dataset(df, images, suffix="_roi")


if __name__ == "__main__":
    main()
