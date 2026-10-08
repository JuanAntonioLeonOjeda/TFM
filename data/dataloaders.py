"""
dataloaders.py
==============
Builds train/val/test DataLoaders from the preprocessed files.
Supports both the full dataset (both views) and the ROI dataset (AP only,
cropped), selectable via the img/meta filenames.

NUEVO: soporte opcional para covariables clínicas (--covariate-cols), p.ej.
["age", "gender", "BMI"]. Cuando se especifican:
    - Las columnas categóricas de tipo texto (p.ej. "gender": "M"/"F") se
      codifican automáticamente a enteros (0, 1, ...), de forma determinista
      (orden alfabético de las categorías encontradas en TODO el dataset,
      ya que es solo una etiqueta, no una estadística que pueda filtrar
      información numérica de val/test a train).
    - Las estadísticas de normalización (media/std) de TODAS las
      covariables (incluida la ya codificada) se calculan SOLO sobre el
      split de train, y se reutilizan en val/test, para evitar fuga de
      datos (data leakage).
"""

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from data.df_import import PROC_DIR
from data.lumos_dataset import LumosDataset

# Covariables clínicas fijas para los experimentos de este trabajo.
# age, gender, BMI -- factores de riesgo clínicamente justificados (ver
# memoria). Se excluyen deliberadamente ethnicity (constante, sin varianza)
# y cualquier columna derivada de la DXA (bmd_*, T_*, Z_*, bmc_*, s_*,
# avgh_*, avgw_*), por fuga de datos: son la propia medición que define
# la etiqueta a predecir.
MEDICAL_COVARIATES = ["age", "gender", "BMI"]


def _encode_categorical_columns(meta: pd.DataFrame, covariate_cols: list[str]) -> pd.DataFrame:
    """
    Convierte a entero cualquier columna de covariables que NO sea numérica
    (texto clásico dtype=object, el nuevo dtype "string" de pandas, o
    "category"), p.ej. "gender": "M"/"F" -> 0/1. No modifica columnas ya
    numéricas. Devuelve una COPIA de meta, no modifica la original.
    """
    meta = meta.copy()
    for col in covariate_cols:
        if not pd.api.types.is_numeric_dtype(meta[col]):
            categories = sorted(meta[col].dropna().unique())
            mapping = {cat: i for i, cat in enumerate(categories)}
            print(f"  Covariable '{col}' codificada como texto -> entero: {mapping}")
            meta[col] = meta[col].map(mapping).astype(float)
    return meta


def make_dataloaders(batch_size: int = 32, num_workers: int = 0, bmd_col: str = "bmd",
                      img_file: str = "images.npy", meta_file: str = "metadata.csv",
                      covariate_cols: list[str] | None = None):
    """
    img_file / meta_file let you switch datasets, e.g.:
        make_dataloaders()                                       # original (both views)
        make_dataloaders(img_file="images_roi.npy",
                          meta_file="metadata_roi.csv")           # AP + ROI

    covariate_cols: p.ej. ["age", "gender", "BMI"]. None (default) mantiene
        el comportamiento original: cada batch es (image, label, bmd).
        Si se especifica, cada batch pasa a ser (image, label, bmd, covariates).
    """
    images = np.load(PROC_DIR / img_file)
    meta = pd.read_csv(PROC_DIR / meta_file)

    if covariate_cols is not None:
        missing = [c for c in covariate_cols if c not in meta.columns]
        if missing:
            raise ValueError(f"Columnas de covariables no encontradas en {meta_file}: {missing}")
        meta = _encode_categorical_columns(meta, covariate_cols)

    loaders = {}
    covariate_stats = None  # se calcula con el split de train, se reutiliza en val/test

    for split in ["train", "val", "test"]:
        mask = (meta["split"] == split).values
        subset_meta = meta[mask]
        subset_images = images[mask]

        ds = LumosDataset(
            subset_images,
            subset_meta,
            train=(split == "train"),
            bmd_col=bmd_col,
            covariate_cols=covariate_cols,
            covariate_stats=covariate_stats,  # None en train (se calculan dentro); ya fijadas en val/test
        )
        ds.meta_subset = subset_meta.reset_index(drop=True)  # expuesto para cálculo de pesos de clase, etc.

        if split == "train" and covariate_cols is not None:
            covariate_stats = ds.covariate_stats  # se reutilizan en val/test, ver arriba

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
    parser.add_argument("--covariates", action="store_true",
                         help=f"Usar las covariables médicas fijas del proyecto: {MEDICAL_COVARIATES}")
    args = parser.parse_args()

    covariate_cols = MEDICAL_COVARIATES if args.covariates else None

    if args.roi:
        loaders = make_dataloaders(batch_size=8, img_file="images_roi.npy",
                                    meta_file="metadata_roi.csv", covariate_cols=covariate_cols)
    else:
        loaders = make_dataloaders(batch_size=8, covariate_cols=covariate_cols)

    batch = next(iter(loaders["train"]))
    print("\nOne training batch:")
    if covariate_cols is None:
        images, labels, bmd = batch
    else:
        images, labels, bmd, covariates = batch
        print(f"  covariates: {tuple(covariates.shape)}  -> {covariates[0].tolist()} (ejemplo, 1ª fila)")
    print(f"  images: {tuple(images.shape)}  dtype {images.dtype}")
    print(f"  labels: {tuple(labels.shape)}  -> {labels.tolist()}")
    print(f"  bmd   : {tuple(bmd.shape)}  -> {[round(x, 3) for x in bmd.tolist()]}")