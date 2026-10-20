import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from data.df_import import PROC_DIR
from data.lumos_dataset import LumosDataset

# Se excluyen ethnicity, que no tiene varianza, y las columnas de la DXA (bmd, T y Z),
# porque son la propia medición que define la etiqueta y provocarían fuga de datos
MEDICAL_COVARIATES = ["age", "gender", "BMI"]

# Variantes de dataset generadas por data/df_preprocess.py y data/variants/
DATASETS = {
    "full": {"img_file": "images.npy", "meta_file": "metadata.csv"},
    "roi": {"img_file": "images_roi.npy", "meta_file": "metadata_roi.csv"},
    "roi_hybrid": {"img_file": "images_roi_hybrid.npy", "meta_file": "metadata_roi_hybrid.csv"},
    "hires": {"img_file": "images_320.npy", "meta_file": "metadata_320.csv"},
    "roi_yolo": {"img_file": "images_roi_yolo.npy", "meta_file": "metadata_roi_yolo.csv"},
}


# Convierte las columnas de texto en enteros por orden alfabético. Se hace sobre todo el dataset
# porque es solo una codificación y no filtra información
def _encode_categorical_columns(meta: pd.DataFrame, covariate_cols: list[str]) -> pd.DataFrame:
    meta = meta.copy()
    for col in covariate_cols:
        if not pd.api.types.is_numeric_dtype(meta[col]):
            categories = sorted(meta[col].dropna().unique())
            mapping = {cat: i for i, cat in enumerate(categories)}
            print(f"  Covariable '{col}' codificada como texto -> entero: {mapping}")
            meta[col] = meta[col].map(mapping).astype(float)
    return meta


def load_meta(meta_file: str = "metadata.csv", covariate_cols: list[str] | None = None) -> pd.DataFrame:
    meta = pd.read_csv(PROC_DIR / meta_file)
    if covariate_cols is not None:
        missing = [c for c in covariate_cols if c not in meta.columns]
        if missing:
            raise ValueError(f"Columnas de covariables no encontradas en {meta_file}: {missing}")
        meta = _encode_categorical_columns(meta, covariate_cols)
    return meta


def make_dataloaders(batch_size: int = 32, num_workers: int = 0, bmd_col: str = "bmd",
                     img_file: str = "images.npy", meta_file: str = "metadata.csv",
                     covariate_cols: list[str] | None = None):
    images = np.load(PROC_DIR / img_file)
    meta = load_meta(meta_file, covariate_cols)

    loaders = {}
    covariate_stats = None

    for split in ["train", "val", "test"]:
        mask = (meta["split"] == split).values
        ds = LumosDataset(
            images[mask],
            meta[mask],
            train=(split == "train"),
            bmd_col=bmd_col,
            covariate_cols=covariate_cols,
            covariate_stats=covariate_stats,
        )
        if split == "train" and covariate_cols is not None:
            covariate_stats = ds.covariate_stats

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
    parser.add_argument("--data", choices=list(DATASETS), default="full")
    parser.add_argument("--covariates", action="store_true")
    args = parser.parse_args()

    covariate_cols = MEDICAL_COVARIATES if args.covariates else None
    loaders = make_dataloaders(batch_size=8, covariate_cols=covariate_cols, **DATASETS[args.data])

    batch = next(iter(loaders["train"]))
    print("\nOne training batch:")
    if covariate_cols is not None:
        print(f"  covariates: {tuple(batch[3].shape)}  -> {batch[3][0].tolist()}")
    images, labels, bmd = batch[:3]
    print(f"  images: {tuple(images.shape)}  dtype {images.dtype}")
    print(f"  labels: {tuple(labels.shape)}  -> {labels.tolist()}")
    print(f"  bmd   : {tuple(bmd.shape)}  -> {[round(x, 3) for x in bmd.tolist()]}")
