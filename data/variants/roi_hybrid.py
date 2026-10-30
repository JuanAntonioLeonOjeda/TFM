import pandas as pd

from data.df_import import PROC_DIR
from data.df_preprocess import (
    print_split_summary, process_all, process_image, save_dataset, split_by_patient,
)
from data.variants.roi import crop_roi, load_clinical_per_patient


def main():
    view_labels_path = PROC_DIR / "view_labels.csv"
    if not view_labels_path.exists():
        raise FileNotFoundError(f"No existe {view_labels_path}. Ejecuta antes: python -m data.views")

    views = pd.read_csv(view_labels_path)[["patient_id", "image_path", "view"]]
    df = views.merge(load_clinical_per_patient(), on="patient_id", how="inner")
    print(f"Imágenes totales (ambas vistas): {len(df)}")
    print(df["view"].value_counts().to_string())

    df = split_by_patient(df)
    print_split_summary(df)

    images = process_all(
        df, lambda row: process_image(row.image_path, crop=crop_roi if row.view == "AP" else None)
    )
    save_dataset(df, images, suffix="_roi_hybrid")


if __name__ == "__main__":
    main()
