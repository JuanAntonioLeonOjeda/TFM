"""
check_resolutions.py
=====================
Calcula las dimensiones REALES (ancho, alto) de las 1616 radiografías DICOM
del dataset, en lugar de estimarlas a partir de una muestra pequeña.

Uso:
    python check_resolutions.py
"""

from pathlib import Path
import pydicom
import numpy as np
import pandas as pd

from data.df_import import load_image_index  # AJUSTA al nombre real de tu módulo


def main():
    images = load_image_index()
    print(f"Total de imágenes a comprobar: {len(images)}")

    widths, heights = [], []
    failed = []

    for _, row in images.iterrows():
        try:
            ds = pydicom.dcmread(row["image_path"], stop_before_pixels=True)
            widths.append(int(ds.Columns))
            heights.append(int(ds.Rows))
        except Exception as e:
            failed.append((row["image_path"], str(e)))

    if failed:
        print(f"\nAVISO: {len(failed)} imágenes no se pudieron leer:")
        for path, err in failed[:5]:
            print(f"  {path} -> {err}")

    widths = np.array(widths)
    heights = np.array(heights)

    print("\n=== ANCHO (Columns) ===")
    print(f"  min: {widths.min()}  max: {widths.max()}  media: {widths.mean():.0f}  mediana: {np.median(widths):.0f}")

    print("\n=== ALTO (Rows) ===")
    print(f"  min: {heights.min()}  max: {heights.max()}  media: {heights.mean():.0f}  mediana: {np.median(heights):.0f}")

    # guardar el detalle completo por si se necesita
    df = pd.DataFrame({"path": images["image_path"], "width": widths, "height": heights})
    df.to_csv("resolutions_full.csv", index=False)
    print("\nGuardado detalle completo en resolutions_full.csv")


if __name__ == "__main__":
    main()