"""
show_patient.py
================
Muestra todas las radiografías de un paciente concreto, junto con su
Series Description (vista AP/lateral) y sus dimensiones.

Uso:
    python show_patient.py --patient 1
    python show_patient.py --patient 1 --raw   (sin normalizar/CLAHE, tal cual el DICOM)
"""

import argparse
from pathlib import Path

import numpy as np
import pydicom
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data.df_import import load_image_index, read_image  # AJUSTA al nombre real de tu módulo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patient", type=int, required=True, help="ID del paciente")
    parser.add_argument("--raw", action="store_true",
                         help="Mostrar la imagen sin normalizar (valores DICOM originales)")
    args = parser.parse_args()

    images = load_image_index()
    patient_images = images[images["patient_id"] == args.patient]

    if patient_images.empty:
        print(f"No se han encontrado imágenes para el paciente {args.patient}.")
        return

    n = len(patient_images)
    print(f"Paciente {args.patient}: {n} imagen(es) encontrada(s)\n")

    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    axes = np.atleast_1d(axes)

    for ax, (_, row) in zip(axes, patient_images.iterrows()):
        path = row["image_path"]
        ds = pydicom.dcmread(path)
        series_desc = str(getattr(ds, "SeriesDescription", "SIN_DATO"))
        rows, cols = ds.Rows, ds.Columns

        if args.raw:
            img = ds.pixel_array
        else:
            img = read_image(path)

        print(f"  {Path(path).name}")
        print(f"    Series Description: {series_desc}")
        print(f"    Dimensiones (alto x ancho): {rows} x {cols}")
        print(f"    Rango de valores: [{img.min():.2f}, {img.max():.2f}]")
        print()

        ax.imshow(img, cmap="gray")
        ax.set_title(f"Paciente {args.patient} - {series_desc}\n{rows}x{cols}", fontsize=11)
        ax.axis("off")

    plt.tight_layout()
    out_path = f"patient_{args.patient}_radiographs.png"
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"Guardado: {out_path}")

    import subprocess
    subprocess.run(["open", out_path])


if __name__ == "__main__":
    main()