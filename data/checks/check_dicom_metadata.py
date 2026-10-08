"""
check_dicom_metadata.py
========================
Muestra TODOS los metadatos disponibles en uno o varios archivos DICOM,
para explorar qué información trae la cabecera.

Uso:
    # Ver los metadatos completos de una imagen concreta:
    python check_dicom_metadata.py --path "res/dataset/.../lumos_x_69_1.Dcm"

    # Ver los metadatos de una muestra aleatoria de N imágenes:
    python check_dicom_metadata.py --sample 5

    # Ver los metadatos del caso con altura anómala (5596), si lo tienes
    # identificado en resolutions_full.csv:
    python check_dicom_metadata.py --path "<ruta_del_csv>"
"""

import argparse
from pathlib import Path

import pydicom
import pandas as pd

from data.df_import import load_image_index  # AJUSTA al nombre real de tu módulo


def show_metadata(path):
    print(f"\n{'='*70}")
    print(f"Archivo: {path}")
    print('='*70)
    ds = pydicom.dcmread(path)

    # imprime TODOS los campos disponibles en la cabecera
    for elem in ds:
        if elem.tag == (0x7FE0, 0x0010):   # PixelData, no lo imprimimos (es la imagen en sí)
            print(f"  {elem.name:35s}: [datos de píxeles, {len(elem.value)} bytes]")
            continue
        try:
            print(f"  {elem.name:35s}: {elem.value}")
        except Exception:
            print(f"  {elem.name:35s}: [no se pudo leer]")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, help="Ruta a un DICOM concreto")
    parser.add_argument("--sample", type=int, help="Número de imágenes aleatorias a mostrar")
    args = parser.parse_args()

    if args.path:
        show_metadata(args.path)
        return

    images = load_image_index()
    n = args.sample or 3
    sample = images.sample(n, random_state=42)
    for _, row in sample.iterrows():
        show_metadata(row["image_path"])


if __name__ == "__main__":
    main()