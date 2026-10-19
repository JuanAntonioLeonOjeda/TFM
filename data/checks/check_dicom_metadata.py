"""Imprime todos los metadatos DICOM de un archivo o de una muestra aleatoria.

Uso: python -m data.checks.check_dicom_metadata [--path x.Dcm | --sample 5]
"""

import argparse

import pydicom

from data.df_import import load_image_index


def show_metadata(path):
    print(f"\n{'='*70}")
    print(f"Archivo: {path}")
    print('='*70)
    ds = pydicom.dcmread(path)

    for elem in ds:
        if elem.tag == (0x7FE0, 0x0010):  # PixelData
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