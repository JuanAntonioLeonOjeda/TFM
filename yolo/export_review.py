"""
export_for_review.py
=====================
Prepara los pacientes ACEPTADOS por auto_label_bboxes.py para revisarlos y
corregirlos en makesense.ai (gratis, en el navegador, sin instalar nada:
https://www.makesense.ai/).

Junta las imágenes de train+val en una sola carpeta (más cómodo para
revisar todo de una sentada) y genera classes.txt, que makesense.ai pide
al importar anotaciones YOLO ya existentes.

Uso:
    python export_for_review.py

Salida:
    res/yolo_dataset/review_session/
        images/*.jpg      <- arrastra ESTA carpeta a makesense.ai
        labels/*.txt       <- las anotaciones actuales, para importar
        classes.txt        <- nombres de clase (una por línea)
"""

import shutil
from pathlib import Path

import pandas as pd

from data.df_import import YOLO_DATASET_DIR as DATASET_DIR
REVIEW_DIR = DATASET_DIR / "review_session"


def main():
    review = pd.read_csv(DATASET_DIR / "review.csv")
    accepted = review[review["accepted"] == True]

    images_out = REVIEW_DIR / "images"
    labels_out = REVIEW_DIR / "labels"
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    copied = 0
    for _, row in accepted.iterrows():
        patient = row["patient"]
        split = row["split"]

        src_img = DATASET_DIR / "images" / split / f"{patient}.jpg"
        src_label = DATASET_DIR / "labels" / split / f"{patient}.txt"

        if not src_img.exists() or not src_label.exists():
            print(f"[aviso] Faltan archivos para {patient}, se salta")
            continue

        shutil.copy(src_img, images_out / f"{patient}.jpg")
        shutil.copy(src_label, labels_out / f"{patient}.txt")
        copied += 1

    (REVIEW_DIR / "classes.txt").write_text("columna_lumbar\n")

    print(f"Copiados {copied} pacientes a {REVIEW_DIR}")
    print(f"\nSiguiente paso:")
    print(f"  1. Ve a https://www.makesense.ai/")
    print(f"  2. 'Get Started' -> arrastra la carpeta {images_out}")
    print(f"  3. Elige 'Object Detection'")
    print(f"  4. 'Actions' (menú izquierdo) -> 'Load labels' -> selecciona los .txt de {labels_out} "
          f"y el archivo {REVIEW_DIR / 'classes.txt'}")
    print(f"  5. Revisa cada imagen: si la caja está bien, pasa a la siguiente; si está mal, "
          f"bórrala y redibuja, o borra la imagen entera si el caso es irrecuperable")
    print(f"  6. Al terminar: 'Actions' -> 'Export Labels' -> formato YOLO -> descarga el .zip")


if __name__ == "__main__":
    main()