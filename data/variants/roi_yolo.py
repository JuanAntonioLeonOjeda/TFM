"""
apply_yolo_crop.py
===================
Aplica el detector YOLO ya entrenado (auto_label_bboxes.py + makesense.ai +
entrenamiento YOLO) a TODO el dataset, y genera la variante de recorte
adaptativo: images_roi_yolo.npy + metadata_roi_yolo.csv, con el mismo
formato que usa el resto del pipeline (data/dataloaders.py).

Estrategia (igual que tu experimento roi_hybrid anterior):
    - Solo se recorta la vista AP (view=1, "_1.Dcm"). La vista lateral se
      mantiene completa, sin tocar.
    - Si la detección de YOLO tiene confianza >= CONF_THRESHOLD, se recorta
      la imagen a esa caja (con margen) y se reescala de vuelta a 224x224.
    - Si la confianza es baja o no hay detección, se mantiene la imagen AP
      completa SIN recortar (fallback seguro, en vez de forzar un recorte
      poco fiable).

El recorte se aplica sobre la imagen YA procesada a 224x224 (post-CLAHE),
el mismo espacio de coordenadas en el que se entrenó el detector -- no
sobre el DICOM en resolución original.

Uso:
    python apply_yolo_crop.py
    python apply_yolo_crop.py --weights ruta/a/best.pt --conf 0.5
"""

import argparse
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

from data.df_import import PROC_DIR, PROJECT_ROOT

DEFAULT_WEIGHTS = (
    PROJECT_ROOT
    / "runs" / "detect" / "res" / "yolo_dataset" / "lumbar_detector" / "weights" / "best.pt"
)
IMG_SIZE = 224
DEFAULT_CONF_THRESHOLD = 0.5
MARGIN_FRAC = 0.08  # mismo margen que auto_label_bboxes.py, por consistencia

_VIEW_PATTERN = re.compile(r"_(\d+)\.[Dd][Cc][Mm]$")


def extract_view(image_path: str) -> int:
    """'.../lumos_x_001_1.Dcm' -> 1. Devuelve -1 si no se puede determinar."""
    m = _VIEW_PATTERN.search(image_path)
    return int(m.group(1)) if m else -1


def crop_with_margin(img: np.ndarray, x0: float, y0: float, x1: float, y1: float,
                      margin_frac: float = MARGIN_FRAC) -> np.ndarray:
    """Recorta img a la caja [x0,y0,x1,y1] (en píxeles) con margen, y reescala a IMG_SIZE."""
    h, w = img.shape
    bw, bh = x1 - x0, y1 - y0
    mx, my = bw * margin_frac, bh * margin_frac

    x0 = max(0, int(x0 - mx))
    y0 = max(0, int(y0 - my))
    x1 = min(w, int(x1 + mx))
    y1 = min(h, int(y1 + my))

    crop = img[y0:y1, x0:x1]
    if crop.size == 0:
        return img  # salvaguarda: si algo raro pasa, no rompemos, devolvemos la original

    return cv2.resize(crop, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default=str(DEFAULT_WEIGHTS),
                         help="Ruta a los pesos entrenados de YOLO (best.pt)")
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF_THRESHOLD,
                         help=f"Umbral de confianza mínimo para recortar (default: {DEFAULT_CONF_THRESHOLD}). "
                              f"Por debajo de esto, se mantiene la imagen completa.")
    parser.add_argument("--proc-dir", type=str, default=str(PROC_DIR))
    args = parser.parse_args()

    proc_dir = Path(args.proc_dir)

    print(f"Cargando detector: {args.weights}")
    model = YOLO(args.weights)

    print("Cargando dataset completo (images.npy + metadata.csv)...")
    images = np.load(proc_dir / "images.npy")
    meta = pd.read_csv(proc_dir / "metadata.csv")
    assert len(images) == len(meta), "images.npy y metadata.csv no están alineados"

    views = meta["image_path"].apply(extract_view)
    is_ap = (views == 1)
    print(f"Total filas: {len(meta)}  |  AP (view=1): {is_ap.sum()}  |  Otras vistas: {(~is_ap).sum()}")

    new_images = images.copy()
    cropped_flags = np.zeros(len(meta), dtype=bool)
    confidences = np.full(len(meta), np.nan)

    ap_indices = meta.index[is_ap].tolist()
    print(f"\nAplicando detector a {len(ap_indices)} imágenes AP...")

    for idx in ap_indices:
        gray = images[idx]
        rgb = np.stack([gray, gray, gray], axis=-1)  # YOLO espera 3 canales

        results = model.predict(rgb, verbose=False)[0]

        if len(results.boxes) == 0:
            continue  # sin detección -> se mantiene la imagen completa

        # nos quedamos con la caja de mayor confianza
        best_i = int(results.boxes.conf.argmax())
        conf = float(results.boxes.conf[best_i])
        confidences[idx] = conf

        if conf < args.conf:
            continue  # confianza insuficiente -> se mantiene la imagen completa

        x0, y0, x1, y1 = results.boxes.xyxy[best_i].tolist()
        new_images[idx] = crop_with_margin(gray, x0, y0, x1, y1)
        cropped_flags[idx] = True

    meta_out = meta.copy()
    meta_out["yolo_cropped"] = cropped_flags
    meta_out["yolo_confidence"] = confidences

    np.save(proc_dir / "images_roi_yolo.npy", new_images)
    meta_out.to_csv(proc_dir / "metadata_roi_yolo.csv", index=False)

    n_cropped = cropped_flags.sum()
    n_ap = is_ap.sum()
    print(f"\n--- Resumen ---")
    print(f"Imágenes AP recortadas (confianza >= {args.conf}): {n_cropped} / {n_ap} "
          f"({100 * n_cropped / n_ap:.1f}%)")
    print(f"Imágenes AP sin recortar (fallback a imagen completa): {n_ap - n_cropped} / {n_ap}")
    print(f"Vistas no-AP (sin tocar): {(~is_ap).sum()}")
    print(f"\nGuardado:")
    print(f"  {proc_dir / 'images_roi_yolo.npy'}")
    print(f"  {proc_dir / 'metadata_roi_yolo.csv'}")


if __name__ == "__main__":
    main()