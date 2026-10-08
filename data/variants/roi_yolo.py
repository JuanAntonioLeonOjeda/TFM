"""Variante ROI YOLO: recorte adaptativo de la vista AP con el detector entrenado (yolo/).

Si la confianza es baja o no hay detección, se mantiene la imagen completa.
Uso: python -m data.variants.roi_yolo [--weights best.pt] [--conf 0.5]
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
    PROJECT_ROOT / "runs" / "detect" / "res" / "yolo_dataset" / "lumbar_detector" / "weights" / "best.pt"
)
IMG_SIZE = 224
DEFAULT_CONF_THRESHOLD = 0.5
MARGIN_FRAC = 0.08  # mismo margen que yolo/auto_label.py

_VIEW_PATTERN = re.compile(r"_(\d+)\.[Dd][Cc][Mm]$")


def extract_view(image_path: str) -> int:
    """'.../lumos_x_001_1.Dcm' -> 1; -1 si no se puede determinar."""
    m = _VIEW_PATTERN.search(image_path)
    return int(m.group(1)) if m else -1


def crop_with_margin(img: np.ndarray, x0: float, y0: float, x1: float, y1: float,
                     margin_frac: float = MARGIN_FRAC) -> np.ndarray:
    h, w = img.shape
    mx, my = (x1 - x0) * margin_frac, (y1 - y0) * margin_frac

    x0 = max(0, int(x0 - mx))
    y0 = max(0, int(y0 - my))
    x1 = min(w, int(x1 + mx))
    y1 = min(h, int(y1 + my))

    crop = img[y0:y1, x0:x1]
    if crop.size == 0:
        return img
    return cv2.resize(crop, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default=str(DEFAULT_WEIGHTS))
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF_THRESHOLD,
                        help="Confianza mínima para recortar; por debajo se mantiene la imagen completa")
    parser.add_argument("--proc-dir", type=str, default=str(PROC_DIR))
    args = parser.parse_args()

    proc_dir = Path(args.proc_dir)

    print(f"Cargando detector: {args.weights}")
    model = YOLO(args.weights)

    # Se recorta sobre la imagen ya procesada a 224x224: es el espacio en el que se entrenó el detector.
    images = np.load(proc_dir / "images.npy")
    meta = pd.read_csv(proc_dir / "metadata.csv")
    assert len(images) == len(meta), "images.npy y metadata.csv no están alineados"

    is_ap = meta["image_path"].apply(extract_view) == 1
    print(f"Total filas: {len(meta)}  |  AP (view=1): {is_ap.sum()}  |  Otras vistas: {(~is_ap).sum()}")

    new_images = images.copy()
    cropped_flags = np.zeros(len(meta), dtype=bool)
    confidences = np.full(len(meta), np.nan)

    ap_indices = meta.index[is_ap].tolist()
    print(f"\nAplicando detector a {len(ap_indices)} imágenes AP...")

    for idx in ap_indices:
        gray = images[idx]
        results = model.predict(np.stack([gray] * 3, axis=-1), verbose=False)[0]
        if len(results.boxes) == 0:
            continue

        best_i = int(results.boxes.conf.argmax())
        conf = float(results.boxes.conf[best_i])
        confidences[idx] = conf
        if conf < args.conf:
            continue

        x0, y0, x1, y1 = results.boxes.xyxy[best_i].tolist()
        new_images[idx] = crop_with_margin(gray, x0, y0, x1, y1)
        cropped_flags[idx] = True

    meta_out = meta.copy()
    meta_out["yolo_cropped"] = cropped_flags
    meta_out["yolo_confidence"] = confidences

    np.save(proc_dir / "images_roi_yolo.npy", new_images)
    meta_out.to_csv(proc_dir / "metadata_roi_yolo.csv", index=False)

    n_cropped, n_ap = cropped_flags.sum(), is_ap.sum()
    print(f"\n--- Resumen ---")
    print(f"Imágenes AP recortadas (confianza >= {args.conf}): {n_cropped} / {n_ap} "
          f"({100 * n_cropped / n_ap:.1f}%)")
    print(f"Imágenes AP sin recortar (imagen completa): {n_ap - n_cropped} / {n_ap}")
    print(f"Vistas no-AP (sin tocar): {(~is_ap).sum()}")
    print(f"\nGuardado:")
    print(f"  {proc_dir / 'images_roi_yolo.npy'}")
    print(f"  {proc_dir / 'metadata_roi_yolo.csv'}")


if __name__ == "__main__":
    main()
