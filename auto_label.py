"""
auto_label.py
=====================
Genera cajas delimitadoras (bounding boxes) automáticas para la columna
lumbar en radiografías AP, usando el mapa de calor Grad-CAM del modelo YA
ENTRENADO como pseudo-etiqueta -- sin anotación manual.

Proceso por cada paciente (vista AP):
  1. Preprocesa la imagen igual que en infer.py (224x224, CLAHE).
  2. Calcula el mapa de calor Grad-CAM crudo (HiResCAM, clase predicha).
  3. Umbraliza el mapa y toma la componente conexa más grande como caja.
  4. Aplica un margen y comprueba que el tamaño, posición horizontal y
     proporción alto/ancho sean razonables para una columna vertebral (si
     no, el paciente se descarta y se anota en el CSV de revisión).

Salida (formato esperado por Ultralytics YOLO):
    res/yolo_dataset/
        images/train/*.jpg, images/val/*.jpg
        labels/train/*.txt, labels/val/*.txt   (formato YOLO: clase x y w h)
        dataset.yaml
        review.csv   <- inspecciona esto para detectar cajas raras antes de entrenar

Uso:
    python auto_label_bboxes.py
    python auto_label_bboxes.py --threshold 0.6 --margin 0.1
    python auto_label_bboxes.py --patients 1-50   # prueba rápida con pocos pacientes
"""

import argparse
import random
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from infer import (
    DATASET_ROOT,
    build_covariate_lookup,
    get_device,
    get_patient_covariates,
    load_model,
    patient_id_to_int,
    preprocess,
    resolve_dicom_path,
)
from evaluation.gradcam import compute_grayscale_cam

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "res" / "yolo_dataset"
SEED = 42

DEFAULT_THRESHOLD = 0.5       # umbral del mapa de calor para considerarlo "columna"
DEFAULT_MARGIN_FRAC = 0.08    # margen añadido alrededor de la caja detectada
MIN_AREA_FRAC = 0.03          # rechaza cajas absurdamente pequeñas (probable fallo)
MAX_AREA_FRAC = 0.75          # rechaza cajas absurdamente grandes (probable fallo)
X_CENTER_MIN = 0.30           # el centro de la caja debe caer en la franja central del ancho
X_CENTER_MAX = 0.70           # (coincide con la franja 25-75% que ya validaste en tu ROI fijo)
MIN_ASPECT_RATIO = 1.0        # alto/ancho -- la columna es una estructura alargada verticalmente
VAL_SPLIT = 0.15              # fracción de imágenes aceptadas para validación del detector


def find_all_patients(dataset_root: Path = DATASET_ROOT) -> list[str]:
    """Escanea dataset_root y devuelve todos los IDs de paciente 'lumos_x_XXX' encontrados."""
    ids = set()
    for p in dataset_root.rglob("lumos_x_*"):
        if p.is_dir() and re.match(r"^lumos_x_\d+$", p.name):
            ids.add(p.name)

    def sort_key(pid):
        m = re.search(r"(\d+)$", pid)
        return int(m.group(1)) if m else pid

    return sorted(ids, key=sort_key)


def parse_patient_list(spec: str) -> list[str]:
    """Igual que en batch_heatmaps.py: '1-50', '1,5,7', etc."""
    patients = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk and chunk.replace("-", "").isdigit():
            start, end = chunk.split("-")
            patients.extend(str(i) for i in range(int(start), int(end) + 1))
        else:
            patients.append(chunk)
    return patients


def extract_bbox_from_cam(grayscale_cam: np.ndarray, threshold: float, margin_frac: float):
    """
    Umbraliza el mapa de calor y extrae la caja de la componente conexa más
    grande, con margen añadido.

    Devuelve (x0, y0, x1, y1, area_frac) en píxeles, o None si no se
    encontró ninguna componente.
    """
    h, w = grayscale_cam.shape
    mask = (grayscale_cam >= threshold).astype(np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return None  # todo fondo, ninguna zona por encima del umbral

    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_idx = 1 + int(np.argmax(areas))
    x, y, bw, bh, area = stats[largest_idx]

    mx, my = int(bw * margin_frac), int(bh * margin_frac)
    x0, y0 = max(0, x - mx), max(0, y - my)
    x1, y1 = min(w, x + bw + mx), min(h, y + bh + my)

    area_frac = ((x1 - x0) * (y1 - y0)) / (h * w)
    return x0, y0, x1, y1, area_frac


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patients", type=str, default=None,
                         help="Subconjunto de pacientes a procesar, p.ej. '1-50'. "
                              "Por defecto procesa TODOS los encontrados en res/dataset/")
    parser.add_argument("--model", type=str, choices=["reference", "covars"], default="covars",
                         help="Qué modelo entrenado usar para generar el Grad-CAM. "
                              "'covars' (default) es el único checkpoint compatible con la "
                              "arquitectura actual (capa compartida); 'reference' requiere haber "
                              "entrenado antes best_phased_full.pt CON la capa compartida.")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                         help=f"Umbral del mapa de calor (default: {DEFAULT_THRESHOLD})")
    parser.add_argument("--margin", type=float, default=DEFAULT_MARGIN_FRAC,
                         help=f"Margen añadido a la caja, como fracción de su tamaño (default: {DEFAULT_MARGIN_FRAC})")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR))
    args = parser.parse_args()

    random.seed(SEED)
    output_dir = Path(args.output_dir)

    patients = parse_patient_list(args.patients) if args.patients else find_all_patients()
    print(f"Pacientes a procesar: {len(patients)}")

    device = get_device()
    num_covariates = 3 if args.model == "covars" else 0
    model, model_path = load_model(device, num_covariates=num_covariates)
    print(f"Modelo usado para Grad-CAM: {model_path}\n")

    covariate_meta, covariate_stats = (build_covariate_lookup() if args.model == "covars" else (None, None))

    for split in ["train", "val"]:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    review_rows = []
    accepted = 0
    rejected = 0

    for patient in patients:
        try:
            dicom_path = resolve_dicom_path(patient, view=1)  # solo AP (view=1)
        except (FileNotFoundError, ValueError) as e:
            review_rows.append({"patient": patient, "accepted": False, "reason": f"sin vista AP: {e}"})
            rejected += 1
            continue

        image_tensor, base_img = preprocess(str(dicom_path))

        covariates = None
        if args.model == "covars":
            try:
                patient_id_int = patient_id_to_int(patient)
                covariates = get_patient_covariates(patient_id_int, covariate_meta, covariate_stats)
            except ValueError as e:
                review_rows.append({"patient": patient, "accepted": False, "reason": f"sin covariables: {e}"})
                rejected += 1
                continue

        grayscale_cam, _ = compute_grayscale_cam(
            model, image_tensor, target="classification", use_hires=True, covariates=covariates
        )

        bbox = extract_bbox_from_cam(grayscale_cam, args.threshold, args.margin)

        if bbox is None:
            review_rows.append({"patient": patient, "accepted": False, "reason": "sin componente por encima del umbral"})
            rejected += 1
            continue

        x0, y0, x1, y1, area_frac = bbox

        if not (MIN_AREA_FRAC <= area_frac <= MAX_AREA_FRAC):
            review_rows.append({
                "patient": patient, "accepted": False,
                "reason": f"área fuera de rango ({area_frac:.3f}, esperado {MIN_AREA_FRAC}-{MAX_AREA_FRAC})",
                "x0": x0, "y0": y0, "x1": x1, "y1": y1, "area_frac": round(area_frac, 4),
            })
            rejected += 1
            continue

        h, w = grayscale_cam.shape
        x_center_frac = (x0 + x1) / 2 / w
        bw, bh = (x1 - x0), (y1 - y0)
        aspect_ratio = bh / bw if bw > 0 else 0

        if not (X_CENTER_MIN <= x_center_frac <= X_CENTER_MAX):
            review_rows.append({
                "patient": patient, "accepted": False,
                "reason": f"centro fuera de la franja central ({x_center_frac:.2f}, esperado {X_CENTER_MIN}-{X_CENTER_MAX})",
                "x0": x0, "y0": y0, "x1": x1, "y1": y1, "area_frac": round(area_frac, 4),
            })
            rejected += 1
            continue

        if aspect_ratio < MIN_ASPECT_RATIO:
            review_rows.append({
                "patient": patient, "accepted": False,
                "reason": f"caja demasiado ancha para ser columna (alto/ancho={aspect_ratio:.2f}, mínimo {MIN_ASPECT_RATIO})",
                "x0": x0, "y0": y0, "x1": x1, "y1": y1, "area_frac": round(area_frac, 4),
            })
            rejected += 1
            continue

        # --- Guardar imagen + etiqueta en formato YOLO ---
        xc, yc = (x0 + x1) / 2 / w, (y0 + y1) / 2 / h
        bw_norm, bh_norm = (x1 - x0) / w, (y1 - y0) / h

        split = "val" if random.random() < VAL_SPLIT else "train"

        img_path = output_dir / "images" / split / f"{patient}.jpg"
        label_path = output_dir / "labels" / split / f"{patient}.txt"

        cv2.imwrite(str(img_path), base_img)  # base_img ya es uint8 en escala de grises
        with open(label_path, "w") as f:
            f.write(f"0 {xc:.6f} {yc:.6f} {bw_norm:.6f} {bh_norm:.6f}\n")

        review_rows.append({
            "patient": patient, "accepted": True, "reason": "",
            "x0": x0, "y0": y0, "x1": x1, "y1": y1, "area_frac": round(area_frac, 4),
            "x_center_frac": round(x_center_frac, 3), "aspect_ratio": round(aspect_ratio, 2),
            "split": split,
        })
        accepted += 1

    review_df = pd.DataFrame(review_rows)
    review_df.to_csv(output_dir / "review.csv", index=False)

    dataset_yaml = output_dir / "dataset.yaml"
    dataset_yaml.write_text(
        f"path: {output_dir.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"names:\n"
        f"  0: columna_lumbar\n"
    )

    print("--- Resumen ---")
    print(f"Aceptados: {accepted}")
    print(f"Rechazados: {rejected}")
    print(f"Revisa {output_dir / 'review.csv'} para ver los casos rechazados y por qué.")
    print(f"dataset.yaml generado en: {dataset_yaml}")
    print(f"\nSiguiente paso: entrenar YOLO con este dataset.")


if __name__ == "__main__":
    main()