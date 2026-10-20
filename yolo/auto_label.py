import argparse
import random
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from data.dataloaders import MEDICAL_COVARIATES
from data.df_import import YOLO_DATASET_DIR
from evaluation.gradcam import compute_grayscale_cam
from infer import (
    DATASET_ROOT,
    build_covariate_lookup,
    get_device,
    get_patient_covariates,
    load_model,
    parse_patient_list,
    patient_id_to_int,
    preprocess,
    resolve_dicom_path,
)

SEED = 42

DEFAULT_THRESHOLD = 0.5
DEFAULT_MARGIN_FRAC = 0.08
MIN_AREA_FRAC = 0.03
MAX_AREA_FRAC = 0.75
X_CENTER_MIN = 0.30           # franja central, coherente con el ROI fijo del 25 al 75 % del ancho
X_CENTER_MAX = 0.70
MIN_ASPECT_RATIO = 1.0        # relación entre alto y ancho; la columna es vertical
VAL_SPLIT = 0.15


def find_all_patients(dataset_root: Path = DATASET_ROOT) -> list[str]:
    ids = {p.name for p in dataset_root.rglob("lumos_x_*") if p.is_dir() and re.match(r"^lumos_x_\d+$", p.name)}
    return sorted(ids, key=lambda pid: int(pid.rsplit("_", 1)[1]))


# Caja de la mayor componente conexa con margen y fracción de área que ocupa,
# o None si no hay ninguna por encima del umbral
def extract_bbox_from_cam(grayscale_cam: np.ndarray, threshold: float, margin_frac: float):
    h, w = grayscale_cam.shape
    mask = (grayscale_cam >= threshold).astype(np.uint8)

    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return None

    largest_idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))  # la etiqueta 0 es el fondo
    x, y, bw, bh, _ = stats[largest_idx]

    mx, my = int(bw * margin_frac), int(bh * margin_frac)
    x0, y0 = max(0, x - mx), max(0, y - my)
    x1, y1 = min(w, x + bw + mx), min(h, y + bh + my)

    area_frac = ((x1 - x0) * (y1 - y0)) / (h * w)
    return x0, y0, x1, y1, area_frac


# Devuelve el motivo de rechazo, o una cadena vacía si la caja es plausible
def check_bbox(x0, y0, x1, y1, area_frac, w) -> str:
    if not (MIN_AREA_FRAC <= area_frac <= MAX_AREA_FRAC):
        return f"área fuera de rango ({area_frac:.3f}, esperado {MIN_AREA_FRAC}-{MAX_AREA_FRAC})"
    x_center_frac = (x0 + x1) / 2 / w
    if not (X_CENTER_MIN <= x_center_frac <= X_CENTER_MAX):
        return f"centro fuera de la franja central ({x_center_frac:.2f}, esperado {X_CENTER_MIN}-{X_CENTER_MAX})"
    bw, bh = x1 - x0, y1 - y0
    aspect_ratio = bh / bw if bw > 0 else 0
    if aspect_ratio < MIN_ASPECT_RATIO:
        return f"caja demasiado ancha para ser columna (alto/ancho={aspect_ratio:.2f}, mínimo {MIN_ASPECT_RATIO})"
    return ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patients", type=str, default=None, help="p.ej. '1-50'. Por defecto, todos")
    parser.add_argument("--model", type=str, choices=["reference", "covars"], default="covars",
                        help="'reference' requiere reentrenar best_phased_full.pt con la arquitectura actual")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--margin", type=float, default=DEFAULT_MARGIN_FRAC)
    parser.add_argument("--output-dir", type=str, default=str(YOLO_DATASET_DIR))
    args = parser.parse_args()

    random.seed(SEED)
    output_dir = Path(args.output_dir)
    use_covariates = args.model == "covars"

    patients = parse_patient_list(args.patients) if args.patients else find_all_patients()
    print(f"Pacientes a procesar: {len(patients)}")

    device = get_device()
    model, model_path = load_model(device, num_covariates=len(MEDICAL_COVARIATES) if use_covariates else 0)
    print(f"Modelo usado para Grad-CAM: {model_path}\n")

    covariate_meta, covariate_stats = build_covariate_lookup() if use_covariates else (None, None)

    for split in ["train", "val"]:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    review_rows = []

    def reject(patient, reason, bbox=None):
        row = {"patient": patient, "accepted": False, "reason": reason}
        if bbox is not None:
            x0, y0, x1, y1, area_frac = bbox
            row.update(x0=x0, y0=y0, x1=x1, y1=y1, area_frac=round(area_frac, 4))
        review_rows.append(row)

    for patient in patients:
        try:
            dicom_path = resolve_dicom_path(patient, view=1)  # se asume que la primera vista es la AP
        except (FileNotFoundError, ValueError) as e:
            reject(patient, f"sin vista AP: {e}")
            continue

        image_tensor, base_img = preprocess(str(dicom_path))

        covariates = None
        if use_covariates:
            try:
                covariates = get_patient_covariates(patient_id_to_int(patient), covariate_meta, covariate_stats)
            except ValueError as e:
                reject(patient, f"sin covariables: {e}")
                continue

        grayscale_cam, _ = compute_grayscale_cam(
            model, image_tensor, target="classification", use_hires=True, covariates=covariates
        )

        bbox = extract_bbox_from_cam(grayscale_cam, args.threshold, args.margin)
        if bbox is None:
            reject(patient, "sin componente por encima del umbral")
            continue

        h, w = grayscale_cam.shape
        reason = check_bbox(*bbox, w)
        if reason:
            reject(patient, reason, bbox)
            continue

        x0, y0, x1, y1, area_frac = bbox
        split = "val" if random.random() < VAL_SPLIT else "train"

        cv2.imwrite(str(output_dir / "images" / split / f"{patient}.jpg"), base_img)
        xc, yc = (x0 + x1) / 2 / w, (y0 + y1) / 2 / h
        bw_norm, bh_norm = (x1 - x0) / w, (y1 - y0) / h
        (output_dir / "labels" / split / f"{patient}.txt").write_text(
            f"0 {xc:.6f} {yc:.6f} {bw_norm:.6f} {bh_norm:.6f}\n"
        )

        review_rows.append({
            "patient": patient, "accepted": True, "reason": "",
            "x0": x0, "y0": y0, "x1": x1, "y1": y1, "area_frac": round(area_frac, 4),
            "x_center_frac": round(xc, 3), "aspect_ratio": round((y1 - y0) / (x1 - x0), 2),
            "split": split,
        })

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

    accepted = int(review_df["accepted"].sum())
    print("--- Resumen ---")
    print(f"Aceptados: {accepted}")
    print(f"Rechazados: {len(review_df) - accepted}")
    print(f"Revisa {output_dir / 'review.csv'} para ver los casos rechazados y por qué.")
    print(f"dataset.yaml generado en: {dataset_yaml}")


if __name__ == "__main__":
    main()
