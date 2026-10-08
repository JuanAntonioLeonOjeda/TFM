"""Inferencia sobre una radiografía DICOM con el modelo entrenado (phased, full).

Uso: python infer.py --patient 001 [--view 2] [--covariates] [--heatmap]
     python infer.py --dicom ruta/a/imagen.dcm [--heatmap]
Cada ejecución se añade a outputs/inference_log.csv; con --heatmap guarda PNG + JSON en heatmaps/.
"""

import argparse
import csv
import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import cv2
import torch

from data.dataloaders import MEDICAL_COVARIATES, load_meta
from data.df_import import CLASS_NAMES, DATA_DIR as DATASET_ROOT, PROC_DIR, PROJECT_ROOT
from data.df_preprocess import process_image
from data.lumos_dataset import build_transforms, compute_covariate_stats, gray_to_rgb, normalize_covariates
from evaluation.gradcam import generate_heatmap
from models.lumosnet import LumosNet, get_device
from training.train import OUT_DIR, run_tag

RESULTS_CSV = OUT_DIR / "inference_log.csv"
HEATMAPS_DIR = PROJECT_ROOT / "heatmaps"


def normalize_patient_id(patient: str) -> str:
    """'001' o '1' -> 'lumos_x_001'."""
    return f"lumos_x_{int(patient):03d}" if patient.isdigit() else patient


def patient_id_to_int(patient: str) -> int:
    """'001', '1' o 'lumos_x_001' -> 1."""
    if patient.isdigit():
        return int(patient)
    m = re.match(r"lumos_x_0*(\d+)$", patient)
    if m:
        return int(m.group(1))
    raise ValueError(f"No se pudo extraer un ID numérico de paciente de '{patient}'")


def parse_patient_list(spec: str) -> list[str]:
    """'1-5,8,lumos_x_010' -> ['1', '2', '3', '4', '5', '8', 'lumos_x_010']."""
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


def resolve_dicom_path(patient: str, view: int = 1, dataset_root: Path = DATASET_ROOT) -> Path:
    """Ruta al DICOM número `view` del paciente (lumos_x_001_1.Dcm, _2.Dcm...)."""
    patient_id = normalize_patient_id(patient)

    # La carpeta contenedora varía (p.ej. lumos_x_001_280_dcm/), por eso rglob
    candidate_dirs = [p for p in dataset_root.rglob(patient_id) if p.is_dir()]
    if not candidate_dirs:
        raise FileNotFoundError(
            f"No se encontró ninguna carpeta '{patient_id}' dentro de {dataset_root}. "
            f"Revisa el ID de paciente o usa --dicom con la ruta completa."
        )

    patient_dir = candidate_dirs[0]
    files = sorted(patient_dir.glob("*.[Dd][Cc][Mm]"))
    if not files:
        raise FileNotFoundError(f"La carpeta {patient_dir} no contiene archivos .dcm/.Dcm")

    if not (1 <= view <= len(files)):
        raise ValueError(
            f"--view {view} fuera de rango: {patient_dir.name} tiene {len(files)} imagen(es) (usa 1..{len(files)})"
        )
    return files[view - 1]


def build_covariate_lookup():
    """Misma codificación y z-score (stats de train) que en entrenamiento. Devuelve (meta, stats)."""
    meta = load_meta("metadata.csv", MEDICAL_COVARIATES)
    stats = compute_covariate_stats(meta[meta["split"] == "train"], MEDICAL_COVARIATES)
    return meta, stats


def get_patient_covariates(patient_id_int: int, meta, stats: dict) -> torch.Tensor:
    """Vector de covariables normalizado (1, num_covariates)."""
    rows = meta[meta["patient_id"] == patient_id_int]
    if rows.empty:
        raise ValueError(f"Paciente {patient_id_int} no encontrado en {PROC_DIR / 'metadata.csv'}.")
    values = normalize_covariates(rows.iloc[0], MEDICAL_COVARIATES, stats)
    return torch.tensor(values, dtype=torch.float32).unsqueeze(0)


def load_model(device, num_covariates: int = 0):
    model_path = OUT_DIR / f"best_{run_tag('phased', 'full', use_covariates=num_covariates > 0)}.pt"
    if not model_path.exists():
        raise FileNotFoundError(
            f"No se encontró {model_path}. Entrena primero con training.train --strategy phased --data full"
            + (" --covariates" if num_covariates > 0 else "")
        )

    model = LumosNet(pretrained=False, num_covariates=num_covariates)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model, model_path


def preprocess(dicom_path: str):
    """Mismo preprocesado que en entrenamiento. Devuelve (tensor (1,3,224,224), imagen uint8 tras CLAHE)."""
    img = process_image(dicom_path)
    tensor = build_transforms(train=False)(gray_to_rgb(img)).unsqueeze(0)
    return tensor, img


def predict(model, image_tensor, device, covariates=None):
    image_tensor = image_tensor.to(device)
    if covariates is not None:
        covariates = covariates.to(device)
    with torch.no_grad():
        logits, bmd_pred = model(image_tensor, covariates)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    return probs, bmd_pred.item()


def heatmap_path(name: str, view: int | None = None, covariates: bool = False) -> str:
    view_part = f"_imagen{view}" if view is not None else ""
    suffix = "_covars" if covariates else ""
    HEATMAPS_DIR.mkdir(parents=True, exist_ok=True)
    return str(HEATMAPS_DIR / f"heatmap_{name}{view_part}{suffix}.png")


def save_heatmap(model, image_tensor, base_img_uint8_gray, target, class_idx, out_path,
                 covariates=None):
    """Guarda el overlay y devuelve la clase explicada (None si target='bmd')."""
    base_rgb_float = gray_to_rgb(base_img_uint8_gray).astype(np.float32) / 255.0
    overlay, used_class_idx = generate_heatmap(
        model, image_tensor, base_rgb_float,
        target=target, class_idx=class_idx, covariates=covariates,
    )
    cv2.imwrite(out_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    explained = CLASS_NAMES[used_class_idx] if used_class_idx is not None else "BMD"
    print(f"Heatmap guardado en: {out_path}  (explica: {explained})")
    return used_class_idx


def build_result(dicom_path, patient: str, view, used_covariates: bool, probs, bmd) -> dict:
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "dicom_path": str(dicom_path),
        "patient": patient,
        "view": view,
        "used_covariates": used_covariates,
        "prob_sano": round(float(probs[0]), 4),
        "prob_osteopenia": round(float(probs[1]), 4),
        "prob_osteoporosis": round(float(probs[2]), 4),
        "predicted_class": CLASS_NAMES[int(np.argmax(probs))],
        "bmd_estimated": round(float(bmd), 4),
        "heatmap_generated": False,
        "heatmap_target": "",
        "heatmap_explained_class": "",
        "heatmap_out": "",
    }


def save_result_json(result: dict, out_path: str):
    json_path = Path(out_path).with_suffix(".json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"Resultados guardados en: {json_path}")


def append_result_csv(result: dict, csv_path: Path = RESULTS_CSV):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(result.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dicom", type=str, default=None,
                        help="Ruta a un DICOM (prioridad sobre --patient). Incompatible con --covariates.")
    parser.add_argument("--patient", type=str, default=None, help="'001' o 'lumos_x_001'")
    parser.add_argument("--view", type=int, default=1, help="1=primera imagen del paciente, 2=segunda...")
    parser.add_argument("--covariates", action="store_true",
                        help=f"Modelo con {MEDICAL_COVARIATES}. Requiere --patient.")
    parser.add_argument("--heatmap", action="store_true")
    parser.add_argument("--heatmap-target", type=str, default="classification", choices=["classification", "bmd"])
    parser.add_argument("--heatmap-class", type=int, default=None,
                        help="0=Sano, 1=Osteopenia, 2=Osteoporosis. Por defecto, la clase predicha.")
    parser.add_argument("--heatmap-out", type=str, default=None)
    parser.add_argument("--no-log", action="store_true", help="No guardar .json ni inference_log.csv")
    args = parser.parse_args()

    if args.covariates and not args.patient:
        parser.error("--covariates requiere --patient (para leer edad/sexo/IMC de metadata.csv)")

    if args.dicom:
        dicom_path = Path(args.dicom)
    elif args.patient:
        dicom_path = resolve_dicom_path(args.patient, args.view)
    else:
        parser.error("Debes indicar --dicom <ruta> o --patient <id>")

    device = get_device()
    num_covariates = len(MEDICAL_COVARIATES) if args.covariates else 0
    model, model_path = load_model(device, num_covariates=num_covariates)
    print(f"Cargando modelo desde: {model_path}")
    print(f"Device: {device}\n")

    covariates = None
    if args.covariates:
        patient_id_int = patient_id_to_int(args.patient)
        meta, stats = build_covariate_lookup()
        covariates = get_patient_covariates(patient_id_int, meta, stats)
        print(f"Covariables del paciente {patient_id_int}: "
              f"{dict(zip(MEDICAL_COVARIATES, [round(v, 3) for v in covariates[0].tolist()]))}\n")

    image_tensor, base_img = preprocess(str(dicom_path))
    probs, bmd = predict(model, image_tensor, device, covariates)
    pred_class = int(np.argmax(probs))

    print(f"Radiografía: {dicom_path}\n")
    print("--- Predicción de clasificación ---")
    for i, (name, p) in enumerate(zip(CLASS_NAMES, probs)):
        marker = " <-- predicción" if i == pred_class else ""
        print(f"  {name:13s}: {p*100:5.1f}%{marker}")

    print(f"\n--- Predicción de regresión ---")
    print(f"  BMD estimado: {bmd:.3f} g/cm²")

    result = build_result(
        dicom_path,
        patient=normalize_patient_id(args.patient) if args.patient else "",
        view=args.view if args.patient else "",
        used_covariates=bool(args.covariates),
        probs=probs, bmd=bmd,
    )

    if args.heatmap:
        print()
        if args.heatmap_out:
            out_path = args.heatmap_out
        elif args.patient:
            out_path = heatmap_path(normalize_patient_id(args.patient), args.view, args.covariates)
        else:
            out_path = heatmap_path(dicom_path.stem, covariates=args.covariates)

        used_class_idx = save_heatmap(
            model, image_tensor, base_img,
            target=args.heatmap_target, class_idx=args.heatmap_class,
            out_path=out_path, covariates=covariates,
        )
        result.update(
            heatmap_generated=True,
            heatmap_target=args.heatmap_target,
            heatmap_explained_class=CLASS_NAMES[used_class_idx] if used_class_idx is not None else "",
            heatmap_out=out_path,
        )
        if not args.no_log:
            save_result_json(result, out_path)

    if not args.no_log:
        append_result_csv(result)


if __name__ == "__main__":
    main()
