"""
infer.py
========
Script de INFERENCIA: carga el modelo ya entrenado (best_phased_full.pt) y
lo usa para predecir sobre una radiografía DICOM nueva, fuera del proceso
de entrenamiento.

Esto es lo mínimo necesario para "desplegar" el modelo:
  1. La arquitectura (models/lumosnet.py)
  2. Los pesos entrenados (outputs/best_phased_full.pt)
  3. El mismo preprocesamiento usado en entrenamiento (CLAHE, resize, CLAHE,
     normalización ImageNet)

Uso:
    # Ruta directa a un archivo .dcm concreto
    python infer.py --dicom "ruta/a/una/radiografia.dcm"
    python infer.py --dicom "ruta/a/una/radiografia.dcm" --heatmap

    # O, más cómodo: solo el ID de paciente del dataset LUMOS.
    # Busca automáticamente dentro de res/dataset/ la carpeta del paciente
    # y usa su primera radiografía (o la que indiques con --view).
    python infer.py --patient 001 --heatmap
    python infer.py --patient lumos_x_001 --view 2 --heatmap

Salida:
    - Clase predicha (Sano / Osteopenia / Osteoporosis) con su probabilidad
    - Valor de BMD predicho
    - (opcional, con --heatmap) imagen PNG + JSON de resultados, guardados en
      heatmaps/ (vía pytorch-grad-cam: pip install grad-cam)
    - Cada ejecución se añade también como fila en outputs/inference_log.csv
"""

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import cv2
import torch
import torchvision.transforms as T

from data.df_import import read_image          # AJUSTA al nombre real de tu módulo
from models.lumosnet import LumosNet, get_device
from evaluation.gradcam import generate_heatmap

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "outputs" / "best_phased_full.pt"
DATASET_ROOT = ROOT / "res" / "dataset"
RESULTS_CSV = ROOT / "outputs" / "inference_log.csv"
HEATMAPS_DIR = ROOT / "heatmaps"

IMG_SIZE = 224
CLASS_NAMES = ["Sano", "Osteopenia", "Osteoporosis"]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def normalize_patient_id(patient: str) -> str:
    """'001' o '1' -> 'lumos_x_001'. Si ya viene con el formato completo, lo deja igual."""
    return f"lumos_x_{int(patient):03d}" if patient.isdigit() else patient


def resolve_dicom_path(patient: str, view: int = 1, dataset_root: Path = DATASET_ROOT) -> Path:
    """
    Dado un ID de paciente del dataset LUMOS (p.ej. "001" o "lumos_x_001"),
    busca automáticamente su carpeta dentro de dataset_root y devuelve la
    ruta al archivo .dcm/.Dcm correspondiente a `view` (1 = primera vista,
    2 = segunda, etc. -- coincide con lumos_x_001_1.Dcm, lumos_x_001_2.Dcm...).

    No hace falta escribir la ruta completa cada vez: solo el número de
    paciente.
    """
    patient_id = normalize_patient_id(patient)

    # La carpeta intermedia (p.ej. "lumos_x_001_280_dcm") puede variar, así
    # que buscamos cualquier carpeta que se llame exactamente patient_id
    # dentro de dataset_root, sea cual sea su carpeta contenedora.
    candidate_dirs = [p for p in dataset_root.rglob(patient_id) if p.is_dir()]
    if not candidate_dirs:
        raise FileNotFoundError(
            f"No se encontró ninguna carpeta '{patient_id}' dentro de {dataset_root}. "
            f"Revisa el ID de paciente o usa --dicom con la ruta completa."
        )

    patient_dir = candidate_dirs[0]
    files = sorted(patient_dir.glob("*.[Dd][Cc][Mm]"))  # coincide con .dcm y .Dcm
    if not files:
        raise FileNotFoundError(f"La carpeta {patient_dir} no contiene archivos .dcm/.Dcm")

    if not (1 <= view <= len(files)):
        raise ValueError(
            f"--view {view} fuera de rango: {patient_dir.name} tiene {len(files)} imagen(es) (usa 1..{len(files)})"
        )

    return files[view - 1]


def load_model(device):
    """Reconstruye la arquitectura y carga los pesos entrenados."""
    model = LumosNet(pretrained=False)   # arquitectura vacía: los pesos vienen del .pt
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    return model


def preprocess(dicom_path: str):
    """
    Aplica EXACTAMENTE el mismo preprocesamiento que en entrenamiento:
    DICOM -> uint8 -> resize 224x224 -> CLAHE -> tensor normalizado.

    Devuelve DOS cosas:
      - tensor: (1, 3, 224, 224) normalizado, listo para el modelo.
      - base_img: array uint8 de 1 canal tras CLAHE (SIN normalizar), usado
                   como fondo del overlay de Grad-CAM.
    """
    img = read_image(dicom_path)                      # float32 [0,1], corrige MONOCHROME1
    img = (img * 255).astype(np.uint8)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)                             # <- esta es la imagen "base"

    rgb = np.stack([img, img, img], axis=-1)            # 1 canal -> 3 canales

    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    tensor = transform(rgb).unsqueeze(0)                # añade la dimensión de batch

    return tensor, img


def predict(model, image_tensor, device):
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        logits, bmd_pred = model(image_tensor)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        bmd = bmd_pred.item()
    return probs, bmd


def save_heatmap(model, image_tensor, base_img_uint8_gray, target, class_idx, out_path):
    """
    base_img_uint8_gray: array (224,224) uint8, tras CLAHE, SIN normalizar.
    pytorch-grad-cam espera la imagen de fondo como RGB float en [0,1].

    Devuelve used_class_idx (None si target="bmd"), útil para registrar
    qué clase se explicó exactamente en el log de resultados.
    """
    base_rgb_float = np.stack([base_img_uint8_gray] * 3, axis=-1).astype(np.float32) / 255.0

    overlay, used_class_idx = generate_heatmap(
        model,
        image_tensor,
        base_rgb_float,
        target=target,
        class_idx=class_idx,
    )

    # show_cam_on_image devuelve RGB; cv2.imwrite espera BGR.
    overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    cv2.imwrite(out_path, overlay_bgr)

    if used_class_idx is not None:
        print(f"Heatmap guardado en: {out_path}  (clase explicada: {CLASS_NAMES[used_class_idx]})")
    else:
        print(f"Heatmap guardado en: {out_path}  (target: BMD)")

    return used_class_idx


def save_result_json(result: dict, out_path: str):
    """Guarda los resultados numéricos de ESTA ejecución en un .json individual."""
    json_path = Path(out_path).with_suffix(".json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"Resultados guardados en: {json_path}")


def append_result_csv(result: dict, csv_path: Path = RESULTS_CSV):
    """Añade una fila con los resultados de ESTA ejecución a un CSV acumulativo,
    para poder tabular/filtrar resultados de varios pacientes más adelante."""
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
                         help="Ruta directa a un archivo DICOM (tiene prioridad sobre --patient)")
    parser.add_argument("--patient", type=str, default=None,
                         help="ID de paciente del dataset LUMOS (p.ej. '001' o 'lumos_x_001'). "
                              "Busca automáticamente el archivo en res/dataset/, sin escribir la ruta completa")
    parser.add_argument("--view", type=int, default=1,
                         help="Qué radiografía del paciente usar si --patient tiene varias vistas "
                              "(1=primera, 2=segunda...). Default: 1")
    parser.add_argument("--heatmap", action="store_true",
                         help="Genera además un mapa de calor Grad-CAM sobre la radiografía")
    parser.add_argument("--heatmap-target", type=str, default="classification",
                         choices=["classification", "bmd"],
                         help="Qué cabeza explicar con el heatmap (default: classification)")
    parser.add_argument("--heatmap-class", type=int, default=None,
                         help="Índice de clase a explicar (0=Sano,1=Osteopenia,2=Osteoporosis). "
                              "Por defecto usa la clase predicha. Ignorado si --heatmap-target=bmd")
    parser.add_argument("--heatmap-out", type=str, default=None,
                         help="Ruta de salida del PNG con el heatmap (solo si --heatmap). "
                              "Por defecto se genera automáticamente incluyendo el paciente y la vista, "
                              "p.ej. heatmap_lumos_x_001_imagen1.png")
    parser.add_argument("--no-log", action="store_true",
                         help="No guardar los resultados numéricos (ni el .json ni el .csv acumulativo)")
    args = parser.parse_args()

    if args.dicom:
        dicom_path = Path(args.dicom)
    elif args.patient:
        dicom_path = resolve_dicom_path(args.patient, args.view)
    else:
        parser.error("Debes indicar --dicom <ruta> o --patient <id>")

    device = get_device()
    print(f"Cargando modelo desde: {MODEL_PATH}")
    print(f"Device: {device}\n")

    model = load_model(device)
    image_tensor, base_img = preprocess(str(dicom_path))

    probs, bmd = predict(model, image_tensor, device)
    pred_class = int(np.argmax(probs))

    print(f"Radiografía: {dicom_path}\n")
    print("--- Predicción de clasificación ---")
    for name, p in zip(CLASS_NAMES, probs):
        marker = " <-- predicción" if name == CLASS_NAMES[pred_class] else ""
        print(f"  {name:13s}: {p*100:5.1f}%{marker}")

    print(f"\n--- Predicción de regresión ---")
    print(f"  BMD estimado: {bmd:.3f} g/cm²")

    result = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "dicom_path": str(dicom_path),
        "patient": normalize_patient_id(args.patient) if args.patient else "",
        "view": args.view if args.patient else "",
        "prob_sano": round(float(probs[0]), 4),
        "prob_osteopenia": round(float(probs[1]), 4),
        "prob_osteoporosis": round(float(probs[2]), 4),
        "predicted_class": CLASS_NAMES[pred_class],
        "bmd_estimated": round(float(bmd), 4),
        "heatmap_generated": bool(args.heatmap),
        "heatmap_target": args.heatmap_target if args.heatmap else "",
        "heatmap_explained_class": "",  # se rellena abajo si aplica
        "heatmap_out": "",
    }

    if args.heatmap:
        print()
        if args.heatmap_out:
            out_path = args.heatmap_out  # ruta explícita del usuario: se respeta tal cual
        else:
            HEATMAPS_DIR.mkdir(parents=True, exist_ok=True)
            if args.patient:
                patient_id = normalize_patient_id(args.patient)
                filename = f"heatmap_{patient_id}_imagen{args.view}.png"
            else:
                # --dicom directo, sin --patient: usa el nombre del propio archivo dicom
                filename = f"heatmap_{dicom_path.stem}.png"
            out_path = str(HEATMAPS_DIR / filename)

        used_class_idx = save_heatmap(
            model, image_tensor, base_img,
            target=args.heatmap_target,
            class_idx=args.heatmap_class,
            out_path=out_path,
        )

        result["heatmap_out"] = out_path
        if used_class_idx is not None:
            result["heatmap_explained_class"] = CLASS_NAMES[used_class_idx]

        if not args.no_log:
            save_result_json(result, out_path)

    if not args.no_log:
        append_result_csv(result)


if __name__ == "__main__":
    main()