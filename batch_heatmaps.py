"""
batch_heatmaps.py
==================
Genera heatmaps Grad-CAM para VARIOS pacientes y vistas de golpe, para poder
comparar patrones (p.ej. si las vistas AP y laterales se comportan distinto).

Reutiliza las mismas funciones que infer.py (carga el modelo una sola vez,
en vez de relanzar el script por cada paciente).

Uso:
    # Pacientes 1 al 10, ambas vistas (1=AP, 2=lateral)
    python batch_heatmaps.py --patients 1-10

    # Lista concreta de pacientes
    python batch_heatmaps.py --patients 1,5,7,12

    # Mezcla de rangos y sueltos
    python batch_heatmaps.py --patients 1-5,8,10-12

    # Solo la vista lateral, para centrarte en el patrón que viste
    python batch_heatmaps.py --patients 1-15 --views 2

Salida:
    - Un heatmap .png + .json por cada (paciente, vista) en heatmaps/
    - Una fila más por cada uno en outputs/inference_log.csv
    - Un resumen final por consola: cuántos se generaron, cuántos se saltaron
"""

import argparse

import numpy as np

from infer import (
    CLASS_NAMES,
    HEATMAPS_DIR,
    append_result_csv,
    get_device,
    load_model,
    normalize_patient_id,
    predict,
    preprocess,
    resolve_dicom_path,
    save_heatmap,
    save_result_json,
)
from datetime import datetime


def parse_patient_list(spec: str) -> list[str]:
    """
    Convierte "1-5,8,10-12" en ["1","2","3","4","5","8","10","11","12"].
    También acepta IDs ya formateados como "lumos_x_001" sueltos (sin rango).
    """
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patients", type=str, required=True,
                         help="Pacientes a procesar: '1-10', '1,5,7', '1-5,8,10-12', "
                              "o IDs completos separados por comas ('lumos_x_001,lumos_x_003')")
    parser.add_argument("--views", type=str, default="1,2",
                         help="Vistas a generar por paciente (default: '1,2' = AP y lateral). "
                              "Usa '2' para procesar solo laterales")
    parser.add_argument("--heatmap-target", type=str, default="classification",
                         choices=["classification", "bmd"])
    args = parser.parse_args()

    patients = parse_patient_list(args.patients)
    views = [int(v.strip()) for v in args.views.split(",") if v.strip()]

    print(f"Pacientes a procesar: {len(patients)}")
    print(f"Vistas por paciente:  {views}\n")

    device = get_device()
    print(f"Device: {device}")
    print("Cargando modelo...\n")
    model = load_model(device)

    HEATMAPS_DIR.mkdir(parents=True, exist_ok=True)

    generated = 0
    skipped = 0

    for patient in patients:
        patient_id = normalize_patient_id(patient)

        for view in views:
            try:
                dicom_path = resolve_dicom_path(patient, view)
            except (FileNotFoundError, ValueError) as e:
                print(f"[SALTADO] {patient_id} vista {view}: {e}")
                skipped += 1
                continue

            image_tensor, base_img = preprocess(str(dicom_path))
            probs, bmd = predict(model, image_tensor, device)
            pred_class = int(np.argmax(probs))

            out_path = str(HEATMAPS_DIR / f"heatmap_{patient_id}_imagen{view}.png")

            used_class_idx = save_heatmap(
                model, image_tensor, base_img,
                target=args.heatmap_target,
                class_idx=None,  # explica siempre la clase predicha
                out_path=out_path,
            )

            result = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "dicom_path": str(dicom_path),
                "patient": patient_id,
                "view": view,
                "prob_sano": round(float(probs[0]), 4),
                "prob_osteopenia": round(float(probs[1]), 4),
                "prob_osteoporosis": round(float(probs[2]), 4),
                "predicted_class": CLASS_NAMES[pred_class],
                "bmd_estimated": round(float(bmd), 4),
                "heatmap_generated": True,
                "heatmap_target": args.heatmap_target,
                "heatmap_explained_class": CLASS_NAMES[used_class_idx] if used_class_idx is not None else "",
                "heatmap_out": out_path,
            }
            append_result_csv(result)
            save_result_json(result, out_path)

            print(f"[OK] {patient_id} vista {view}: {CLASS_NAMES[pred_class]} "
                  f"({probs[pred_class]*100:.1f}%) -> {out_path}\n")

            generated += 1

    print("--- Resumen ---")
    print(f"Heatmaps generados: {generated}")
    print(f"Saltados (no encontrados): {skipped}")
    print(f"Revisa outputs/inference_log.csv y la carpeta heatmaps/ para comparar patrones.")


if __name__ == "__main__":
    main()