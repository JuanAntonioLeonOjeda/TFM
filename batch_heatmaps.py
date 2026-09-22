"""
batch_heatmaps.py
==================
Genera heatmaps Grad-CAM para VARIOS pacientes y vistas de golpe, para poder
comparar patrones (p.ej. si las vistas AP y laterales se comportan distinto).

Reutiliza las mismas funciones que infer.py (carga el modelo una sola vez,
en vez de relanzar el script por cada paciente).

Uso:
    # Pacientes 1 al 10, ambas vistas (1=AP, 2=lateral), modelo de referencia
    python batch_heatmaps.py --patients 1-10

    # Lo mismo, con el modelo de covariables médicas
    python batch_heatmaps.py --patients 1-10 --covariates

    # Lista concreta de pacientes
    python batch_heatmaps.py --patients 1,5,7,12

    # Mezcla de rangos y sueltos
    python batch_heatmaps.py --patients 1-5,8,10-12

    # Solo la vista lateral, para centrarte en el patrón que viste
    python batch_heatmaps.py --patients 1-15 --views 2

Salida:
    - Un heatmap .png + .json por cada (paciente, vista) en heatmaps/
      (con sufijo _covars si se usa --covariates)
    - Una fila más por cada uno en outputs/inference_log.csv
    - Un resumen final por consola: cuántos se generaron, cuántos se saltaron
"""

import argparse
from datetime import datetime

import numpy as np

from infer import (
    CLASS_NAMES,
    HEATMAPS_DIR,
    MEDICAL_COVARIATES,
    append_result_csv,
    build_covariate_lookup,
    get_device,
    get_patient_covariates,
    load_model,
    normalize_patient_id,
    patient_id_to_int,
    predict,
    preprocess,
    resolve_dicom_path,
    save_heatmap,
    save_result_json,
)


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
    parser.add_argument("--covariates", action="store_true",
                         help=f"Usa el modelo con covariables médicas ({MEDICAL_COVARIATES}) "
                              f"en vez del modelo de referencia")
    parser.add_argument("--heatmap-target", type=str, default="classification",
                         choices=["classification", "bmd"])
    args = parser.parse_args()

    patients = parse_patient_list(args.patients)
    views = [int(v.strip()) for v in args.views.split(",") if v.strip()]

    print(f"Pacientes a procesar: {len(patients)}")
    print(f"Vistas por paciente:  {views}")
    print(f"Covariables médicas:  {'sí, ' + str(MEDICAL_COVARIATES) if args.covariates else 'no'}\n")

    device = get_device()
    print(f"Device: {device}")
    print("Cargando modelo...\n")

    num_covariates = len(MEDICAL_COVARIATES) if args.covariates else 0
    model, model_path = load_model(device, num_covariates=num_covariates)
    print(f"Modelo: {model_path}\n")

    covariate_meta, covariate_stats = (build_covariate_lookup() if args.covariates else (None, None))

    HEATMAPS_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "_covars" if args.covariates else ""

    generated = 0
    skipped = 0

    for patient in patients:
        patient_id = normalize_patient_id(patient)

        covariates = None
        if args.covariates:
            try:
                patient_id_int = patient_id_to_int(patient)
                covariates = get_patient_covariates(patient_id_int, covariate_meta, covariate_stats)
            except ValueError as e:
                print(f"[SALTADO] {patient_id}: {e}")
                skipped += len(views)
                continue

        for view in views:
            try:
                dicom_path = resolve_dicom_path(patient, view)
            except (FileNotFoundError, ValueError) as e:
                print(f"[SALTADO] {patient_id} vista {view}: {e}")
                skipped += 1
                continue

            image_tensor, base_img = preprocess(str(dicom_path))
            probs, bmd = predict(model, image_tensor, device, covariates)
            pred_class = int(np.argmax(probs))

            out_path = str(HEATMAPS_DIR / f"heatmap_{patient_id}_imagen{view}{suffix}.png")

            used_class_idx = save_heatmap(
                model, image_tensor, base_img,
                target=args.heatmap_target,
                class_idx=None,  # explica siempre la clase predicha
                out_path=out_path,
                covariates=covariates,
            )

            result = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "dicom_path": str(dicom_path),
                "patient": patient_id,
                "view": view,
                "used_covariates": bool(args.covariates),
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
    variant = "covars" if args.covariates else "reference"
    print(f"Revisa outputs/inference_log.csv y la carpeta heatmaps/ para comparar patrones.")
    print(f"Para montar la cuadrícula: python make_grid.py --variant {variant}")


if __name__ == "__main__":
    main()