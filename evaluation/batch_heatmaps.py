import argparse

import numpy as np

from data.dataloaders import MEDICAL_COVARIATES
from data.df_import import CLASS_NAMES
from infer import (
    append_result_csv,
    build_covariate_lookup,
    build_result,
    get_device,
    get_patient_covariates,
    heatmap_path,
    load_model,
    normalize_patient_id,
    parse_patient_list,
    patient_id_to_int,
    predict,
    preprocess,
    resolve_dicom_path,
    save_heatmap,
    save_result_json,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patients", type=str, required=True, help="'1-10', '1,5,7' o 'lumos_x_001,...'")
    parser.add_argument("--views", type=str, default="1,2", help="'1,2' = AP y lateral")
    parser.add_argument("--covariates", action="store_true", help=f"Modelo con {MEDICAL_COVARIATES}")
    parser.add_argument("--heatmap-target", type=str, default="classification", choices=["classification", "bmd"])
    args = parser.parse_args()

    patients = parse_patient_list(args.patients)
    views = [int(v.strip()) for v in args.views.split(",") if v.strip()]

    print(f"Pacientes a procesar: {len(patients)}")
    print(f"Vistas por paciente:  {views}")
    print(f"Covariables médicas:  {'sí, ' + str(MEDICAL_COVARIATES) if args.covariates else 'no'}\n")

    device = get_device()
    num_covariates = len(MEDICAL_COVARIATES) if args.covariates else 0
    model, model_path = load_model(device, num_covariates=num_covariates)
    print(f"Device: {device}\nModelo: {model_path}\n")

    covariate_meta, covariate_stats = build_covariate_lookup() if args.covariates else (None, None)

    generated = 0
    skipped = 0

    for patient in patients:
        patient_id = normalize_patient_id(patient)

        covariates = None
        if args.covariates:
            try:
                covariates = get_patient_covariates(patient_id_to_int(patient), covariate_meta, covariate_stats)
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

            out_path = heatmap_path(patient_id, view, args.covariates)
            used_class_idx = save_heatmap(
                model, image_tensor, base_img,
                target=args.heatmap_target, class_idx=None,
                out_path=out_path, covariates=covariates,
            )

            result = build_result(dicom_path, patient_id, view, bool(args.covariates), probs, bmd)
            result.update(
                heatmap_generated=True,
                heatmap_target=args.heatmap_target,
                heatmap_explained_class=CLASS_NAMES[used_class_idx] if used_class_idx is not None else "",
                heatmap_out=out_path,
            )
            append_result_csv(result)
            save_result_json(result, out_path)

            print(f"[OK] {patient_id} vista {view}: {CLASS_NAMES[pred_class]} "
                  f"({probs[pred_class]*100:.1f}%) -> {out_path}\n")
            generated += 1

    variant = "covars" if args.covariates else "reference"
    print("--- Resumen ---")
    print(f"Heatmaps generados: {generated}")
    print(f"Saltados (no encontrados): {skipped}")
    print(f"Para montar la cuadrícula: python -m evaluation.heatmap_grid --variant {variant}")


if __name__ == "__main__":
    main()
