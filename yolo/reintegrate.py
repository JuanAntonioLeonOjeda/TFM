import argparse
import shutil
from pathlib import Path

import pandas as pd

from data.df_import import YOLO_DATASET_DIR as DATASET_DIR


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True,
                         help="Ruta al CSV exportado de makesense.ai")
    parser.add_argument("--dataset-dir", type=str, default=str(DATASET_DIR))
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    review_path = dataset_dir / "review.csv"

    corrections = pd.read_csv(args.csv)
    # Las filas sin lumos en el nombre (1.jpg, 2.jpg, etc.) son restos de una importación anterior, no pacientes
    corrections = corrections[corrections["image_name"].str.contains("lumos", case=False)].copy()
    corrections["patient"] = corrections["image_name"].str.replace(".jpg", "", regex=False)

    dup_counts = corrections.groupby("patient").size()
    duplicated = dup_counts[dup_counts > 1]
    if len(duplicated) > 0:
        print(f"[aviso] {len(duplicated)} paciente(s) con más de una caja en el CSV -- "
              f"se usa solo la primera: {list(duplicated.index)}")
        corrections = corrections.drop_duplicates(subset="patient", keep="first")

    corrected_patients = set(corrections["patient"])

    review = pd.read_csv(review_path)
    previously_accepted = review[review["accepted"] == True].copy()

    # Los aceptados que no aparecen en el CSV son los que se borraron a mano durante la revisión
    discarded_by_user = set(previously_accepted["patient"]) - corrected_patients
    if discarded_by_user:
        print(f"\n{len(discarded_by_user)} paciente(s) estaban aceptados pero NO aparecen "
              f"en el CSV corregido -- se interpretan como descartados en la revisión manual:")
        print(f"  {sorted(discarded_by_user)}")

    patient_to_split = dict(zip(previously_accepted["patient"], previously_accepted["split"]))

    updated_rows = []
    updated = 0
    removed = 0

    for _, row in corrections.iterrows():
        patient = row["patient"]

        if patient not in patient_to_split:
            print(f"[aviso] {patient} aparece en el CSV pero no estaba en review.csv como "
                  f"aceptado -- se ignora (¿paciente nuevo añadido a mano en makesense?)")
            continue

        split = patient_to_split[patient]
        w, h = row["image_width"], row["image_height"]

        # En makesense.ai, bbox_x y bbox_y son la esquina superior izquierda, en píxeles
        x0, y0 = row["bbox_x"], row["bbox_y"]
        bw, bh = row["bbox_width"], row["bbox_height"]

        xc = (x0 + bw / 2) / w
        yc = (y0 + bh / 2) / h
        bw_norm = bw / w
        bh_norm = bh / h

        label_path = dataset_dir / "labels" / split / f"{patient}.txt"
        with open(label_path, "w") as f:
            f.write(f"0 {xc:.6f} {yc:.6f} {bw_norm:.6f} {bh_norm:.6f}\n")

        updated_rows.append({
            "patient": patient, "accepted": True, "reason": "corregido manualmente en makesense.ai",
            "split": split,
        })
        updated += 1

    for patient in discarded_by_user:
        split = patient_to_split[patient]
        img_path = dataset_dir / "images" / split / f"{patient}.jpg"
        label_path = dataset_dir / "labels" / split / f"{patient}.txt"

        img_path.unlink(missing_ok=True)
        label_path.unlink(missing_ok=True)

        updated_rows.append({
            "patient": patient, "accepted": False,
            "reason": "descartado manualmente durante la revisión en makesense.ai",
        })
        removed += 1

    still_rejected = review[review["accepted"] == False]
    final_review = pd.concat([pd.DataFrame(updated_rows), still_rejected], ignore_index=True)
    final_review.to_csv(review_path, index=False)

    print(f"\n--- Resumen ---")
    print(f"Etiquetas actualizadas: {updated}")
    print(f"Pacientes descartados (imagen+etiqueta eliminadas): {removed}")
    print(f"Dataset final: {updated} pacientes listos para entrenar YOLO")
    print(f"review.csv actualizado en: {review_path}")


if __name__ == "__main__":
    main()