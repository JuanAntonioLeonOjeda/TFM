"""
filter_ap_view.py
==================
A partir de res/processed/view_labels.csv, se queda con UNA imagen AP por
paciente (la primera, si hubiera varias). Guarda el resultado en
res/processed/ap_only.csv, listo para usarse en el preprocesamiento con ROI.
"""

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
PROC_DIR = ROOT / "res" / "processed"


def main():
    df = pd.read_csv(PROC_DIR / "view_labels.csv")
    print(f"Total imágenes: {len(df)}")

    ap = df[df["view"] == "AP"].copy()
    print(f"Imágenes AP: {len(ap)}")

    # si un paciente tiene varias AP, nos quedamos con la primera
    ap_one_per_patient = ap.groupby("patient_id").first().reset_index()
    print(f"Pacientes con AP: {ap_one_per_patient['patient_id'].nunique()}")

    dup = ap.groupby("patient_id").size()
    print(f"Pacientes con más de 1 AP (se usó la primera): {(dup > 1).sum()}")

    # pacientes SIN ninguna vista AP (se perderían)
    all_patients = set(df["patient_id"])
    ap_patients = set(ap["patient_id"])
    sin_ap = all_patients - ap_patients
    print(f"Pacientes SIN vista AP (se pierden): {len(sin_ap)}  {sorted(sin_ap)}")

    ap_one_per_patient.to_csv(PROC_DIR / "ap_only.csv", index=False)
    print(f"\nGuardado: {PROC_DIR / 'ap_only.csv'}")


if __name__ == "__main__":
    main()