"""Clasifica cada imagen como AP/LAT y genera view_labels.csv y ap_only.csv (una AP por paciente).

Uso: python -m data.views
"""

from collections import Counter

import pandas as pd
import pydicom

from data.df_import import PROC_DIR, load_image_index

# ViewPosition está vacío en ~87% de los DICOM; SeriesDescription sí es fiable.
AP_KEYWORDS = ["ap", "腰椎前后位", "extension", "flexion", "bending"]
LAT_KEYWORDS = ["lat", "腰椎侧位"]


def classify_view(series_description: str) -> str:
    text = series_description.lower().strip()
    if any(k in text for k in LAT_KEYWORDS):
        return "LAT"
    if any(k in text for k in AP_KEYWORDS):
        return "AP"
    return "UNKNOWN"


def build_view_labels() -> pd.DataFrame:
    rows = []
    for _, row in load_image_index().iterrows():
        path = row["image_path"]
        try:
            ds = pydicom.dcmread(path, stop_before_pixels=True)
        except Exception as e:
            print(f"ERROR leyendo {path}: {e}")
            continue
        desc = str(getattr(ds, "SeriesDescription", ""))
        rows.append({"patient_id": row["patient_id"], "image_path": path,
                     "series_description": desc, "view": classify_view(desc)})
    return pd.DataFrame(rows)


def build_ap_only(views: pd.DataFrame) -> pd.DataFrame:
    ap = views[views["view"] == "AP"]
    per_patient = ap.groupby("patient_id").size()
    missing = sorted(set(views["patient_id"]) - set(ap["patient_id"]))
    print(f"\nPacientes con AP: {len(per_patient)}")
    print(f"Pacientes con más de 1 AP (se usa la primera): {(per_patient > 1).sum()}")
    print(f"Pacientes SIN vista AP (se pierden): {len(missing)}  {missing}")
    return ap.groupby("patient_id").first().reset_index()


def main():
    views = build_view_labels()
    print(f"Total de imágenes: {len(views)}\n")

    print("Clasificación resultante:")
    for view, n in views["view"].value_counts().items():
        print(f"  {view:10s}: {n}")

    unknown = views[views["view"] == "UNKNOWN"]
    if len(unknown):
        print(f"\nCasos UNKNOWN ({len(unknown)}), primeros 10:")
        for _, row in unknown.head(10).iterrows():
            print(f"  '{row['series_description']}' -> {row['image_path']}")

    combos = Counter(views.groupby("patient_id")["view"].apply(lambda v: tuple(sorted(v))))
    print("\nCombinaciones de vistas por paciente:")
    for combo, n in combos.most_common():
        print(f"  {combo}: {n} pacientes")

    PROC_DIR.mkdir(parents=True, exist_ok=True)
    views.to_csv(PROC_DIR / "view_labels.csv", index=False)
    build_ap_only(views).to_csv(PROC_DIR / "ap_only.csv", index=False)
    print(f"\nGuardado: {PROC_DIR / 'view_labels.csv'}")
    print(f"Guardado: {PROC_DIR / 'ap_only.csv'}")


if __name__ == "__main__":
    main()
