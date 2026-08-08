"""
check_view_position2.py
========================
Clasifica cada imagen como AP o LATERAL usando SeriesDescription
(ViewPosition no es fiable: está vacío en el 87% de los casos).
"""

from collections import Counter
import pydicom

from data.df_import import load_image_index   # AJUSTA al nombre real de tu módulo


AP_KEYWORDS = ["ap", "腰椎前后位", "extension", "flexion", "bending"]
LAT_KEYWORDS = ["lat", "腰椎侧位"]


def classify_view(series_description: str) -> str:
    """Devuelve 'AP', 'LAT' o 'UNKNOWN' según el texto de SeriesDescription."""
    text = series_description.lower().strip()
    if any(k in text for k in LAT_KEYWORDS):
        return "LAT"
    if any(k in text for k in AP_KEYWORDS):
        return "AP"
    return "UNKNOWN"


def main():
    images = load_image_index()
    print(f"Total de imágenes: {len(images)}\n")

    counts = Counter()
    unknown_examples = []

    for _, row in images.iterrows():
        path = row["image_path"]
        try:
            ds = pydicom.dcmread(path, stop_before_pixels=True)
        except Exception as e:
            print(f"ERROR leyendo {path}: {e}")
            continue

        desc = str(getattr(ds, "SeriesDescription", ""))
        view = classify_view(desc)
        counts[view] += 1

        if view == "UNKNOWN":
            unknown_examples.append((path, desc))

    print("Clasificación resultante:")
    for view, n in counts.most_common():
        print(f"  {view:10s}: {n}")

    if unknown_examples:
        print(f"\nCasos UNKNOWN ({len(unknown_examples)}), primeros 10:")
        for path, desc in unknown_examples[:10]:
            print(f"  '{desc}' -> {path}")

    # cuántos pacientes tienen exactamente 1 AP y 1 LAT
    images["view"] = None
    print("\nGuardando clasificación completa en res/processed/view_labels.csv ...")

    rows = []
    for _, row in images.iterrows():
        path = row["image_path"]
        ds = pydicom.dcmread(path, stop_before_pixels=True)
        desc = str(getattr(ds, "SeriesDescription", ""))
        rows.append({"patient_id": row["patient_id"], "image_path": path,
                     "series_description": desc, "view": classify_view(desc)})

    import pandas as pd
    from pathlib import Path
    out = pd.DataFrame(rows)
    out_dir = Path(__file__).resolve().parent / "res" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_dir / "view_labels.csv", index=False)
    print(f"Guardado: {out_dir / 'view_labels.csv'}")

    per_patient = out.groupby("patient_id")["view"].apply(lambda v: sorted(v.tolist()))
    dist = Counter(tuple(v) for v in per_patient)
    print("\nCombinaciones de vistas por paciente:")
    for combo, n in dist.most_common():
        print(f"  {combo}: {n} pacientes")


if __name__ == "__main__":
    main()