import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from data.df_import import CLASS_NAMES, PROC_DIR, PROJECT_ROOT

# Radiografías de ejemplo del split de test de LUMOS (el modelo no las vio al entrenar).
# La API las lee de EXAMPLES_DIR para poder montarlas como volumen, igual que el modelo
EXAMPLES_DIR = Path(os.environ.get("EXAMPLES_DIR", PROJECT_ROOT / "res" / "examples"))
MANIFEST = "examples.json"
SEED = 42


# Devuelve los ejemplos y las métricas del modelo en test que se calcularon al prepararlos
def load_examples(examples_dir: Path = EXAMPLES_DIR) -> tuple[list[dict], dict | None]:
    manifest = examples_dir / MANIFEST
    if not manifest.exists():
        return [], None
    spec = json.loads(manifest.read_text(encoding="utf-8"))
    for ex in spec["examples"]:
        ex["path"] = examples_dir / ex["file"]
    return spec["examples"], spec["test_metrics"]


# Predice todas las imágenes de test con el mismo servicio que usa la API
def predict_test_split(meta: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    from api.service import PredictionService

    service = PredictionService()
    test = meta[meta["split"] == "test"].copy()
    test["pred"] = [
        CLASS_NAMES.index(service.predict_dicom(Path(r.image_path).read_bytes(), r.age, r.gender, r.BMI)
                          ["predicted_class"])
        for r in test.itertuples()
    ]
    y, p = test["label"].to_numpy(), test["pred"].to_numpy()
    errors = y != p
    metrics = {
        "model": service.model_path.name,
        "n_images": len(test),
        "accuracy": round(float((~errors).mean()), 4),
        "adjacent_error_share": round(float((np.abs(y - p)[errors] == 1).mean()), 4),
        "confusion_matrix": confusion_matrix(y, p, labels=range(len(CLASS_NAMES))).tolist(),
    }
    return test, metrics


# Por cada clase elige `correct` pacientes bien clasificados y `errors` confundidos con una clase
# vecina, para que los ejemplos muestren aciertos y fallos en una proporción parecida a la real
def build_examples(correct: int, errors: int, out_dir: Path = EXAMPLES_DIR):
    meta = pd.read_csv(PROC_DIR / "metadata.csv")
    test, metrics = predict_test_split(meta)
    first_images = test.sort_values("image_path").groupby("patient_id").first()
    hit = first_images["label"] == first_images["pred"]
    adjacent_miss = (first_images["label"] - first_images["pred"]).abs() == 1
    chosen = pd.concat([
        first_images[hit].groupby("label").sample(n=correct, random_state=SEED),
        first_images[adjacent_miss].groupby("label").sample(n=errors, random_state=SEED),
    ]).sort_values(["label", "T_L1-L4"], ascending=[True, False])

    # Los DICOM del manifiesto anterior se borran para no dejar ejemplos huérfanos
    old_examples, _ = load_examples(out_dir)
    for ex in old_examples:
        ex["path"].unlink(missing_ok=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    examples = []
    for patient_id, row in chosen.iterrows():
        src = Path(row["image_path"])
        shutil.copy2(src, out_dir / src.name)
        examples.append({
            "id": f"lumos_x_{patient_id:03d}",
            "file": src.name,
            "age": float(row["age"]),
            "gender": row["gender"].lower(),
            "bmi": round(float(row["BMI"]), 2),
            "true_class": CLASS_NAMES[int(row["label"])],
            "true_bmd": float(row["bmd"]),
            "true_t_score": float(row["T_L1-L4"]),
        })
    spec = {"test_metrics": metrics, "examples": examples}
    (out_dir / MANIFEST).write_text(json.dumps(spec, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"{len(examples)} ejemplos guardados en {out_dir} "
          f"(exactitud del modelo en test: {metrics['accuracy']:.1%} sobre {metrics['n_images']} imágenes)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepara las radiografías de ejemplo que ofrece la API")
    parser.add_argument("--correct", type=int, default=2, help="Aciertos por clase")
    parser.add_argument("--errors", type=int, default=1, help="Fallos entre clases vecinas por clase")
    parser.add_argument("--out", type=Path, default=EXAMPLES_DIR)
    args = parser.parse_args()
    build_examples(args.correct, args.errors, args.out)
