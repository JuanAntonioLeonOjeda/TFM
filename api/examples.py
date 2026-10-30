import argparse
import json
import os
import shutil
from pathlib import Path

import pandas as pd

from data.df_import import CLASS_NAMES, PROC_DIR, PROJECT_ROOT

# Radiografías de ejemplo del split de test de LUMOS (el modelo no las vio al entrenar).
# La API las lee de EXAMPLES_DIR para poder montarlas como volumen, igual que el modelo
EXAMPLES_DIR = Path(os.environ.get("EXAMPLES_DIR", PROJECT_ROOT / "res" / "examples"))
MANIFEST = "examples.json"
SEED = 42


def load_examples(examples_dir: Path = EXAMPLES_DIR) -> list[dict]:
    manifest = examples_dir / MANIFEST
    if not manifest.exists():
        return []
    examples = json.loads(manifest.read_text(encoding="utf-8"))
    for ex in examples:
        ex["path"] = examples_dir / ex["file"]
    return examples


# Elige per_class pacientes de test por clase y copia su primera imagen junto con un manifiesto
# que lleva las covariables y la verdad terreno
def build_examples(per_class: int, out_dir: Path = EXAMPLES_DIR):
    meta = pd.read_csv(PROC_DIR / "metadata.csv")
    first_images = meta[meta["split"] == "test"].sort_values("image_path").groupby("patient_id").first()
    chosen = (first_images.groupby("label").sample(n=per_class, random_state=SEED)
              .sort_values(["label", "bmd"]))

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
        })
    (out_dir / MANIFEST).write_text(json.dumps(examples, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"{len(examples)} ejemplos guardados en {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepara las radiografías de ejemplo que ofrece la API")
    parser.add_argument("--per-class", type=int, default=3)
    parser.add_argument("--out", type=Path, default=EXAMPLES_DIR)
    args = parser.parse_args()
    build_examples(args.per_class, args.out)
