"""
preview_bboxes.py
==================
Dibuja las cajas delimitadoras auto-generadas (por auto_label_bboxes.py)
sobre sus radiografías correspondientes, y las monta en una cuadrícula,
para poder inspeccionar visualmente su calidad ANTES de entrenar YOLO con
ellas.

Es el paso equivalente a la figura roi_grid_preview de tu memoria (donde
comprobaste visualmente si el recorte fijo tenía sentido), pero para las
cajas adaptativas.

Uso:
    python preview_bboxes.py
    python preview_bboxes.py --n 30
    python preview_bboxes.py --only-split train
"""

import argparse
from pathlib import Path

import cv2
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

YOLO_DATASET_DIR = Path(__file__).resolve().parent / "res" / "yolo_dataset"

THUMB_SIZE = 220
LABEL_HEIGHT = 20
COLS = 6


def draw_box_on_image(img_path: Path, x0: int, y0: int, x1: int, y1: int) -> Image.Image:
    img = cv2.imread(str(img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.rectangle(img, (int(x0), int(y0)), (int(x1), int(y1)), (255, 60, 60), 3)
    return Image.fromarray(img)


def build_grid(rows: list[dict], images_dir_by_split: dict, out_path: Path, cols: int = COLS):
    n = len(rows)
    if n == 0:
        raise SystemExit("No hay filas aceptadas que mostrar.")

    n_rows = (n + cols - 1) // cols
    cell_w = THUMB_SIZE
    cell_h = THUMB_SIZE + LABEL_HEIGHT
    grid = Image.new("RGB", (cols * cell_w, n_rows * cell_h), "white")
    draw = ImageDraw.Draw(grid)
    font = ImageFont.load_default()

    for i, row in enumerate(rows):
        r, c = divmod(i, cols)
        x, y = c * cell_w, r * cell_h

        split = row["split"]
        img_path = images_dir_by_split[split] / f"{row['patient']}.jpg"
        thumb = draw_box_on_image(img_path, row["x0"], row["y0"], row["x1"], row["y1"])
        thumb.thumbnail((THUMB_SIZE, THUMB_SIZE))
        grid.paste(thumb, (x, y))

        label = f"{row['patient']}  ({row['area_frac']:.2f})"
        draw.text((x + 4, y + THUMB_SIZE + 2), label, font=font, fill="black")

    grid.save(out_path)
    print(f"Vista previa guardada en: {out_path}  ({n} cajas)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=str, default=str(YOLO_DATASET_DIR))
    parser.add_argument("--n", type=int, default=50,
                         help="Cuántas cajas mostrar como máximo (muestreo aleatorio si hay más). Default: 50")
    parser.add_argument("--only-split", type=str, choices=["train", "val"], default=None,
                         help="Filtra solo un split concreto (por defecto, mezcla ambos)")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    review = pd.read_csv(dataset_dir / "review.csv")

    accepted = review[review["accepted"] == True].copy()
    if args.only_split:
        accepted = accepted[accepted["split"] == args.only_split]

    if len(accepted) == 0:
        raise SystemExit("No hay cajas aceptadas en review.csv para mostrar.")

    if len(accepted) > args.n:
        accepted = accepted.sample(n=args.n, random_state=args.seed)

    rows = accepted.to_dict("records")

    images_dir_by_split = {
        "train": dataset_dir / "images" / "train",
        "val": dataset_dir / "images" / "val",
    }

    out_path = Path(args.output) if args.output else dataset_dir / "preview_grid.png"
    build_grid(rows, images_dir_by_split, out_path)


if __name__ == "__main__":
    main()