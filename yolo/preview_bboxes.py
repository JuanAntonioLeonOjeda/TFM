"""Dibuja las cajas aceptadas de review.csv sobre sus imágenes, en cuadrícula, para revisarlas.

Uso: python -m yolo.preview_bboxes [--n 30] [--only-split train]
"""

import argparse
from pathlib import Path

import cv2
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from data.df_import import YOLO_DATASET_DIR

THUMB_SIZE = 220
LABEL_HEIGHT = 20
COLS = 6


def draw_yolo_box(img_path: Path, label_path: Path) -> tuple[Image.Image, float]:
    """Dibuja la caja de labels/*.txt (no de review.csv: tras reintegrate ya no tiene píxeles)."""
    img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    _, xc, yc, bw, bh = map(float, label_path.read_text().split())
    x0, y0 = int((xc - bw / 2) * w), int((yc - bh / 2) * h)
    x1, y1 = int((xc + bw / 2) * w), int((yc + bh / 2) * h)
    cv2.rectangle(img, (x0, y0), (x1, y1), (255, 60, 60), 3)
    return Image.fromarray(img), bw * bh


def build_grid(rows: list[dict], dataset_dir: Path, out_path: Path, cols: int = COLS):
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

        patient, split = row["patient"], row["split"]
        thumb, area_frac = draw_yolo_box(dataset_dir / "images" / split / f"{patient}.jpg",
                                         dataset_dir / "labels" / split / f"{patient}.txt")
        thumb.thumbnail((THUMB_SIZE, THUMB_SIZE))
        grid.paste(thumb, (x, y))

        label = f"{patient}  ({area_frac:.2f})"
        draw.text((x + 4, y + THUMB_SIZE + 2), label, font=font, fill="black")

    grid.save(out_path)
    print(f"Vista previa guardada en: {out_path}  ({n} cajas)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=str, default=str(YOLO_DATASET_DIR))
    parser.add_argument("--n", type=int, default=50, help="Máximo de cajas (muestra aleatoria)")
    parser.add_argument("--only-split", type=str, choices=["train", "val"], default=None)
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

    out_path = Path(args.output) if args.output else dataset_dir / "preview_grid.png"
    build_grid(accepted.to_dict("records"), dataset_dir, out_path)


if __name__ == "__main__":
    main()