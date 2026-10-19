"""Monta los heatmaps de heatmaps/ en una cuadrícula: una fila por paciente, una columna por vista.

Uso: python -m evaluation.heatmap_grid [--patients 1-10] [--variant covars]
"""

import argparse
import json
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from infer import HEATMAPS_DIR, normalize_patient_id, parse_patient_list

THUMB_SIZE = 260
LABEL_HEIGHT = 60
HEADER_HEIGHT = 40
ROW_LABEL_WIDTH = 0
PADDING = 8
FILENAME_RE = re.compile(r"heatmap_(?P<patient>.+)_imagen(?P<view>\d+)(?P<variant>_covars)?\.png$")


def natural_key(patient_id: str):
    m = re.search(r"(\d+)$", patient_id)
    return int(m.group(1)) if m else patient_id


def load_entries(heatmaps_dir: Path, variant: str = "reference"):
    """{patient_id: {view: (png_path, meta_del_json)}} solo de la variante pedida ('reference' o 'covars')."""
    entries = {}
    for png_path in heatmaps_dir.glob("heatmap_*_imagen*.png"):
        m = FILENAME_RE.match(png_path.name)
        if not m:
            continue

        file_variant = "covars" if m.group("variant") else "reference"
        if file_variant != variant:
            continue

        patient_id = m.group("patient")
        view = int(m.group("view"))

        meta = {}
        json_path = png_path.with_suffix(".json")
        if json_path.exists():
            with open(json_path, "r", encoding="utf-8") as f:
                meta = json.load(f)

        entries.setdefault(patient_id, {})[view] = (png_path, meta)
    return entries


def label_for(meta: dict) -> str:
    if not meta:
        return ""
    pred = meta.get("predicted_class", "")
    probs = {
        "Sano": meta.get("prob_sano"),
        "Osteopenia": meta.get("prob_osteopenia"),
        "Osteoporosis": meta.get("prob_osteoporosis"),
    }
    p = probs.get(pred)
    if pred and p is not None:
        return f"{pred} ({p*100:.0f}%)"
    return pred


def draw_centered_text(draw, xy, text, font, box_width):
    x, y = xy
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    draw.text((x + (box_width - text_w) / 2, y), text, font=font, fill="black")


def build_grid(entries: dict, out_path: Path, view_labels: dict):
    patients = sorted(entries.keys(), key=natural_key)
    views = sorted({v for views in entries.values() for v in views})

    if not patients or not views:
        raise ValueError("No se encontraron heatmaps que combinar. Genera antes con infer.py --heatmap o evaluation.batch_heatmaps.")

    n_rows = len(patients)
    n_cols = len(views)

    cell_w = THUMB_SIZE
    cell_h = THUMB_SIZE + LABEL_HEIGHT

    grid_w = ROW_LABEL_WIDTH + n_cols * (cell_w + PADDING) + PADDING
    grid_h = HEADER_HEIGHT + n_rows * (cell_h + PADDING) + PADDING

    grid = Image.new("RGB", (grid_w, grid_h), "white")
    draw = ImageDraw.Draw(grid)
    font = ImageFont.load_default()

    for c, view in enumerate(views):
        x = ROW_LABEL_WIDTH + c * (cell_w + PADDING) + PADDING
        header_text = view_labels.get(view, f"Vista {view}")
        draw_centered_text(draw, (x, PADDING), header_text, font, cell_w)

    for r, patient_id in enumerate(patients):
        y = HEADER_HEIGHT + r * (cell_h + PADDING) + PADDING

        for c, view in enumerate(views):
            x = ROW_LABEL_WIDTH + c * (cell_w + PADDING) + PADDING

            if view in entries[patient_id]:
                png_path, meta = entries[patient_id][view]
                thumb = Image.open(png_path).convert("RGB")
                thumb.thumbnail((THUMB_SIZE, THUMB_SIZE))
                paste_x = x + (THUMB_SIZE - thumb.width) // 2
                paste_y = y + (THUMB_SIZE - thumb.height) // 2
                grid.paste(thumb, (paste_x, paste_y))

                label = label_for(meta)
                if label:
                    draw_centered_text(draw, (x, y + THUMB_SIZE + 4), label, font, cell_w)
            else:
                draw.rectangle([x, y, x + THUMB_SIZE, y + THUMB_SIZE], outline="lightgray")
                draw_centered_text(draw, (x, y + THUMB_SIZE / 2), "(sin datos)", font, cell_w)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(out_path)
    print(f"Cuadrícula guardada en: {out_path}  ({n_rows} pacientes x {n_cols} vistas)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--patients", type=str, default=None, help="p.ej. '1-10' o '001,003,007'")
    parser.add_argument("--variant", type=str, choices=["reference", "covars"], default="reference")
    parser.add_argument("--view-labels", type=str, default="1=AP,2=Lateral")
    args = parser.parse_args()

    heatmaps_dir = Path(args.input_dir) if args.input_dir else HEATMAPS_DIR
    out_path = Path(args.output) if args.output else heatmaps_dir / f"grid_{args.variant}.png"

    entries = load_entries(heatmaps_dir, variant=args.variant)

    if not entries:
        heatmap_flag = " --covariates" if args.variant == "covars" else ""
        raise SystemExit(
            f"No se encontró ningún heatmap de la variante '{args.variant}' en {heatmaps_dir}. "
            f"Genera algunos primero con: python infer.py --patient <id>{heatmap_flag} --heatmap"
        )

    if args.patients:
        wanted = {normalize_patient_id(p) for p in parse_patient_list(args.patients)}

        filtered = {pid: v for pid, v in entries.items() if pid in wanted}
        if not filtered:
            raise SystemExit(
                f"Hay {len(entries)} paciente(s) con heatmaps en {heatmaps_dir}, "
                f"pero ninguno coincide con --patients {args.patients}. "
                f"Pacientes disponibles: {sorted(entries.keys(), key=natural_key)}. "
                f"Genera los que faltan con: python -m evaluation.batch_heatmaps --patients {args.patients}"
            )
        entries = filtered

    view_labels = {}
    for pair in args.view_labels.split(","):
        if "=" in pair:
            k, v = pair.split("=")
            view_labels[int(k.strip())] = v.strip()

    build_grid(entries, out_path, view_labels)


if __name__ == "__main__":
    main()