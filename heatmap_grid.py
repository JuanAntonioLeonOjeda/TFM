"""
make_grid.py
============
Monta todos los heatmaps generados en heatmaps/ en una ÚNICA imagen tipo
cuadrícula: una fila por paciente, una columna por vista (AP / lateral...),
con la predicción y confianza como etiqueta bajo cada imagen.

Pensado para comparar de un vistazo el patrón que viste (AP centrado en la
columna vs. lateral desviado), en vez de abrir imagen por imagen.

Uso:
    python make_grid.py
    python make_grid.py --patients 1-10
    python make_grid.py --output outputs/grid_ap_vs_lateral.png

Requiere: Pillow (pip install Pillow) -- muy probable que ya la tengas
instalada como dependencia indirecta de torchvision/opencv.
"""

import argparse
import json
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from infer import HEATMAPS_DIR

THUMB_SIZE = 260          # tamaño de cada imagen dentro de la cuadrícula
LABEL_HEIGHT = 60         # alto del área de texto bajo cada imagen
HEADER_HEIGHT = 40        # alto de la fila de cabeceras (Vista 1, Vista 2...)
ROW_LABEL_WIDTH = 0       # sin columna de ID de paciente
PADDING = 8
FILENAME_RE = re.compile(r"heatmap_(?P<patient>.+)_imagen(?P<view>\d+)\.png$")


def natural_key(patient_id: str):
    """Ordena 'lumos_x_002' antes que 'lumos_x_010' (orden numérico, no alfabético)."""
    m = re.search(r"(\d+)$", patient_id)
    return int(m.group(1)) if m else patient_id


def load_entries(heatmaps_dir: Path):
    """
    Escanea heatmaps_dir y devuelve un dict {patient_id: {view: (png_path, meta_dict)}}
    meta_dict viene del .json correspondiente si existe (predicted_class, probs...).
    """
    entries = {}
    for png_path in heatmaps_dir.glob("heatmap_*_imagen*.png"):
        m = FILENAME_RE.match(png_path.name)
        if not m:
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
        raise ValueError("No se encontraron heatmaps que combinar. ¿Ejecutaste infer.py/batch_heatmaps.py con --heatmap antes?")

    n_rows = len(patients)
    n_cols = len(views)

    cell_w = THUMB_SIZE
    cell_h = THUMB_SIZE + LABEL_HEIGHT

    grid_w = ROW_LABEL_WIDTH + n_cols * (cell_w + PADDING) + PADDING
    grid_h = HEADER_HEIGHT + n_rows * (cell_h + PADDING) + PADDING

    grid = Image.new("RGB", (grid_w, grid_h), "white")
    draw = ImageDraw.Draw(grid)
    font = ImageFont.load_default()

    # Cabeceras de columna (nombre de cada vista)
    for c, view in enumerate(views):
        x = ROW_LABEL_WIDTH + c * (cell_w + PADDING) + PADDING
        header_text = view_labels.get(view, f"Vista {view}")
        draw_centered_text(draw, (x, PADDING), header_text, font, cell_w)

    # Filas: una por paciente
    for r, patient_id in enumerate(patients):
        y = HEADER_HEIGHT + r * (cell_h + PADDING) + PADDING

        for c, view in enumerate(views):
            x = ROW_LABEL_WIDTH + c * (cell_w + PADDING) + PADDING

            if view in entries[patient_id]:
                png_path, meta = entries[patient_id][view]
                thumb = Image.open(png_path).convert("RGB")
                thumb.thumbnail((THUMB_SIZE, THUMB_SIZE))
                # centra la miniatura si no es cuadrada tras el resize
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
    parser.add_argument("--input-dir", type=str, default=None,
                         help="Carpeta con los heatmaps (default: heatmaps/, la misma que usa infer.py)")
    parser.add_argument("--output", type=str, default=None,
                         help="Ruta de la imagen final (default: heatmaps/grid.png)")
    parser.add_argument("--patients", type=str, default=None,
                         help="Filtra solo estos pacientes, p.ej. '1-10' o '001,003,007'. "
                              "Por defecto incluye TODOS los encontrados en input-dir")
    parser.add_argument("--view-labels", type=str, default="1=AP,2=Lateral",
                         help="Etiquetas de cabecera por vista, p.ej. '1=AP,2=Lateral'")
    args = parser.parse_args()

    heatmaps_dir = Path(args.input_dir) if args.input_dir else HEATMAPS_DIR
    out_path = Path(args.output) if args.output else heatmaps_dir / "grid.png"

    entries = load_entries(heatmaps_dir)

    if not entries:
        raise SystemExit(
            f"No se encontró ningún heatmap en {heatmaps_dir}. "
            f"Genera algunos primero con: python infer.py --patient <id> --heatmap "
            f"o python batch_heatmaps.py --patients <rango>"
        )

    if args.patients:
        from infer import normalize_patient_id
        wanted = set()
        for chunk in args.patients.split(","):
            chunk = chunk.strip()
            if "-" in chunk and chunk.replace("-", "").isdigit():
                start, end = chunk.split("-")
                wanted.update(normalize_patient_id(str(i)) for i in range(int(start), int(end) + 1))
            else:
                wanted.add(normalize_patient_id(chunk))

        filtered = {pid: v for pid, v in entries.items() if pid in wanted}
        if not filtered:
            raise SystemExit(
                f"Hay {len(entries)} paciente(s) con heatmaps en {heatmaps_dir}, "
                f"pero ninguno coincide con --patients {args.patients}. "
                f"Pacientes disponibles: {sorted(entries.keys(), key=natural_key)}. "
                f"Genera los que faltan con: python batch_heatmaps.py --patients {args.patients}"
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