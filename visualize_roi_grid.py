"""
visualize_roi_grid.py
======================
Muestra una muestra de radiografías AP (en su resolución ORIGINAL, antes del
resize a 224) con una rejilla vertical superpuesta, para decidir a ojo dónde
recortar la columna vertebral (ROI heurístico).

Uso:
    python visualize_roi_grid.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data.df_import import read_image   # AJUSTA al nombre real de tu módulo

ROOT = Path(__file__).resolve().parent
PROC_DIR = ROOT / "res" / "processed"

# líneas verticales de referencia, como fracción del ancho de la imagen
GRID_FRACTIONS = [0.30, 0.40, 0.50, 0.60, 0.70]
N_SAMPLES = 9
SEED = 20


def main():
    ap = pd.read_csv(PROC_DIR / "ap_only.csv")
    sample = ap.sample(min(N_SAMPLES, len(ap)), random_state=SEED)

    n = len(sample)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 6 * rows))
    axes = np.atleast_1d(axes).flatten()

    for ax, (_, row) in zip(axes, sample.iterrows()):
        img = read_image(row["image_path"])   # array (H, W), float [0,1]
        h, w = img.shape

        ax.imshow(img, cmap="gray")
        for frac in GRID_FRACTIONS:
            x = frac * w
            ax.axvline(x, color="red", linestyle="--", linewidth=0.8, alpha=0.7)
            ax.text(x, 15, f"{frac:.2f}", color="red", fontsize=8,
                    ha="center", va="top")

        ax.set_title(f"Patient {row['patient_id']}  ({w}x{h})", fontsize=10)
        ax.axis("off")

    for ax in axes[n:]:
        ax.axis("off")

    plt.tight_layout()
    out_path = ROOT / "roi_grid_preview.png"
    plt.savefig(out_path, dpi=110, bbox_inches="tight")
    print(f"Guardado: {out_path}")

    import subprocess
    subprocess.run(["open", str(out_path)])


if __name__ == "__main__":
    main()