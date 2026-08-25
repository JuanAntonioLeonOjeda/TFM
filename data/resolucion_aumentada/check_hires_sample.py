"""
check_hires_sample.py
======================
Muestra una comparativa visual entre imágenes procesadas a 224x224 y a
320x320, y una muestra general del dataset de mayor resolución.

Requiere tener generados ambos conjuntos:
    res/processed/images.npy      (224x224, baseline)
    res/processed/images_320.npy  (320x320, alta resolución)

Uso:
    python check_hires_sample.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
PROC_DIR = ROOT / "res" / "processed"


def compare_resolutions(n_samples=4, seed=42):
    """Muestra las mismas radiografías a 224 y a 320, lado a lado."""
    images_224 = np.load(PROC_DIR / "images.npy")
    meta_224 = pd.read_csv(PROC_DIR / "metadata.csv")

    images_320 = np.load(PROC_DIR / "images_320.npy")
    meta_320 = pd.read_csv(PROC_DIR / "metadata_320.csv")

    # emparejar por patient_id (el orden puede no coincidir entre los dos npy)
    common_patients = sorted(
        set(meta_224["patient_id"]) & set(meta_320["patient_id"])
    )
    rng = np.random.default_rng(seed)
    sample_patients = rng.choice(common_patients, size=n_samples, replace=False)

    fig, axes = plt.subplots(2, n_samples, figsize=(4 * n_samples, 8))
    names = {0: "Sano", 1: "Osteopenia", 2: "Osteoporosis"}

    for col, pid in enumerate(sample_patients):
        idx_224 = meta_224.index[meta_224["patient_id"] == pid][0]
        idx_320 = meta_320.index[meta_320["patient_id"] == pid][0]

        label = meta_224.loc[idx_224, "label"]

        axes[0, col].imshow(images_224[idx_224], cmap="gray")
        axes[0, col].set_title(f"ID {pid} - {names.get(label,'?')}\n224x224", fontsize=10)
        axes[0, col].axis("off")

        axes[1, col].imshow(images_320[idx_320], cmap="gray")
        axes[1, col].set_title("320x320", fontsize=10)
        axes[1, col].axis("off")

    plt.tight_layout()
    out_path = ROOT / "resolution_comparison.png"
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"Guardado: {out_path}")


def preview_hires_grid(n_samples=9, seed=7):
    """Muestra una rejilla general de imágenes a 320x320, con su etiqueta."""
    images = np.load(PROC_DIR / "images_320.npy")
    meta = pd.read_csv(PROC_DIR / "metadata_320.csv")

    sample = meta.sample(n_samples, random_state=seed)
    names = {0: "Sano", 1: "Osteopenia", 2: "Osteoporosis"}

    cols = 3
    rows = (n_samples + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = axes.flatten()

    for ax, (_, row) in zip(axes, sample.iterrows()):
        img = images[row["idx"]]
        ax.imshow(img, cmap="gray")
        ax.set_title(f"ID {row['patient_id']} - {names.get(row['label'],'?')}", fontsize=10)
        ax.axis("off")

    plt.tight_layout()
    out_path = ROOT / "hires_preview.png"
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"Guardado: {out_path}")


if __name__ == "__main__":
    print("Generando comparativa 224 vs 320...")
    compare_resolutions()

    print("\nGenerando muestra general a 320x320...")
    preview_hires_grid()

    print("\nListo.")