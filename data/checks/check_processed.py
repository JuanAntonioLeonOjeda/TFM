"""Muestra 6 imágenes de images.npy con su etiqueta.

Uso: python -m data.checks.check_processed
"""

import subprocess

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data.df_import import CLASS_NAMES, PROC_DIR as PROC

images = np.load(PROC / "images.npy")
meta = pd.read_csv(PROC / "metadata.csv")

print("Array:", images.shape, images.dtype)
print("Metadata:", meta.shape)
print(meta.head())

fig, axes = plt.subplots(2, 3, figsize=(10, 7))
for ax, i in zip(axes.flat, range(6)):
    ax.imshow(images[i], cmap="gray")
    fila = meta.iloc[i]
    ax.set_title(f"ID {fila['patient_id']} - {CLASS_NAMES[fila['label']]}")
    ax.axis("off")
plt.tight_layout()
plt.savefig("check_processed.png", dpi=120)
print("Guardado en check_processed.png")

subprocess.run(["open", "check_processed.png"])