import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent   # sube de data/ a tfm/
PROC = ROOT / "res" / "processed"

images = np.load(PROC / "images.npy")
meta = pd.read_csv(PROC / "metadata.csv")

print("Array:", images.shape, images.dtype)   # (1616, 224, 224) uint8
print("Metadata:", meta.shape)
print(meta.head())

# ver 6 imágenes con su etiqueta
names = {0: "Sano", 1: "Osteopenia", 2: "Osteoporosis"}
fig, axes = plt.subplots(2, 3, figsize=(10, 7))
for ax, i in zip(axes.flat, range(6)):
    ax.imshow(images[i], cmap="gray")
    fila = meta.iloc[i]
    ax.set_title(f"ID {fila['patient_id']} - {names[fila['label']]}")
    ax.axis("off")
plt.tight_layout()
plt.savefig("check_processed.png", dpi=120)
print("Guardado en check_processed.png")

import subprocess; subprocess.run(["open", "check_processed.png"])