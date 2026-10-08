"""
roi_crop.py
===========
Recorte heurístico de la región de interés (ROI) para radiografías AP de
columna lumbar. Se queda con una franja vertical central (25%-75% del ancho),
sin recortar en altura.

Este recorte se aplica sobre la imagen en RESOLUCIÓN ORIGINAL, antes de
redimensionar a 224x224.

Limitación conocida: en pacientes con escoliosis pronunciada, la columna
puede desviarse fuera de esta franja en parte de su recorrido, por lo que
el recorte puede perder parte de la anatomía en esos casos.
"""

import numpy as np

LEFT_FRACTION = 0.25
RIGHT_FRACTION = 0.75


def crop_roi(img: np.ndarray) -> np.ndarray:
    """
    Recorta la franja vertical central de la imagen.
    img: array 2D (H, W).
    Devuelve el recorte (H, W_recortado).
    """
    h, w = img.shape
    x0 = int(w * LEFT_FRACTION)
    x1 = int(w * RIGHT_FRACTION)
    return img[:, x0:x1]