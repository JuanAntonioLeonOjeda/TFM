"""
infer.py
========
Script de INFERENCIA: carga el modelo ya entrenado (best_phased_full.pt) y
lo usa para predecir sobre una radiografía DICOM nueva, fuera del proceso
de entrenamiento.

Esto es lo mínimo necesario para "desplegar" el modelo:
  1. La arquitectura (models/resnet.py)
  2. Los pesos entrenados (outputs/best_phased_full.pt)
  3. El mismo preprocesamiento usado en entrenamiento (CLAHE, resize, CLAHE,
     normalización ImageNet)

Uso:
    python infer.py --dicom "ruta/a/una/radiografia.dcm"

Salida:
    - Clase predicha (Sano / Osteopenia / Osteoporosis) con su probabilidad
    - Valor de BMD predicho
"""

import argparse
from pathlib import Path

import numpy as np
import cv2
import torch
import torchvision.transforms as T

from data.loader import read_image          # AJUSTA al nombre real de tu módulo
from models.resnet import LumosNet, get_device

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "outputs" / "best_phased_full.pt"

IMG_SIZE = 224
CLASS_NAMES = ["Sano", "Osteopenia", "Osteoporosis"]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def load_model(device):
    """Reconstruye la arquitectura y carga los pesos entrenados."""
    model = LumosNet(pretrained=False)   # arquitectura vacía: los pesos vienen del .pt
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    return model


def preprocess(dicom_path: str) -> torch.Tensor:
    """
    Aplica EXACTAMENTE el mismo preprocesamiento que en entrenamiento:
    DICOM -> uint8 -> resize 224x224 -> CLAHE -> tensor normalizado.
    """
    img = read_image(dicom_path)                      # float32 [0,1], corrige MONOCHROME1
    img = (img * 255).astype(np.uint8)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)

    rgb = np.stack([img, img, img], axis=-1)           # 1 canal -> 3 canales

    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    tensor = transform(rgb)
    return tensor.unsqueeze(0)                         # añade la dimensión de batch


def predict(model, image_tensor, device):
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        logits, bmd_pred = model(image_tensor)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        bmd = bmd_pred.item()
    return probs, bmd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dicom", type=str, required=True, help="Ruta a una radiografía DICOM")
    args = parser.parse_args()

    device = get_device()
    print(f"Cargando modelo desde: {MODEL_PATH}")
    print(f"Device: {device}\n")

    model = load_model(device)
    image_tensor = preprocess(args.dicom)
    probs, bmd = predict(model, image_tensor, device)

    pred_class = int(np.argmax(probs))

    print(f"Radiografía: {args.dicom}\n")
    print("--- Predicción de clasificación ---")
    for name, p in zip(CLASS_NAMES, probs):
        marker = " <-- predicción" if name == CLASS_NAMES[pred_class] else ""
        print(f"  {name:13s}: {p*100:5.1f}%{marker}")

    print(f"\n--- Predicción de regresión ---")
    print(f"  BMD estimado: {bmd:.3f} g/cm²")


if __name__ == "__main__":
    main()