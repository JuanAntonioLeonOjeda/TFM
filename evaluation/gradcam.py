"""
gradcam.py
================
Interpretabilidad médica para LumosNet mediante la librería pytorch-grad-cam
(pip install grad-cam).

PROBLEMA que resuelve este archivo:
pytorch-grad-cam asume internamente que model(x) devuelve UN SOLO tensor de
forma (batch, ...), y reparte ese tensor entre los "targets" haciendo
zip(targets, outputs) a lo largo de la dimensión de batch.

LumosNet.forward() devuelve una TUPLA (logits, bmd) -- dos cabezas, no dos
muestras de un batch -- lo cual rompe esa asunción interna (la librería
malinterpreta cada elemento de la tupla como si fuera una muestra distinta).

La solución es envolver el modelo en un nn.Module "de un solo tensor de
salida" según qué cabeza queramos explicar, y dejar que la librería
funcione exactamente como espera.
"""

import torch
import torch.nn as nn
from pytorch_grad_cam import GradCAM, HiResCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


class _ClassificationOutputWrapper(nn.Module):
    """Expone SOLO los logits de clasificación, como un modelo normal (batch, num_classes)."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        logits, _bmd = self.model(x)
        return logits


class _BMDOutputWrapper(nn.Module):
    """Expone SOLO la predicción de BMD, con forma (batch, 1) -- 'clasificación' de 1 salida."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        _logits, bmd = self.model(x)
        if bmd.dim() == 1:
            bmd = bmd.unsqueeze(1)  # (batch,) -> (batch, 1)
        return bmd


class _RegressionTarget:
    """Target para explicar una salida de regresión de una sola neurona."""

    def __call__(self, model_output):
        return model_output[0] if model_output.dim() > 0 else model_output


def get_target_layer(model):
    """
    Capa recomendada para Grad-CAM en LumosNet: el último bloque residual
    completo de ResNet50 (incluye BatchNorm + suma residual + ReLU),
    NO una convolución aislada.

    Funciona igual tanto si `model` es el LumosNet "desnudo" como si es uno
    de los wrappers de arriba, porque wrapper.model es literalmente el
    mismo objeto LumosNet (el hook se registra sobre el submódulo real).
    """
    base = getattr(model, "model", model)  # desenvuelve el wrapper si lo hay
    return base.backbone.layer4[-1]


def generate_heatmap(
    model,
    image_tensor: torch.Tensor,
    base_img_rgb_float01,
    target: str = "classification",
    class_idx: int | None = None,
    use_hires: bool = True,
):
    """
    model: LumosNet "desnudo" (sin envolver), ya cargado con pesos, en eval().
    image_tensor: (1, 3, 224, 224) normalizado (salida de tu preprocess()).
    base_img_rgb_float01: imagen RGB en float [0,1], shape (224, 224, 3).
    target: "classification" o "bmd".
    class_idx: para "classification", índice de clase a explicar
               (0=Sano, 1=Osteopenia, 2=Osteoporosis). Si None, usa la clase predicha.
    use_hires: True usa HiResCAM (más fiel matemáticamente); False usa GradCAM clásico.

    Devuelve: (imagen_overlay uint8 HxWx3, class_idx_usado_o_None)
    """
    device = next(model.parameters()).device
    image_tensor = image_tensor.to(device)

    if target == "classification":
        if class_idx is None:
            with torch.no_grad():
                logits, _ = model(image_tensor)
                class_idx = int(torch.argmax(logits, dim=1)[0].item())
        wrapper = _ClassificationOutputWrapper(model).to(device).eval()
        cam_target = ClassifierOutputTarget(class_idx)
        used_class_idx = class_idx

    elif target == "bmd":
        wrapper = _BMDOutputWrapper(model).to(device).eval()
        cam_target = _RegressionTarget()
        used_class_idx = None

    else:
        raise ValueError("target debe ser 'classification' o 'bmd'")

    CAMClass = HiResCAM if use_hires else GradCAM
    target_layer = get_target_layer(model)

    with CAMClass(model=wrapper, target_layers=[target_layer]) as cam:
        grayscale_cam = cam(input_tensor=image_tensor, targets=[cam_target])
        grayscale_cam = grayscale_cam[0, :]  # quita la dimensión de batch

    overlay = show_cam_on_image(base_img_rgb_float01, grayscale_cam, use_rgb=True)
    return overlay, used_class_idx