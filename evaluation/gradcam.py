"""
gradcam_utils.py
================
Interpretabilidad médica para LumosNet mediante la librería pytorch-grad-cam
(pip install grad-cam).

LumosNet.forward() devuelve una TUPLA (logits, bmd), no un tensor único como
espera por defecto la librería. Por eso definimos dos "targets" personalizados:
uno para explicar la cabeza de clasificación y otro para la de regresión BMD.

LumosNet también acepta un segundo argumento opcional `covariates`
(forward(x, covariates=None)). Si el modelo se creó con num_covariates > 0,
covariates es OBLIGATORIO en cada forward. Los wrappers de abajo lo llevan
"fijado" desde su construcción, porque pytorch-grad-cam solo llama a
model(input_tensor) con un único argumento.

NUEVO: se separa el cálculo del mapa de calor CRUDO (compute_grayscale_cam,
un array HxW en [0,1]) de su dibujado como overlay (generate_heatmap). El
array crudo es lo que necesita auto_label_bboxes.py para extraer cajas
delimitadoras automáticas (umbralizando el mapa), sin tener que reimplementar
la parte de pytorch-grad-cam.
"""

import torch
import torch.nn as nn
from pytorch_grad_cam import GradCAM, HiResCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


class _ClassificationOutputWrapper(nn.Module):
    """Expone SOLO los logits de clasificación, como un modelo normal (batch, num_classes)."""

    def __init__(self, model, covariates: torch.Tensor | None = None):
        super().__init__()
        self.model = model
        self.covariates = covariates  # fijas para toda la llamada a GradCAM

    def forward(self, x):
        logits, _bmd = self.model(x, self.covariates)
        return logits


class _BMDOutputWrapper(nn.Module):
    """Expone SOLO la predicción de BMD, con forma (batch, 1) -- 'clasificación' de 1 salida."""

    def __init__(self, model, covariates: torch.Tensor | None = None):
        super().__init__()
        self.model = model
        self.covariates = covariates

    def forward(self, x):
        _logits, bmd = self.model(x, self.covariates)
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
    """
    base = getattr(model, "model", model)  # desenvuelve el wrapper si lo hay
    return base.backbone.layer4[-1]


def compute_grayscale_cam(
    model,
    image_tensor: torch.Tensor,
    target: str = "classification",
    class_idx: int | None = None,
    use_hires: bool = True,
    covariates: torch.Tensor | None = None,
):
    """
    Calcula el mapa de calor CRUDO (sin overlay), como array HxW en [0,1].
    Es la pieza reutilizable tanto para generate_heatmap() (visualización)
    como para auto_label_bboxes.py (extracción de cajas delimitadoras).

    Devuelve: (grayscale_cam: np.ndarray HxW float32 en [0,1], class_idx_usado_o_None)
    """
    device = next(model.parameters()).device
    image_tensor = image_tensor.to(device)
    if covariates is not None:
        covariates = covariates.to(device)

    if model.num_covariates > 0 and covariates is None:
        raise ValueError(
            "Este modelo se creó con num_covariates > 0, pero no se recibió "
            "covariates. Pásalas igual que al hacer predict()."
        )

    if target == "classification":
        if class_idx is None:
            with torch.no_grad():
                logits, _ = model(image_tensor, covariates)
                class_idx = int(torch.argmax(logits, dim=1)[0].item())
        wrapper = _ClassificationOutputWrapper(model, covariates).to(device).eval()
        cam_target = ClassifierOutputTarget(class_idx)
        used_class_idx = class_idx

    elif target == "bmd":
        wrapper = _BMDOutputWrapper(model, covariates).to(device).eval()
        cam_target = _RegressionTarget()
        used_class_idx = None

    else:
        raise ValueError("target debe ser 'classification' o 'bmd'")

    CAMClass = HiResCAM if use_hires else GradCAM
    target_layer = get_target_layer(model)

    with CAMClass(model=wrapper, target_layers=[target_layer]) as cam:
        grayscale_cam = cam(input_tensor=image_tensor, targets=[cam_target])
        grayscale_cam = grayscale_cam[0, :]  # quita la dimensión de batch

    return grayscale_cam, used_class_idx


def generate_heatmap(
    model,
    image_tensor: torch.Tensor,
    base_img_rgb_float01,
    target: str = "classification",
    class_idx: int | None = None,
    use_hires: bool = True,
    covariates: torch.Tensor | None = None,
):
    """
    model: LumosNet "desnudo" (sin envolver), ya cargado con pesos, en eval().
    image_tensor: (1, 3, 224, 224) normalizado (salida de tu preprocess()).
    base_img_rgb_float01: imagen RGB en float [0,1], shape (224, 224, 3).
    target: "classification" o "bmd".
    class_idx: para "classification", índice de clase a explicar. Si None, usa la clase predicha.
    use_hires: True usa HiResCAM (más fiel matemáticamente); False usa GradCAM clásico.
    covariates: (1, num_covariates) normalizado igual que en entrenamiento.
               OBLIGATORIO si el modelo se creó con num_covariates > 0.

    Devuelve: (imagen_overlay uint8 HxWx3, class_idx_usado_o_None)
    """
    grayscale_cam, used_class_idx = compute_grayscale_cam(
        model, image_tensor, target=target, class_idx=class_idx,
        use_hires=use_hires, covariates=covariates,
    )
    overlay = show_cam_on_image(base_img_rgb_float01, grayscale_cam, use_rgb=True)
    return overlay, used_class_idx