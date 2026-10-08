"""Grad-CAM / HiResCAM para LumosNet (pip install grad-cam)."""

import torch
import torch.nn as nn
from pytorch_grad_cam import GradCAM, HiResCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


# pytorch-grad-cam llama a model(x) con un solo tensor de salida y un solo argumento.
# Estos wrappers exponen una de las dos cabezas y llevan las covariables fijadas.
class _ClassificationOutputWrapper(nn.Module):
    def __init__(self, model, covariates: torch.Tensor | None = None):
        super().__init__()
        self.model = model
        self.covariates = covariates

    def forward(self, x):
        logits, _ = self.model(x, self.covariates)
        return logits


class _BMDOutputWrapper(nn.Module):
    def __init__(self, model, covariates: torch.Tensor | None = None):
        super().__init__()
        self.model = model
        self.covariates = covariates

    def forward(self, x):
        _, bmd = self.model(x, self.covariates)
        return bmd.unsqueeze(1) if bmd.dim() == 1 else bmd


class _RegressionTarget:
    def __call__(self, model_output):
        return model_output[0] if model_output.dim() > 0 else model_output


def get_target_layer(model):
    """Último bloque residual completo de ResNet50, no una convolución aislada."""
    base = getattr(model, "model", model)
    return base.backbone.layer4[-1]


def compute_grayscale_cam(
    model,
    image_tensor: torch.Tensor,
    target: str = "classification",
    class_idx: int | None = None,
    use_hires: bool = True,
    covariates: torch.Tensor | None = None,
):
    """Mapa de calor crudo HxW en [0,1]. Devuelve (cam, class_idx usado o None si target='bmd')."""
    device = next(model.parameters()).device
    image_tensor = image_tensor.to(device)
    if covariates is not None:
        covariates = covariates.to(device)

    if model.num_covariates > 0 and covariates is None:
        raise ValueError("Modelo creado con num_covariates > 0: hay que pasar covariates.")

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

    cam_class = HiResCAM if use_hires else GradCAM
    with cam_class(model=wrapper, target_layers=[get_target_layer(model)]) as cam:
        grayscale_cam = cam(input_tensor=image_tensor, targets=[cam_target])[0, :]

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
    """Overlay RGB uint8 del mapa sobre base_img_rgb_float01 (HxWx3 en [0,1]). Devuelve (overlay, class_idx)."""
    grayscale_cam, used_class_idx = compute_grayscale_cam(
        model, image_tensor, target=target, class_idx=class_idx,
        use_hires=use_hires, covariates=covariates,
    )
    overlay = show_cam_on_image(base_img_rgb_float01, grayscale_cam, use_rgb=True)
    return overlay, used_class_idx
