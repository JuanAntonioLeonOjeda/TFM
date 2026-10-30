import base64
import io
import json
import threading

import cv2
import numpy as np
import torch

from api.examples import load_examples
from data.dataloaders import MEDICAL_COVARIATES
from data.df_import import CLASS_NAMES
from data.df_preprocess import process_image
from data.lumos_dataset import build_transforms, gray_to_rgb, normalize_covariates
from evaluation.gradcam import generate_heatmap
from infer import load_model, predict
from models.lumosnet import get_device
from training.train import covariate_stats_path, run_tag

MODEL_TAG = run_tag("phased", "full", use_covariates=True)


def _png_base64(img: np.ndarray) -> str:
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    ok, png = cv2.imencode(".png", img)
    return base64.b64encode(png.tobytes()).decode("ascii")


# Carga una sola vez el modelo con covariables y las estadísticas de train para normalizarlas
# (el JSON que se guarda junto al checkpoint)
class PredictionService:
    def __init__(self):
        self.device = get_device()
        self.model, self.model_path = load_model(self.device, num_covariates=len(MEDICAL_COVARIATES))
        stats_path = covariate_stats_path(MODEL_TAG)
        if not stats_path.exists():
            raise FileNotFoundError(
                f"No se encontró {stats_path}. Genéralo con: python -m training.train --covariates --export-stats")
        spec = json.loads(stats_path.read_text(encoding="utf-8"))
        self.stats = {c: (spec["mean"][c], spec["std"][c]) for c in spec["covariates"]}
        self.gender_codes = spec["codes"]["gender"]
        self.examples = {ex["id"]: ex for ex in load_examples()}
        # Grad-CAM registra hooks en el modelo compartido: las peticiones simultáneas se serializan
        self._lock = threading.Lock()

    def _covariates(self, age: float, gender: str, bmi: float) -> torch.Tensor:
        row = {"age": age, "gender": self.gender_codes[gender.lower()], "BMI": bmi}
        values = normalize_covariates(row, MEDICAL_COVARIATES, self.stats)
        return torch.tensor(values, dtype=torch.float32).unsqueeze(0)

    # Mismo preprocesado que infer.preprocess, pero desde bytes: el DICOM subido se procesa
    # en memoria y nunca se escribe en disco
    def _preprocess(self, dicom_bytes: bytes):
        img = process_image(io.BytesIO(dicom_bytes))
        tensor = build_transforms(train=False)(gray_to_rgb(img)).unsqueeze(0)
        return tensor, img

    def predict_dicom(self, dicom_bytes: bytes, age: float, gender: str, bmi: float,
                      heatmap: bool = False) -> dict:
        image_tensor, base_img = self._preprocess(dicom_bytes)
        covariates = self._covariates(age, gender, bmi)
        with self._lock:
            probs, bmd = predict(self.model, image_tensor, self.device, covariates)
            pred_idx = int(np.argmax(probs))
            result = {
                "predicted_class": CLASS_NAMES[pred_idx],
                "probabilities": {name: round(float(p), 4) for name, p in zip(CLASS_NAMES, probs)},
                "bmd_estimated": round(float(bmd), 4),
            }
            if heatmap:
                # Grad-CAM de la clase predicha sobre la imagen preprocesada (224×224 tras CLAHE)
                base_rgb_float = gray_to_rgb(base_img).astype(np.float32) / 255.0
                overlay, _ = generate_heatmap(self.model, image_tensor, base_rgb_float,
                                              class_idx=pred_idx, covariates=covariates)
                result["heatmap_png_base64"] = _png_base64(overlay)
                result["image_png_base64"] = _png_base64(base_img)
        return result

    def list_examples(self) -> list[dict]:
        return [{k: v for k, v in ex.items() if k not in ("file", "path")} for ex in self.examples.values()]

    def example_image_png(self, example_id: str) -> bytes:
        ok, png = cv2.imencode(".png", process_image(str(self.examples[example_id]["path"])))
        return png.tobytes()

    def predict_example(self, example_id: str, heatmap: bool = True) -> dict:
        ex = self.examples[example_id]
        result = self.predict_dicom(ex["path"].read_bytes(), ex["age"], ex["gender"], ex["bmi"], heatmap)
        result["ground_truth"] = {"class": ex["true_class"], "bmd": ex["true_bmd"]}
        return result
