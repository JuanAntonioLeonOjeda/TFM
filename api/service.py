import json
import tempfile
from pathlib import Path

import numpy as np
import torch

from data.dataloaders import MEDICAL_COVARIATES
from data.df_import import CLASS_NAMES
from data.lumos_dataset import normalize_covariates
from infer import load_model, predict, preprocess
from models.lumosnet import get_device
from training.train import covariate_stats_path, run_tag

MODEL_TAG = run_tag("phased", "full", use_covariates=True)


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

    def _covariates(self, age: float, gender: str, bmi: float) -> torch.Tensor:
        row = {"age": age, "gender": self.gender_codes[gender.lower()], "BMI": bmi}
        values = normalize_covariates(row, MEDICAL_COVARIATES, self.stats)
        return torch.tensor(values, dtype=torch.float32).unsqueeze(0)

    def predict_dicom(self, dicom_bytes: bytes, age: float, gender: str, bmi: float) -> dict:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "upload.dcm"
            path.write_bytes(dicom_bytes)
            image_tensor, _ = preprocess(str(path))

        probs, bmd = predict(self.model, image_tensor, self.device, self._covariates(age, gender, bmi))
        return {
            "predicted_class": CLASS_NAMES[int(np.argmax(probs))],
            "probabilities": {name: round(float(p), 4) for name, p in zip(CLASS_NAMES, probs)},
            "bmd_estimated": round(float(bmd), 4),
        }
