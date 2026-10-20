import tempfile
from pathlib import Path

import numpy as np
import torch

from data.dataloaders import MEDICAL_COVARIATES
from data.df_import import CLASS_NAMES
from data.lumos_dataset import normalize_covariates
from infer import build_covariate_lookup, load_model, predict, preprocess
from models.lumosnet import get_device

# Codificación alfabética usada en el entrenamiento: Female = 0, Male = 1
GENDER_CODES = {"female": 0, "male": 1}


# Carga una sola vez el modelo con covariables y las estadísticas de train para normalizarlas
class PredictionService:
    def __init__(self):
        self.device = get_device()
        self.model, self.model_path = load_model(self.device, num_covariates=len(MEDICAL_COVARIATES))
        _, self.stats = build_covariate_lookup()

    def _covariates(self, age: float, gender: str, bmi: float) -> torch.Tensor:
        row = {"age": age, "gender": GENDER_CODES[gender.lower()], "BMI": bmi}
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
