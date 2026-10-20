from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from api.service import PredictionService

services: dict[str, PredictionService] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    services["prediction"] = PredictionService()
    yield
    services.clear()


app = FastAPI(
    title="LumosNet API",
    description="Clasificación Sano / Osteopenia / Osteoporosis y estimación de BMD a partir de una radiografía lumbar.",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {"status": "ok", "model": services["prediction"].model_path.name}


@app.post("/predict")
async def predict_endpoint(
    dicom: UploadFile = File(..., description="Radiografía lumbar en formato DICOM"),
    age: float = Form(..., ge=0, le=120),
    gender: Literal["female", "male"] = Form(...),
    bmi: float = Form(..., gt=0, le=80),
    heatmap: bool = Form(False, description="Incluye el mapa Grad-CAM de la clase predicha (PNG en base64)"),
):
    data = await dicom.read()
    try:
        return services["prediction"].predict_dicom(data, age, gender, bmi, heatmap)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"No se pudo procesar el DICOM: {e}")
