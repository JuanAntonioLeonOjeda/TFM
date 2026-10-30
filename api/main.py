from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import Response
from starlette.formparsers import MultiPartParser

from api.service import PredictionService

MAX_UPLOAD_MB = 50

# Por defecto Starlette vuelca a un fichero temporal las subidas de más de 1 MB. Con esto
# la radiografía se queda en memoria durante toda la petición y nunca toca el disco
MultiPartParser.spool_max_size = MAX_UPLOAD_MB * 1024 * 1024

PRIVACY_NOTICE = ("Las radiografías y los datos que se envían se procesan solo en memoria para "
                  "devolver la predicción: no se guardan, no se registran y no se usan para entrenar el modelo.")

services: dict[str, PredictionService] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    services["prediction"] = PredictionService()
    yield
    services.clear()


app = FastAPI(
    title="LumosNet API",
    description="Clasificación Sano / Osteopenia / Osteoporosis y estimación de BMD a partir de una "
                f"radiografía lumbar.\n\n**Privacidad:** {PRIVACY_NOTICE}\n\n"
                "Herramienta de investigación (TFM): no es un producto sanitario ni sustituye a una DXA.",
    lifespan=lifespan,
)


# Ni el navegador ni los proxies intermedios deben guardar copia de las predicciones
@app.middleware("http")
async def no_store(request: Request, call_next):
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-store"
    return response


@app.get("/health")
def health():
    return {"status": "ok", "model": services["prediction"].model_path.name}


@app.get("/privacy")
def privacy():
    return {"notice": PRIVACY_NOTICE}


@app.get("/model-info")
def model_info():
    """Rendimiento del modelo en el split de test de LUMOS (se calcula con python -m api.examples)."""
    return {"model": services["prediction"].model_path.name,
            "test_metrics": services["prediction"].test_metrics}


@app.post("/predict")
async def predict_endpoint(
    dicom: UploadFile = File(..., description="Radiografía lumbar en formato DICOM"),
    age: float = Form(..., ge=0, le=120),
    gender: Literal["female", "male"] = Form(...),
    bmi: float = Form(..., gt=0, le=80),
    heatmap: bool = Form(False, description="Incluye el mapa Grad-CAM de la clase predicha y la imagen "
                                            "preprocesada (PNG en base64)"),
):
    if dicom.size is not None and dicom.size > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"El archivo supera el límite de {MAX_UPLOAD_MB} MB")
    data = await dicom.read()
    try:
        return await run_in_threadpool(services["prediction"].predict_dicom, data, age, gender, bmi, heatmap)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"No se pudo procesar el DICOM: {e}")
    finally:
        del data
        await dicom.close()


def _get_example(example_id: str) -> str:
    if example_id not in services["prediction"].examples:
        raise HTTPException(status_code=404, detail=f"No existe el ejemplo '{example_id}'")
    return example_id


@app.get("/examples")
def list_examples():
    """Radiografías del split de test de LUMOS, con sus covariables y su diagnóstico real."""
    return services["prediction"].list_examples()


@app.get("/examples/{example_id}/image", response_class=Response,
         responses={200: {"content": {"image/png": {}}}})
async def example_image(example_id: str):
    png = await run_in_threadpool(services["prediction"].example_image_png, _get_example(example_id))
    return Response(png, media_type="image/png")


@app.post("/examples/{example_id}/predict")
async def predict_example(example_id: str, heatmap: bool = True):
    return await run_in_threadpool(services["prediction"].predict_example, _get_example(example_id), heatmap)
