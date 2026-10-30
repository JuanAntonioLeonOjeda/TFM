# TFM – Osteoporosis en radiografías lumbares (LUMOS)

LumosNet (ResNet50) multitarea: clasificación Sano / Osteopenia / Osteoporosis + regresión de BMD.
Todos los comandos se lanzan desde la raíz del repo.

## MAIN DEPENDENCIES
![Python](https://img.shields.io/badge/PYTHON-3.13-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PYTORCH-2.13.0-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Torchvision](https://img.shields.io/badge/TORCHVISION-0.28.0-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![NumPy](https://img.shields.io/badge/NUMPY-2.5.1-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/PANDAS-3.0.5-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/SCIKIT--LEARN-1.9.0-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![OpenCV](https://img.shields.io/badge/OPENCV-5.0.0.93-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![Pydicom](https://img.shields.io/badge/PYDICOM-3.0.2-blue?style=for-the-badge&logo=python&logoColor=white)
![Matplotlib](https://img.shields.io/badge/MATPLOTLIB-3.11.1-11557c?style=for-the-badge&logo=python&logoColor=white)
![Grad-CAM](https://img.shields.io/badge/GRAD--CAM-1.5.7-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Ultralytics](https://img.shields.io/badge/ULTRALYTICS-8.4.161-111F68?style=for-the-badge&logo=yolo&logoColor=white)
![FastAPI](https://img.shields.io/badge/FASTAPI-0.151.0-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Uvicorn](https://img.shields.io/badge/UVICORN-0.54.0-499848?style=for-the-badge&logo=gunicorn&logoColor=white)
![React](https://img.shields.io/badge/REACT-19.3.1-61DAFB?style=for-the-badge&logo=react&logoColor=black)
![React Router](https://img.shields.io/badge/REACT%20ROUTER-8.5.1-CA4245?style=for-the-badge&logo=reactrouter&logoColor=white)
![Vite](https://img.shields.io/badge/VITE-8.3.5-646CFF?style=for-the-badge&logo=vite&logoColor=white)
![TypeScript](https://img.shields.io/badge/TYPESCRIPT-7.0.2-3178C6?style=for-the-badge&logo=typescript&logoColor=white)
![Nginx](https://img.shields.io/badge/NGINX-1.30.6-009639?style=for-the-badge&logo=nginx&logoColor=white)
![Docker](https://img.shields.io/badge/DOCKER-29.8.2-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![Pytest](https://img.shields.io/badge/PYTEST-9.1.1-0A9EDC?style=for-the-badge&logo=pytest&logoColor=white)

## Estructura

```
data/         carga DICOM, preprocesado, Dataset y DataLoaders
  variants/   un script por variante de dataset (ROI, ROI híbrido, ROI YOLO)
  checks/     scripts exploratorios y de comprobación
models/       LumosNet
training/     bucle de entrenamiento + estrategias de transfer learning
evaluation/   evaluación en test, validación cruzada, Grad-CAM
yolo/         pseudo-etiquetado con Grad-CAM y revisión manual del detector
api/          backend FastAPI
frontend/     interfaz web (React + Vite), servida por nginx
infer.py      predicción sobre un DICOM / paciente
```

## Pipeline

```bash
python -m data.prepare_data              # full: images.npy + metadata.csv
python -m training.train --data full     # + --strategy, --class-weighting, --covariates
python -m evaluation.evaluate --data full
```

## Variantes de dataset (`--data`)

| `--data`     | Cómo se genera                              | Contenido                                   |
|--------------|---------------------------------------------|---------------------------------------------|
| `full`       | `python -m data.df_preprocess`              | ambas vistas, 224×224                        |
| `hires`      | `python -m data.df_preprocess --size 320`   | ambas vistas, 320×320                        |
| `roi`        | `python -m data.views` → `data.variants.roi` | solo AP, franja central 25–75%              |
| `roi_hybrid` | `python -m data.views` → `data.variants.roi_hybrid` | ambas vistas, ROI solo en AP         |
| `roi_yolo`   | `yolo/` → `python -m data.variants.roi_yolo` | ambas vistas, recorte YOLO en AP (si conf ≥ 0.5) |

El resto de variaciones de los experimentos son flags de `training.train` (`--strategy`,
`--class-weighting`, `--covariates`); el checkpoint se guarda como `outputs/best_<tag>.pt`.

## Detector YOLO

```bash
python -m yolo.auto_label         # cajas a partir de Grad-CAM -> res/yolo_dataset/
python -m yolo.preview_bboxes     # revisión visual
python -m yolo.export_review      # preparar para makesense.ai
python -m yolo.reintegrate --csv export.csv
yolo detect train model=yolov8n.pt data=res/yolo_dataset/dataset.yaml epochs=100 imgsz=224 \
    project=res/yolo_dataset name=lumbar_detector
```

## API

Servicio FastAPI que carga `best_phased_full_covars.pt` y su `best_phased_full_covars_covariates.json`
(se genera al entrenar con `--covariates`, o con `python -m training.train --covariates --export-stats`).
La carpeta de los modelos se configura con `MODEL_DIR` (por defecto `outputs/`).

La interfaz web (`frontend/`, React) permite subir un DICOM (con edad, sexo e IMC) o elegir una
radiografía de ejemplo, y muestra la clase predicha, las probabilidades, la BMD estimada y el mapa
Grad-CAM de las zonas en las que se fijó el modelo. Avisa cuando la predicción es dudosa (las dos clases
más probables a menos de 10 puntos) y muestra la exactitud del modelo en test.

En `/proyecto` hay una explicación del TFM en 11 capítulos (problema, estado del arte, datos, modelo,
entrenamiento, experimentos, modelo final, interpretabilidad, despliegue y conclusiones), basada en la
memoria, con gráficas interactivas y Grad-CAM calculado en vivo sobre los ejemplos. Las métricas de cada
experimento están en `frontend/src/docs/metrics.ts` (solo cifras agregadas); las radiografías no van en
la web, se piden a la API.

Los ejemplos son pacientes del split de test de LUMOS y se preparan con `python -m api.examples` en
`EXAMPLES_DIR` (por defecto `res/examples/`); si no existen, la pestaña aparece vacía. El script predice
todo el split de test, guarda las métricas en el manifiesto y elige por clase 2 aciertos y 1 fallo entre
clases vecinas (`--correct`, `--errors`), para que los ejemplos reflejen cómo se comporta el modelo. Hay que
volver a ejecutarlo al cambiar de modelo.

Privacidad: las subidas se procesan solo en memoria (sin ficheros temporales ni en nginx ni en el
backend), no se registran ni se usan para entrenar, y las respuestas de la API llevan `Cache-Control: no-store`.

| Método | Ruta                      | Descripción                                               |
|--------|---------------------------|-----------------------------------------------------------|
| POST   | `/predict`                | DICOM + `age`, `gender`, `bmi` (+ `heatmap`)               |
| GET    | `/examples`               | ejemplos con sus covariables y diagnóstico real            |
| GET    | `/examples/{id}/image`    | radiografía preprocesada (PNG)                             |
| POST   | `/examples/{id}/predict`  | predicción + heatmap + verdad terreno                      |
| GET    | `/model-info`             | exactitud y matriz de confusión del modelo en test         |
| GET    | `/privacy`, `/health`     | aviso de privacidad, estado                                |

### Docker (backend + frontend)

nginx sirve el frontend y redirige `/api/*` al backend, que no se expone fuera de la red de Compose.

```bash
python -m api.examples                    # 2 aciertos + 1 fallo por clase -> res/examples/
docker compose up --build                 # http://localhost:8080
# otras rutas o puerto:
MODEL_DIR=/ruta/a/modelos EXAMPLES_DIR=/ruta/a/examples PORT=9000 docker compose up --build
```

### Desarrollo

```bash
uvicorn api.main:app --port 8000          # documentación en /docs
cd frontend && npm install && npm run dev # http://localhost:5173, /api -> localhost:8000
pytest tests

curl -X POST localhost:8000/predict -F dicom=@radiografia.dcm \
     -F age=65 -F gender=female -F bmi=24.5 -F heatmap=true
```
