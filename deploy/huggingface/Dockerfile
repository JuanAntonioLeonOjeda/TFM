# Imagen para Hugging Face Spaces: API y web en un solo contenedor, en el puerto 7860.
# Se construye desde la carpeta que genera deploy/huggingface/build_space.sh, que trae el código,
# el modelo (outputs/) y las radiografías de ejemplo (res/examples/)
FROM node:24.22.2-alpine AS frontend
WORKDIR /app
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ .
RUN npm run build

FROM python:3.13-slim

# Librerías del sistema que necesita OpenCV
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Spaces ejecuta el contenedor con el uid 1000
RUN useradd --create-home --uid 1000 user
WORKDIR /app

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY api ./api
COPY data ./data
COPY evaluation ./evaluation
COPY models ./models
COPY training ./training
COPY infer.py .
COPY outputs/best_phased_full_covars.pt outputs/best_phased_full_covars_covariates.json /models/
COPY res/examples /examples
COPY --from=frontend /app/dist ./frontend/dist

ENV MODEL_DIR=/models \
    EXAMPLES_DIR=/examples \
    FRONTEND_DIR=/app/frontend/dist \
    MPLBACKEND=Agg \
    PYTHONUNBUFFERED=1

USER user
EXPOSE 7860
CMD ["uvicorn", "api.web:app", "--host", "0.0.0.0", "--port", "7860"]
