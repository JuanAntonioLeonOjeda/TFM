FROM python:3.13-slim

# Librerías del sistema que necesita OpenCV
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY api ./api
COPY data ./data
COPY evaluation ./evaluation
COPY models ./models
COPY training ./training
COPY infer.py .

# El modelo (best_phased_full_covars.pt y su JSON de covariables) se monta como volumen en /models
# y las radiografías de ejemplo (python -m api.examples) en /examples, que es opcional
ENV MODEL_DIR=/models \
    EXAMPLES_DIR=/examples \
    MPLBACKEND=Agg \
    PYTHONUNBUFFERED=1
VOLUME /models

RUN useradd --create-home appuser
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
