# TFM – Osteoporosis en radiografías lumbares (LUMOS)

LumosNet (ResNet50) multitarea: clasificación Sano / Osteopenia / Osteoporosis + regresión de BMD.
Todos los comandos se lanzan desde la raíz del repo.

## Estructura

```
data/         carga DICOM, preprocesado, Dataset y DataLoaders
  variants/   un script por variante de dataset (ROI, ROI híbrido, ROI YOLO)
  checks/     scripts exploratorios y de comprobación
models/       LumosNet
training/     bucle de entrenamiento + estrategias de transfer learning
evaluation/   evaluación en test, validación cruzada, Grad-CAM
yolo/         pseudo-etiquetado con Grad-CAM y revisión manual del detector
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
