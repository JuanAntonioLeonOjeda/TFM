import type { CSSProperties } from 'react'
import { CLASS_NAMES, type Prediction } from '../api'
import { CLASS_COLORS, fmt } from '../format'

const color = (c: string) => ({ '--c': c }) as CSSProperties

interface Props {
  result: Prediction
  exampleId: string | null
}

export function ResultPanel({ result, exampleId }: Props) {
  const { predicted_class, probabilities, bmd_estimated, ground_truth } = result

  return (
    <div>
      <h2>Resultado</h2>
      <div className="summary">
        <div>
          <div className="kpi-label">Clase predicha</div>
          <div className="big" style={color(CLASS_COLORS[predicted_class])}>{predicted_class}</div>
        </div>
        <div>
          <div className="kpi-label">BMD estimada</div>
          <div className="big">{fmt(bmd_estimated)} <small>g/cm²</small></div>
        </div>
      </div>

      <div className="bars">
        {CLASS_NAMES.map((name) => (
          <div key={name} className="bar" style={color(CLASS_COLORS[name])}>
            <span>{name}</span>
            <div className="track"><div className="fill" style={{ width: `${probabilities[name] * 100}%` }} /></div>
            <span className="pct">{fmt(probabilities[name] * 100, 1)} %</span>
          </div>
        ))}
      </div>

      {ground_truth && (
        <div className="truth">
          Diagnóstico real (DXA) de {exampleId}: {ground_truth.class}, BMD {fmt(ground_truth.bmd)} g/cm² —{' '}
          {ground_truth.class === predicted_class ? '✓ la predicción coincide' : '✗ la predicción no coincide'}{' '}
          (error de BMD {fmt(Math.abs(bmd_estimated - ground_truth.bmd))} g/cm²)
        </div>
      )}

      <div className="images">
        <figure>
          <img src={`data:image/png;base64,${result.image_png_base64}`} alt="Radiografía preprocesada" />
          <figcaption>Radiografía tal como la ve el modelo (224×224, CLAHE)</figcaption>
        </figure>
        <figure>
          <img src={`data:image/png;base64,${result.heatmap_png_base64}`} alt="Mapa de calor Grad-CAM" />
          <figcaption>Grad-CAM: zonas en las que se fijó el modelo para predecir «{predicted_class}»</figcaption>
        </figure>
      </div>
      <div className="legend"><span>Menos influencia</span><div className="scale" /><span>Más influencia</span></div>
      <p className="hint">
        El mapa de calor (HiResCAM sobre el último bloque de ResNet50) resalta las zonas de la imagen que más
        pesaron en la clase predicha. Indica dónde se fijó el modelo, no una lesión.
      </p>
    </div>
  )
}
