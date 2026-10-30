import type { CSSProperties } from 'react'
import { CLASS_NAMES, type Prediction, type TestMetrics } from '../api'
import { CLASS_COLORS, DOUBTFUL_MARGIN, fmt, fmtSigned } from '../format'

const color = (c: string) => ({ '--c': c }) as CSSProperties

interface Props {
  result: Prediction
  exampleId: string | null
  metrics: TestMetrics | null
}

export function ResultPanel({ result, exampleId, metrics }: Props) {
  const { predicted_class, probabilities, bmd_estimated, ground_truth } = result
  const [first, second] = [...CLASS_NAMES].sort((a, b) => probabilities[b] - probabilities[a])
  const doubtful = probabilities[first] - probabilities[second] < DOUBTFUL_MARGIN

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

      {doubtful && (
        <div className="warning">
          <strong>Predicción dudosa:</strong> {first} ({fmt(probabilities[first] * 100, 1)} %) y {second}{' '}
          ({fmt(probabilities[second] * 100, 1)} %) están casi empatadas. El modelo no distingue con claridad
          entre ambas clases en esta radiografía.
        </div>
      )}

      {ground_truth && (
        <div className={`truth ${ground_truth.class === predicted_class ? 'hit' : 'miss'}`}>
          <strong>{ground_truth.class === predicted_class ? '✓ Acierto' : '✗ Fallo'}</strong> · Diagnóstico real
          (DXA) de {exampleId}: <strong>{ground_truth.class}</strong>, T-score {fmtSigned(ground_truth.t_score)},
          BMD {fmt(ground_truth.bmd)} g/cm² (error de la BMD estimada: {fmt(Math.abs(bmd_estimated - ground_truth.bmd))}).
          {ground_truth.class !== predicted_class && (
            <> Las clases se definen por el T-score (Sano ≥ −1, Osteopenia entre −1 y −2,5, Osteoporosis ≤ −2,5),
              y los fallos se concentran en pacientes cerca de esos límites.</>
          )}
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
      {metrics && (
        <p className="hint">
          Fiabilidad: en el conjunto de test de LUMOS ({metrics.n_images} radiografías) el modelo acierta el{' '}
          {fmt(metrics.accuracy * 100, 1)} %. El {fmt(metrics.adjacent_error_share * 100, 0)} % de sus fallos son
          entre clases vecinas (Sano ↔ Osteopenia u Osteopenia ↔ Osteoporosis).
        </p>
      )}
    </div>
  )
}
