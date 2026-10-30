import { useEffect, useState, type CSSProperties } from 'react'
import { exampleImageUrl, getExamples, predictExample, type Example, type Prediction } from '../../api'
import { CLASS_COLORS, GENDER_LABELS, fmt, fmtSigned } from '../../format'

function useExamples() {
  const [examples, setExamples] = useState<Example[] | null>(null)
  const [error, setError] = useState(false)
  useEffect(() => {
    getExamples().then(setExamples).catch(() => setError(true))
  }, [])
  return { examples, error }
}

const Unavailable = () => (
  <p className="chart-note">Las radiografías de ejemplo no están disponibles en este servidor (se montan aparte con
    <code> python -m api.examples</code>).</p>
)

// Una radiografía de ejemplo por clase, tal como la recibe el modelo. Las imágenes vienen de la API,
// no del bundle: son datos de pacientes y solo existen donde se monta el volumen de ejemplos
export function ExampleStrip() {
  const { examples, error } = useExamples()
  if (error || (examples && examples.length === 0)) return <Unavailable />
  if (!examples) return <p className="chart-note">Cargando radiografías…</p>
  const byClass = ['Sano', 'Osteopenia', 'Osteoporosis'].map((c) => examples.find((e) => e.true_class === c)).filter(Boolean) as Example[]
  return (
    <div className="example-strip">
      {byClass.map((ex) => (
        <figure key={ex.id}>
          <img src={exampleImageUrl(ex.id)} alt={`Radiografía de ${ex.id}, ${ex.true_class}`} loading="lazy" />
          <figcaption>
            <span className="badge" style={{ '--c': CLASS_COLORS[ex.true_class] } as CSSProperties}>{ex.true_class}</span>
            <span>T-score {fmtSigned(ex.true_t_score)} · {GENDER_LABELS[ex.gender]}, {Math.round(ex.age)} años</span>
          </figcaption>
        </figure>
      ))}
    </div>
  )
}

// Grad-CAM calculado en vivo por la API para el ejemplo que se elija
export function LiveGradCam() {
  const { examples, error } = useExamples()
  const [chosen, setChosen] = useState<string | null>(null)
  const [loaded, setLoaded] = useState<{ id: string; result: Prediction | null } | null>(null)
  const selected = chosen ?? examples?.[0]?.id ?? null
  const loading = selected !== null && loaded?.id !== selected
  const result = loaded?.result ?? null

  useEffect(() => {
    if (!selected) return
    let cancelled = false
    predictExample(selected)
      .then((r) => { if (!cancelled) setLoaded({ id: selected, result: r }) })
      .catch(() => { if (!cancelled) setLoaded({ id: selected, result: null }) })
    return () => { cancelled = true }
  }, [selected])

  if (error || (examples && examples.length === 0)) return <Unavailable />
  if (!examples) return <p className="chart-note">Cargando ejemplos…</p>

  return (
    <div className="live-cam">
      <div className="chips" role="radiogroup" aria-label="Radiografía de ejemplo">
        {examples.map((ex) => (
          <button key={ex.id} type="button" role="radio" aria-checked={ex.id === selected}
                  className={`chip${ex.id === selected ? ' active' : ''}`} onClick={() => setChosen(ex.id)}>
            <i style={{ background: CLASS_COLORS[ex.true_class] }} />{ex.id.replace('lumos_x_', '#')}
          </button>
        ))}
      </div>
      <div className={`live-cam-body${loading ? ' loading' : ''}`}>
        {result ? (
          <>
            <div className="images">
              <figure>
                <img src={`data:image/png;base64,${result.image_png_base64}`} alt="Radiografía preprocesada" />
                <figcaption>Entrada del modelo</figcaption>
              </figure>
              <figure>
                <img src={`data:image/png;base64,${result.heatmap_png_base64}`} alt="Mapa Grad-CAM" />
                <figcaption>Grad-CAM de «{result.predicted_class}»</figcaption>
              </figure>
            </div>
            {result.ground_truth && (
              <p className="chart-note">
                Real: <strong>{result.ground_truth.class}</strong> (T {fmtSigned(result.ground_truth.t_score)}) ·
                Predicho: <strong>{result.predicted_class}</strong> con un{' '}
                {fmt(result.probabilities[result.predicted_class] * 100, 1)} % ·{' '}
                {result.ground_truth.class === result.predicted_class ? '✓ acierto' : '✗ fallo'}
              </p>
            )}
          </>
        ) : (
          <div className="empty"><div className="spinner" />Calculando Grad-CAM…</div>
        )}
      </div>
    </div>
  )
}
