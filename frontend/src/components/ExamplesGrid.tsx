import { useEffect, useState, type CSSProperties } from 'react'
import { exampleImageUrl, getExamples, type Example } from '../api'
import { CLASS_COLORS, GENDER_LABELS, fmt } from '../format'

interface Props {
  selectedId: string | null
  busy: boolean
  onSelect: (example: Example) => void
}

export function ExamplesGrid({ selectedId, busy, onSelect }: Props) {
  const [examples, setExamples] = useState<Example[] | null>(null)
  const [error, setError] = useState(false)

  useEffect(() => {
    getExamples().then(setExamples).catch(() => setError(true))
  }, [])

  if (error) return <p className="empty">No se pudieron cargar los ejemplos.</p>
  if (examples === null) return <p className="empty">Cargando ejemplos…</p>
  if (examples.length === 0) return <p className="empty">No hay ejemplos configurados en este servidor.</p>

  return (
    <>
      <p className="hint intro">
        Radiografías del conjunto de test de LUMOS, que el modelo no vio durante el entrenamiento. Elige una
        para comparar la predicción con el diagnóstico real.
      </p>
      <div className="examples">
        {examples.map((ex) => (
          <button key={ex.id} type="button" disabled={busy}
                  className={`example${ex.id === selectedId ? ' active' : ''}`}
                  onClick={() => onSelect(ex)}>
            <img loading="lazy" src={exampleImageUrl(ex.id)} alt={`Radiografía ${ex.id}`} />
            <div className="badge-row">
              <span className="badge" style={{ '--c': CLASS_COLORS[ex.true_class] } as CSSProperties}>
                {ex.true_class}
              </span>
            </div>
            <div className="meta">
              {ex.id} · {GENDER_LABELS[ex.gender]}, {Math.round(ex.age)} años · IMC {fmt(ex.bmi, 1)}
            </div>
          </button>
        ))}
      </div>
    </>
  )
}
