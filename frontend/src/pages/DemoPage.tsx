import { useEffect, useRef, useState } from 'react'
import { Link } from 'react-router'
import {
  getTestMetrics, predictExample, predictUpload, type Example, type PatientData, type Prediction, type TestMetrics,
} from '../api'
import { EmptyResult } from '../components/EmptyResult'
import { ExamplesGrid } from '../components/ExamplesGrid'
import { PrivacyNotice } from '../components/PrivacyNotice'
import { ResultPanel } from '../components/ResultPanel'
import { UploadForm } from '../components/UploadForm'

type Tab = 'upload' | 'examples'

export function DemoPage() {
  const [tab, setTab] = useState<Tab>('upload')
  const [result, setResult] = useState<Prediction | null>(null)
  const [exampleId, setExampleId] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [metrics, setMetrics] = useState<TestMetrics | null>(null)
  // Altura de la columna izquierda con la pestaña de subida: el panel vacío la conserva en la de ejemplos
  // para que el dibujo no cambie de sitio al cambiar de pestaña
  const sideRef = useRef<HTMLDivElement>(null)
  const [uploadHeight, setUploadHeight] = useState<number>()
  const resultRef = useRef<HTMLElement>(null)

  useEffect(() => {
    const side = sideRef.current
    if (tab !== 'upload' || !side) return
    const observer = new ResizeObserver(() => setUploadHeight(side.offsetHeight))
    observer.observe(side)
    return () => observer.disconnect()
  }, [tab])

  // En una sola columna (móvil) el resultado queda debajo del formulario o de los ejemplos: se lleva a la vista
  useEffect(() => {
    if (!result && !error) return
    if (!window.matchMedia('(max-width: 860px)').matches) return
    const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches
    resultRef.current?.scrollIntoView({ behavior: reduceMotion ? 'auto' : 'smooth', block: 'start' })
  }, [result, error])

  useEffect(() => {
    getTestMetrics().then(setMetrics).catch(() => {})
  }, [])

  async function run(request: () => Promise<Prediction>, id: string | null) {
    setBusy(true)
    setError(null)
    setResult(null)
    setExampleId(id)
    try {
      setResult(await request())
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Error inesperado')
    } finally {
      setBusy(false)
    }
  }

  const handleUpload = (file: File, patient: PatientData) => run(() => predictUpload(file, patient), null)
  const handleExample = (ex: Example) => run(() => predictExample(ex.id), ex.id)

  return (
    <main>
      <h1>Probar el modelo</h1>
      <p className="demo-lead">
        LumosNet es el modelo de mi Trabajo Fin de Máster: una red neuronal (ResNet50) que, a partir de una sola
        radiografía lumbar convencional, estima si el hueso está sano o tiene osteopenia u osteoporosis. La
        prueba que lo diagnostica hoy, la densitometría (DXA), es cara y no está en todos los centros; una
        radiografía sí. Se entrenó con las radiografías y densitometrías de 803 pacientes del dataset público
        LUMOS.
      </p>
      <ol className="how">
        <li>
          <strong>Entrada</strong>
          Una radiografía lumbar en DICOM, frontal o lateral, junto con la edad, el sexo y el IMC del paciente.
        </li>
        <li>
          <strong>Predicción</strong>
          La clase (Sano, Osteopenia u Osteoporosis) con su probabilidad y una estimación de la densidad mineral
          ósea (BMD).
        </li>
        <li>
          <strong>Explicación</strong>
          Un mapa de calor Grad-CAM que muestra en qué zonas de la imagen se fijó el modelo para decidir.
        </li>
      </ol>
      <p className="disclaimer">
        En test acierta el 66,9 % de los pacientes y detecta el 85,3 % de los casos de osteoporosis. Es una
        herramienta de investigación: no es un producto sanitario ni sustituye a una DXA ni al criterio de un
        profesional. <Link to="/proyecto">Cómo se hizo →</Link>
      </p>

      {/* Con la pestaña de ejemplos la columna izquierda es muy alta: el panel vacío no se estira y se
          queda fijo al hacer scroll */}
      <div className={`layout${tab === 'examples' ? ' examples-tab' : ''}`}>
        <div className="side" ref={sideRef}>
          <section className="panel">
            <div className="tabs" role="tablist">
              <button type="button" role="tab" className="tab" aria-selected={tab === 'upload'}
                      onClick={() => setTab('upload')}>Subir radiografía</button>
              <button type="button" role="tab" className="tab" aria-selected={tab === 'examples'}
                      onClick={() => setTab('examples')}>Ejemplos LUMOS</button>
            </div>
            {/* Se ocultan en vez de desmontarse para conservar el formulario al cambiar de pestaña */}
            <div hidden={tab !== 'upload'}><UploadForm busy={busy} onSubmit={handleUpload} /></div>
            <div hidden={tab !== 'examples'}>
              <ExamplesGrid selectedId={exampleId} metrics={metrics} busy={busy} onSelect={handleExample} />
            </div>
          </section>
          <PrivacyNotice />
        </div>

        <section ref={resultRef} className={`panel result-panel${!busy && !result ? ' is-empty' : ''}`} aria-live="polite"
                 style={tab === 'examples' && !busy && !result ? { minHeight: uploadHeight } : undefined}>
          {busy && <div className="empty"><div className="spinner" />Analizando…</div>}
          {!busy && result && <ResultPanel result={result} exampleId={exampleId} metrics={metrics} />}
          {!busy && !result && <EmptyResult />}
          {error && <p className="error">{error}</p>}
        </section>
      </div>
    </main>
  )
}
