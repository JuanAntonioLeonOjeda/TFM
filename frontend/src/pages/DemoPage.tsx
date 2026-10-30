import { useEffect, useState } from 'react'
import {
  getTestMetrics, predictExample, predictUpload, type Example, type PatientData, type Prediction, type TestMetrics,
} from '../api'
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
      <p className="subtitle">
        Clasificación Sano / Osteopenia / Osteoporosis y estimación de la densidad mineral ósea (BMD) a partir
        de una radiografía lumbar.
      </p>
      <PrivacyNotice />

      <div className="layout">
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

        <section className="panel" aria-live="polite">
          {busy && <div className="empty"><div className="spinner" />Analizando…</div>}
          {!busy && result && <ResultPanel result={result} exampleId={exampleId} metrics={metrics} />}
          {!busy && !result && (
            <div className="empty">Sube una radiografía o elige un ejemplo para ver el resultado.</div>
          )}
          {error && <p className="error">{error}</p>}
        </section>
      </div>
    </main>
  )
}
