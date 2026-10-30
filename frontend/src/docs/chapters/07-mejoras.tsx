import { useState, type ReactNode } from 'react'
import { EXPERIMENT_METRICS, type ExperimentKey } from '../metrics'
import { DotPlot } from '../components/charts'
import { num } from '../components/chart-utils'
import { CompareTable, SensSpecTable } from '../components/tables'
import { experimentRows } from '../components/table-rows'
import { Callout, Figure, Summary } from '../components/ui'

const ROWS: { key: ExperimentKey; label: string; emphasis?: 'reference' | 'final' }[] = [
  { key: 'v2-phased', label: 'Referencia (por fases)', emphasis: 'reference' },
  { key: 'v3', label: 'Más regularización' },
  { key: 'v7-inverse', label: 'Balanceo inverso' },
  { key: 'v7-effective', label: 'Balanceo efectivo' },
  { key: 'v4-roi', label: 'ROI fijo, solo AP' },
  { key: 'v4-hybrid', label: 'ROI fijo híbrido' },
  { key: 'v6', label: 'Resolución 320 × 320' },
  { key: 'final-roi-yolo', label: 'Covariables + ROI YOLO' },
  { key: 'v8', label: 'Covariables clínicas', emphasis: 'final' },
]

const METRICS = {
  sensOp: { label: 'Sensibilidad a osteoporosis', get: (k: ExperimentKey) => EXPERIMENT_METRICS[k].patient.sens[2], domain: [0.3, 0.9] },
  acc: { label: 'Accuracy', get: (k: ExperimentKey) => EXPERIMENT_METRICS[k].patient.acc, domain: [0.45, 0.7] },
  f1: { label: 'F1 macro', get: (k: ExperimentKey) => EXPERIMENT_METRICS[k].patient.f1, domain: [0.45, 0.7] },
  auc: { label: 'AUC', get: (k: ExperimentKey) => EXPERIMENT_METRICS[k].patient.auc, domain: [0.7, 0.86] },
  pcc: { label: 'PCC de la BMD', get: (k: ExperimentKey) => EXPERIMENT_METRICS[k].patient.pcc, domain: [0.4, 0.8] },
} as const

type MetricKey = keyof typeof METRICS

function ExperimentExplorer() {
  const [metric, setMetric] = useState<MetricKey>('sensOp')
  const m = METRICS[metric]
  return (
    <>
      <div className="chips" role="radiogroup" aria-label="Métrica">
        {(Object.keys(METRICS) as MetricKey[]).map((k) => (
          <button key={k} type="button" role="radio" aria-checked={k === metric}
                  className={`chip${k === metric ? ' active' : ''}`} onClick={() => setMetric(k)}>{METRICS[k].label}</button>
        ))}
      </div>
      <DotPlot rows={ROWS.map((r) => ({ ...r, value: m.get(r.key) }))} domain={m.domain as unknown as [number, number]}
               format={(v) => num(v)} />
    </>
  )
}

function Experiment({ title, hypothesis, result, verdict, children }: {
  title: string
  hypothesis: string
  result: string
  verdict: 'descartado' | 'adoptado'
  children?: ReactNode
}) {
  return (
    <section className={`experiment ${verdict}`}>
      <header>
        <h3>{title}</h3>
        <span className="verdict">{verdict === 'adoptado' ? '✓ Adoptado' : '✗ Descartado'}</span>
      </header>
      <p><strong>Hipótesis.</strong> {hypothesis}</p>
      <p><strong>Resultado.</strong> {result}</p>
      {children}
    </section>
  )
}

export default function Mejoras() {
  return (
    <>
      <p>
        Con la estrategia por fases como referencia, se probaron <strong>seis vías de mejora</strong>, siempre cambiando
        una sola cosa respecto a la referencia y con la misma semilla. Cinco actúan sobre la imagen; una añade
        información clínica. La regla para quedarse con una variante es la misma de siempre: que mejore la
        sensibilidad a la osteoporosis.
      </p>

      <Figure caption="Cada experimento en test, por paciente. La línea vertical es la referencia; el punto grande, el modelo final. Cambia de métrica con los botones." wide>
        <ExperimentExplorer />
      </Figure>

      <Callout kind="idea" title="El patrón que deja la bitácora">
        Ninguna mejora centrada en la imagen supera a la referencia en sensibilidad a la osteoporosis. La única que lo
        consigue es la que añade información que no está en la imagen: las covariables clínicas.
      </Callout>

      <h2>Mejoras sobre la imagen</h2>

      <Experiment title="1 · Más regularización" verdict="descartado"
        hypothesis="Aún hay algo de sobreajuste: más dropout (0,3 → 0,5) y weight decay (10⁻⁴) deberían ayudar a generalizar."
        result="La accuracy baja de 0,661 a 0,636 y el F1 de 0,668 a 0,640. Mejora la sensibilidad a sano (0,605 → 0,721), pero a costa de la osteoporosis (0,794 → 0,706)." />

      <Experiment title="2 · Balanceo de clases" verdict="descartado"
        hypothesis="Hay menos pacientes con osteoporosis (223 frente a ~290): pesar más esa clase en la pérdida podría detectarla mejor."
        result="Se probaron dos esquemas: inverso (peso ∝ 1 / frecuencia) y efectivo (Cui et al., 2019, más suave). El inverso logra el mejor AUC (0,846) y PCC (0,766) de la serie, pero la osteopenia cae a 0,477 y ninguno mejora la sensibilidad a la osteoporosis.">
        <SensSpecTable columns={['Referencia', 'Inverso', 'Efectivo']} keys={['v2-phased', 'v7-inverse', 'v7-effective']} />
      </Experiment>

      <Experiment title="3 · Región de interés (ROI) fija" verdict="descartado"
        hypothesis="Recortar la franja central de la columna (del 25 % al 75 % del ancho) quita ruido: tejido blando, prótesis, bordes."
        result="Se hizo en dos pasos para separar dos causas. Con solo la vista AP recortada, todo empeora (accuracy 0,500), pero también se pierde la mitad de las imágenes. El ROI híbrido (recorte en la AP, lateral completa) mantiene el volumen de datos y mejora mucho (0,603), pero sigue por debajo de no recortar.">
        <CompareTable columns={['Referencia', 'ROI solo AP', 'ROI híbrido']} rows={experimentRows(['v2-phased', 'v4-roi', 'v4-hybrid'])} />
        <p>
          El recorte afecta sobre todo a la clasificación: la correlación de la BMD del ROI híbrido (0,726) queda
          cerca de la referencia. La explicación más probable es la <strong>pérdida de contexto</strong>: el tamaño
          relativo del paciente, la pelvis o las costillas, relacionados con factores de riesgo como el IMC. Y un
          recorte fijo falla con columnas desviadas, como un caso de escoliosis visto en la muestra.
        </p>
      </Experiment>

      <Experiment title="4 · Más resolución" verdict="descartado"
        hypothesis="Pasar de 224 × 224 a 320 × 320 conserva más textura trabecular."
        result="Las cinco métricas quedan cerca de la referencia sin superarla. La sensibilidad a sano cae de 0,605 a 0,488 y la de osteopenia sube a 0,750: con más resolución el modelo parece leer cualquier textura irregular como osteopenia. El backbone se preentrenó a 224, y con tan pocos datos no compensa el cambio de escala." />

      <Experiment title="5 · ROI adaptativo con YOLO" verdict="descartado"
        hypothesis="Un detector entrenado recorta la columna de cada radiografía, en vez de una franja fija."
        result="La accuracy cae de 0,669 a 0,554 y la sensibilidad a la osteoporosis, de 0,853 a 0,500. Se detalla en el capítulo de interpretabilidad, porque el detector se entrenó a partir de los mapas de Grad-CAM." />

      <h2>Información clínica</h2>
      <Experiment title="6 · Covariables clínicas" verdict="adoptado"
        hypothesis="La edad, el sexo y el IMC son factores de riesgo conocidos (Khan et al., 2024); dárselos al modelo junto a la imagen debería ayudar."
        result="Es la única vía que mejora la sensibilidad a la osteoporosis: de 0,794 a 0,853, el mejor valor de todo el trabajo. También suben la sensibilidad a sano (0,605 → 0,767), el AUC y el PCC. A cambio, la osteopenia cae a 0,432 y el RMSE de la BMD sube a 0,168.">
        <CompareTable columns={['Referencia', 'Covariables']} rows={experimentRows(['v2-phased', 'v8'])} highlight={1} />
      </Experiment>

      <Summary items={[
        'Regularización, balanceo, ROI fijo, más resolución y ROI con YOLO: ninguna supera a la referencia.',
        'Recortar la imagen quita contexto útil; la vía que funciona añade información de fuera de la imagen.',
        'Lectura de conjunto: el límite del proyecto es el tamaño del dataset, no la arquitectura.',
      ]} />
    </>
  )
}
