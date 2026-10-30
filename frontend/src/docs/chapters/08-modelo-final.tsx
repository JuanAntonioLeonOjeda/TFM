import { useState } from 'react'
import { EXPERIMENT_METRICS } from '../metrics'
import { BarList, ConfusionMatrix, PairedBars } from '../components/charts'
import { num } from '../components/chart-utils'
import { SensSpecTable } from '../components/tables'
import { Callout, Figure, StatGrid, Summary } from '../components/ui'

const FINAL = EXPERIMENT_METRICS.v8
const REF = EXPERIMENT_METRICS['v2-phased']

function FinalConfusion() {
  const [level, setLevel] = useState<'patient' | 'image'>('patient')
  const m = FINAL[level]
  return (
    <>
      <div className="chips" role="radiogroup" aria-label="Nivel de evaluación">
        <button type="button" role="radio" aria-checked={level === 'patient'} className={`chip${level === 'patient' ? ' active' : ''}`} onClick={() => setLevel('patient')}>Por paciente (121)</button>
        <button type="button" role="radio" aria-checked={level === 'image'} className={`chip${level === 'image' ? ' active' : ''}`} onClick={() => setLevel('image')}>Por imagen (244)</button>
      </div>
      <ConfusionMatrix matrix={m.confusion}
        caption={`Accuracy ${num(m.acc)} ${level === 'patient' ? 'por paciente' : 'por imagen'}. Cada celda indica cuántos casos de la clase real (fila) acabaron en cada clase predicha (columna), y qué parte de la fila son.`} />
    </>
  )
}

export default function ModeloFinal() {
  return (
    <>
      <p>
        El modelo final es la <strong>estrategia por fases con covariables clínicas</strong>, a 224 × 224, sin recorte y
        agregando las dos vistas de cada paciente. Es el que sirve la aplicación.
      </p>

      <StatGrid stats={[
        { value: '66,9 %', label: 'accuracy', detail: 'por paciente' },
        { value: '0,836', label: 'AUC macro' },
        { value: '76,2 %', label: 'correlación (PCC) de la BMD' },
        { value: '85,3 %', label: 'sensibilidad a osteoporosis', detail: 'la métrica prioritaria' },
      ]} />

      <Figure caption="Matriz de confusión del modelo final en test. Por paciente se promedian las dos vistas; por imagen, cada radiografía cuenta por separado (es lo que hace la aplicación con una sola radiografía).">
        <FinalConfusion />
      </Figure>

      <Callout kind="info" title="Por paciente o por imagen">
        Combinar las dos vistas de un paciente sube la accuracy de 0,627 a 0,669 sin cambiar el modelo. La aplicación
        analiza una radiografía cada vez, así que lo que verás al probarla se parece más a la cifra por imagen.
      </Callout>

      <h2>Un compromiso asumido</h2>
      <p>
        Las covariables mejoran justo las dos clases que más importan clínicamente, a cambio de la osteopenia:
      </p>
      <Figure caption="Sensibilidad por clase (test, por paciente) antes y después de añadir edad, sexo e IMC.">
        <PairedBars categories={['Sano', 'Osteopenia', 'Osteoporosis']} series={[
          { label: 'Referencia (solo imagen)', values: [...REF.patient.sens], color: 'var(--s-muted)' },
          { label: 'Con covariables (final)', values: [...FINAL.patient.sens], color: 'var(--s1)' },
        ]} />
      </Figure>
      <p>
        El matiz importante está en <strong>hacia dónde</strong> se equivoca. De los 44 pacientes con osteopenia, 17 se
        clasifican como osteoporosis y solo 8 como sanos. Confundir osteopenia con osteoporosis lleva, como mucho, a
        una prueba de más; confundirla con sano dejaría a alguien sin seguimiento. Con el criterio de priorizar la
        osteoporosis, es un intercambio aceptable.
      </p>
      <SensSpecTable columns={['Referencia', 'Final']} keys={['v2-phased', 'v8']} />

      <h2>La regresión de la BMD</h2>
      <p>
        Las dos métricas de regresión cuentan cosas distintas. La <strong>correlación de Pearson (0,762)</strong> mide si
        el modelo ordena bien a los pacientes de menos a más densidad: lo hace razonablemente bien. El
        <strong> RMSE (0,168 g/cm²)</strong> mide el error absoluto: es casi tan grande como toda la franja de osteopenia,
        y el modelo tiende a infraestimar las densidades altas. Aumentar el peso de la regresión en la pérdida no
        mejoró el conjunto.
      </p>

      <h2>Frente al estado del arte</h2>
      <Figure caption="Modelos entrenados con LUMOS solo con radiografía. Este trabajo usa una ResNet50 genérica; FCoTNet y OsteoNet son arquitecturas diseñadas para el problema.">
        <div className="small-multiples three">
          {([['Accuracy', [66.9, 72.3, 73.6]], ['F1 macro', [66.2, 69.8, 73.5]], ['PCC de la BMD', [76.2, 74.7, 77.8]]] as const).map(([metric, v]) => (
            <div key={metric}>
              <p className="chart-title">{metric}</p>
              <BarList title={metric} max={100} format={(x) => `${num(x, 1)} %`} bars={[
                { label: 'Este trabajo', value: v[0], color: 'var(--s1)', emphasis: true },
                { label: 'FCoTNet', value: v[1], color: 'var(--s-muted)' },
                { label: 'OsteoNet', value: v[2], color: 'var(--s-muted)' },
              ]} />
            </div>
          ))}
        </div>
      </Figure>
      <p>
        En clasificación queda por debajo (66,9 % frente a 72–74 %), pero en el mismo orden de magnitud. En la
        regresión de la BMD es <strong>competitivo</strong>: supera a FCoTNet (76,2 frente a 74,7) y se acerca a OsteoNet
        (77,8). Fuera de LUMOS, supera el PCC de 0,63 de Sato et al. y queda por debajo del AUC de 0,91–0,94 de Mao
        et al., que entrenaron con 6.908 pacientes.
      </p>

      <Summary items={[
        'Final: por fases + covariables. Accuracy 0,669, AUC 0,836, PCC 0,762 y sensibilidad a osteoporosis 0,853.',
        'Gana en sano y osteoporosis a costa de la osteopenia, y cuando falla en osteopenia tiende a la clase más grave.',
        'Por debajo de FCoTNet y OsteoNet en clasificación; a su altura en la estimación de la BMD.',
      ]} />
    </>
  )
}
