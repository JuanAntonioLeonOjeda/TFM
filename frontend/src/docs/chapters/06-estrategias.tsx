import { EXPERIMENT_METRICS } from '../metrics'
import { ConfusionMatrix, CvFolds } from '../components/charts'
import { CompareTable, SensSpecTable } from '../components/tables'
import { experimentRows } from '../components/table-rows'
import { Callout, Figure, StatGrid, Summary } from '../components/ui'

const CV_FOLDS = [
  { fold: 1, value: 0.678 },
  { fold: 2, value: 0.636 },
  { fold: 3, value: 0.438, note: 'paró en la época 3' },
  { fold: 4, value: 0.678 },
  { fold: 5, value: 0.694 },
]

export default function Estrategias() {
  return (
    <>
      <p>
        Con la parada temprana resuelta, las dos estrategias se evalúan en el conjunto de test, agregando las dos
        vistas de cada paciente. Antes de mirar los números conviene fijar el <strong>criterio</strong>: en un cribado,
        el error caro es decirle a alguien con osteoporosis que está bien. Por eso la métrica que decide es la
        <strong> sensibilidad a la osteoporosis</strong>, por encima de la exactitud global.
      </p>

      <CompareTable columns={['Diferencial', 'Por fases']} rows={experimentRows(['v2-differential', 'v2-phased'])}
                    caption="Test, por paciente (121 pacientes)" highlight={1} />

      <SensSpecTable columns={['Diferencial', 'Por fases']} keys={['v2-differential', 'v2-phased']}
                     caption="Sensibilidad y especificidad por clase" />

      <StatGrid stats={[
        { value: '0,794', label: 'sensibilidad a la osteoporosis por fases' },
        { value: '0,529', label: 'sensibilidad a la osteoporosis diferencial', detail: 'dejaría sin detectar casi la mitad de los casos' },
      ]} />

      <div className="two-col">
        <Figure caption="Diferencial">
          <ConfusionMatrix matrix={EXPERIMENT_METRICS['v2-differential'].patient.confusion} />
        </Figure>
        <Figure caption="Por fases">
          <ConfusionMatrix matrix={EXPERIMENT_METRICS['v2-phased'].patient.confusion} />
        </Figure>
      </div>

      <p>
        La estrategia <strong>por fases</strong> gana en exactitud, F1 y AUC, y sobre todo en sensibilidad a la
        osteoporosis. La diferencial solo la supera en la correlación de la BMD y en sensibilidad a la osteopenia.
        Se elige la estrategia por fases como <strong>modelo de referencia</strong> para todo lo que viene después.
      </p>

      <Callout kind="idea" title="Por qué congelar primero funciona">
        Mientras el backbone está congelado, las cabezas aprenden a usar las características de ImageNet sin
        estropearlas. Cuando se descongela, los gradientes ya son pequeños y el ajuste fino es suave. Con pocos datos,
        proteger lo preentrenado generaliza mejor.
      </Callout>

      <h2>¿Basta con una sola partición?</h2>
      <p>
        Todos los experimentos usan una única partición fija (<em>holdout</em>). Para comprobar que no es una
        partición con suerte, se repitió el entrenamiento por fases con <strong>validación cruzada de 5 bolsas</strong> sobre
        entrenamiento y validación, separadas por paciente, evaluando siempre sobre el mismo test.
      </p>
      <Figure caption="Accuracy por paciente en test de cada bolsa, con la media ± una desviación típica sombreada. La bolsa 3 se detuvo en la época 3: un entrenamiento que no llegó a arrancar.">
        <CvFolds folds={CV_FOLDS} holdout={0.661} />
      </Figure>
      <CompareTable columns={['Holdout', 'Validación cruzada (k = 5)']} rows={[
        { label: 'Accuracy', values: [0.661, '0,625 ± 0,107'] },
        { label: 'F1 macro', values: [0.668, '0,609 ± 0,151'] },
        { label: 'AUC macro', values: [0.822, '0,805 ± 0,074'] },
        { label: 'RMSE de la BMD', values: [0.1379, '0,154 ± 0,021'], digits: 4 },
        { label: 'PCC de la BMD', values: [0.75, '0,657 ± 0,148'] },
      ]} />
      <p>
        La media de la validación cruzada es coherente con el holdout, así que se mantiene la partición simple y se
        evita multiplicar por cinco el coste de cada experimento. La desviación, eso sí, avisa de algo importante:
        con 121 pacientes de test, <strong>un punto de accuracy son 1,2 pacientes</strong>, y las diferencias de uno o dos
        puntos entre experimentos hay que leerlas con prudencia.
      </p>

      <Summary items={[
        'El criterio de decisión es la sensibilidad a la osteoporosis.',
        'Por fases: 0,794 frente a 0,529, además de mejor accuracy, F1 y AUC. Se elige como referencia.',
        'La validación cruzada (0,625 ± 0,107) respalda usar una partición fija.',
      ]} />
    </>
  )
}
