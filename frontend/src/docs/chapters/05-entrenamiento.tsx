import { LrSchedule } from '../components/charts'
import { Callout, Figure, Summary } from '../components/ui'

const CONFIG = [
  ['Modelo base', 'ResNet50 preentrenada (ImageNet)'],
  ['Tarea', 'Clasificación (3 clases) + regresión de la BMD'],
  ['Entrada', '224 × 224, CLAHE y normalización ImageNet'],
  ['Optimizador', 'Adam'],
  ['Pérdida', 'Entropía cruzada + 10 × MSE'],
  ['Lote', '32 imágenes'],
  ['Épocas', 'Hasta 50, con parada temprana (paciencia 7)'],
  ['Reproducibilidad', 'Semilla 42 en todo lo aleatorio'],
  ['Hardware', 'MacBook Pro M4, 16 GB de RAM'],
]

export default function Entrenamiento() {
  return (
    <>
      <p>
        Con pocos datos y un backbone preentrenado, <strong>cómo</strong> se entrena importa tanto como el modelo. El
        riesgo es el <em>olvido catastrófico</em>: al principio las cabezas nuevas tienen pesos aleatorios y mandan
        gradientes enormes que pueden estropear lo que el backbone ya sabía. Se compararon dos estrategias de
        <em> fine-tuning</em>, ambas inspiradas en ULMFiT (Howard y Ruder, 2018).
      </p>

      <div className="two-col">
        <div className="card">
          <h3>Diferencial</h3>
          <p>Todo se entrena desde la primera época, pero el backbone con una tasa de aprendizaje diez veces menor
            (10⁻⁴) que las cabezas (10⁻³), para protegerlo.</p>
        </div>
        <div className="card">
          <h3>Por fases</h3>
          <p>Durante 8 épocas el backbone está congelado y solo aprenden las cabezas (10⁻³). Después se descongela todo
            y se sigue con 10⁻⁴ para todo el modelo, con un optimizador nuevo.</p>
        </div>
      </div>

      <Figure caption="Tasa de aprendizaje de cada parte del modelo por época, en escala logarítmica. Pasa el puntero (o el foco) por la gráfica para ver los valores.">
        <LrSchedule />
      </Figure>

      <h2>Primera iteración: 30 épocas fijas</h2>
      <p>
        El primer intento entrenó las dos estrategias 30 épocas sin más, y cada una falló de una forma distinta:
      </p>
      <div className="two-col">
        <div className="card warn">
          <h3>Diferencial: sobreajuste</h3>
          <p>A partir de la época 10 la pérdida de validación sube mientras la de entrenamiento sigue bajando: el
            modelo empieza a memorizar.</p>
        </div>
        <div className="card warn">
          <h3>Por fases: subajuste</h3>
          <p>Las curvas siguen bajando en la época 30: el modelo aún estaba aprendiendo cuando se cortó.</p>
        </div>
      </div>

      <h2>Segunda iteración: parada temprana</h2>
      <p>
        La solución sirvió para las dos: <strong>hasta 50 épocas</strong>, pero parando si la pérdida de validación no
        mejora en <strong>7 épocas seguidas</strong> y quedándose con el modelo de la mejor época. La diferencial paró en
        la época 19 (mejor: 12) y la de fases en la 29 (mejor: 22), con una pérdida de validación mínima casi igual
        (1,06 y 1,08).
      </p>

      <Callout kind="idea" title="La semilla también importa">
        Sin una semilla fija, dos ejecuciones del mismo experimento daban resultados de test distintos, por la
        inicialización aleatoria de las cabezas. Desde entonces todo se entrena con <code>seed = 42</code>, para que
        las diferencias entre experimentos se deban a lo que se cambia y no al azar.
      </Callout>

      <h2>Configuración común</h2>
      <div className="table-wrap">
        <table className="compare left">
          <tbody>
            {CONFIG.map(([k, v]) => <tr key={k}><th scope="row">{k}</th><td>{v}</td></tr>)}
          </tbody>
        </table>
      </div>

      <Summary items={[
        'Dos estrategias: diferencial (todo a la vez, tasas distintas) y por fases (congelar y luego ajustar).',
        'Sin parada temprana, una sobreajusta y la otra se queda corta; con paciencia 7, las dos convergen.',
        'Semilla fija para que los experimentos sean comparables.',
      ]} />
    </>
  )
}
