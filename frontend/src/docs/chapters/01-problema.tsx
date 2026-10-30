import { TScoreExplorer } from '../components/charts'
import { Callout, Figure, StatGrid, Summary } from '../components/ui'

export default function Problema() {
  return (
    <>
      <p>
        La osteoporosis es una pérdida de masa y calidad del hueso que lo vuelve frágil: una caída tonta basta
        para romper una vértebra o una cadera. Es <strong>silenciosa</strong>, porque no da síntomas hasta que llega
        la primera fractura, y en personas mayores esa fractura puede significar dependencia permanente. Su
        antesala es la <strong>osteopenia</strong>, una densidad ósea baja pero todavía no tanto.
      </p>

      <StatGrid stats={[
        { value: '×2', label: 'riesgo de fractura con osteopenia' },
        { value: '×4', label: 'riesgo de fractura con osteoporosis' },
        { value: 'hasta 80 %', label: 'de las fracturas, sin diagnóstico previo' },
        { value: '+27 %', label: 'gasto anual en fracturas por fragilidad hasta 2030', detail: 'previsión de la OMS' },
      ]} />

      <h2>Cómo se diagnostica hoy</h2>
      <p>
        El estándar es la <strong>densitometría ósea (DXA)</strong>: una prueba de rayos X específica que mide la
        <strong> densidad mineral ósea (BMD)</strong>, en g/cm². La BMD se compara con la de un adulto joven sano
        del mismo sexo y se expresa en desviaciones típicas: el <strong>T-score</strong>. La OMS fija los umbrales
        de diagnóstico: <strong>sano</strong> si T ≥ −1, <strong>osteopenia</strong> entre −1 y −2,5 y
        <strong> osteoporosis</strong> si T ≤ −2,5.
      </p>

      <Figure caption={<>Mueve la BMD y mira cómo cambian el T-score y el diagnóstico. En LUMOS, T = (BMD − 1,114) / 0,120,
        así que los límites caen en una BMD de 0,994 y 0,814 g/cm².</>}>
        <TScoreExplorer />
      </Figure>

      <Callout kind="idea" title="Por qué la osteopenia es la clase difícil">
        Toda la franja de osteopenia ocupa solo 0,18 g/cm² de BMD. Cualquier paciente con osteopenia está muy cerca
        de alguna de las dos fronteras, y un error pequeño en la imagen lo empuja a la clase vecina. Esta geometría
        explica buena parte de los resultados del proyecto.
      </Callout>

      <h2>La brecha que se quiere cerrar</h2>
      <div className="two-col">
        <div className="card">
          <h3>Densitometría (DXA)</h3>
          <ul>
            <li>Es el estándar diagnóstico: mide BMD y T-score.</li>
            <li>Necesita un equipo específico y costoso.</li>
            <li>No está disponible en muchos centros rurales o con pocos recursos.</li>
          </ul>
        </div>
        <div className="card">
          <h3>Radiografía convencional</h3>
          <ul>
            <li>Barata y disponible en casi cualquier centro.</li>
            <li>Se hace a diario por otros motivos.</li>
            <li>La densidad ósea no se puede leer a simple vista: aquí entra la IA.</li>
          </ul>
        </div>
      </div>

      <h2>Objetivo: un modelo, dos tareas</h2>
      <p>
        A partir de <strong>una única radiografía lumbar convencional</strong>, el modelo debe:
      </p>
      <ol>
        <li><strong>Clasificar</strong> la fragilidad ósea en tres niveles: sano, osteopenia y osteoporosis.</li>
        <li><strong>Estimar la BMD</strong> numéricamente, para dar un valor y no solo una etiqueta.</li>
      </ol>
      <p>
        Se trabaja con la columna lumbar porque, junto con la cadera, es la zona más predictiva del riesgo de
        fractura, y con tres clases en vez de una clasificación binaria porque la mayoría de fracturas por
        fragilidad ocurren en pacientes con osteopenia, no con osteoporosis.
      </p>

      <Callout kind="info" title="Alcance">
        Es una prueba de concepto: un prototipo entrenado y validado sobre un conjunto de datos público. No se
        aborda la integración en un hospital ni la certificación como producto sanitario.
      </Callout>

      <Summary items={[
        'La osteoporosis no avisa hasta la fractura, y la DXA que la diagnostica no está en todas partes.',
        'El T-score separa las clases en −1 y −2,5; la osteopenia es una franja estrecha entre las dos.',
        'Objetivo: clasificar en tres niveles y estimar la BMD desde una radiografía lumbar convencional.',
      ]} />
    </>
  )
}
