import { ArchitectureDiagram } from '../components/diagrams'
import { Callout, Figure, StatGrid, Summary } from '../components/ui'

export default function Modelo() {
  return (
    <>
      <p>
        El modelo, <strong>LumosNet</strong>, es deliberadamente sencillo: una red preentrenada que convierte la
        radiografía en un vector de características, una rama pequeña para los datos clínicos y dos "cabezas", una
        por tarea. Esta es la versión final, con covariables clínicas; la de referencia es idéntica sin la rama de
        la derecha.
      </p>

      <Figure caption="Arquitectura de LumosNet con covariables clínicas. Los números de las flechas son el tamaño del vector que pasa de un bloque al siguiente." wide>
        <ArchitectureDiagram />
      </Figure>

      <h2>El backbone: ResNet50</h2>
      <p>
        ResNet50 es una red convolucional de 50 capas cuyas <strong>conexiones residuales</strong> (cada bloque suma su
        entrada a su salida) dejan pasar el gradiente aunque la red sea profunda. Se carga con los pesos de
        <strong> ImageNet</strong> y se le quita la última capa: lo que queda convierte cualquier imagen en
        <strong> 2048 números</strong> que resumen lo que contiene. Las primeras capas detectan bordes y texturas, y
        las últimas, formas y regiones; el ajuste fino las reorienta hacia la anatomía lumbar.
      </p>

      <h2>Dos tareas a la vez</h2>
      <p>
        Las dos cabezas comparten casi todo el modelo. Tiene sentido clínico, porque un médico querrá el número de
        BMD y no solo la etiqueta, y también técnico: las dos tareas dependen de lo mismo (la densidad ósea visible),
        así que entrenarlas juntas actúa como regularización. La señal continua de la BMD es más rica que una
        etiqueta de tres valores y ayuda al backbone a aprender rasgos útiles.
      </p>
      <p>
        Las dos se entrenan con una sola pérdida: <strong>L = entropía cruzada + 10 · MSE</strong>. El factor 10 compensa
        la escala: la BMD vale alrededor de 1 y su error cuadrático es diminuto (unas centésimas) frente a una
        entropía cruzada de alrededor de 1; sin él, la regresión apenas aportaría gradiente.
      </p>

      <h2>La rama de covariables</h2>
      <p>
        La edad, el sexo y el IMC entran normalizados con la media y la desviación del conjunto de
        <strong> entrenamiento</strong>, que se guardan junto al modelo para poder normalizar igual a un paciente nuevo.
        Se proyectan a 32 dimensiones y se concatenan con las 2048 de la imagen antes del bloque compartido.
      </p>
      <Callout kind="warn" title="Fuga de datos evitada">
        La tabla de LUMOS también trae la BMD, el T-score y el Z-score, pero no se usan como entrada: son la propia
        medición de la DXA que define la etiqueta, y usarlas sería darle la respuesta al modelo. La etnia también se
        descarta porque no varía en el dataset.
      </Callout>

      <StatGrid stats={[
        { value: '24,8 M', label: 'parámetros en total' },
        { value: '23,5 M', label: 'en el backbone preentrenado' },
        { value: '≈ 1,3 M', label: 'en la fusión y las cabezas', detail: 'lo que se aprende desde cero' },
      ]} />

      <Summary items={[
        'ResNet50 preentrenada → 2048 características; con covariables, 2080 tras concatenar.',
        'Un bloque compartido de 512 y dos cabezas: clase (3 salidas) y BMD (1 salida).',
        'Pérdida conjunta CE + 10 · MSE, y covariables sin nada que filtre la etiqueta.',
      ]} />
    </>
  )
}
