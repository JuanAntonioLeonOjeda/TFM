import { BarList, SplitBar } from '../components/charts'
import { pct } from '../components/chart-utils'
import { ExampleStrip } from '../components/live'
import { Callout, Figure, StatGrid, Steps, Summary } from '../components/ui'

export default function Datos() {
  return (
    <>
      <p>
        El proyecto usa <strong>LUMOS</strong> (Shi et al., 2025), el primer conjunto de datos público y multimodal
        pensado para el cribado de osteoporosis lumbar. Junta, para cada paciente, sus radiografías y una tabla
        clínica con 52 columnas: edad, sexo, IMC, diagnóstico y BMD y T-score por vértebra (L1–L4), medidos con DXA.
        Todos los datos están anonimizados y se relacionan por un <code>patient_id</code>.
      </p>

      <StatGrid stats={[
        { value: '803', label: 'pacientes' },
        { value: '1.616', label: 'radiografías', detail: 'anteroposterior (AP) y lateral' },
        { value: '792', label: 'pacientes con las dos vistas' },
        { value: '64 años', label: 'edad media', detail: '646 mujeres y 155 hombres, de los 801 con datos completos' },
      ]} />

      <h2>Cuántos pacientes hay de cada clase</h2>
      <Figure caption="Pacientes por diagnóstico. El desbalanceo es moderado: hay algo menos de osteoporosis.">
        <BarList title="Pacientes por clase" bars={[
          { label: 'Sano', value: 289, color: 'var(--mark-sano)', detail: 'pacientes sanos · BMD media 1,17' },
          { label: 'Osteopenia', value: 291, color: 'var(--mark-osteopenia)', detail: 'pacientes con osteopenia · BMD media 0,91' },
          { label: 'Osteoporosis', value: 223, color: 'var(--mark-osteoporosis)', detail: 'pacientes con osteoporosis · BMD media 0,72' },
        ]} format={(v) => `${v}`} />
      </Figure>

      <Callout kind="warn" title="Limitación de origen">
        Todos los pacientes son de origen asiático y la mayoría son mujeres de entre 60 y 79 años. El modelo no se
        puede dar por válido en otras poblaciones sin probarlo antes con ellas.
      </Callout>

      <h2>Del DICOM a la entrada del modelo</h2>
      <p>
        Las radiografías llegan en formato <strong>DICOM</strong>, el estándar de imagen médica, y con tamaños muy
        distintos: de media 1.773 × 2.866 píxeles, desde 1.152 × 1.860 hasta 3.008 × 5.596. Cada una pasa por el
        mismo preprocesado, que se reutiliza tal cual en la aplicación:
      </p>
      <Steps steps={[
        { title: 'Leer el DICOM', text: 'Con pydicom. Si la imagen es MONOCHROME1 se invierte para que el hueso salga claro.' },
        { title: 'Normalizar', text: 'Min-max a [0, 1] y paso a 8 bits.' },
        { title: 'Redimensionar', text: 'A 224 × 224 píxeles, el tamaño con el que se preentrenó ResNet50.' },
        { title: 'CLAHE', text: 'Ecualización adaptativa del contraste (clip 2,0, rejilla 8 × 8) para realzar la estructura ósea.' },
        { title: 'Formato ImageNet', text: 'El canal gris se replica en tres y se normaliza con la media y desviación de ImageNet.' },
      ]} />

      <Figure caption="Tres radiografías del conjunto de test, una por clase, tal como las recibe el modelo. Las sirve la API desde el volumen de ejemplos; no forman parte de la web.">
        <ExampleStrip />
      </Figure>

      <h2>Una partición por paciente</h2>
      <p>
        El dataset se divide en entrenamiento, validación y test (70/15/15), <strong>estratificando por clase</strong> y
        <strong> por paciente</strong>: las dos radiografías de una persona van siempre al mismo conjunto. Si la vista
        AP cayera en entrenamiento y la lateral en test, el modelo podría reconocer al paciente por su anatomía y el
        resultado de test saldría inflado.
      </p>
      <Figure caption="Pacientes e imágenes de cada conjunto, con semilla fija (42).">
        <SplitBar parts={[
          { label: 'Entrenamiento', patients: 559, images: 1131, color: 'var(--s1)' },
          { label: 'Validación', patients: 121, images: 241, color: 'var(--s2)' },
          { label: 'Test', patients: 121, images: 244, color: 'var(--s3)' },
        ]} />
      </Figure>
      <p>
        Durante el entrenamiento cada imagen cuenta como un ejemplo independiente, pero la evaluación se hace
        <strong> por paciente</strong>, promediando las predicciones de sus dos vistas, como haría un radiólogo.
      </p>

      <h2>Aumento de datos</h2>
      <p>
        Solo en entrenamiento, cada imagen se altera al azar en cada época para que el modelo no memorice: rotación
        de hasta ±15°, recorte aleatorio del {pct(0.75, 0)} al {pct(1, 0)} del área, volteo horizontal, cambios de
        brillo y contraste de ±20 % y una pequeña traslación. Son transformaciones plausibles: una radiografía
        volteada sigue teniendo la misma densidad ósea.
      </p>

      <Summary items={[
        'LUMOS: 803 pacientes y 1.616 radiografías, casi siempre una AP y una lateral.',
        'Preprocesado: DICOM → 224 × 224 → CLAHE → formato ImageNet, igual en entrenamiento y en la aplicación.',
        'Partición 70/15/15 por paciente y estratificada, y evaluación agregando las dos vistas de cada paciente.',
      ]} />
    </>
  )
}
