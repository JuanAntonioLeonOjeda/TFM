import { EXPERIMENT_METRICS } from '../metrics'
import { LiveGradCam } from '../components/live'
import { CompareTable } from '../components/tables'
import { experimentRows } from '../components/table-rows'
import { Callout, Figure, StatGrid, Steps, Summary } from '../components/ui'

export default function Interpretabilidad() {
  const yolo = EXPERIMENT_METRICS['final-roi-yolo'].patient
  return (
    <>
      <p>
        En medicina no basta con acertar: hay que poder comprobar <strong>dónde mira</strong> el modelo. Si un modelo
        clasifica bien porque se fija en la etiqueta impresa en la placa o en el borde del colimador, ha aprendido un
        atajo que no le servirá en otro hospital. Para revisarlo se usan mapas de calor con <strong>Grad-CAM</strong>.
      </p>

      <h2>Cómo se calcula</h2>
      <Steps steps={[
        { title: 'Gradiente', text: 'Se calcula cuánto cambia la salida (la probabilidad de la clase, o la BMD) al cambiar cada mapa de activación del último bloque de ResNet50.' },
        { title: 'Peso por canal', text: 'Ese gradiente indica qué canales empujan la predicción y cuáles la frenan.' },
        { title: 'Suma ponderada', text: 'Se suman los mapas ponderados y se queda solo la parte positiva: lo que apoya la predicción.' },
        { title: 'Superposición', text: 'El mapa (7 × 7 en ese bloque) se reescala al tamaño de la imagen y se pinta encima.' },
      ]} />
      <p>
        Se usa la librería <code>pytorch-grad-cam</code> con su variante <strong>HiResCAM</strong>, más fiel al modelo en
        redes como ResNet50. Como la librería espera una sola salida, se aísla cada cabeza (clasificación o BMD) y se
        explica por separado.
      </p>

      <Figure caption="Grad-CAM calculado en este momento por la API sobre las radiografías de ejemplo. El color indica las zonas que más pesaron en la clase predicha: rojo, mucho; azul, poco." wide>
        <LiveGradCam />
      </Figure>

      <h2>Lo que muestran los mapas</h2>
      <p>Se analizaron de forma exploratoria las dos vistas de 10 pacientes:</p>
      <div className="three-col">
        <div className="card ok"><h3>Vista AP</h3><p>La atención se concentra de forma consistente sobre la columna lumbar.</p></div>
        <div className="card warn"><h3>Vista lateral</h3><p>Menos fiable: el foco sale a menudo de la columna. Una predicción basada solo en la lateral debe tomarse con cautela.</p></div>
        <div className="card warn"><h3>Material quirúrgico</h3><p>Tornillos y prótesis atraen la atención. Los 2 pacientes con prótesis se clasificaron como sanos, sin tendencia a marcarlos como anómalos.</p></div>
      </div>
      <Callout kind="warn" title="Lo que un mapa de calor no demuestra">
        Que el mapa caiga sobre la columna indica qué zonas influyen en la salida, no qué rasgo usa el modelo (densidad
        trabecular, forma, grosor de la cortical). Sirve para detectar atajos y errores groseros, no como prueba de
        validez clínica.
      </Callout>

      <h2>De Grad-CAM a un detector: ROI adaptativo con YOLO</h2>
      <p>
        El ROI fijo falló, en parte, porque una franja igual para todos no se adapta a cada paciente. Para probar un
        recorte adaptativo hacía falta un detector de la columna, y entrenarlo exige anotar cajas a mano. Los mapas de
        Grad-CAM permitieron ahorrar buena parte de ese trabajo (<em>auto-etiquetado</em>):
      </p>
      <Steps steps={[
        { title: 'Cajas automáticas', text: 'Se umbraliza el mapa de calor y se toma la mayor zona conexa como caja.' },
        { title: 'Filtros', text: 'Área entre el 3 % y el 75 % de la imagen, centro entre el 30 % y el 70 % del ancho y caja más alta que ancha. Pasan 342 de 803 pacientes.' },
        { title: 'Revisión manual', text: 'Cada caja se revisa y corrige en makesense.ai.' },
        { title: 'YOLOv8n', text: 'El detector más pequeño de la familia, 100 épocas a 224 × 224.' },
        { title: 'Recorte', text: 'En la vista AP, si la confianza supera 0,5, se recorta con un 8 % de margen; la lateral no se toca.' },
      ]} />

      <StatGrid stats={[
        { value: '0,751', label: 'mAP50 del detector', detail: 'localiza bien la columna' },
        { value: '0,349', label: 'mAP50-95', detail: 'con bordes poco precisos' },
        { value: '50 / 64', label: 'columnas detectadas en validación', detail: '14 sin detectar y 13 falsas alarmas' },
      ]} />

      <p>Al reentrenar el modelo final con las radiografías recortadas por el detector, todo empeoró:</p>
      <CompareTable columns={['Covariables (final)', 'Covariables + ROI YOLO']} rows={[
        ...experimentRows(['v8', 'final-roi-yolo']),
        { label: 'Sensibilidad a osteoporosis', values: [EXPERIMENT_METRICS.v8.patient.sens[2], yolo.sens[2]] },
      ]} />
      <p>
        La sensibilidad a la osteoporosis cae de 0,853 a 0,500, y la variante se descarta. Junto con el ROI fijo, la
        conclusión es la misma: <strong>recortar quita contexto útil</strong>. Queda abierta una duda razonable: las cajas
        venían de Grad-CAM, con sus sesgos (como el material quirúrgico), y una anotación completamente manual podría
        separar mejor si el problema es recortar o es el detector.
      </p>

      <Summary items={[
        'Grad-CAM confirma que el modelo mira la columna en la vista AP; en la lateral, mucho menos.',
        'Las prótesis atraen la atención del modelo: un sesgo a corregir.',
        'Grad-CAM sirvió para auto-etiquetar un detector YOLO, pero recortar con él empeoró el modelo.',
      ]} />
    </>
  )
}
