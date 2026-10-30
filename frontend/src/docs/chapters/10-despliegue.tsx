import { Link } from 'react-router'
import { DeployDiagram } from '../components/diagrams'
import { Callout, Figure, Summary } from '../components/ui'

const ENDPOINTS = [
  ['POST', '/predict', 'DICOM + edad, sexo e IMC; devuelve clase, probabilidades, BMD y, si se pide, el Grad-CAM'],
  ['GET', '/examples', 'Radiografías de ejemplo del test de LUMOS con su diagnóstico real'],
  ['POST', '/examples/{id}/predict', 'Predicción y Grad-CAM de un ejemplo, con la verdad terreno'],
  ['GET', '/model-info', 'Exactitud y matriz de confusión del modelo en test'],
  ['GET', '/privacy', 'Aviso de privacidad'],
  ['GET', '/health', 'Estado del servicio, para Docker'],
]

export default function Despliegue() {
  return (
    <>
      <p>
        Un modelo en un notebook no le sirve a nadie. La última parte convierte LumosNet en una aplicación: una API que
        sirve el modelo y una web para usarlo, empaquetadas en dos contenedores Docker.{' '}
        <Link to="/">Puedes probarla aquí mismo</Link>.
      </p>

      <Figure caption="Recorrido de una radiografía desde el navegador hasta el modelo y vuelta." wide>
        <DeployDiagram />
      </Figure>

      <h2>Decisiones de diseño</h2>
      <ul>
        <li>
          <strong>Un solo preprocesado.</strong> La API reutiliza el mismo código que el entrenamiento (inversión
          MONOCHROME1, 224 × 224, CLAHE, normalización ImageNet). Dos implementaciones del preprocesado acabarían
          divergiendo, y el modelo vería en producción imágenes distintas de las que aprendió.
        </li>
        <li>
          <strong>Las covariables se normalizan con las estadísticas de entrenamiento</strong>, guardadas en un JSON junto
          al checkpoint. Si se normalizara con los datos que llegan, el mismo paciente "cambiaría" según quién más
          hubiera llegado ese día.
        </li>
        <li>
          <strong>El modelo no va dentro de la imagen de Docker</strong>: se monta como volumen de solo lectura. Se puede
          actualizar sin reconstruir, y la imagen no contiene artefactos derivados de datos de pacientes. Lo mismo con
          las radiografías de ejemplo.
        </li>
        <li>
          <strong>Validación de entradas.</strong> FastAPI rechaza edades, sexos o IMC fuera de rango, y un fichero que no
          sea DICOM devuelve un error claro en vez de un fallo interno.
        </li>
        <li>
          <strong>El backend no se expone</strong>: solo nginx publica un puerto, y el resto viaja por la red interna de
          Docker Compose.
        </li>
      </ul>

      <h2>Privacidad: nada se guarda</h2>
      <p>
        Las radiografías son datos de salud. La aplicación las procesa <strong>solo en memoria</strong> y las descarta al
        responder: ni nginx ni FastAPI las escriben en ficheros temporales (los dos lo harían por defecto con ficheros
        grandes), las respuestas llevan <code>Cache-Control: no-store</code> para que no queden en cachés, y nada de lo
        que se sube se registra ni se usa para entrenar. Un test automático comprueba que una subida grande no toca el
        disco.
      </p>

      <h2>La API</h2>
      <div className="table-wrap">
        <table className="compare left">
          <thead><tr><th scope="col">Método</th><th scope="col">Ruta</th><th scope="col">Qué hace</th></tr></thead>
          <tbody>
            {ENDPOINTS.map(([m, p, d]) => <tr key={p}><td><code>{m}</code></td><th scope="row"><code>{p}</code></th><td>{d}</td></tr>)}
          </tbody>
        </table>
      </div>

      <Callout kind="warn" title="Lo que faltaría para un uso clínico real">
        <ul>
          <li><strong>Validación externa</strong> con radiografías de otros hospitales, equipos y poblaciones.</li>
          <li><strong>Calibración</strong>, para que una probabilidad del 70 % signifique un 70 % de aciertos.</li>
          <li><strong>Umbral de decisión</strong> elegido según el coste de cada error.</li>
          <li><strong>Monitorización</strong> de las entradas y las predicciones, para detectar deriva.</li>
          <li><strong>Regulación</strong>: un software de apoyo al diagnóstico es un producto sanitario (MDR) y un sistema de
            alto riesgo según el Reglamento europeo de IA.</li>
        </ul>
      </Callout>

      <Summary items={[
        'API FastAPI + web React, en dos contenedores; el modelo y los ejemplos se montan como volúmenes.',
        'El mismo preprocesado y las mismas estadísticas de covariables que en entrenamiento.',
        'Las radiografías se procesan en memoria y no se guardan ni se usan para entrenar.',
      ]} />
    </>
  )
}
