import { Link } from 'react-router'
import { Callout, StatGrid } from '../components/ui'

export default function Conclusiones() {
  return (
    <>
      <p>
        El proyecto consigue un <strong>modelo funcional</strong>: a partir de una sola radiografía lumbar convencional y
        tres datos clínicos, clasifica la fragilidad ósea en tres niveles y estima la densidad mineral ósea.
      </p>

      <StatGrid stats={[
        { value: '66,9 %', label: 'accuracy' },
        { value: '0,836', label: 'AUC' },
        { value: '76,2 %', label: 'PCC de la BMD' },
        { value: '85,3 %', label: 'sensibilidad a osteoporosis' },
      ]} />

      <h2>Lo que se ha aprendido</h2>
      <div className="findings">
        <div className="card">
          <h3>El límite son los datos</h3>
          <p>Ninguna de las cinco mejoras centradas en la imagen supera a la referencia, y la única que sí lo hace añade
            información de fuera de la imagen. Con 803 pacientes, el modelo ya aprovecha razonablemente bien lo que la
            radiografía le puede dar.</p>
        </div>
        <div className="card">
          <h3>Priorizar el error caro</h3>
          <p>Elegir por la sensibilidad a la osteoporosis, y no por la accuracy, cambia las decisiones: descarta la
            estrategia diferencial y acepta perder osteopenia a cambio de no pasar por alto los casos graves.</p>
        </div>
        <div className="card">
          <h3>La osteopenia es la clase difícil</h3>
          <p>Como anticipaba el estado del arte: es una franja estrecha entre dos fronteras, y su sensibilidad o su
            especificidad es la peor en todos los experimentos.</p>
        </div>
        <div className="card">
          <h3>Competitivo en regresión</h3>
          <p>Con una ResNet50 genérica, la estimación de la BMD supera a FCoTNet y se acerca a OsteoNet, aunque la
            clasificación queda por debajo de ambas.</p>
        </div>
        <div className="card">
          <h3>Resultados negativos documentados</h3>
          <p>Cada vía descartada se probó cambiando una sola cosa y se documentó con el mismo detalle que las que
            funcionaron. El ROI en dos pasos es el mejor ejemplo: separó la pérdida de datos del efecto del recorte.</p>
        </div>
        <div className="card">
          <h3>Interpretabilidad útil</h3>
          <p>Grad-CAM validó que el modelo mira la columna en la vista AP, destapó el sesgo hacia el material quirúrgico
            y sirvió para auto-etiquetar el detector YOLO.</p>
        </div>
      </div>

      <h2>Limitaciones</h2>
      <ul>
        <li><strong>Tamaño del dataset.</strong> 803 pacientes, frente a 1.730–6.908 en estudios similares.</li>
        <li><strong>Población.</strong> Solo pacientes de origen asiático, sobre todo mujeres de 60 a 79 años.</li>
        <li><strong>Sesgo hacia el material quirúrgico</strong>, que además se propagó a las cajas del detector YOLO.</li>
        <li><strong>Vista lateral</strong> menos fiable que la anteroposterior.</li>
        <li><strong>Arquitectura genérica</strong>, con un ajuste de hiperparámetros limitado al alcance de un TFM.</li>
      </ul>

      <h2>Trabajo futuro</h2>
      <ul>
        <li>Ampliar el dataset combinando LUMOS con otros conjuntos públicos o con datos de hospitales.</li>
        <li>Anotar a mano, con más rigor, las regiones de interés, para saber si el problema es recortar o el detector.</li>
        <li>Mitigar el sesgo hacia prótesis y tornillos (excluirlos o representarlos de forma controlada).</li>
        <li>Validar con otras poblaciones antes de cualquier uso clínico.</li>
        <li>Entrenar un modelo específico para la vista lateral.</li>
      </ul>

      <Callout kind="idea" title="Pruébalo">
        El modelo final está desplegado en esta misma aplicación: <Link to="/">sube una radiografía o elige un ejemplo
        de LUMOS</Link> y mira la predicción, la BMD estimada y dónde se fijó el modelo.
      </Callout>
    </>
  )
}
