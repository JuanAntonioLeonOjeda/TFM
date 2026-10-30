import type { ReactNode } from 'react'
import { Callout, Summary } from '../components/ui'

// Referencias de la memoria del TFM, con su DOI. FCoTNet y OsteoNet son los modelos de referencia
// publicados en el artículo de LUMOS
const REFERENCES = {
  pignolo: {
    citation: 'Pignolo, R. J., Connell, J. J., Briggs, W., Kelly, C. J., Tromans, C., Sultana, N. y Brady, J. M. (2025). Opportunistic assessment of osteoporosis using hip and pelvic X-rays with OsteoSight: validation of an AI-based tool in a US population. Osteoporosis International.',
    doi: '10.1007/s00198-025-07487-0',
  },
  gharehmohammadi: {
    citation: 'Gharehmohammadi, F. y Sebro, R. (2025). Deep learning opportunistic screening for osteoporosis and osteopenia using radiographs of the foot or ankle - a pilot study. European Journal of Radiology, 184, 111980.',
    doi: '10.1016/j.ejrad.2025.111980',
  },
  mao: {
    citation: 'Mao, L., Xia, Z., Pan, L., Chen, J., Liu, X., Li, Z., Yan, Z., Lin, G., Wen, H. y Liu, B. (2022). Deep learning for screening primary osteopenia and osteoporosis using spine radiographs and patient clinical covariates in a Chinese population. Frontiers in Endocrinology, 13, 971877.',
    doi: '10.3389/fendo.2022.971877',
  },
  sato: {
    citation: 'Sato, Y., Yamamoto, N., Inagaki, N., Iesaki, Y., Asamoto, T., Suzuki, T. y Takahara, S. (2022). Deep learning for bone mineral density and T-score prediction from chest X-rays: a multicenter study. Biomedicines, 10(9), 2323.',
    doi: '10.3390/biomedicines10092323',
  },
  shi: {
    citation: 'Shi, K., Shen, Q., Ye, Z., Jiang, L., Bu, J. y Wang, H. (2025). LUMOS: A lumbar multimodal osteoporosis screening dataset with X-ray and CT images. Proceedings of the 33rd ACM International Conference on Multimedia, 13250–13257.',
    doi: '10.1145/3746027.3758282',
  },
} as const

type RefKey = keyof typeof REFERENCES

const doiUrl = (key: RefKey) => `https://doi.org/${REFERENCES[key].doi}`

const STUDIES: { name: string; ref: RefKey; image: string; task: string; result: string }[] = [
  { name: 'OsteoSight (Pignolo et al., 2025)', ref: 'pignolo', image: 'Cadera', task: 'Binaria', result: 'Sensibilidad 0,628' },
  { name: 'Gharehmohammadi y Sebro (2025)', ref: 'gharehmohammadi', image: 'Pie y tobillo', task: 'Binaria', result: 'Sensibilidad 0,899 · VPN 0,741' },
  { name: 'Mao et al. (2022)', ref: 'mao', image: 'Lumbar', task: 'Osteoporosis', result: 'AUC 0,91–0,94 con 6.908 pacientes' },
  { name: 'Sato et al. (2022)', ref: 'sato', image: 'Tórax', task: '3 clases + BMD', result: 'AUC 0,84 · PCC 0,63' },
  { name: 'FCoTNet (LUMOS)', ref: 'shi', image: 'Lumbar', task: '3 clases + BMD', result: 'Accuracy 72,3 % · PCC 74,7 %' },
  { name: 'OsteoNet (LUMOS)', ref: 'shi', image: 'Lumbar', task: '3 clases + BMD', result: 'Accuracy 73,6 % · PCC 77,8 %' },
]

function PaperLink({ refKey, children }: { refKey: RefKey; children: ReactNode }) {
  return <a href={doiUrl(refKey)} target="_blank" rel="noopener noreferrer">{children}</a>
}

export default function EstadoDelArte() {
  return (
    <>
      <p>
        Detectar baja densidad ósea en radiografías convencionales es un campo de investigación activo. Los
        trabajos se diferencian en tres cosas: <strong>qué zona del cuerpo</strong> miran, <strong>qué tarea</strong> resuelven
        (detectar sí o no, tres clases o estimar la BMD) y con <strong>cuántos pacientes</strong> se entrenan.
      </p>

      <div className="table-wrap">
        <table className="compare left">
          <thead>
            <tr><th scope="col">Estudio</th><th scope="col">Imagen</th><th scope="col">Tarea</th><th scope="col">Resultado</th></tr>
          </thead>
          <tbody>
            {STUDIES.map((s) => (
              <tr key={s.name} className={s.name.includes('LUMOS') ? 'key-row' : undefined}>
                <th scope="row"><PaperLink refKey={s.ref}>{s.name}</PaperLink></th><td>{s.image}</td><td>{s.task}</td><td>{s.result}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <h2>Lo que enseñan estos trabajos</h2>
      <ul>
        <li>
          <strong>La clasificación binaria se queda corta.</strong> OsteoSight deja sin detectar casi un 40 % de los
          pacientes con baja densidad, y aunque el estudio de pie y tobillo alcanza una sensibilidad de 0,899, uno de
          cada cuatro negativos es en realidad un paciente con baja densidad. Además, separar "osteoporosis sí o no"
          ignora la osteopenia, donde ocurren la mayoría de fracturas.
        </li>
        <li>
          <strong>El volumen de datos manda.</strong> Mao et al. alcanzan un AUC de hasta 0,94, pero con 6.908
          pacientes: más de ocho veces los de LUMOS.
        </li>
        <li>
          <strong>La osteopenia es la clase más difícil</strong> en todos los trabajos de tres clases.
        </li>
        <li>
          <strong>FCoTNet y OsteoNet</strong> son la referencia directa: se entrenaron con el mismo dataset LUMOS y
          con arquitecturas diseñadas específicamente para la osteoporosis.
        </li>
      </ul>

      <Callout kind="idea" title="La aportación de este trabajo">
        Clasificar en tres niveles y estimar la BMD a la vez, con una arquitectura de propósito general (ResNet50)
        en lugar de una diseñada para el problema; estudiar de forma sistemática qué vías de mejora funcionan y
        cuáles no; y analizar con Grad-CAM dónde se fija el modelo.
      </Callout>

      <h2>Por qué ResNet50 y no un Vision Transformer</h2>
      <p>
        Los Vision Transformers dominan muchas tareas de visión, pero necesitan grandes volúmenes de datos para
        entrenarse bien. Con unos 800 pacientes, una red convolucional preentrenada en ImageNet y ajustada con
        <em> transfer learning</em> es una opción más segura: casi todo lo que sabe "ver" ya viene aprendido, y solo
        hay que adaptarlo a las radiografías.
      </p>

      <h2>Referencias</h2>
      <ol className="references">
        {(Object.keys(REFERENCES) as RefKey[]).map((key) => (
          <li key={key}>
            {REFERENCES[key].citation}{' '}
            <PaperLink refKey={key}>doi:{REFERENCES[key].doi}</PaperLink>
          </li>
        ))}
      </ol>

      <Summary items={[
        'La clasificación binaria pierde la osteopenia, que es donde ocurren la mayoría de fracturas.',
        'FCoTNet (72,3 %) y OsteoNet (73,6 %) marcan la referencia en LUMOS.',
        'Este trabajo usa una arquitectura genérica y mide qué mejoras ayudan de verdad.',
      ]} />
    </>
  )
}
