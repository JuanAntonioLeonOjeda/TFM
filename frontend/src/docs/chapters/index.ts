import type { ComponentType } from 'react'
import Problema from './01-problema'
import EstadoDelArte from './02-estado-del-arte'
import Datos from './03-datos'
import Modelo from './04-modelo'
import Entrenamiento from './05-entrenamiento'
import Estrategias from './06-estrategias'
import Mejoras from './07-mejoras'
import ModeloFinal from './08-modelo-final'
import Interpretabilidad from './09-interpretabilidad'
import Despliegue from './10-despliegue'
import Conclusiones from './11-conclusiones'

export interface Chapter {
  slug: string
  title: string
  summary: string
  Content: ComponentType
}

// Capítulos del proyecto, en el orden de la memoria del TFM
export const CHAPTERS: Chapter[] = [
  { slug: 'el-problema', title: 'El problema: una enfermedad silenciosa', Content: Problema,
    summary: 'Qué es la osteoporosis, cómo se diagnostica con el T-score y qué se quiere conseguir con una radiografía convencional.' },
  { slug: 'estado-del-arte', title: 'Estado del arte', Content: EstadoDelArte,
    summary: 'Qué consiguen otros trabajos, cuáles son las referencias en LUMOS y qué aporta este.' },
  { slug: 'los-datos', title: 'Los datos: LUMOS', Content: Datos,
    summary: 'El dataset, el camino del DICOM a la entrada del modelo y una partición por paciente.' },
  { slug: 'el-modelo', title: 'El modelo: LumosNet', Content: Modelo,
    summary: 'ResNet50 preentrenada, covariables clínicas y dos cabezas: una para la clase y otra para la BMD.' },
  { slug: 'entrenamiento', title: 'Cómo se entrena', Content: Entrenamiento,
    summary: 'Dos estrategias de fine-tuning, la parada temprana y una configuración reproducible.' },
  { slug: 'estrategias', title: 'Diferencial frente a por fases', Content: Estrategias,
    summary: 'Qué estrategia generaliza mejor, con qué criterio se decide y si basta con una sola partición.' },
  { slug: 'vias-de-mejora', title: 'Seis vías de mejora', Content: Mejoras,
    summary: 'Regularización, balanceo, recortes, resolución y covariables: qué se probó, qué pasó y qué se quedó.' },
  { slug: 'modelo-final', title: 'El modelo final', Content: ModeloFinal,
    summary: 'Sus resultados, el compromiso entre clases y la comparación con FCoTNet y OsteoNet.' },
  { slug: 'interpretabilidad', title: 'Interpretabilidad: Grad-CAM y YOLO', Content: Interpretabilidad,
    summary: 'Dónde mira el modelo, qué sesgos tiene y cómo los mapas de calor sirvieron para entrenar un detector.' },
  { slug: 'despliegue', title: 'De la memoria a la aplicación', Content: Despliegue,
    summary: 'La API, la web y Docker, y cómo se protege lo que se sube.' },
  { slug: 'conclusiones', title: 'Conclusiones y trabajo futuro', Content: Conclusiones,
    summary: 'Lo que se ha conseguido, lo que se ha aprendido, sus límites y por dónde seguir.' },
]
