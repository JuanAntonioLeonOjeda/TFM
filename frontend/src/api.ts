export const CLASS_NAMES = ['Sano', 'Osteopenia', 'Osteoporosis'] as const
export type ClassName = (typeof CLASS_NAMES)[number]
export type Gender = 'female' | 'male'

export interface Example {
  id: string
  age: number
  gender: Gender
  bmi: number
  true_class: ClassName
  true_bmd: number
  true_t_score: number
}

export interface Prediction {
  predicted_class: ClassName
  probabilities: Record<ClassName, number>
  bmd_estimated: number
  heatmap_png_base64: string
  image_png_base64: string
  ground_truth?: { class: ClassName; bmd: number; t_score: number }
}

export interface TestMetrics {
  model: string
  n_images: number
  accuracy: number
  adjacent_error_share: number
  confusion_matrix: number[][]
}

export interface PatientData {
  age: number
  gender: Gender
  bmi: number
}

const BASE = '/api'

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, init)
  const body = await res.json().catch(() => null)
  if (!res.ok) throw new Error(errorMessage(body))
  return body as T
}

// FastAPI devuelve detail como texto o como lista de errores de validación
function errorMessage(body: unknown): string {
  const detail = (body as { detail?: unknown } | null)?.detail
  if (typeof detail === 'string') return detail
  if (Array.isArray(detail)) {
    return detail.map((d: { loc: string[]; msg: string }) => `${d.loc.at(-1)}: ${d.msg}`).join(' · ')
  }
  return 'No se pudo obtener la predicción'
}

export const getPrivacyNotice = () => request<{ notice: string }>('/privacy').then((r) => r.notice)

export const getTestMetrics = () =>
  request<{ test_metrics: TestMetrics | null }>('/model-info').then((r) => r.test_metrics)

export const getExamples = () => request<Example[]>('/examples')

export const exampleImageUrl = (id: string) => `${BASE}/examples/${encodeURIComponent(id)}/image`

export const predictExample = (id: string) =>
  request<Prediction>(`/examples/${encodeURIComponent(id)}/predict`, { method: 'POST' })

export function predictUpload(dicom: File, patient: PatientData) {
  const form = new FormData()
  form.append('dicom', dicom)
  form.append('age', String(patient.age))
  form.append('gender', patient.gender)
  form.append('bmi', String(patient.bmi))
  form.append('heatmap', 'true')
  return request<Prediction>('/predict', { method: 'POST', body: form })
}
