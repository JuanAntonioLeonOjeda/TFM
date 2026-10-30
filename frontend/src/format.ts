import type { ClassName, Gender } from './api'

export const CLASS_COLORS: Record<ClassName, string> = {
  Sano: 'var(--sano)',
  Osteopenia: 'var(--osteopenia)',
  Osteoporosis: 'var(--osteoporosis)',
}

export const GENDER_LABELS: Record<Gender, string> = { female: 'Mujer', male: 'Hombre' }

// Si las dos clases más probables están a menos de esto, la predicción se marca como dudosa
export const DOUBTFUL_MARGIN = 0.1

export const fmtSigned = (x: number, digits = 1) => `${x > 0 ? '+' : ''}${fmt(x, digits)}`

export const fmt = (x: number, digits = 3) =>
  x.toLocaleString('es-ES', { minimumFractionDigits: digits, maximumFractionDigits: digits })
