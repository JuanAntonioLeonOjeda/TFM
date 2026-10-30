import type { ClassName, Gender } from './api'

export const CLASS_COLORS: Record<ClassName, string> = {
  Sano: 'var(--sano)',
  Osteopenia: 'var(--osteopenia)',
  Osteoporosis: 'var(--osteoporosis)',
}

export const GENDER_LABELS: Record<Gender, string> = { female: 'Mujer', male: 'Hombre' }

export const fmt = (x: number, digits = 3) =>
  x.toLocaleString('es-ES', { minimumFractionDigits: digits, maximumFractionDigits: digits })
