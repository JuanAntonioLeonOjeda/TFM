import { useLayoutEffect, useRef, useState } from 'react'

// Las gráficas se dibujan al ancho real del contenedor para que el texto no encoja en móvil
export function useWidth<T extends HTMLElement>(fallback = 640) {
  const ref = useRef<T>(null)
  const [width, setWidth] = useState(fallback)
  useLayoutEffect(() => {
    if (!ref.current) return
    const observer = new ResizeObserver(([entry]) => setWidth(Math.round(entry.contentRect.width)))
    observer.observe(ref.current)
    return () => observer.disconnect()
  }, [])
  return [ref, width] as const
}

export const pct = (x: number, digits = 1) => `${(x * 100).toLocaleString('es-ES', { maximumFractionDigits: digits, minimumFractionDigits: digits })} %`
export const num = (x: number, digits = 3) => x.toLocaleString('es-ES', { maximumFractionDigits: digits, minimumFractionDigits: digits })

// T-score en LUMOS: T = (BMD − 1,114) / 0,120
export const tScore = (bmd: number) => (bmd - 1.114) / 0.12
export const classOf = (t: number) => (t >= -1 ? 'Sano' : t > -2.5 ? 'Osteopenia' : 'Osteoporosis')
