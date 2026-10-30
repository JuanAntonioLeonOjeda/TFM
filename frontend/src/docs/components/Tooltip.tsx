import { useCallback, useState, type PointerEvent, type ReactNode } from 'react'

export interface TooltipState {
  x: number
  y: number
  content: ReactNode
}

// Un único tooltip por gráfica, posicionado respecto a su contenedor. Las marcas lo activan con
// puntero o con foco de teclado, y todo lo que muestra también está en etiquetas o en una tabla
export function useTooltip() {
  const [tip, setTip] = useState<TooltipState | null>(null)

  const bind = useCallback((content: ReactNode) => ({
    tabIndex: 0,
    onPointerMove: (e: PointerEvent<Element>) => {
      const box = (e.currentTarget as Element).closest('.chart')!.getBoundingClientRect()
      setTip({ x: e.clientX - box.left, y: e.clientY - box.top, content })
    },
    onPointerLeave: () => setTip(null),
    onFocus: (e: { currentTarget: Element }) => {
      const box = e.currentTarget.closest('.chart')!.getBoundingClientRect()
      const mark = e.currentTarget.getBoundingClientRect()
      setTip({ x: mark.left - box.left + mark.width / 2, y: mark.top - box.top, content })
    },
    onBlur: () => setTip(null),
  }), [])

  const node = tip && (
    <div className="chart-tooltip" style={{ left: tip.x, top: tip.y }} role="status">{tip.content}</div>
  )
  return { bind, node }
}
