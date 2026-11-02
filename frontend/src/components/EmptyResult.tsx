// Estado inicial del panel de resultados: una radiografía lumbar esquemática con un foco de "atención"
// sobre las vértebras, a modo de anticipo del mapa Grad-CAM
const VERTEBRAE = [
  { y: 34, w: 46 },
  { y: 66, w: 50 },
  { y: 98, w: 54 },
  { y: 130, w: 58 },
  { y: 162, w: 62 },
]

export function EmptyResult() {
  return (
    <div className="empty-result">
      <svg className="empty-art" viewBox="0 0 160 220" width="160" height="220" aria-hidden="true">
        <defs>
          <radialGradient id="empty-cam">
            <stop offset="0" stopColor="var(--s2)" stopOpacity=".55" />
            <stop offset=".55" stopColor="var(--osteopenia)" stopOpacity=".22" />
            <stop offset="1" stopColor="var(--osteopenia)" stopOpacity="0" />
          </radialGradient>
        </defs>
        <rect x="8" y="8" width="144" height="204" rx="14" fill="var(--bg)" stroke="var(--border)" strokeWidth="2" />
        <ellipse cx="80" cy="128" rx="58" ry="52" fill="url(#empty-cam)" />
        {VERTEBRAE.map(({ y, w }) => (
          <g key={y}>
            <rect x={80 - w / 2} y={y} width={w} height="24" rx="7" fill="var(--panel)" stroke="var(--muted)"
                  strokeWidth="2" />
            <line x1="80" y1={y + 24} x2="80" y2={y + 32} stroke="var(--border)" strokeWidth="6" />
          </g>
        ))}
        <circle cx="122" cy="176" r="15" fill="none" stroke="var(--accent)" strokeWidth="3" />
        <line x1="133" y1="187" x2="145" y2="199" stroke="var(--accent)" strokeWidth="4" strokeLinecap="round" />
      </svg>
      <p className="empty-title">Aquí aparecerá el resultado</p>
      <p className="empty-text">
        Sube una radiografía o elige un ejemplo de LUMOS. Verás la clase predicha con sus probabilidades, la
        densidad mineral ósea estimada y el mapa de calor de las zonas en las que se fijó el modelo.
      </p>
    </div>
  )
}
