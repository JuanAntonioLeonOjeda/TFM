import { useState, type ReactNode } from 'react'
import { classOf, num, pct, tScore, useWidth } from './chart-utils'
import { TipRow } from './TipRow'
import { useTooltip } from './Tooltip'

export interface Bar {
  label: string
  value: number
  color?: string
  detail?: string
  emphasis?: boolean
}

// Barras horizontales con la etiqueta a la izquierda y el valor al final de cada barra
export function BarList({ bars, max, format = (v) => String(v), title }: {
  bars: Bar[]
  max?: number
  format?: (v: number) => string
  title?: string
}) {
  const [ref, width] = useWidth<HTMLDivElement>()
  const { bind, node } = useTooltip()
  const labelW = Math.min(170, Math.max(96, width * 0.3))
  const valueW = 64
  const rowH = 30
  const top = max ?? Math.max(...bars.map((b) => b.value))
  const plotW = Math.max(40, width - labelW - valueW)
  const height = bars.length * rowH

  return (
    <div className="chart" ref={ref}>
      <svg width={width} height={height} role="img" aria-label={title}>
        {bars.map((b, i) => {
          const w = Math.max(2, (b.value / top) * plotW)
          const y = i * rowH
          return (
            <g key={b.label} className="mark" {...bind(<TipRow value={format(b.value)} label={b.detail ?? b.label} color={b.color ?? 'var(--s1)'} />)}>
              <rect x={0} y={y} width={width} height={rowH} fill="transparent" />
              <text x={labelW - 10} y={y + rowH / 2} className={`axis-label${b.emphasis ? ' strong' : ''}`} textAnchor="end" dominantBaseline="middle">{b.label}</text>
              <rect x={labelW} y={y + 7} width={w} height={rowH - 14} rx={4} fill={b.color ?? 'var(--s1)'} className="bar" />
              <text x={labelW + w + 8} y={y + rowH / 2} className={`value-label${b.emphasis ? ' strong' : ''}`} dominantBaseline="middle">{format(b.value)}</text>
            </g>
          )
        })}
      </svg>
      {node}
    </div>
  )
}

// Escala de T-score con las tres franjas de la OMS y un deslizador de BMD (T = (BMD − 1,114) / 0,120 en LUMOS)
const T_MIN = -5.5
const T_MAX = 3

export function TScoreExplorer({ rmse = 0.168 }: { rmse?: number }) {
  const [bmd, setBmd] = useState(0.9)
  const [ref, width] = useWidth<HTMLDivElement>()
  const t = tScore(bmd)
  const cls = classOf(t)
  const x = (v: number) => ((v - T_MIN) / (T_MAX - T_MIN)) * width
  const errT = rmse / 0.12
  const bands = [
    { from: T_MIN, to: -2.5, label: 'Osteoporosis', color: 'var(--mark-osteoporosis)' },
    { from: -2.5, to: -1, label: 'Osteopenia', color: 'var(--mark-osteopenia)' },
    { from: -1, to: T_MAX, label: 'Sano', color: 'var(--mark-sano)' },
  ]

  return (
    <div className="tscore">
      <div className="tscore-readout">
        <div><span className="kpi-label">BMD (DXA)</span><strong>{num(bmd)} g/cm²</strong></div>
        <div><span className="kpi-label">T-score</span><strong>{t > 0 ? '+' : ''}{num(t, 1)}</strong></div>
        <div><span className="kpi-label">Diagnóstico OMS</span><strong className={`cls-${cls.toLowerCase()}`}>{cls}</strong></div>
      </div>
      <div className="chart" ref={ref}>
        <svg width={width} height={86} role="img" aria-label={`Escala de T-score; el valor actual es ${num(t, 1)}, ${cls}`}>
          {bands.map((b) => (
            <g key={b.label}>
              <rect x={x(b.from) + 1} y={26} width={x(b.to) - x(b.from) - 2} height={22} rx={4} fill={b.color} opacity={0.85} />
              <text x={(x(b.from) + x(b.to)) / 2} y={16} textAnchor="middle" className="axis-label">{b.label}</text>
            </g>
          ))}
          <rect x={x(t - errT)} y={22} width={x(t + errT) - x(t - errT)} height={30} rx={4} className="err-band" />
          <line x1={x(t)} x2={x(t)} y1={18} y2={56} className="marker-line" />
          <circle cx={x(t)} cy={37} r={6} className="marker-dot" />
          {[-5, -4, -3, -2.5, -2, -1, 0, 1, 2, 3].map((v) => (
            <text key={v} x={x(v)} y={72} textAnchor="middle" className={`tick-label${v === -1 || v === -2.5 ? ' strong' : ''}`}>
              {v === -2.5 ? '−2,5' : v < 0 ? `−${-v}` : v}
            </text>
          ))}
        </svg>
      </div>
      <input type="range" min={0.48} max={1.45} step={0.005} value={bmd} aria-label="BMD en g/cm²"
             onChange={(e) => setBmd(Number(e.target.value))} />
      <p className="chart-note">
        La franja sombreada alrededor del punto es ± el error típico (RMSE) con el que el modelo final estima
        la BMD ({num(rmse)} g/cm², unas {num(errT, 1)} unidades de T-score): más ancha que toda la franja de
        osteopenia.
      </p>
    </div>
  )
}

// Barra apilada de train / validación / test con 2 px de separación entre segmentos
export function SplitBar({ parts }: { parts: { label: string; patients: number; images: number; color: string }[] }) {
  const [ref, width] = useWidth<HTMLDivElement>()
  const { bind, node } = useTooltip()
  const total = parts.reduce((s, p) => s + p.patients, 0)
  const offsets = parts.map((_, i) => parts.slice(0, i).reduce((s, p) => s + p.patients, 0))
  return (
    <div className="chart" ref={ref}>
      <svg width={width} height={64} role="img" aria-label="Partición por paciente 70/15/15">
        {parts.map((p, i) => {
          const x = (offsets[i] / total) * width
          const w = (p.patients / total) * width - 2
          return (
            <g key={p.label} className="mark"
               {...bind(<><TipRow value={`${p.patients} pacientes`} label={p.label} color={p.color} /><TipRow value={`${p.images} imágenes`} label="" /></>)}>
              <rect x={x} y={4} width={w} height={36} rx={4} fill={p.color} className="bar" />
              <text x={x + 10} y={27} className="bar-inner-label">{w > 120 ? `${p.label} · ${Math.round((p.patients / total) * 100)} %` : `${Math.round((p.patients / total) * 100)} %`}</text>
              <text x={x} y={58} className="tick-label">{w > 110 ? `${p.patients} pac. · ${p.images} img.` : p.label}</text>
            </g>
          )
        })}
      </svg>
      {node}
    </div>
  )
}

// Learning rate por época de las dos estrategias, en dos gráficas pequeñas con el mismo eje
export function LrSchedule() {
  const [ref, width] = useWidth<HTMLDivElement>()
  const small = width < 560
  const chartW = small ? width : (width - 24) / 2
  const epochs = 30
  const series = {
    differential: { backbone: () => 1e-4, heads: () => 1e-3 },
    phased: { backbone: (e: number) => (e <= 8 ? 0 : 1e-4), heads: (e: number) => (e <= 8 ? 1e-3 : 1e-4) },
  }
  return (
    <div ref={ref} className={`small-multiples${small ? ' stacked' : ''}`}>
      <LrPanel title="Diferencial" width={chartW} epochs={epochs} s={series.differential} />
      <LrPanel title="Por fases" width={chartW} epochs={epochs} s={series.phased} freeze={8} />
    </div>
  )
}

function LrPanel({ title, width, epochs, s, freeze }: {
  title: string
  width: number
  epochs: number
  s: { backbone: (e: number) => number; heads: (e: number) => number }
  freeze?: number
}) {
  const { bind, node } = useTooltip()
  const m = { l: 52, r: 74, t: 12, b: 30 }
  const h = 180
  const pw = width - m.l - m.r
  const ph = h - m.t - m.b
  const x = (e: number) => m.l + ((e - 1) / (epochs - 1)) * pw
  // Escala logarítmica de 1e-5 a 2e-3; "congelado" (lr = 0) se dibuja en una banda propia abajo
  const [lo, hi] = [Math.log10(3e-5), Math.log10(2e-3)]
  const y = (v: number) => (v === 0 ? m.t + ph : m.t + ((hi - Math.log10(v)) / (hi - lo)) * ph * 0.85)
  // Escalonada: la tasa cambia de golpe al empezar una época, no de forma gradual
  const path = (f: (e: number) => number) =>
    Array.from({ length: epochs }, (_, i) => i + 1)
      .map((e, i) => (i ? `H${x(e - 0.5)}V${y(f(e))}` : `M${x(e)},${y(f(e))}`))
      .join(' ') + `H${x(epochs)}`

  return (
    <div className="chart">
      <p className="chart-title">{title}</p>
      <svg width={width} height={h} role="img" aria-label={`Learning rate por época, estrategia ${title.toLowerCase()}`}>
        {[1e-3, 1e-4].map((v) => (
          <g key={v}>
            <line x1={m.l} x2={m.l + pw} y1={y(v)} y2={y(v)} className="grid" />
            <text x={m.l - 8} y={y(v)} textAnchor="end" dominantBaseline="middle" className="tick-label">{v === 1e-3 ? '10⁻³' : '10⁻⁴'}</text>
          </g>
        ))}
        <text x={m.l - 8} y={m.t + ph} textAnchor="end" dominantBaseline="middle" className="tick-label">0</text>
        {freeze && (
          <g>
            <rect x={x(1)} y={m.t} width={x(freeze + 0.5) - x(1)} height={ph} className="phase-band" />
            <text x={x(1) + 6} y={m.t + ph - 8} className="tick-label">backbone congelado</text>
          </g>
        )}
        <path d={path(s.heads)} className="line" stroke="var(--s2)" />
        <path d={path(s.backbone)} className="line" stroke="var(--s1)" />
        <text x={m.l + pw + 6} y={y(s.heads(epochs))} dominantBaseline="middle" className="series-label" fill="var(--text)">Cabezas</text>
        <text x={m.l + pw + 6} y={y(s.backbone(epochs)) + (s.heads(epochs) === s.backbone(epochs) ? 14 : 0)} dominantBaseline="middle" className="series-label" fill="var(--text)">Backbone</text>
        {[1, 8, 15, 22, 30].map((e) => (
          <text key={e} x={x(e)} y={h - 10} textAnchor="middle" className="tick-label">{e}</text>
        ))}
        {Array.from({ length: epochs }, (_, i) => i + 1).map((e) => (
          <rect key={e} x={x(e) - pw / epochs / 2} y={m.t} width={pw / epochs} height={ph} fill="transparent"
                {...bind(<><div className="tip-head">Época {e}</div>
                  <TipRow value={fmtLr(s.backbone(e))} label="backbone" color="var(--s1)" />
                  <TipRow value={fmtLr(s.heads(e))} label="cabezas" color="var(--s2)" /></>)} />
        ))}
      </svg>
      {node}
    </div>
  )
}

const fmtLr = (v: number) => (v === 0 ? 'congelado' : v === 1e-3 ? '10⁻³' : '10⁻⁴')

// Diagrama de puntos de una métrica por experimento, con la referencia marcada
export interface DotRow {
  key: string
  label: string
  value: number
  group?: string
  emphasis?: 'reference' | 'final'
}

export function DotPlot({ rows, domain, format, referenceLabel = 'Referencia' }: {
  rows: DotRow[]
  domain: [number, number]
  format: (v: number) => string
  referenceLabel?: string
}) {
  const [ref, width] = useWidth<HTMLDivElement>()
  const { bind, node } = useTooltip()
  const labelW = Math.min(210, Math.max(120, width * 0.38))
  const m = { r: 56, t: 22 }
  const rowH = 32
  const pw = width - labelW - m.r
  const h = m.t + rows.length * rowH + 24
  const x = (v: number) => labelW + ((v - domain[0]) / (domain[1] - domain[0])) * pw
  const reference = rows.find((r) => r.emphasis === 'reference')
  const ticks = niceTicks(domain)

  return (
    <div className="chart" ref={ref}>
      <svg width={width} height={h} role="img" aria-label="Comparación de experimentos">
        {ticks.map((v) => (
          <g key={v}>
            <line x1={x(v)} x2={x(v)} y1={m.t - 6} y2={h - 22} className="grid" />
            <text x={x(v)} y={h - 6} textAnchor="middle" className="tick-label">{format(v)}</text>
          </g>
        ))}
        {reference && (
          <g>
            <line x1={x(reference.value)} x2={x(reference.value)} y1={m.t - 10} y2={h - 22} className="ref-line" />
            <text x={x(reference.value)} y={m.t - 12} textAnchor="middle" className="tick-label strong">{referenceLabel}</text>
          </g>
        )}
        {rows.map((r, i) => {
          const cy = m.t + i * rowH + rowH / 2
          const better = reference && r !== reference && r.value > reference.value
          return (
            <g key={r.key} className="mark" {...bind(<TipRow value={format(r.value)} label={r.label} color="var(--s1)" />)}>
              <rect x={0} y={cy - rowH / 2} width={width} height={rowH} fill="transparent" />
              <text x={labelW - 12} y={cy} textAnchor="end" dominantBaseline="middle" className={`axis-label${r.emphasis ? ' strong' : ''}`}>{r.label}</text>
              <line x1={x(domain[0])} x2={x(r.value)} y1={cy} y2={cy} className="stem" />
              <circle cx={x(r.value)} cy={cy} r={r.emphasis === 'final' ? 8 : 6} className={`dot${r.emphasis ? ` dot-${r.emphasis}` : ''}`} />
              <text x={x(r.value) + 12} y={cy} dominantBaseline="middle" className={`value-label${better ? ' better' : ''}`}>{format(r.value)}</text>
            </g>
          )
        })}
      </svg>
      {node}
    </div>
  )
}

function niceTicks([a, b]: [number, number]) {
  const step = (b - a) / 4
  return Array.from({ length: 5 }, (_, i) => +(a + i * step).toFixed(4))
}

// Matriz de confusión: filas = clase real, columnas = predicha; el sombreado es el % de cada fila
const CLASSES = ['Sano', 'Osteopenia', 'Osteoporosis'] as const

export function ConfusionMatrix({ matrix, caption }: { matrix: readonly (readonly number[])[]; caption?: ReactNode }) {
  const { bind, node } = useTooltip()
  return (
    <div className="chart confusion">
      <div className="cm-grid" role="table" aria-label="Matriz de confusión">
        <div className="cm-corner" role="columnheader"><span>Real ↓ / Predicha →</span></div>
        {CLASSES.map((c) => <div key={c} className="cm-head" role="columnheader">{c}</div>)}
        {matrix.map((row, i) => {
          const total = row.reduce((s, v) => s + v, 0)
          return [
            <div key={`h${i}`} className="cm-head row" role="rowheader">{CLASSES[i]}</div>,
            ...row.map((v, j) => {
              const share = v / total
              return (
                <div key={`${i}-${j}`} role="cell" className={`cm-cell${i === j ? ' diag' : ''}`}
                     style={{ background: `color-mix(in oklab, var(--seq-hi) ${Math.round(share * 100)}%, var(--seq-lo))`, color: share > 0.45 ? 'var(--seq-text-on)' : 'var(--text)' }}
                     {...bind(<><TipRow value={`${v} de ${total}`} label={`reales ${CLASSES[i]} → ${CLASSES[j]}`} /><TipRow value={pct(share)} label="de la fila" /></>)}>
                  <strong>{v}</strong>
                  <span>{pct(share, 0)}</span>
                </div>
              )
            }),
          ]
        })}
      </div>
      {caption && <p className="chart-note">{caption}</p>}
      {node}
    </div>
  )
}

// Puntos por bolsa de la validación cruzada con la media ± desviación y el valor del holdout
export function CvFolds({ folds, holdout, domain = [0.4, 0.75] }: {
  folds: { fold: number; value: number; note?: string }[]
  holdout: number
  domain?: [number, number]
}) {
  const [ref, width] = useWidth<HTMLDivElement>()
  const { bind, node } = useTooltip()
  const m = { l: 16, r: 16 }
  const h = 120
  const x = (v: number) => m.l + ((v - domain[0]) / (domain[1] - domain[0])) * (width - m.l - m.r)
  const mean = folds.reduce((s, f) => s + f.value, 0) / folds.length
  const sd = Math.sqrt(folds.reduce((s, f) => s + (f.value - mean) ** 2, 0) / (folds.length - 1))
  const cy = 58

  return (
    <div className="chart" ref={ref}>
      <svg width={width} height={h} role="img" aria-label="Exactitud por bolsa de la validación cruzada">
        <rect x={x(mean - sd)} y={cy - 18} width={x(mean + sd) - x(mean - sd)} height={36} rx={4} className="err-band" />
        <line x1={x(mean)} x2={x(mean)} y1={cy - 24} y2={cy + 24} className="ref-line" />
        <text x={x(mean)} y={cy - 30} textAnchor="middle" className="tick-label strong">media CV {num(mean)}</text>
        <line x1={x(holdout)} x2={x(holdout)} y1={cy - 24} y2={cy + 24} className="marker-line" />
        <text x={x(holdout)} y={cy + 38} textAnchor="middle" className="tick-label strong">holdout {num(holdout)}</text>
        {folds.map((f) => (
          <g key={f.fold} className="mark" {...bind(<TipRow value={num(f.value)} label={`bolsa ${f.fold}${f.note ? ` · ${f.note}` : ''}`} color="var(--s1)" />)}>
            <circle cx={x(f.value)} cy={cy} r={14} fill="transparent" />
            <circle cx={x(f.value)} cy={cy} r={7} className="dot" />
            {f.note && <text x={x(f.value)} y={cy + 38} textAnchor="middle" className="tick-label">bolsa {f.fold}</text>}
          </g>
        ))}
        {niceTicks(domain).map((v) => (
          <text key={v} x={x(v)} y={h - 4} textAnchor="middle" className="tick-label">{num(v, 2)}</text>
        ))}
      </svg>
      {node}
    </div>
  )
}

// Barras agrupadas de dos series (antes / después) por categoría, con leyenda y etiquetas directas
export function PairedBars({ categories, series, format = (v: number) => num(v) }: {
  categories: string[]
  series: { label: string; values: number[]; color: string }[]
  format?: (v: number) => string
}) {
  const [ref, width] = useWidth<HTMLDivElement>()
  const { bind, node } = useTooltip()
  const labelW = Math.min(130, Math.max(90, width * 0.26))
  const valueW = 52
  const barH = 16
  const groupH = series.length * (barH + 2) + 14
  const pw = width - labelW - valueW
  const h = categories.length * groupH

  return (
    <div className="chart" ref={ref}>
      <div className="legend">
        {series.map((s) => <span key={s.label}><i style={{ background: s.color }} />{s.label}</span>)}
      </div>
      <svg width={width} height={h} role="img" aria-label={series.map((s) => s.label).join(' frente a ')}>
        {categories.map((c, ci) => {
          const gy = ci * groupH
          return (
            <g key={c}>
              <text x={labelW - 10} y={gy + (groupH - 14) / 2} textAnchor="end" dominantBaseline="middle" className="axis-label">{c}</text>
              {series.map((s, si) => {
                const v = s.values[ci]
                const y = gy + si * (barH + 2)
                return (
                  <g key={s.label} className="mark" {...bind(<TipRow value={format(v)} label={`${s.label} · ${c}`} color={s.color} />)}>
                    <rect x={labelW} y={y} width={pw} height={barH + 2} fill="transparent" />
                    <rect x={labelW} y={y} width={Math.max(2, v * pw)} height={barH} rx={4} fill={s.color} className="bar" />
                    <text x={labelW + v * pw + 6} y={y + barH / 2} dominantBaseline="middle" className="value-label">{format(v)}</text>
                  </g>
                )
              })}
            </g>
          )
        })}
      </svg>
      {node}
    </div>
  )
}
