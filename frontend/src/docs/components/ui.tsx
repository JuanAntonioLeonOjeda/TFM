import type { ReactNode } from 'react'

type CalloutKind = 'idea' | 'warn' | 'info'
const CALLOUT_LABEL: Record<CalloutKind, string> = { idea: 'Idea clave', warn: 'Ojo', info: 'Contexto' }

export function Callout({ kind = 'info', title, children }: { kind?: CalloutKind; title?: string; children: ReactNode }) {
  return (
    <aside className={`callout callout-${kind}`}>
      <p className="callout-title">{title ?? CALLOUT_LABEL[kind]}</p>
      <div>{children}</div>
    </aside>
  )
}

export function Figure({ caption, children, wide }: { caption?: ReactNode; children: ReactNode; wide?: boolean }) {
  return (
    <figure className={`doc-figure${wide ? ' wide' : ''}`}>
      {children}
      {caption && <figcaption>{caption}</figcaption>}
    </figure>
  )
}

export interface Stat {
  value: string
  label: string
  detail?: string
}

export function StatGrid({ stats }: { stats: Stat[] }) {
  return (
    <div className="stat-grid">
      {stats.map((s) => (
        <div key={s.label} className="stat">
          <div className="stat-value">{s.value}</div>
          <div className="stat-label">{s.label}</div>
          {s.detail && <div className="stat-detail">{s.detail}</div>}
        </div>
      ))}
    </div>
  )
}

export function Summary({ items }: { items: ReactNode[] }) {
  return (
    <section className="summary-box">
      <h2>En resumen</h2>
      <ul>{items.map((item, i) => <li key={i}>{item}</li>)}</ul>
    </section>
  )
}

// Pasos en horizontal unidos por flechas (en móvil se apilan)
export function Steps({ steps }: { steps: { title: string; text: ReactNode }[] }) {
  return (
    <ol className="steps">
      {steps.map((s, i) => (
        <li key={s.title} className="step">
          <span className="step-num">{i + 1}</span>
          <strong>{s.title}</strong>
          <span className="step-text">{s.text}</span>
        </li>
      ))}
    </ol>
  )
}
