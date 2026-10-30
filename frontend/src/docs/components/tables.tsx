import { EXPERIMENT_METRICS, type ExperimentKey } from '../metrics'
import { num } from './chart-utils'
import type { CompareRow } from './table-rows'

// Métricas en filas y modelos en columnas, como en la memoria; el mejor valor de cada fila va en negrita
export function CompareTable({ columns, rows, caption, highlight }: {
  columns: string[]
  rows: CompareRow[]
  caption?: string
  highlight?: number
}) {
  return (
    <div className="table-wrap">
      <table className="compare">
        {caption && <caption>{caption}</caption>}
        <thead>
          <tr>
            <th scope="col">Métrica</th>
            {columns.map((c, i) => <th key={c} scope="col" className={i === highlight ? 'hl' : undefined}>{c}</th>)}
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => {
            const nums = r.values.filter((v): v is number => typeof v === 'number')
            const best = nums.length > 1 ? (r.lowerIsBetter ? Math.min(...nums) : Math.max(...nums)) : null
            return (
              <tr key={r.label}>
                <th scope="row">{r.label}</th>
                {r.values.map((v, i) => (
                  <td key={i} className={[v === best ? 'best' : '', i === highlight ? 'hl' : ''].join(' ').trim() || undefined}>
                    {typeof v === 'number' ? num(v, r.digits ?? 3) : v}
                  </td>
                ))}
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}

// Sensibilidad y especificidad por clase, una pareja de columnas por experimento
export function SensSpecTable({ columns, keys, caption }: { columns: string[]; keys: ExperimentKey[]; caption?: string }) {
  const classes = ['Sano', 'Osteopenia', 'Osteoporosis']
  const p = keys.map((k) => EXPERIMENT_METRICS[k].patient)
  return (
    <div className="table-wrap">
      <table className="compare">
        {caption && <caption>{caption}</caption>}
        <thead>
          <tr>
            <th rowSpan={2} scope="col">Clase</th>
            {columns.map((c) => <th key={c} colSpan={2} scope="colgroup">{c}</th>)}
          </tr>
          <tr>{columns.map((c) => [<th key={`${c}s`} scope="col" className="sub">Sens.</th>, <th key={`${c}e`} scope="col" className="sub">Espec.</th>])}</tr>
        </thead>
        <tbody>
          {classes.map((cls, ci) => {
            const bestSens = Math.max(...p.map((m) => m.sens[ci]))
            const bestSpec = Math.max(...p.map((m) => m.spec[ci]))
            return (
              <tr key={cls} className={cls === 'Osteoporosis' ? 'key-row' : undefined}>
                <th scope="row">{cls}</th>
                {p.map((m, i) => [
                  <td key={`${i}s`} className={m.sens[ci] === bestSens && p.length > 1 ? 'best' : undefined}>{num(m.sens[ci])}</td>,
                  <td key={`${i}e`} className={m.spec[ci] === bestSpec && p.length > 1 ? 'best' : undefined}>{num(m.spec[ci])}</td>,
                ])}
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}
