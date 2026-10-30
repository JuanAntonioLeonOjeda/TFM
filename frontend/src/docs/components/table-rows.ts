import { EXPERIMENT_METRICS, type ExperimentKey } from '../metrics'

export interface CompareRow {
  label: string
  values: (number | string)[]
  digits?: number
  lowerIsBetter?: boolean
}

// Las cinco métricas de test por paciente de varios experimentos
export function experimentRows(keys: ExperimentKey[]): CompareRow[] {
  const p = keys.map((k) => EXPERIMENT_METRICS[k].patient)
  return [
    { label: 'Accuracy', values: p.map((m) => m.acc) },
    { label: 'F1 macro', values: p.map((m) => m.f1) },
    { label: 'AUC macro', values: p.map((m) => m.auc) },
    { label: 'RMSE de la BMD', values: p.map((m) => m.rmse), digits: 4, lowerIsBetter: true },
    { label: 'PCC de la BMD', values: p.map((m) => m.pcc) },
  ]
}
