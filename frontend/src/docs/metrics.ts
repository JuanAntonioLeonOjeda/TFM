/**
 * Métricas de test de cada checkpoint del TFM, reevaluadas con el código del TFM el 2026-10-29 (copiadas
 * de repaso/src/content/tfm/metrics.ts, generado por scripts/tfm/experiment_metrics.py). Solo cifras
 * agregadas, sin datos por paciente.
 */
export interface SplitMetrics {
  n: number
  acc: number
  f1: number
  auc: number
  rmse: number
  pcc: number
  /** Sensitivity and specificity per class: [sano, osteopenia, osteoporosis]. */
  sens: readonly [number, number, number]
  spec: readonly [number, number, number]
  confusion: readonly (readonly number[])[]
}

export interface ExperimentMetrics {
  data: string
  covariates: boolean
  patient: SplitMetrics
  image: SplitMetrics
}

export const EXPERIMENT_METRICS = {
  v0: {
    data: 'full',
    covariates: false,
    patient: {
      n: 121,
      acc: 0.678,
      f1: 0.681,
      auc: 0.822,
      rmse: 0.1461,
      pcc: 0.752,
      sens: [0.698, 0.682, 0.647],
      spec: [0.91, 0.714, 0.885],
      confusion: [
        [30, 11, 2],
        [6, 30, 8],
        [1, 11, 22],
      ],
    },
    image: {
      n: 244,
      acc: 0.607,
      f1: 0.611,
      auc: 0.785,
      rmse: 0.1598,
      pcc: 0.68,
      sens: [0.64, 0.589, 0.588],
      spec: [0.886, 0.656, 0.858],
      confusion: [
        [55, 28, 3],
        [15, 53, 22],
        [3, 25, 40],
      ],
    },
  },
  'v1-differential': {
    data: 'full',
    covariates: false,
    patient: {
      n: 121,
      acc: 0.628,
      f1: 0.623,
      auc: 0.832,
      rmse: 0.1374,
      pcc: 0.776,
      sens: [0.814, 0.409, 0.676],
      spec: [0.808, 0.766, 0.862],
      confusion: [
        [35, 8, 0],
        [14, 18, 12],
        [1, 10, 23],
      ],
    },
    image: {
      n: 244,
      acc: 0.615,
      f1: 0.613,
      auc: 0.795,
      rmse: 0.1517,
      pcc: 0.69,
      sens: [0.744, 0.5, 0.603],
      spec: [0.816, 0.747, 0.852],
      confusion: [
        [64, 18, 4],
        [23, 45, 22],
        [6, 21, 41],
      ],
    },
  },
  'v1-phased': {
    data: 'full',
    covariates: false,
    patient: {
      n: 121,
      acc: 0.512,
      f1: 0.506,
      auc: 0.72,
      rmse: 0.1712,
      pcc: 0.641,
      sens: [0.674, 0.295, 0.588],
      spec: [0.821, 0.74, 0.713],
      confusion: [
        [29, 8, 6],
        [12, 13, 19],
        [2, 12, 20],
      ],
    },
    image: {
      n: 244,
      acc: 0.504,
      f1: 0.502,
      auc: 0.691,
      rmse: 0.1797,
      pcc: 0.55,
      sens: [0.616, 0.367, 0.544],
      spec: [0.81, 0.734, 0.716],
      confusion: [
        [53, 18, 15],
        [22, 33, 35],
        [8, 23, 37],
      ],
    },
  },
  'v2-differential': {
    data: 'full',
    covariates: false,
    patient: {
      n: 121,
      acc: 0.628,
      f1: 0.632,
      auc: 0.805,
      rmse: 0.1434,
      pcc: 0.764,
      sens: [0.605, 0.727, 0.529],
      spec: [0.949, 0.584, 0.897],
      confusion: [
        [26, 17, 0],
        [3, 32, 9],
        [1, 15, 18],
      ],
    },
    image: {
      n: 244,
      acc: 0.574,
      f1: 0.581,
      auc: 0.773,
      rmse: 0.1561,
      pcc: 0.676,
      sens: [0.547, 0.589, 0.588],
      spec: [0.924, 0.604, 0.824],
      confusion: [
        [47, 35, 4],
        [10, 53, 27],
        [2, 26, 40],
      ],
    },
  },
  'v2-phased': {
    data: 'full',
    covariates: false,
    patient: {
      n: 121,
      acc: 0.661,
      f1: 0.668,
      auc: 0.822,
      rmse: 0.1379,
      pcc: 0.749,
      sens: [0.605, 0.614, 0.794],
      spec: [0.91, 0.714, 0.862],
      confusion: [
        [26, 16, 1],
        [6, 27, 11],
        [1, 6, 27],
      ],
    },
    image: {
      n: 244,
      acc: 0.611,
      f1: 0.615,
      auc: 0.78,
      rmse: 0.1551,
      pcc: 0.665,
      sens: [0.581, 0.578, 0.691],
      spec: [0.905, 0.701, 0.807],
      confusion: [
        [50, 28, 8],
        [12, 52, 26],
        [3, 18, 47],
      ],
    },
  },
  v3: {
    data: 'full',
    covariates: false,
    patient: {
      n: 121,
      acc: 0.636,
      f1: 0.64,
      auc: 0.824,
      rmse: 0.1413,
      pcc: 0.751,
      sens: [0.721, 0.5, 0.706],
      spec: [0.859, 0.727, 0.862],
      confusion: [
        [31, 12, 0],
        [10, 22, 12],
        [1, 9, 24],
      ],
    },
    image: {
      n: 244,
      acc: 0.611,
      f1: 0.615,
      auc: 0.792,
      rmse: 0.1536,
      pcc: 0.681,
      sens: [0.64, 0.489, 0.735],
      spec: [0.854, 0.734, 0.824],
      confusion: [
        [55, 26, 5],
        [20, 44, 26],
        [3, 15, 50],
      ],
    },
  },
  'v4-roi': {
    data: 'roi',
    covariates: false,
    patient: {
      n: 120,
      acc: 0.5,
      f1: 0.492,
      auc: 0.729,
      rmse: 0.1959,
      pcc: 0.443,
      sens: [0.605, 0.523, 0.333],
      spec: [0.805, 0.539, 0.885],
      confusion: [
        [26, 17, 0],
        [11, 23, 10],
        [4, 18, 11],
      ],
    },
    image: {
      n: 120,
      acc: 0.5,
      f1: 0.492,
      auc: 0.729,
      rmse: 0.1959,
      pcc: 0.443,
      sens: [0.605, 0.523, 0.333],
      spec: [0.805, 0.539, 0.885],
      confusion: [
        [26, 17, 0],
        [11, 23, 10],
        [4, 18, 11],
      ],
    },
  },
  'v4-hybrid': {
    data: 'roi_hybrid',
    covariates: false,
    patient: {
      n: 121,
      acc: 0.603,
      f1: 0.613,
      auc: 0.797,
      rmse: 0.1443,
      pcc: 0.726,
      sens: [0.581, 0.568, 0.676],
      spec: [0.91, 0.636, 0.851],
      confusion: [
        [25, 18, 0],
        [6, 25, 13],
        [1, 10, 23],
      ],
    },
    image: {
      n: 244,
      acc: 0.594,
      f1: 0.602,
      auc: 0.764,
      rmse: 0.1603,
      pcc: 0.621,
      sens: [0.57, 0.567, 0.662],
      spec: [0.918, 0.649, 0.818],
      confusion: [
        [49, 34, 3],
        [10, 51, 29],
        [3, 20, 45],
      ],
    },
  },
  v6: {
    data: 'hires',
    covariates: false,
    patient: {
      n: 121,
      acc: 0.645,
      f1: 0.651,
      auc: 0.809,
      rmse: 0.1428,
      pcc: 0.744,
      sens: [0.488, 0.75, 0.706],
      spec: [0.962, 0.597, 0.897],
      confusion: [
        [21, 22, 0],
        [2, 33, 9],
        [1, 9, 24],
      ],
    },
    image: {
      n: 244,
      acc: 0.586,
      f1: 0.593,
      auc: 0.783,
      rmse: 0.1518,
      pcc: 0.679,
      sens: [0.465, 0.644, 0.662],
      spec: [0.949, 0.584, 0.835],
      confusion: [
        [40, 44, 2],
        [5, 58, 27],
        [3, 20, 45],
      ],
    },
  },
  'v7-inverse': {
    data: 'full',
    covariates: false,
    patient: {
      n: 121,
      acc: 0.636,
      f1: 0.637,
      auc: 0.846,
      rmse: 0.1356,
      pcc: 0.766,
      sens: [0.698, 0.477, 0.765],
      spec: [0.923, 0.792, 0.747],
      confusion: [
        [30, 9, 4],
        [5, 21, 18],
        [1, 7, 26],
      ],
    },
    image: {
      n: 244,
      acc: 0.619,
      f1: 0.621,
      auc: 0.803,
      rmse: 0.1485,
      pcc: 0.687,
      sens: [0.674, 0.544, 0.647],
      spec: [0.886, 0.727, 0.812],
      confusion: [
        [58, 21, 7],
        [15, 49, 26],
        [3, 21, 44],
      ],
    },
  },
  'v7-effective': {
    data: 'full',
    covariates: false,
    patient: {
      n: 121,
      acc: 0.661,
      f1: 0.665,
      auc: 0.829,
      rmse: 0.1398,
      pcc: 0.743,
      sens: [0.651, 0.591, 0.765],
      spec: [0.885, 0.766, 0.839],
      confusion: [
        [28, 12, 3],
        [7, 26, 11],
        [2, 6, 26],
      ],
    },
    image: {
      n: 244,
      acc: 0.602,
      f1: 0.607,
      auc: 0.795,
      rmse: 0.1504,
      pcc: 0.687,
      sens: [0.605, 0.511, 0.721],
      spec: [0.886, 0.721, 0.795],
      confusion: [
        [52, 27, 7],
        [15, 46, 29],
        [3, 16, 49],
      ],
    },
  },
  v8: {
    data: 'full',
    covariates: true,
    patient: {
      n: 121,
      acc: 0.669,
      f1: 0.662,
      auc: 0.836,
      rmse: 0.1682,
      pcc: 0.762,
      sens: [0.767, 0.432, 0.853],
      spec: [0.885, 0.87, 0.759],
      confusion: [
        [33, 6, 4],
        [8, 19, 17],
        [1, 4, 29],
      ],
    },
    image: {
      n: 244,
      acc: 0.627,
      f1: 0.624,
      auc: 0.809,
      rmse: 0.1773,
      pcc: 0.707,
      sens: [0.721, 0.433, 0.765],
      spec: [0.88, 0.825, 0.744],
      confusion: [
        [62, 13, 11],
        [17, 39, 34],
        [2, 14, 52],
      ],
    },
  },
  'final-roi-yolo': {
    data: 'roi_yolo',
    covariates: true,
    patient: {
      n: 121,
      acc: 0.554,
      f1: 0.544,
      auc: 0.797,
      rmse: 0.1656,
      pcc: 0.75,
      sens: [0.814, 0.341, 0.5],
      spec: [0.756, 0.688, 0.874],
      confusion: [
        [35, 8, 0],
        [18, 15, 11],
        [1, 16, 17],
      ],
    },
    image: {
      n: 244,
      acc: 0.57,
      f1: 0.556,
      auc: 0.777,
      rmse: 0.1771,
      pcc: 0.697,
      sens: [0.837, 0.411, 0.441],
      spec: [0.772, 0.688, 0.881],
      confusion: [
        [72, 13, 1],
        [33, 37, 20],
        [3, 35, 30],
      ],
    },
  },
} as const satisfies Record<string, ExperimentMetrics>

export type ExperimentKey = keyof typeof EXPERIMENT_METRICS
