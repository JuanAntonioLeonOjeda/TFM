import type { ReactNode } from 'react'

function Node({ kind = 'layer', title, children }: { kind?: 'input' | 'layer' | 'tensor' | 'output' | 'infra'; title: ReactNode; children?: ReactNode }) {
  return (
    <div className={`node node-${kind}`}>
      <strong>{title}</strong>
      {children && <span>{children}</span>}
    </div>
  )
}

const Down = ({ label }: { label?: string }) => <div className="flow-arrow" aria-hidden="true">{label && <span>{label}</span>}</div>

// LumosNet de arriba abajo: dos entradas, fusión, bloque compartido y dos cabezas
export function ArchitectureDiagram() {
  return (
    <div className="flow" role="img" aria-label="Arquitectura de LumosNet: ResNet50 y covariables, bloque compartido y dos cabezas">
      <div className="flow-row two">
        <Node kind="input" title="Radiografía">224 × 224 × 3 · CLAHE · normalización ImageNet</Node>
        <Node kind="input" title="Covariables clínicas">edad · sexo · IMC (z-score con medias de train)</Node>
      </div>
      <div className="flow-row two"><Down /><Down /></div>
      <div className="flow-row two">
        <Node title="ResNet50 preentrenada">conv1 → layer1…layer4 → average pooling · 23,5 M parámetros</Node>
        <Node title="Linear 3 → 32 + ReLU">embedding de las covariables</Node>
      </div>
      <div className="flow-row two"><Down label="2048" /><Down label="32" /></div>
      <div className="flow-row"><Node kind="tensor" title="Concatenación">vector de 2080 características</Node></div>
      <div className="flow-row"><Down /></div>
      <div className="flow-row"><Node title="Bloque compartido">Linear 2080 → 512 + ReLU + Dropout</Node></div>
      <div className="flow-row two"><Down /><Down /></div>
      <div className="flow-row two">
        <Node title="Cabeza de clasificación">Linear 512 → 256 → 3 · ReLU · Dropout</Node>
        <Node title="Cabeza de regresión">Linear 512 → 256 → 1 · ReLU · Dropout</Node>
      </div>
      <div className="flow-row two"><Down /><Down /></div>
      <div className="flow-row two">
        <Node kind="output" title="Sano · Osteopenia · Osteoporosis">probabilidades (softmax)</Node>
        <Node kind="output" title="BMD estimada">g/cm², sin activación</Node>
      </div>
      <div className="flow-row"><div className="flow-loss">Pérdida conjunta: L = entropía cruzada + 10 · MSE</div></div>
    </div>
  )
}

// Cómo viaja una radiografía desde el navegador hasta el modelo y vuelta
export function DeployDiagram() {
  return (
    <div className="flow" role="img" aria-label="Arquitectura de despliegue: navegador, nginx, FastAPI y modelo">
      <div className="flow-row"><Node kind="input" title="Navegador">React · sube el DICOM con edad, sexo e IMC, o elige un ejemplo</Node></div>
      <div className="flow-row"><Down label="HTTP · /api/*" /></div>
      <div className="flow-row"><Node kind="infra" title="nginx (contenedor frontend)">sirve la web compilada y reenvía /api al backend · la subida pasa en streaming, sin ficheros temporales</Node></div>
      <div className="flow-row"><Down label="red interna de Docker Compose" /></div>
      <div className="flow-row"><Node kind="infra" title="FastAPI (contenedor backend)">valida los datos · procesa el DICOM en memoria · Cache-Control: no-store</Node></div>
      <div className="flow-row"><Down /></div>
      <div className="flow-row two">
        <Node title="Preprocesado">mismo código que en entrenamiento: 224 × 224 + CLAHE</Node>
        <Node title="LumosNet + Grad-CAM">clase, probabilidades, BMD y mapa de calor</Node>
      </div>
      <div className="flow-row two"><Down label="volumen /models" /><Down label="volumen /examples" /></div>
      <div className="flow-row two">
        <Node kind="tensor" title="Checkpoint + estadísticas">best_phased_full_covars.pt y su JSON de covariables (solo lectura)</Node>
        <Node kind="tensor" title="Ejemplos de LUMOS">9 radiografías de test con su diagnóstico real (solo lectura)</Node>
      </div>
    </div>
  )
}
