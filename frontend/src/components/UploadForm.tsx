import { useState, type DragEvent, type FormEvent } from 'react'
import type { Gender, PatientData } from '../api'

interface Props {
  busy: boolean
  onSubmit: (file: File, patient: PatientData) => void
}

export function UploadForm({ busy, onSubmit }: Props) {
  const [file, setFile] = useState<File | null>(null)
  const [age, setAge] = useState('')
  const [gender, setGender] = useState<Gender>('female')
  const [bmi, setBmi] = useState('')
  const [dragging, setDragging] = useState(false)

  const ready = file !== null && age !== '' && bmi !== '' && !busy

  function handleDrop(e: DragEvent) {
    e.preventDefault()
    setDragging(false)
    if (e.dataTransfer.files[0]) setFile(e.dataTransfer.files[0])
  }

  function handleSubmit(e: FormEvent) {
    e.preventDefault()
    if (file) onSubmit(file, { age: Number(age), gender, bmi: Number(bmi) })
  }

  return (
    <form onSubmit={handleSubmit}>
      <label
        className={`drop${dragging ? ' over' : ''}`}
        onDragOver={(e) => { e.preventDefault(); setDragging(true) }}
        onDragLeave={() => setDragging(false)}
        onDrop={handleDrop}
      >
        <input type="file" accept=".dcm,.dicom,application/dicom"
               onChange={(e) => setFile(e.target.files?.[0] ?? null)} />
        {file ? (
          <span><strong>{file.name}</strong><br />Haz clic para cambiarla</span>
        ) : (
          <span><strong>Arrastra un DICOM</strong> o haz clic para elegirlo<br />
            <small>Radiografía lumbar en formato .dcm (máx. 50 MB)</small></span>
        )}
      </label>
      <div className="fields">
        <label className="field">Edad
          <input type="number" min={0} max={120} step={1} required placeholder="65"
                 value={age} onChange={(e) => setAge(e.target.value)} />
        </label>
        <label className="field">Sexo
          <select value={gender} onChange={(e) => setGender(e.target.value as Gender)}>
            <option value="female">Mujer</option>
            <option value="male">Hombre</option>
          </select>
        </label>
        <label className="field">IMC (kg/m²)
          <input type="number" min={1} max={80} step={0.1} required placeholder="24.5"
                 value={bmi} onChange={(e) => setBmi(e.target.value)} />
        </label>
      </div>
      <button className="primary" type="submit" disabled={!ready}>Analizar radiografía</button>
      <p className="hint">El modelo usa la edad, el sexo y el IMC junto con la imagen.</p>
    </form>
  )
}
