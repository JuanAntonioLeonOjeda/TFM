import { useEffect, useState } from 'react'
import { getPrivacyNotice } from '../api'

// Texto por defecto; se sustituye por el del servidor para que no se desincronicen
const FALLBACK =
  'La radiografía y los datos que envíes se procesan solo en memoria para devolver la predicción: ' +
  'no se guardan, no se registran y no se usan para entrenar el modelo.'

export function PrivacyNotice() {
  const [notice, setNotice] = useState(FALLBACK)

  useEffect(() => {
    getPrivacyNotice().then(setNotice).catch(() => {})
  }, [])

  return (
    <>
      <div className="notice" role="note">
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"
             strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
          <rect x="4" y="11" width="16" height="10" rx="2" />
          <path d="M8 11V7a4 4 0 0 1 8 0v4" />
        </svg>
        <div><strong>Tus datos no se guardan.</strong> {notice}</div>
      </div>
      <p className="disclaimer">
        Herramienta de investigación (Trabajo Fin de Máster). No es un producto sanitario ni sustituye a una
        densitometría (DXA) ni al criterio de un profesional.
      </p>
    </>
  )
}
