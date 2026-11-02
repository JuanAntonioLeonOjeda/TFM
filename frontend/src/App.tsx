import { NavLink, Navigate, Route, Routes } from 'react-router'
import { ThemeToggle } from './components/ThemeToggle'
import { CHAPTERS } from './docs/chapters'
import { DocsPage } from './docs/DocsPage'
import { DemoPage } from './pages/DemoPage'

export default function App() {
  return (
    <>
      <header className="topbar">
        <div className="topbar-inner">
          <NavLink to="/" className="brand" aria-label="LumosNet">
            <svg width="22" height="22" viewBox="0 0 24 24" aria-hidden="true">
              <rect x="9" y="2" width="6" height="4.5" rx="1.5" fill="currentColor" opacity=".55" />
              <rect x="8.5" y="7.5" width="7" height="4.5" rx="1.5" fill="currentColor" opacity=".75" />
              <rect x="8" y="13" width="8" height="4.5" rx="1.5" fill="currentColor" />
              <rect x="7.5" y="18.5" width="9" height="4" rx="1.5" fill="currentColor" />
            </svg>
            <span className="brand-name">LumosNet</span>
          </NavLink>
          <nav className="topnav" aria-label="Secciones">
            <NavLink to="/" end>Probar el modelo</NavLink>
            <NavLink to="/proyecto">El proyecto</NavLink>
          </nav>
          <ThemeToggle />
        </div>
      </header>
      <Routes>
        <Route path="/" element={<DemoPage />} />
        <Route path="/proyecto" element={<Navigate to={`/proyecto/${CHAPTERS[0].slug}`} replace />} />
        <Route path="/proyecto/:slug" element={<DocsPage />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
      <footer className="site-footer">
        Radiografías y datos de entrenamiento:{' '}
        <a href="https://doi.org/10.1145/3746027.3758282" target="_blank" rel="noopener noreferrer">LUMOS</a>{' '}
        (Shi et al., 2025), con licencia{' '}
        <a href="https://creativecommons.org/licenses/by-nc/4.0/" target="_blank" rel="noopener noreferrer">CC BY-NC 4.0</a>.
        Las imágenes se muestran preprocesadas (224×224, CLAHE). Uso no comercial.
      </footer>
    </>
  )
}
