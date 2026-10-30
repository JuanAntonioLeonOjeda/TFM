import { NavLink, Navigate, Route, Routes } from 'react-router'
import { CHAPTERS } from './docs/chapters'
import { DocsPage } from './docs/DocsPage'
import { DemoPage } from './pages/DemoPage'

export default function App() {
  return (
    <>
      <header className="topbar">
        <div className="topbar-inner">
          <NavLink to="/" className="brand">
            <svg width="22" height="22" viewBox="0 0 24 24" aria-hidden="true">
              <rect x="9" y="2" width="6" height="4.5" rx="1.5" fill="currentColor" opacity=".55" />
              <rect x="8.5" y="7.5" width="7" height="4.5" rx="1.5" fill="currentColor" opacity=".75" />
              <rect x="8" y="13" width="8" height="4.5" rx="1.5" fill="currentColor" />
              <rect x="7.5" y="18.5" width="9" height="4" rx="1.5" fill="currentColor" />
            </svg>
            LumosNet
          </NavLink>
          <nav className="topnav" aria-label="Secciones">
            <NavLink to="/" end>Probar el modelo</NavLink>
            <NavLink to="/proyecto">El proyecto</NavLink>
          </nav>
        </div>
      </header>
      <Routes>
        <Route path="/" element={<DemoPage />} />
        <Route path="/proyecto" element={<Navigate to={`/proyecto/${CHAPTERS[0].slug}`} replace />} />
        <Route path="/proyecto/:slug" element={<DocsPage />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </>
  )
}
