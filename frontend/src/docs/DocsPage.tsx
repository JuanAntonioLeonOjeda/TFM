import { useEffect } from 'react'
import { Link, NavLink, Navigate, useParams } from 'react-router'
import { CHAPTERS } from './chapters'

export function DocsPage() {
  const { slug } = useParams()
  const index = CHAPTERS.findIndex((c) => c.slug === slug)
  const chapter = CHAPTERS[index]

  useEffect(() => {
    if (!chapter) return
    window.scrollTo({ top: 0 })
    document.title = `${chapter.title} · LumosNet`
    return () => { document.title = 'LumosNet' }
  }, [chapter])

  if (!chapter) return <Navigate to={`/proyecto/${CHAPTERS[0].slug}`} replace />

  const prev = CHAPTERS[index - 1]
  const next = CHAPTERS[index + 1]
  const { Content } = chapter

  return (
    <div className="docs">
      <aside className="docs-sidebar">
        {/* En móvil el índice se pliega; en escritorio el <details> se ve siempre abierto */}
        <details className="chapter-menu" open>
          <summary>
            <span className="sidebar-title">El proyecto</span>
            <span className="chapter-menu-current">Capítulo {index + 1} de {CHAPTERS.length}</span>
          </summary>
          <ol className="chapter-list">
            {CHAPTERS.map((c, i) => (
              <li key={c.slug}>
                <NavLink to={`/proyecto/${c.slug}`}>
                  <span className="chapter-num" aria-hidden="true">{i + 1}</span>
                  <span>{c.title}</span>
                </NavLink>
              </li>
            ))}
          </ol>
        </details>
      </aside>

      <article className="docs-content">
        <p className="crumb">El proyecto · Capítulo {index + 1}</p>
        <h1>{chapter.title}</h1>
        <p className="lead">{chapter.summary}</p>
        <div className="prose">
          <Content />
        </div>
        <nav className="pager" aria-label="Capítulos">
          {prev ? (
            <Link to={`/proyecto/${prev.slug}`} className="pager-link">
              <span className="pager-dir">← Anterior</span>{prev.title}
            </Link>
          ) : <span />}
          {next ? (
            <Link to={`/proyecto/${next.slug}`} className="pager-link next">
              <span className="pager-dir">Siguiente →</span>{next.title}
            </Link>
          ) : (
            <Link to="/" className="pager-link next">
              <span className="pager-dir">Y ahora →</span>Probar el modelo
            </Link>
          )}
        </nav>
      </article>
    </div>
  )
}
