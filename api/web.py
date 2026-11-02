"""API y web de React en un solo proceso, para desplegar en un único contenedor (Hugging Face Spaces).

La API queda en /api, igual que detrás del nginx de docker compose, y el resto de rutas sirven la build
de Vite (FRONTEND_DIR), con index.html como respaldo para las rutas de React Router.
"""
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from api.main import app as api_app, lifespan as api_lifespan

FRONTEND_DIR = Path(os.environ.get("FRONTEND_DIR", Path(__file__).resolve().parent.parent / "frontend" / "dist"))


# Starlette no ejecuta el lifespan de las apps montadas: el modelo se carga desde aquí
@asynccontextmanager
async def lifespan(app: FastAPI):
    async with api_lifespan(api_app):
        yield


app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)


@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"
    return response


app.mount("/api", api_app)


class ImmutableStaticFiles(StaticFiles):
    """Los ficheros de Vite llevan hash en el nombre: se pueden cachear sin límite."""

    def file_response(self, *args, **kwargs):
        response = super().file_response(*args, **kwargs)
        response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
        return response


app.mount("/assets", ImmutableStaticFiles(directory=FRONTEND_DIR / "assets", check_dir=False), name="assets")


@app.get("/{path:path}", include_in_schema=False)
def frontend(path: str):
    root = FRONTEND_DIR.resolve()
    file = (root / path).resolve()
    if not file.is_file() or not file.is_relative_to(root):
        file = root / "index.html"
    return FileResponse(file, headers={"Cache-Control": "no-cache"})
