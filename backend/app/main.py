from __future__ import annotations

import logging
import os
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse as FastJSONResponse
from fastapi.staticfiles import StaticFiles

from .api import auth_routes, dl_routes, explain_routes, ml_routes, pricing_routes
from .config import settings

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)s | %(message)s",
)

START_TIME = time.time()

app = FastAPI(title=settings.app_name, version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ROOT_DIR = Path(
    os.getenv("APP_ROOT_DIR", str(Path(__file__).resolve().parents[2]))
)
FRONTEND_DIR = (ROOT_DIR / settings.frontend_dir).resolve()

app.include_router(auth_routes.router)
app.include_router(pricing_routes.router)
app.include_router(ml_routes.router)
app.include_router(dl_routes.router)
app.include_router(explain_routes.router)


@app.middleware("http")
async def request_logger(request: Request, call_next):
    request_id = str(uuid.uuid4())
    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception as exc:
        logging.exception("Unhandled error [%s] %s", request_id, exc)
        return FastJSONResponse(
            status_code=500,
            content={"error": "Internal Server Error", "request_id": request_id},
        )
    duration = (time.perf_counter() - start) * 1000
    response.headers["X-Request-Id"] = request_id
    logging.info(
        "%s %s %s %0.2fms",
        request.method,
        request.url.path,
        response.status_code,
        duration,
    )
    return response


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_: Request, exc: RequestValidationError):
    return FastJSONResponse(
        status_code=422,
        content={"error": "Validation error", "details": exc.errors()},
    )


@app.on_event("startup")
def startup_check() -> None:
    model_dir = (ROOT_DIR / settings.model_dir).resolve()
    model_dir.mkdir(parents=True, exist_ok=True)
    kb_dir = Path(__file__).resolve().parent / "rag" / "knowledge_base"
    kb_dir.mkdir(parents=True, exist_ok=True)


@app.get("/health")
def health() -> dict:
    uptime = int(time.time() - START_TIME)
    return {
        "status": "ok",
        "app": settings.app_name,
        "environment": settings.environment,
        "uptime_seconds": uptime,
        "version": "1.0.0",
    }


@app.get("/ready")
def readiness() -> dict:
    return {"status": "ready"}


# ── Static files (must be LAST so API routes match first) ─────────
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
