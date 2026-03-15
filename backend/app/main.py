"""
OptionQuant — Enterprise FastAPI Application
═══════════════════════════════════════════════
Production-grade API with:
  • Centralized exception handling
  • Request ID tracking
  • Latency monitoring
  • CORS configuration
  • Health & readiness probes
  • Graceful startup/shutdown
"""
from __future__ import annotations

import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.staticfiles import StaticFiles

from .api import auth_routes, dl_routes, explain_routes, ml_routes, pricing_routes, market_routes, ws_routes, quant_routes, pricing_api
from .config import settings
from .prometheus_metrics import (
    PROMETHEUS_AVAILABLE,
    get_metrics_output,
    set_app_info,
)

APP_VERSION = "2.0.0"

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("optiquant")

START_TIME = time.time()

ROOT_DIR = Path(
    os.getenv("APP_ROOT_DIR", str(Path(__file__).resolve().parents[2]))
)
FRONTEND_DIR = (ROOT_DIR / settings.frontend_dir).resolve()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Application startup/shutdown lifecycle."""
    model_dir = (ROOT_DIR / settings.model_dir).resolve()
    model_dir.mkdir(parents=True, exist_ok=True)
    kb_dir = Path(__file__).resolve().parent / "rag" / "knowledge_base"
    kb_dir.mkdir(parents=True, exist_ok=True)
    logger.info("OptionQuant v%s started — %s mode", APP_VERSION, settings.environment)
    set_app_info(APP_VERSION, settings.environment)
    yield
    logger.info("OptionQuant v%s shutting down", APP_VERSION)


app = FastAPI(
    title=settings.app_name,
    version=APP_VERSION,
    description="Enterprise option pricing platform with Monte Carlo simulation & deep learning",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)


# ═══════════════════════════════════════════════════════════════
#  Middleware
# ═══════════════════════════════════════════════════════════════

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_logger(request: Request, call_next):
    """Request logging with UUID tracking, latency measurement, and Prometheus metrics."""
    request_id = str(uuid.uuid4())[:8]
    start = time.perf_counter()

    # Track in-flight requests
    if PROMETHEUS_AVAILABLE:
        from .prometheus_metrics import HTTP_IN_FLIGHT, HTTP_REQUESTS_TOTAL, HTTP_REQUEST_DURATION
        HTTP_IN_FLIGHT.inc()

    try:
        response = await call_next(request)
    except Exception as exc:
        logger.exception("[%s] Unhandled error: %s", request_id, exc)
        if PROMETHEUS_AVAILABLE:
            HTTP_IN_FLIGHT.dec()
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal Server Error",
                "request_id": request_id,
                "message": "An unexpected error occurred. Please try again.",
            },
        )

    duration = (time.perf_counter() - start) * 1000
    response.headers["X-Request-Id"] = request_id
    response.headers["X-Response-Time"] = f"{duration:.2f}ms"

    # Prometheus metrics
    if PROMETHEUS_AVAILABLE:
        HTTP_IN_FLIGHT.dec()
        path = request.url.path
        # Normalize path for metric labels
        endpoint = path.split("?")[0]
        if endpoint.startswith("/api/v1/"):
            parts = endpoint.split("/")
            endpoint = "/".join(parts[:5]) if len(parts) > 4 else endpoint
        HTTP_REQUESTS_TOTAL.labels(
            method=request.method,
            endpoint=endpoint,
            status=str(response.status_code),
        ).inc()
        HTTP_REQUEST_DURATION.labels(
            method=request.method,
            endpoint=endpoint,
        ).observe((time.perf_counter() - start + duration / 1000))

    # Log non-static requests
    path = request.url.path
    if not path.startswith(("/static", "/favicon")) and not path.endswith((".css", ".js", ".png", ".ico")):
        level = logging.WARNING if response.status_code >= 400 else logging.INFO
        logger.log(
            level,
            "[%s] %s %s → %s (%.1fms)",
            request_id, request.method, path, response.status_code, duration,
        )
    return response


# ═══════════════════════════════════════════════════════════════
#  Centralized Exception Handlers
# ═══════════════════════════════════════════════════════════════

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_: Request, exc: RequestValidationError):
    """Structured validation error response."""
    return JSONResponse(
        status_code=422,
        content={
            "error": "Validation error",
            "message": "Invalid request parameters. Check the details below.",
            "details": exc.errors(),
        },
    )


@app.exception_handler(ValueError)
async def value_error_handler(_: Request, exc: ValueError):
    return JSONResponse(
        status_code=400,
        content={"error": "Bad Request", "message": str(exc)},
    )


@app.exception_handler(Exception)
async def global_exception_handler(_: Request, exc: Exception):
    logger.exception("Unhandled exception: %s", exc)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": "An unexpected error occurred. Please try again.",
        },
    )


# ═══════════════════════════════════════════════════════════════
#  Routes
# ═══════════════════════════════════════════════════════════════

app.include_router(auth_routes.router)
app.include_router(pricing_routes.router)
app.include_router(pricing_api.router)
app.include_router(ml_routes.router)
app.include_router(dl_routes.router)
app.include_router(explain_routes.router)
app.include_router(market_routes.router)
app.include_router(ws_routes.router)
app.include_router(quant_routes.router)


# ═══════════════════════════════════════════════════════════════
#  Health & Readiness
# ═══════════════════════════════════════════════════════════════

@app.get("/health")
def health() -> dict:
    """Health check with system info."""
    uptime = int(time.time() - START_TIME)
    return {
        "status": "ok",
        "app": settings.app_name,
        "version": APP_VERSION,
        "environment": settings.environment,
        "uptime_seconds": uptime,
        "endpoints": {
            "pricing": "/api/v1/pricing",
            "ml": "/api/v1/ml",
            "dl": "/api/v1/dl",
            "ai": "/api/v1/ai",
            "auth": "/api/v1/auth",
            "market": "/api/v1/market",
            "websocket": "/ws/market/{symbol}",
            "quant": "/api/v1/quant",
        },
    }


@app.get("/ready")
def readiness() -> dict:
    """Kubernetes-style readiness probe."""
    return {"status": "ready"}


@app.get("/metrics")
def metrics():
    """Prometheus metrics endpoint."""
    if not PROMETHEUS_AVAILABLE:
        return JSONResponse(
            status_code=503,
            content={"error": "Metrics unavailable", "message": "prometheus-client not installed"},
        )
    from starlette.responses import Response as StarletteResponse
    data, content_type = get_metrics_output()
    return StarletteResponse(content=data, media_type=content_type)


# ── Static files (must be LAST so API routes match first) ─────────
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
