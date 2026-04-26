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

import httpx
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.staticfiles import StaticFiles
from starlette.responses import Response as StarletteResponse

from .api import auth_routes, dl_routes, explain_routes, ml_routes, pricing_routes, market_routes, ws_routes, quant_routes, pricing_api
from .api import pinns_routes
from .config import settings
from .database import init_pool, init_db, close_pool
from .runtime_lockstep import validate_dependency_lock
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
GRAFANA_UPSTREAM = os.getenv("GRAFANA_UPSTREAM", "http://127.0.0.1:3000")


def _load_pretrained_models(model_dir: Path) -> None:
    """Load pre-trained ML/DL/PINNs models from disk at startup."""
    loaded = []

    # 1. Volatility engine (sklearn ML models)
    try:
        from .vol_engine import get_engine
        engine = get_engine()
        if engine.load(model_dir, strict_compatibility=settings.enforce_model_compatibility):
            loaded.append("vol_engine")
    except Exception as e:
        logger.warning("Could not load vol engine models: %s", e)

    # 2. Hybrid DL predictor (LSTM)
    try:
        from .dl import get_predictor
        predictor = get_predictor()
        if predictor.load(model_dir, strict_compatibility=settings.enforce_model_compatibility):
            loaded.append("dl_lstm")
    except Exception as e:
        logger.warning("Could not load DL models: %s", e)

    # 3. PINNs
    try:
        pinns_path = model_dir / "pinns_model.pkl"
        if pinns_path.exists():
            from .pinns import PINNsOptionPricer
            pricer = PINNsOptionPricer.load(
                pinns_path,
                strict_compatibility=settings.enforce_model_compatibility,
            )
            # Replace the singleton
            from . import pinns as pinns_mod
            pinns_mod._pinns_instance = pricer
            loaded.append("pinns")
    except Exception as e:
        logger.warning("Could not load PINNs model: %s", e)

    if loaded:
        logger.info("Pre-trained models loaded: %s", ", ".join(loaded))
    else:
        logger.info("No pre-trained models found — models will train on first request")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Application startup/shutdown lifecycle."""
    lock_file = (ROOT_DIR / settings.dependency_lock_file).resolve()
    dep_report = validate_dependency_lock(lock_file)
    if dep_report.checked == 0:
        logger.warning("Dependency lock check skipped: no pinned entries in %s", lock_file)
    elif dep_report.ok:
        logger.info("Dependency lock check passed (%d pinned packages)", dep_report.checked)
    else:
        msg = (
            f"Dependency lock check failed ({dep_report.checked} checked). "
            f"Mismatches={len(dep_report.mismatches)}, Missing={len(dep_report.missing)}"
        )
        details = dep_report.mismatches + dep_report.missing
        for item in details:
            logger.error("Dependency violation: %s", item)
        if settings.enforce_dependency_pins:
            raise RuntimeError(msg)
        logger.warning("%s; continuing because ENFORCE_DEPENDENCY_PINS is disabled", msg)

    # Initialize Neon PostgreSQL connection pool and schema
    if init_pool():
        init_db()
        logger.info("Neon PostgreSQL database connected and schema initialized")
    else:
        logger.warning("Running without database — auth and history features unavailable")

    model_dir = (ROOT_DIR / settings.model_dir).resolve()
    model_dir.mkdir(parents=True, exist_ok=True)
    kb_dir = Path(__file__).resolve().parent / "rag" / "knowledge_base"
    kb_dir.mkdir(parents=True, exist_ok=True)

    # Load pre-trained models if available
    _load_pretrained_models(model_dir)

    logger.info("OptionQuant v%s started — %s mode", APP_VERSION, settings.environment)
    set_app_info(APP_VERSION, settings.environment)
    yield
    # Shutdown: close database pool
    close_pool()
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
        ).observe(time.perf_counter() - start)

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
app.include_router(pinns_routes.router)


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


async def _proxy_grafana_request(request: Request, path: str = ""):
    """Proxy Grafana through the main app so it can be embedded in the UI.

    Grafana is configured with GF_SERVER_SERVE_FROM_SUB_PATH=true and
    GF_SERVER_ROOT_URL=.../grafana/, so on port 3000 all its routes are
    under the /grafana/ prefix.  We must keep that prefix when forwarding.
    Using follow_redirects=False prevents an infinite redirect loop where
    Grafana redirects /d/... → /grafana/d/... which points back at us.
    """
    # Always keep the /grafana/ prefix so we hit the right path on port 3000
    upstream_path = f"/grafana/{path}" if path else "/grafana/"
    upstream_url = httpx.URL(f"{GRAFANA_UPSTREAM.rstrip('/')}{upstream_path}").copy_merge_params(request.query_params)

    body = await request.body()
    headers = {
        key: value
        for key, value in request.headers.items()
        if key.lower() not in {"host", "content-length", "connection", "accept-encoding"}
    }

    try:
        async with httpx.AsyncClient(follow_redirects=False, timeout=30.0) as client:
            upstream = await client.request(
                request.method,
                upstream_url,
                headers=headers,
                content=body,
            )
    except httpx.HTTPError as exc:
        logger.warning("Grafana proxy unavailable: %s", exc)
        return JSONResponse(
            status_code=503,
            content={
                "error": "Grafana unavailable",
                "message": "Start the monitoring stack to use the embedded Grafana view.",
            },
        )

    response_headers = {
        key: value
        for key, value in upstream.headers.items()
        if key.lower() not in {"content-length", "transfer-encoding", "connection", "x-frame-options", "content-security-policy"}
    }
    response_headers["Cache-Control"] = response_headers.get("Cache-Control", "no-store")

    return StarletteResponse(
        content=upstream.content,
        status_code=upstream.status_code,
        headers=response_headers,
        media_type=upstream.headers.get("content-type"),
    )


@app.api_route("/grafana", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"])
async def grafana_root_proxy(request: Request):
    return await _proxy_grafana_request(request)


@app.api_route("/grafana/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"])
async def grafana_proxy(path: str, request: Request):
    return await _proxy_grafana_request(request, path)


# ── Static files (must be LAST so API routes match first) ─────────
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
