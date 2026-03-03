# ══════════════════════════════════════════════════════════════
#  OptiQuant Backend — Production Multi-Stage Build
#  Optimized for: security, size, caching, observability
# ══════════════════════════════════════════════════════════════

# ── Stage 1: Dependencies (cached layer) ─────────────────────
FROM python:3.11-slim AS deps

WORKDIR /build

RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc g++ && \
    rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt ./
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt && \
    pip install --no-cache-dir --prefix=/install \
        prometheus-client==0.21.0 \
        psycopg2-binary==2.9.9 \
        redis==5.0.8 \
        gunicorn==22.0.0 \
        celery==5.4.0

# ── Stage 2: Production Runtime ──────────────────────────────
FROM python:3.11-slim AS runtime

LABEL maintainer="OptiQuant Team" \
      version="2.0.0" \
      description="OptiQuant Backend — FastAPI Quant Engine"

# Security: non-root user
RUN groupadd -r optiquant && \
    useradd -r -g optiquant -d /app -s /sbin/nologin optiquant && \
    apt-get update && \
    apt-get install -y --no-install-recommends curl tini && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy installed dependencies
COPY --from=deps /install /usr/local

# Copy application code
COPY backend/app ./app
COPY backend/models ./models
COPY frontend ./frontend

# Create required directories
RUN mkdir -p /app/models /app/data /tmp/optiquant_logs /app/prometheus && \
    chown -R optiquant:optiquant /app /tmp/optiquant_logs

# Environment
ENV APP_ROOT_DIR=/app \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    PROMETHEUS_MULTIPROC_DIR=/app/prometheus

USER optiquant

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --retries=3 --start-period=20s \
    CMD curl -f http://localhost:8000/health || exit 1

# Use tini as init system for proper signal handling
ENTRYPOINT ["tini", "--"]

# Gunicorn with Uvicorn workers for production
CMD ["gunicorn", "app.main:app", \
     "--bind", "0.0.0.0:8000", \
     "--workers", "4", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--timeout", "120", \
     "--graceful-timeout", "30", \
     "--keep-alive", "5", \
     "--max-requests", "1000", \
     "--max-requests-jitter", "50", \
     "--access-logfile", "-", \
     "--error-logfile", "-", \
     "--log-level", "info"]
