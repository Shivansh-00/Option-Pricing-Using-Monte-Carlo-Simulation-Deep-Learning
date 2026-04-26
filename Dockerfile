# ─────────────────────────────────────────────────────────────
#  OptiQuant — Production Dockerfile (Render / Fly / DO / K8s)
#  Single image: FastAPI backend + static frontend + models
# ─────────────────────────────────────────────────────────────
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=8000

WORKDIR /app

# System deps for psycopg2, numpy, scipy
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first (layer cache)
COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --upgrade pip && \
    pip install -r /app/backend/requirements.txt

# Copy application
COPY backend /app/backend
COPY frontend /app/frontend

# Render/Fly inject PORT env var; fallback to 8000
ENV APP_ROOT_DIR=/app
ENV PYTHONPATH=/app

# Disable strict dependency-pin check in container (image is already locked)
ENV ENFORCE_DEPENDENCY_PINS=0

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

EXPOSE 8000

# Use shell form so $PORT is expanded at runtime
CMD uvicorn backend.app.main:app --host 0.0.0.0 --port ${PORT} --workers 1 --proxy-headers --forwarded-allow-ips="*"
