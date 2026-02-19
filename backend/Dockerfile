# ══════════════════════════════════════════════════════════════
#  OptionQuant Backend — Multi-Stage Optimized Build
# ══════════════════════════════════════════════════════════════

# ── Stage 1: Dependencies ────────────────────────────────────
FROM python:3.11-slim AS deps

WORKDIR /build

RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc g++ && \
    rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt ./
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ── Stage 2: Production Runtime ──────────────────────────────
FROM python:3.11-slim AS runtime

RUN groupadd -r optiquant && useradd -r -g optiquant -d /app -s /sbin/nologin optiquant

WORKDIR /app

COPY --from=deps /install /usr/local

COPY backend/app ./app
COPY backend/models ./models
COPY frontend ./frontend

RUN mkdir -p /app/models /app/data /tmp/optiquant_logs && \
    chown -R optiquant:optiquant /app /tmp/optiquant_logs

ENV APP_ROOT_DIR=/app \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

USER optiquant

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2", "--access-log", "--log-level", "info"]
