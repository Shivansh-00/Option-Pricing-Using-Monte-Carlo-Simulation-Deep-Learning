"""
OptiQuant — Prometheus Metrics Instrumentation
═══════════════════════════════════════════════
Production-grade observability with:
  • HTTP request metrics (count, latency, in-flight)
  • Pricing engine metrics (duration by method)
  • WebSocket connection tracking
  • Model prediction counters
  • System health gauges
"""
from __future__ import annotations

import time
from functools import wraps
from typing import Callable

try:
    from prometheus_client import (
        Counter,
        Gauge,
        Histogram,
        Info,
        generate_latest,
        CONTENT_TYPE_LATEST,
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False


# ── Metric Definitions ───────────────────────────────────────

if PROMETHEUS_AVAILABLE:
    # HTTP metrics
    HTTP_REQUESTS_TOTAL = Counter(
        "optiquant_http_requests_total",
        "Total HTTP requests",
        ["method", "endpoint", "status"],
    )

    HTTP_REQUEST_DURATION = Histogram(
        "optiquant_http_request_duration_seconds",
        "HTTP request latency",
        ["method", "endpoint"],
        buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
    )

    HTTP_IN_FLIGHT = Gauge(
        "optiquant_http_requests_in_flight",
        "Currently in-flight HTTP requests",
    )

    # Pricing engine metrics
    PRICING_DURATION = Histogram(
        "optiquant_pricing_duration_seconds",
        "Option pricing computation time",
        ["method"],
        buckets=(0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0),
    )

    PRICING_REQUESTS = Counter(
        "optiquant_pricing_requests_total",
        "Total pricing computations",
        ["method", "option_type"],
    )

    # Monte Carlo specific
    MC_PATHS = Histogram(
        "optiquant_mc_simulation_paths",
        "Number of Monte Carlo paths used",
        ["variance_reduction"],
        buckets=(1000, 5000, 10000, 50000, 100000, 250000, 500000),
    )

    # WebSocket metrics
    WS_CONNECTIONS = Gauge(
        "optiquant_websocket_connections",
        "Active WebSocket connections",
        ["channel"],
    )

    WS_MESSAGES = Counter(
        "optiquant_websocket_messages_total",
        "Total WebSocket messages",
        ["direction", "channel"],
    )

    WS_ERRORS = Counter(
        "optiquant_websocket_errors_total",
        "WebSocket errors",
        ["channel", "error_type"],
    )

    # Model metrics
    MODEL_PREDICTIONS = Counter(
        "optiquant_model_predictions_total",
        "Model prediction count",
        ["model_name"],
    )

    MODEL_PREDICTION_DURATION = Histogram(
        "optiquant_model_prediction_duration_seconds",
        "Model prediction latency",
        ["model_name"],
        buckets=(0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0),
    )

    # RAG metrics
    RAG_QUERIES = Counter(
        "optiquant_rag_queries_total",
        "RAG query count",
        ["cache_hit"],
    )

    RAG_LATENCY = Histogram(
        "optiquant_rag_query_duration_seconds",
        "RAG query latency",
        buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0),
    )

    # Application info
    APP_INFO = Info(
        "optiquant_app",
        "Application build information",
    )


# ── Helper Functions ─────────────────────────────────────────

def track_pricing(method: str):
    """Decorator to track pricing computation metrics."""
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not PROMETHEUS_AVAILABLE:
                return func(*args, **kwargs)
            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                option_type = kwargs.get("option_type", "unknown")
                PRICING_REQUESTS.labels(method=method, option_type=option_type).inc()
                return result
            finally:
                duration = time.perf_counter() - start
                PRICING_DURATION.labels(method=method).observe(duration)
        return wrapper
    return decorator


def track_model(model_name: str):
    """Decorator to track model prediction metrics."""
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not PROMETHEUS_AVAILABLE:
                return func(*args, **kwargs)
            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                MODEL_PREDICTIONS.labels(model_name=model_name).inc()
                return result
            finally:
                duration = time.perf_counter() - start
                MODEL_PREDICTION_DURATION.labels(model_name=model_name).observe(duration)
        return wrapper
    return decorator


def get_metrics_output() -> tuple[bytes, str]:
    """Generate Prometheus metrics output."""
    if not PROMETHEUS_AVAILABLE:
        return b"# prometheus_client not installed\n", "text/plain"
    return generate_latest(), CONTENT_TYPE_LATEST


def set_app_info(version: str, environment: str) -> None:
    """Set application build info."""
    if PROMETHEUS_AVAILABLE:
        APP_INFO.info({
            "version": version,
            "environment": environment,
            "framework": "fastapi",
            "language": "python",
        })
