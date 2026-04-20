from __future__ import annotations

import time

from fastapi import APIRouter, Depends

from .. import explain
from ..auth import UserRecord, get_current_user
from ..prometheus_metrics import PROMETHEUS_AVAILABLE
from ..schemas import (
    ExplainRequest,
    ExplainResponse,
    RAGHealthResponse,
    RAGMetricsResponse,
    RAGStatsResponse,
)

router = APIRouter(prefix="/api/v1/ai", tags=["explain"])


def _record_rag(cache_hit: bool, duration: float):
    """Record RAG query metrics to Prometheus."""
    if not PROMETHEUS_AVAILABLE:
        return
    from ..prometheus_metrics import RAG_QUERIES, RAG_LATENCY
    RAG_QUERIES.labels(cache_hit=str(cache_hit).lower()).inc()
    RAG_LATENCY.observe(duration)


@router.post("/explain", response_model=ExplainResponse)
def ai_explain(
    request: ExplainRequest,
    _user: UserRecord = Depends(get_current_user),
) -> ExplainResponse:
    t0 = time.perf_counter()
    result = explain.build_explanation(request, chat_history=request.chat_history)
    _record_rag(cache_hit=result.get("cached", False), duration=time.perf_counter() - t0)
    return ExplainResponse(**result)


@router.get("/rag/health", response_model=RAGHealthResponse)
def rag_health(
    _user: UserRecord = Depends(get_current_user),
) -> RAGHealthResponse:
    """RAG subsystem health and configuration."""
    return RAGHealthResponse(**explain.get_rag_health())


@router.get("/rag/metrics", response_model=RAGMetricsResponse)
def rag_metrics(
    _user: UserRecord = Depends(get_current_user),
) -> RAGMetricsResponse:
    """RAG evaluation metrics and quality dashboard."""
    from ..rag.evaluation import get_metrics_tracker

    summary = get_metrics_tracker().summary
    return RAGMetricsResponse(**summary)


@router.get("/rag/stats", response_model=RAGStatsResponse)
def rag_stats(
    _user: UserRecord = Depends(get_current_user),
) -> RAGStatsResponse:
    """RAG index statistics and performance metrics."""
    health = explain.get_rag_health()
    idx = health["index"]
    cache = health["cache"]
    return RAGStatsResponse(
        total_chunks=idx["total_chunks"],
        unique_sources=idx["unique_sources"],
        source_files=idx["source_files"],
        vocab_size=idx["vocab_size"],
        queries_served=idx["queries_served"],
        avg_search_ms=idx["avg_search_ms"],
        cache_hit_rate=cache["hit_rate"],
    )
