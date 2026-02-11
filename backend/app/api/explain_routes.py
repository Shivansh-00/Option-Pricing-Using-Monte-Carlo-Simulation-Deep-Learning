from __future__ import annotations

from fastapi import APIRouter, Depends

from .. import explain
from ..auth import UserRecord, get_current_user
from ..schemas import ExplainRequest, ExplainResponse, RAGHealthResponse, RAGStatsResponse

router = APIRouter(prefix="/api/v1/ai", tags=["explain"])


@router.post("/explain", response_model=ExplainResponse)
def ai_explain(
    request: ExplainRequest,
    _user: UserRecord = Depends(get_current_user),
) -> ExplainResponse:
    result = explain.build_explanation(request, chat_history=request.chat_history)
    return ExplainResponse(**result)


@router.get("/rag/health", response_model=RAGHealthResponse)
def rag_health(
    _user: UserRecord = Depends(get_current_user),
) -> RAGHealthResponse:
    """RAG subsystem health and configuration."""
    return RAGHealthResponse(**explain.get_rag_health())


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
