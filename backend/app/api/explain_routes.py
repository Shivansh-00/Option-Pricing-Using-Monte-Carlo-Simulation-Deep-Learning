from __future__ import annotations

from fastapi import APIRouter

from .. import explain
from ..schemas import ExplainRequest, ExplainResponse

router = APIRouter(prefix="/api/v1/ai", tags=["explain"])


@router.post("/explain", response_model=ExplainResponse)
def ai_explain(request: ExplainRequest) -> ExplainResponse:
    answer, sources = explain.build_explanation(request)
    return ExplainResponse(answer=answer, sources=sources)
