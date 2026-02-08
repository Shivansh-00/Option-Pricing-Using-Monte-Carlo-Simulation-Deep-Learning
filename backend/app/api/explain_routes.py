from __future__ import annotations

from fastapi import APIRouter, Depends

from .. import explain
from ..auth import UserRecord, get_current_user
from ..schemas import ExplainRequest, ExplainResponse

router = APIRouter(prefix="/api/v1/ai", tags=["explain"])


@router.post("/explain", response_model=ExplainResponse)
def ai_explain(
    request: ExplainRequest,
    _user: UserRecord = Depends(get_current_user),
) -> ExplainResponse:
    answer, sources, confidence = explain.build_explanation(request)
    return ExplainResponse(answer=answer, sources=sources, confidence=confidence)
