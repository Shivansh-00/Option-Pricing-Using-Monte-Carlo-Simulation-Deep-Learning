from __future__ import annotations

from fastapi import APIRouter

from .. import ml
from ..schemas import VolatilityRequest, VolatilityResponse

router = APIRouter(prefix="/api/v1/ml", tags=["ml"])


@router.post("/iv-predict", response_model=VolatilityResponse)
def ml_iv(request: VolatilityRequest) -> VolatilityResponse:
    iv, regime, drivers = ml.predict_iv(request)
    return VolatilityResponse(implied_vol=iv, regime=regime, drivers=drivers)
