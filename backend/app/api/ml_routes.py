from __future__ import annotations

from fastapi import APIRouter, Depends

from .. import ml
from ..auth import UserRecord, get_current_user
from ..schemas import VolatilityRequest, VolatilityResponse

router = APIRouter(prefix="/api/v1/ml", tags=["ml"])


@router.post("/iv-predict", response_model=VolatilityResponse)
def ml_iv(
    request: VolatilityRequest,
    _user: UserRecord = Depends(get_current_user),
) -> VolatilityResponse:
    iv, regime, drivers = ml.predict_iv(request)
    return VolatilityResponse(implied_vol=iv, regime=regime, drivers=drivers)
