from __future__ import annotations

from fastapi import APIRouter, Depends

from .. import dl, pricing
from ..auth import UserRecord, get_current_user
from ..schemas import PricingRequest

router = APIRouter(prefix="/api/v1/dl", tags=["dl"])


@router.post("/forecast")
def dl_forecast(
    request: PricingRequest,
    _user: UserRecord = Depends(get_current_user),
) -> dict:
    inputs = pricing.PricingInputs(**request.model_dump())
    mc_price = pricing.monte_carlo_gbm(inputs, seed=42)
    bs_price = pricing.black_scholes(inputs)
    hybrid = dl.residual_learning(bs_price, mc_price)
    return {
        "model": hybrid.model,
        "forecast_price": hybrid.forecast_price,
        "forecast_vol": hybrid.forecast_vol,
        "residual": hybrid.residual,
        "benchmarks": {"mc": mc_price, "bs": bs_price},
    }
