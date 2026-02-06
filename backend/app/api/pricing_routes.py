from __future__ import annotations

from fastapi import APIRouter

from .. import pricing
from ..schemas import GreeksResponse, PricingRequest, PricingResponse

router = APIRouter(prefix="/api/v1/pricing", tags=["pricing"])


@router.post("/bs", response_model=PricingResponse)
def pricing_bs(request: PricingRequest) -> PricingResponse:
    inputs = pricing.PricingInputs(**request.model_dump())
    price = pricing.black_scholes(inputs)
    return PricingResponse(model="black-scholes", price=price, metadata={"inputs": request.model_dump()})


@router.post("/mc", response_model=PricingResponse)
def pricing_mc(request: PricingRequest) -> PricingResponse:
    inputs = pricing.PricingInputs(**request.model_dump())
    price = pricing.monte_carlo_gbm(inputs, seed=7)
    return PricingResponse(model="monte-carlo-gbm", price=price, metadata={"paths": inputs.paths})


@router.post("/greeks", response_model=GreeksResponse)
def pricing_greeks(request: PricingRequest) -> GreeksResponse:
    inputs = pricing.PricingInputs(**request.model_dump())
    greeks = pricing.greeks_fd(inputs)
    return GreeksResponse(**greeks)
