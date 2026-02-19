"""
OptionQuant — Pricing API Routes (Enterprise)
═══════════════════════════════════════════════
Endpoints:
  POST /bs           — Black-Scholes analytical pricing
  POST /mc           — Monte Carlo GBM pricing
  POST /mc/detailed  — MC with full diagnostics & paths
  POST /mc/compare   — Compare all pricing methods
  POST /greeks       — Greeks (analytical)
"""
from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, Depends, HTTPException

from .. import pricing
from ..auth import UserRecord, get_current_user
from ..schemas import (
    GreeksResponse,
    MCComparisonResponse,
    MCDetailedResponse,
    MCSimulationRequest,
    PricingRequest,
    PricingResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/pricing", tags=["pricing"])


@router.post("/bs", response_model=PricingResponse)
async def pricing_bs(
    request: PricingRequest,
    _user: UserRecord = Depends(get_current_user),
) -> PricingResponse:
    """Black-Scholes closed-form option pricing."""
    try:
        inputs = pricing.PricingInputs(**request.model_dump())
        price = pricing.black_scholes(inputs)
        return PricingResponse(
            model="black-scholes",
            price=round(price, 8),
            metadata={"inputs": request.model_dump(), "method": "analytical"},
        )
    except Exception as e:
        logger.error("BS pricing error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Pricing error: {e}")


@router.post("/mc", response_model=PricingResponse)
async def pricing_mc(
    request: PricingRequest,
    _user: UserRecord = Depends(get_current_user),
) -> PricingResponse:
    """Standard Monte Carlo GBM pricing."""
    try:
        inputs = pricing.PricingInputs(**request.model_dump())
        result = await asyncio.to_thread(pricing.monte_carlo_engine, inputs, 42)
        return PricingResponse(
            model="monte-carlo-gbm",
            price=round(result.price, 8),
            metadata={
                "paths": result.paths_used,
                "steps": result.steps_used,
                "std_error": round(result.std_error, 8),
                "ci_lower": round(result.ci_lower, 8),
                "ci_upper": round(result.ci_upper, 8),
                "elapsed_ms": result.elapsed_ms,
            },
        )
    except Exception as e:
        logger.error("MC pricing error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Pricing error: {e}")


@router.post("/mc/detailed", response_model=MCDetailedResponse)
async def pricing_mc_detailed(
    request: MCSimulationRequest,
    _user: UserRecord = Depends(get_current_user),
) -> MCDetailedResponse:
    """
    Detailed Monte Carlo simulation with variance reduction.
    Methods: standard, antithetic, control_variate, stratified
    Optionally returns sample paths for visualization.
    """
    try:
        inputs = pricing.PricingInputs(
            spot=request.spot, strike=request.strike,
            maturity=request.maturity, rate=request.rate,
            volatility=request.volatility, option_type=request.option_type,
            steps=request.steps, paths=request.paths,
        )
        result = await asyncio.to_thread(
            pricing.monte_carlo_engine,
            inputs,
            request.seed,
            request.method,
            request.return_paths,
        )
        return MCDetailedResponse(
            price=round(result.price, 8),
            std_error=round(result.std_error, 8),
            ci_lower=round(result.ci_lower, 8),
            ci_upper=round(result.ci_upper, 8),
            paths_used=result.paths_used,
            steps_used=result.steps_used,
            variance_reduction=result.variance_reduction,
            elapsed_ms=result.elapsed_ms,
            convergence=result.convergence,
            sample_paths=result.sample_paths[:50],  # cap paths for response size
        )
    except Exception as e:
        logger.error("MC detailed error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Simulation error: {e}")


@router.post("/mc/compare", response_model=MCComparisonResponse)
async def pricing_mc_compare(
    request: PricingRequest,
    _user: UserRecord = Depends(get_current_user),
) -> MCComparisonResponse:
    """Compare BS with multiple Monte Carlo methods."""
    try:
        inputs = pricing.PricingInputs(**request.model_dump())
        result = await asyncio.to_thread(pricing.price_all_methods, inputs)
        return MCComparisonResponse(**result)
    except Exception as e:
        logger.error("MC compare error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Comparison error: {e}")


@router.post("/greeks", response_model=GreeksResponse)
async def pricing_greeks(
    request: PricingRequest,
    _user: UserRecord = Depends(get_current_user),
) -> GreeksResponse:
    """Analytical Black-Scholes Greeks."""
    try:
        inputs = pricing.PricingInputs(**request.model_dump())
        greeks = pricing.greeks_fd(inputs)
        return GreeksResponse(**greeks)
    except Exception as e:
        logger.error("Greeks error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Greeks error: {e}")
