"""
OptionQuant — PINNs (Physics-Informed Neural Networks) API Routes
═════════════════════════════════════════════════════════════════
Endpoints:
  POST /price          — Price an option using trained PINNs model
  POST /price-greeks   — Price with Greeks via finite differences
  POST /train          — Train PINNs model (background)
  GET  /status         — Model build & training status
"""
from __future__ import annotations

import asyncio
import logging
import threading

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ..auth import UserRecord, get_current_user
from ..pinns import get_pinns_pricer

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/pricing/pinns", tags=["pinns"])


# ─── Request / Response schemas ───────────────────────────────

class PINNsPriceRequest(BaseModel):
    spot: float = Field(..., gt=0, description="Spot price")
    strike: float = Field(..., gt=0, description="Strike price")
    time_to_expiry: float = Field(..., gt=0, description="Time to expiry (years)")
    volatility: float = Field(..., gt=0, description="Implied volatility")
    rate: float = Field(0.05, ge=0, description="Risk-free rate")


class PINNsPriceResponse(BaseModel):
    model: str = "pinns"
    price: float
    metadata: dict = {}


class PINNsGreeksResponse(BaseModel):
    model: str = "pinns"
    price: float
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float


class PINNsTrainRequest(BaseModel):
    epochs: int = Field(200, ge=10, le=2000, description="Training epochs")
    learning_rate: float = Field(0.001, gt=0, le=0.1)
    n_samples: int = Field(5000, ge=100, le=50000, description="Synthetic training samples")


class PINNsTrainResponse(BaseModel):
    status: str
    message: str
    final_loss: float | None = None


# ─── Background training state ───────────────────────────────

_train_lock = threading.Lock()
_train_status: dict = {"running": False, "last_loss": None, "epochs_done": 0}


# ─── Endpoints ────────────────────────────────────────────────

@router.post("/price", response_model=PINNsPriceResponse)
async def pinns_price(
    req: PINNsPriceRequest,
    _user: UserRecord = Depends(get_current_user),
):
    """Price an option using the PINNs model."""
    pricer = get_pinns_pricer()
    if not pricer._built:
        raise HTTPException(status_code=503, detail="PINNs model not trained yet")

    try:
        price = await asyncio.to_thread(  # type: ignore[arg-type]
            pricer.predict,
            S=req.spot,
            K=req.strike,
            tau=req.time_to_expiry,
            sigma=req.volatility,
            r=req.rate,
        )
        return PINNsPriceResponse(
            price=round(float(price[0]), 8),
            metadata={
                "spot": req.spot,
                "strike": req.strike,
                "time_to_expiry": req.time_to_expiry,
                "volatility": req.volatility,
                "rate": req.rate,
            },
        )
    except Exception as e:
        logger.error("PINNs pricing error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"PINNs pricing error: {e}")


@router.post("/price-greeks", response_model=PINNsGreeksResponse)
async def pinns_price_greeks(
    req: PINNsPriceRequest,
    _user: UserRecord = Depends(get_current_user),
):
    """Price an option with Greeks computed via finite differences."""
    pricer = get_pinns_pricer()
    if not pricer._built:
        raise HTTPException(status_code=503, detail="PINNs model not trained yet")

    try:
        result = await asyncio.to_thread(
            pricer.compute_greeks,
            S=req.spot,
            K=req.strike,
            tau=req.time_to_expiry,
            sigma=req.volatility,
            r=req.rate,
        )
        return PINNsGreeksResponse(**result)  # type: ignore[arg-type]
    except Exception as e:
        logger.error("PINNs greeks error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"PINNs error: {e}")


@router.post("/train", response_model=PINNsTrainResponse)
async def pinns_train(
    req: PINNsTrainRequest,
    _user: UserRecord = Depends(get_current_user),
):
    """Trigger PINNs training in a background thread."""
    if _train_status["running"]:
        raise HTTPException(status_code=409, detail="Training already in progress")

    def _do_train():
        with _train_lock:
            _train_status["running"] = True
            _train_status["epochs_done"] = 0
            try:
                pricer = get_pinns_pricer(reset=True)
                data = pricer.generate_training_data(n_samples=req.n_samples)
                pricer.train(
                    data,
                    epochs=req.epochs,
                    learning_rate=req.learning_rate,
                )
                _train_status["last_loss"] = float(pricer._train_history[-1]) if pricer._train_history else None
                _train_status["epochs_done"] = req.epochs
            except Exception as e:
                logger.error("PINNs training error: %s", e, exc_info=True)
                _train_status["last_loss"] = None
            finally:
                _train_status["running"] = False

    thread = threading.Thread(target=_do_train, daemon=True)
    thread.start()

    return PINNsTrainResponse(
        status="started",
        message=f"Training started: {req.epochs} epochs, {req.n_samples} samples",
    )


@router.get("/status")
async def pinns_status(
    _user: UserRecord = Depends(get_current_user),
):
    """Return PINNs model status."""
    pricer = get_pinns_pricer()
    status = pricer.get_status()
    status["training_in_progress"] = _train_status["running"]
    status["last_training_loss"] = _train_status.get("last_loss")
    return status
