from __future__ import annotations

import logging
import threading
from dataclasses import asdict
from fastapi import APIRouter, Depends, HTTPException

from .. import ml
from ..auth import UserRecord, get_current_user
from ..schemas import (
    VolatilityRequest,
    VolatilityResponse,
    VolTrainRequest,
    VolTrainResponse,
    VolModelComparison,
    VolModelMetrics,
    VolEngineStatusResponse,
)
from ..vol_engine import get_engine

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/ml", tags=["ml"])

# ─── Training state for async tracking ─────────────────────────
_training_lock = threading.Lock()
_training_in_progress = False
_training_error: str | None = None


@router.post("/iv-predict", response_model=VolatilityResponse)
def ml_iv(
    request: VolatilityRequest,
    _user: UserRecord = Depends(get_current_user),
) -> VolatilityResponse:
    """Predict implied volatility — uses trained ML model if available, else analytical fallback."""
    engine = get_engine()
    if engine.is_trained:
        result = engine.predict_single(
            spot=request.spot,
            rate=request.rate,
            maturity=request.maturity,
            realized_vol=request.realized_vol,
            vix=request.vix,
            skew=request.skew,
        )
        return VolatilityResponse(
            implied_vol=result["implied_vol"],
            regime=result["regime"],
            drivers=result["drivers"],
            model_used=result["model_used"],
            confidence=result["confidence"],
        )
    # Fallback to original analytical
    iv, regime, drivers = ml.predict_iv(request)
    return VolatilityResponse(implied_vol=iv, regime=regime, drivers=drivers)


@router.post("/vol/train", response_model=VolTrainResponse)
def train_vol_engine(
    request: VolTrainRequest,
    _user: UserRecord = Depends(get_current_user),
):
    """
    Train the ML volatility engine.
    Runs the full pipeline: data → features → targets → models → evaluation.
    """
    global _training_in_progress, _training_error
    with _training_lock:
        if _training_in_progress:
            raise HTTPException(status_code=409, detail="Training already in progress")
        _training_in_progress = True
        _training_error = None
    try:
        engine = get_engine()
        result = engine.train_and_evaluate(
            model_names=request.models,
            target_name=request.target,
            forward_window=request.forward_window,
            n_cv_folds=request.cv_folds,
            n_days=request.n_days,
            seed=request.seed,
        )
        # Convert dataclass result → Pydantic response
        comparisons = []
        for c in result.comparisons:
            comparisons.append(VolModelComparison(
                model_name=c.model_name,
                target_name=c.target_name,
                train_metrics=VolModelMetrics(**asdict(c.train_metrics)),
                test_metrics=VolModelMetrics(**asdict(c.test_metrics)),
                cv_metrics=VolModelMetrics(**asdict(c.cv_metrics)) if c.cv_metrics else None,
                train_time_ms=c.train_time_ms,
                inference_time_ms=c.inference_time_ms,
                feature_importance=c.feature_importance,
                improvement_vs_historical=c.improvement_vs_historical,
                improvement_vs_garch=c.improvement_vs_garch,
                improvement_vs_ewma=c.improvement_vs_ewma,
            ))

        return VolTrainResponse(
            comparisons=comparisons,
            best_model=result.best_model,
            best_target=result.best_target,
            best_test_rmse=result.best_test_rmse,
            best_test_r2=result.best_test_r2,
            baseline_rmse=result.baseline_rmse,
            feature_names=result.feature_names,
            top_features=result.top_features,
            n_train=result.n_train,
            n_val=result.n_val,
            n_test=result.n_test,
            total_time_ms=result.total_time_ms,
        )
    except Exception as e:
        _training_error = str(e)
        logger.error("Training failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Training failed: {e}")
    finally:
        _training_in_progress = False


@router.get("/vol/status", response_model=VolEngineStatusResponse)
def vol_engine_status(
    _user: UserRecord = Depends(get_current_user),
):
    """Get the current status of the volatility engine."""
    engine = get_engine()
    result = engine.last_result

    return VolEngineStatusResponse(
        is_trained=engine.is_trained,
        models_available=list(engine._trained_models.keys()),
        best_model=result.best_model if result else None,
        best_rmse=result.best_test_rmse if result else None,
        best_r2=result.best_test_r2 if result else None,
        n_features=len(engine._feature_names),
        target=result.best_target if result else None,
    )


@router.get("/vol/models")
def list_available_models(
    _user: UserRecord = Depends(get_current_user),
):
    """List all available model types in the registry."""
    from ..vol_models import MODEL_REGISTRY
    return {
        "models": list(MODEL_REGISTRY.keys()),
        "targets": ["realized_vol", "parkinson_vol", "garman_klass_vol"],
        "training_in_progress": _training_in_progress,
    }
