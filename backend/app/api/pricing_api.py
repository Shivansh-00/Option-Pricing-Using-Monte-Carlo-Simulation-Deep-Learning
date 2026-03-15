from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import numpy as np
from fastapi import APIRouter, Depends, HTTPException

from ..auth import UserRecord, get_current_user
from ..data.market_data_loader import MarketDataPipeline, PipelineConfig
from ..pricing_engine.greeks_calculator import NeuralSDEGreeksCalculator
from ..pricing_engine.monte_carlo_simulator import (
    NeuralSDEMonteCarloPricer,
    OptionContract,
    benchmark_vs_baselines,
)
from ..pricing_engine.neural_sde_model import (
    NeuralSDE,
    NeuralSDEConfig,
    TrainingConfig,
)
from ..schemas import (
    NeuralSDEBenchmarkRequest,
    NeuralSDEBenchmarkResponse,
    NeuralSDEGreeksRequest,
    NeuralSDEGreeksResponse,
    NeuralSDEPriceRequest,
    NeuralSDEPriceResponse,
    NeuralSDESimRequest,
    NeuralSDESimResponse,
    NeuralSDETrainRequest,
    NeuralSDETrainResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/neural-sde", tags=["neural-sde"])

_MODEL_CACHE: dict[str, NeuralSDE] = {}


def _checkpoint_path(tag: str) -> Path:
    model_dir = Path(__file__).resolve().parents[3] / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir / f"neural_sde_{tag}.pt"


def _load_model(tag: str) -> NeuralSDE:
    if tag in _MODEL_CACHE:
        return _MODEL_CACHE[tag]
    path = _checkpoint_path(tag)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Model '{tag}' not found. Train first.")
    model, _ = NeuralSDE.load(path)
    _MODEL_CACHE[tag] = model
    return model


@router.post("/train", response_model=NeuralSDETrainResponse)
async def train_neural_sde(
    request: NeuralSDETrainRequest,
    _user: UserRecord = Depends(get_current_user),
) -> NeuralSDETrainResponse:
    try:
        config = PipelineConfig(
            timestamp_col=request.timestamp_col,
            price_col=request.price_col,
            freq=request.freq,
            lookback=request.lookback,
            batch_size=request.batch_size,
            val_split=request.val_split,
            test_split=request.test_split,
            random_seed=request.seed,
        )
        pipeline = MarketDataPipeline(config)

        split = await asyncio.to_thread(
            pipeline.build_datasets,
            request.prices_csv,
            request.implied_vol_csv,
            request.option_chain_csv,
            request.indicators_csv,
        )

        model = NeuralSDE(
            NeuralSDEConfig(
                hidden_dim=request.hidden_dim,
                num_layers=request.num_layers,
                dropout=request.dropout,
                drift_scale=request.drift_scale,
                sigma_floor=request.sigma_floor,
                market_feature_dim=len(split.feature_columns),
            )
        )

        history = await asyncio.to_thread(
            model.fit,
            split.train_loader,
            split.val_loader,
            TrainingConfig(
                epochs=request.epochs,
                learning_rate=request.learning_rate,
                weight_decay=request.weight_decay,
                grad_clip=request.grad_clip,
                seed=request.seed,
            ),
        )

        checkpoint = _checkpoint_path(request.model_tag)
        await asyncio.to_thread(
            model.save,
            checkpoint,
            {
                "feature_columns": split.feature_columns,
                "scaler": pipeline.scaler_state(),
                "train_size": split.train_size,
                "val_size": split.val_size,
                "test_size": split.test_size,
            },
        )
        _MODEL_CACHE[request.model_tag] = model

        return NeuralSDETrainResponse(
            model_tag=request.model_tag,
            checkpoint_path=str(checkpoint),
            feature_columns=split.feature_columns,
            train_size=split.train_size,
            val_size=split.val_size,
            test_size=split.test_size,
            best_val_loss=history["best_val_loss"],
            train_loss=history["train_loss"],
            val_loss=history["val_loss"],
            loss_components=history["components"],
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Neural SDE training failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Neural SDE training error: {e}")


@router.post("/price", response_model=NeuralSDEPriceResponse)
async def price_with_neural_sde(
    request: NeuralSDEPriceRequest,
    _user: UserRecord = Depends(get_current_user),
) -> NeuralSDEPriceResponse:
    try:
        model = _load_model(request.model_tag)
        pricer = NeuralSDEMonteCarloPricer(model=model, max_batch_paths=request.max_batch_paths)
        contract = OptionContract(
            spot=request.spot,
            strike=request.strike,
            maturity=request.maturity,
            rate=request.rate,
            option_type=request.option_type,
            dividend_yield=request.dividend_yield,
        )
        result = await asyncio.to_thread(
            pricer.price_european,
            contract,
            request.paths,
            request.steps,
            request.seed,
        )
        return NeuralSDEPriceResponse(
            model="neural-sde-monte-carlo",
            model_tag=request.model_tag,
            price=result.price,
            std_error=result.std_error,
            ci_lower=result.ci_lower,
            ci_upper=result.ci_upper,
            paths=result.paths,
            steps=result.steps,
            metadata={
                "discount_factor": result.discount_factor,
                "risk_neutral": True,
            },
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Neural SDE pricing failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Neural SDE pricing error: {e}")


@router.post("/greeks", response_model=NeuralSDEGreeksResponse)
async def neural_sde_greeks(
    request: NeuralSDEGreeksRequest,
    _user: UserRecord = Depends(get_current_user),
) -> NeuralSDEGreeksResponse:
    try:
        model = _load_model(request.model_tag)
        calc = NeuralSDEGreeksCalculator(model=model, n_paths=request.paths, steps=request.steps)
        greeks = await asyncio.to_thread(
            calc.compute,
            request.spot,
            request.strike,
            request.maturity,
            request.rate,
            request.option_type,
            request.seed,
        )
        return NeuralSDEGreeksResponse(
            model_tag=request.model_tag,
            delta=greeks.delta,
            gamma=greeks.gamma,
            vega=greeks.vega,
            theta=greeks.theta,
            rho=greeks.rho,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Neural SDE greeks failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Neural SDE greeks error: {e}")


@router.post("/benchmark", response_model=NeuralSDEBenchmarkResponse)
async def neural_sde_benchmark(
    request: NeuralSDEBenchmarkRequest,
    _user: UserRecord = Depends(get_current_user),
) -> NeuralSDEBenchmarkResponse:
    try:
        model = _load_model(request.model_tag)
        pricer = NeuralSDEMonteCarloPricer(model=model, max_batch_paths=request.max_batch_paths)
        contract = OptionContract(
            spot=request.spot,
            strike=request.strike,
            maturity=request.maturity,
            rate=request.rate,
            option_type=request.option_type,
            dividend_yield=request.dividend_yield,
        )
        neural = await asyncio.to_thread(
            pricer.price_european,
            contract,
            request.paths,
            request.steps,
            request.seed,
        )
        summary = await asyncio.to_thread(
            benchmark_vs_baselines,
            contract,
            neural,
            request.implied_vol,
            request.paths,
            request.steps,
        )
        return NeuralSDEBenchmarkResponse(
            model_tag=request.model_tag,
            benchmarks=summary,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Neural SDE benchmark failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Neural SDE benchmark error: {e}")


@router.post("/simulate", response_model=NeuralSDESimResponse)
async def neural_sde_simulate(
    request: NeuralSDESimRequest,
    _user: UserRecord = Depends(get_current_user),
) -> NeuralSDESimResponse:
    try:
        model = _load_model(request.model_tag)
        pricer = NeuralSDEMonteCarloPricer(model=model, max_batch_paths=request.max_batch_paths)
        contract = OptionContract(
            spot=request.spot,
            strike=request.strike,
            maturity=request.maturity,
            rate=request.rate,
            option_type=request.option_type,
            dividend_yield=request.dividend_yield,
        )
        paths = await asyncio.to_thread(
            pricer.simulate_paths,
            contract,
            request.paths,
            request.steps,
            request.seed,
            True,
        )
        sample_count = min(request.sample_paths, paths.shape[0])
        sampled = paths[:sample_count].tolist()
        terminal = paths[:, -1]
        return NeuralSDESimResponse(
            model_tag=request.model_tag,
            paths=request.paths,
            steps=request.steps,
            terminal_mean=float(terminal.mean()),
            terminal_std=float(terminal.std()),
            terminal_p05=float(np.quantile(terminal, 0.05)),
            terminal_p95=float(np.quantile(terminal, 0.95)),
            sample_paths=sampled,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Neural SDE simulate failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Neural SDE simulation error: {e}")
