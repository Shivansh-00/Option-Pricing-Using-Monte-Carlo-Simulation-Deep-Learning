"""
OptionQuant — Quant Intelligence Engine API Routes
════════════════════════════════════════════════════
Unified API surface for the cohesive Quant Ecosystem:

  PINNs:           POST /pinns/train, /pinns/predict, /pinns/greeks
  RL Hedging:      POST /hedging/train, /hedging/backtest, /hedging/suggest
  Vol Surface:     POST /vol-surface/predict, /vol-surface/train
  Jump Diffusion:  POST /jump-diffusion/price, /jump-diffusion/calibrate
  Arbitrage:       POST /arbitrage/scan, /arbitrage/put-call-parity
  Uncertainty:     POST /uncertainty/quantify, /uncertainty/train
  GPU MC:          POST /gpu-mc/price, /gpu-mc/benchmark
  Portfolio Risk:  POST /portfolio/risk-report, /portfolio/stress-test
  Explainer:       POST /explain/decision
  Ecosystem:       GET  /status
"""
from __future__ import annotations

import asyncio
import logging
import math
import time
import threading
import uuid

import numpy as np
from fastapi import APIRouter, Depends, HTTPException

from ..auth import UserRecord, get_current_user
from ..quant_schemas import (
    # PINNs
    PINNsTrainRequest, PINNsTrainResponse,
    PINNsPredictRequest, PINNsPredictResponse,
    PINNsGreeksRequest, PINNsGreeksResponse,
    PINNsStatusResponse,
    # RL Hedging
    HedgingTrainRequest, HedgingTrainResponse, HedgingStatusResponse,
    HedgingBacktestRequest, HedgingBacktestResponse,
    HedgeSuggestRequest, HedgeSuggestResponse,
    # Vol Surface
    VolSurfaceTrainRequest, VolSurfaceTrainResponse,
    VolSurfacePredictRequest, VolSurfacePredictResponse,
    # Jump Diffusion
    JumpDiffusionPriceRequest, JumpDiffusionPriceResponse,
    RegimeCalibrateRequest, RegimeCalibrateResponse,
    ScenarioAnalysisRequest, ScenarioAnalysisResponse,
    # Arbitrage
    ArbitrageScanRequest, ArbitrageScanResponse,
    PutCallParityRequest, PutCallParityResponse,
    # Uncertainty
    UncertaintyRequest, UncertaintyResponse,
    UncertaintyTrainRequest, UncertaintyTrainResponse,
    # GPU MC
    GPUMCPriceRequest, GPUMCPriceResponse,
    GPUMCBenchmarkRequest, GPUMCBenchmarkResponse,
    # Portfolio
    PortfolioRiskRequest, PortfolioRiskResponse,
    PortfolioStressRequest, PortfolioStressResponse,
    # Explainer
    QuantExplainRequest, QuantExplainResponse,
    # Ecosystem
    QuantEcosystemStatusResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/quant", tags=["quant-intelligence"])


# ═══════════════════════════════════════════════════════════════
#  PINNs — Physics-Informed Neural Networks
# ═══════════════════════════════════════════════════════════════

# ---------- Background training manager ----------

class _PINNsTrainingJob:
    """Holds state for a background PINNs training run."""
    def __init__(self, job_id: str, epochs: int, n_samples: int):
        self.job_id = job_id
        self.status = "queued"
        self.total_epochs = epochs
        self.current_epoch = 0
        self.current_loss = 0.0
        self.n_samples = n_samples
        self.result: dict = {}
        self.error = ""
        self.started_at = time.time()
        self.finished_at: float | None = None

    def to_dict(self) -> dict:
        elapsed = (self.finished_at or time.time()) - self.started_at
        progress = (self.current_epoch / max(self.total_epochs, 1)) * 100
        return {
            "job_id": self.job_id,
            "status": self.status,
            "progress": round(progress, 1),
            "current_epoch": self.current_epoch,
            "total_epochs": self.total_epochs,
            "current_loss": round(self.current_loss, 8),
            "elapsed_seconds": round(elapsed, 2),
            "result": self.result,
            "error": self.error,
        }


class _PINNsTrainingManager:
    """Thread-safe manager for at-most-one PINNs background job."""
    def __init__(self):
        self._lock = threading.Lock()
        self._job: _PINNsTrainingJob | None = None
        self._thread: threading.Thread | None = None

    @property
    def is_busy(self) -> bool:
        with self._lock:
            return (self._job is not None
                    and self._job.status in ("queued", "training"))

    def get_status(self) -> dict:
        with self._lock:
            if self._job is None:
                return {"status": "idle", "job_id": "", "progress": 0.0,
                        "current_epoch": 0, "total_epochs": 0,
                        "current_loss": 0.0, "elapsed_seconds": 0.0,
                        "result": {}, "error": ""}
            return self._job.to_dict()


_pinns_manager = _PINNsTrainingManager()


@router.post("/pinns/train", response_model=PINNsTrainResponse)
async def pinns_train(
    request: PINNsTrainRequest,
    _user: UserRecord = Depends(get_current_user),
) -> PINNsTrainResponse:
    """Train PINNs model with Black-Scholes PDE constraints."""
    try:
        from ..pinns import get_pinns_pricer, PINNsOptionPricer

        # Fresh pricer each training run to avoid stale weights
        pricer = get_pinns_pricer(reset=True)
        pricer.config.epochs = request.epochs

        train_data = await asyncio.to_thread(
            PINNsOptionPricer.generate_training_data,
            n_samples=request.n_samples,
        )

        start = time.perf_counter()
        result = await asyncio.to_thread(pricer.train, train_data)
        elapsed = (time.perf_counter() - start) * 1000

        history = result.get("history", [])
        final_losses = result.get("final_losses", {})
        last_epoch = history[-1] if history else {}

        return PINNsTrainResponse(
            epochs_trained=int(result.get("epochs", request.epochs)),
            final_loss=float(result.get("best_loss",
                                        last_epoch.get("total_loss", 0))),
            pde_loss=float(final_losses.get("pde_loss",
                           last_epoch.get("pde_loss", 0))),
            data_loss=float(final_losses.get("data_loss",
                            last_epoch.get("data_loss", 0))),
            arbitrage_loss=float(final_losses.get("arb_loss",
                                 last_epoch.get("arb_loss", 0))),
            training_time_ms=round(elapsed, 2),
            loss_history=[float(h.get("total_loss", 0))
                          for h in history[-50:]],
        )
    except Exception as e:
        logger.error("PINNs training error: %s", e, exc_info=True)
        raise HTTPException(status_code=500,
                            detail=f"PINNs training error: {e}")


@router.get("/pinns/status")
async def pinns_status(
    _user: UserRecord = Depends(get_current_user),
) -> PINNsStatusResponse:
    """Return current PINNs training status."""
    d = _pinns_manager.get_status()
    return PINNsStatusResponse(**d)


@router.post("/pinns/predict", response_model=PINNsPredictResponse)
async def pinns_predict(
    request: PINNsPredictRequest,
    _user: UserRecord = Depends(get_current_user),
) -> PINNsPredictResponse:
    """Price option using trained PINNs model with PDE residual check."""
    try:
        from ..pinns import get_pinns_pricer

        pricer = get_pinns_pricer()
        if not pricer._built:
            raise HTTPException(
                status_code=400,
                detail="PINNs model not trained yet. Train first via POST /pinns/train.")

        S_arr = np.array([request.spot]).reshape(-1, 1)
        K_arr = np.array([request.strike]).reshape(-1, 1)
        tau_arr = np.array([request.maturity]).reshape(-1, 1)
        sigma_arr = np.array([request.volatility]).reshape(-1, 1)
        r_arr = np.array([request.rate]).reshape(-1, 1)

        prices = await asyncio.to_thread(
            pricer.predict, S_arr, K_arr, tau_arr, sigma_arr, r_arr,
        )
        pinns_price = float(prices[0])

        greeks = await asyncio.to_thread(
            pricer.compute_greeks,
            request.spot, request.strike, request.maturity,
            request.volatility, request.rate,
        )

        # Black-Scholes benchmark
        from .. import pricing
        inputs = pricing.PricingInputs(
            spot=request.spot, strike=request.strike,
            maturity=request.maturity, rate=request.rate,
            volatility=request.volatility, option_type=request.option_type,
        )
        bs_price = pricing.black_scholes(inputs)
        deviation = abs(pinns_price - bs_price) / max(bs_price, 1e-8) * 100

        pde_res = pricer._pde_residual(
            S_arr, K_arr, tau_arr, sigma_arr, r_arr,
        )
        pde_residual = float(np.mean(pde_res ** 2))

        return PINNsPredictResponse(
            pinns_price=round(pinns_price, 6),
            bs_price=round(bs_price, 6),
            deviation_pct=round(deviation, 4),
            pde_residual=round(pde_residual, 8),
            greeks=greeks,
            metadata={"is_trained": pricer._built},
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("PINNs predict error: %s", e, exc_info=True)
        raise HTTPException(status_code=500,
                            detail=f"PINNs predict error: {e}")


@router.post("/pinns/greeks", response_model=PINNsGreeksResponse)
async def pinns_greeks(
    request: PINNsGreeksRequest,
    _user: UserRecord = Depends(get_current_user),
) -> PINNsGreeksResponse:
    """Compute PDE-informed Greeks via PINNs finite differences."""
    try:
        from ..pinns import get_pinns_pricer

        pricer = get_pinns_pricer()
        greeks = await asyncio.to_thread(
            pricer.compute_greeks,
            request.spot, request.strike, request.maturity,
            request.volatility, request.rate,
        )
        return PINNsGreeksResponse(
            delta=round(float(greeks.get("delta", 0)), 6),
            gamma=round(float(greeks.get("gamma", 0)), 8),
            theta=round(float(greeks.get("theta", 0)), 6),
            vega=round(float(greeks.get("vega", 0)), 6),
        )
    except Exception as e:
        logger.error("PINNs greeks error: %s", e, exc_info=True)
        raise HTTPException(status_code=500,
                            detail=f"PINNs greeks error: {e}")


# ═══════════════════════════════════════════════════════════════
#  RL Dynamic Hedging — Background Training Manager
# ═══════════════════════════════════════════════════════════════

class _RLTrainingJob:
    """Holds state for a background RL training run."""
    def __init__(self, job_id: str, agent_type: str, episodes: int):
        self.job_id = job_id
        self.agent_type = agent_type
        self.status = "queued"           # queued → training → completed | failed
        self.total_episodes = episodes
        self.current_episode = 0
        self.avg_reward = 0.0
        self.reward_history: list[float] = []
        self.result: dict = {}
        self.error = ""
        self.started_at = time.time()
        self.finished_at: float | None = None

    def to_dict(self) -> dict:
        elapsed = (self.finished_at or time.time()) - self.started_at
        progress = (self.current_episode / max(self.total_episodes, 1)) * 100
        d = {
            "job_id": self.job_id,
            "status": self.status,
            "progress": round(progress, 1),
            "current_episode": self.current_episode,
            "total_episodes": self.total_episodes,
            "avg_reward": round(self.avg_reward, 6),
            "reward_history": self.reward_history[-50:],
            "elapsed_seconds": round(elapsed, 1),
            "result": self.result,
            "error": self.error,
        }
        return d


class _RLTrainingManager:
    """Thread-safe manager for RL background training."""
    def __init__(self):
        self._lock = threading.Lock()
        self._current: _RLTrainingJob | None = None
        self._thread: threading.Thread | None = None

    @property
    def is_busy(self) -> bool:
        with self._lock:
            return (self._current is not None
                    and self._current.status in ("queued", "training"))

    def start(self, agent_type: str, episodes: int, env_config) -> _RLTrainingJob:
        if self.is_busy:
            raise HTTPException(status_code=409,
                                detail="RL training already in progress. Poll /hedging/status.")
        job = _RLTrainingJob(
            job_id=uuid.uuid4().hex[:12],
            agent_type=agent_type,
            episodes=episodes,
        )
        with self._lock:
            self._current = job
        t = threading.Thread(target=self._run, args=(job, env_config), daemon=True)
        self._thread = t
        t.start()
        return job

    def get_status(self) -> dict:
        with self._lock:
            if self._current is None:
                return {"job_id": "", "status": "idle", "progress": 0,
                        "current_episode": 0, "total_episodes": 0,
                        "avg_reward": 0, "reward_history": [],
                        "elapsed_seconds": 0, "result": {}, "error": ""}
            return self._current.to_dict()

    def _run(self, job: _RLTrainingJob, env_config) -> None:
        try:
            from ..rl_hedging import get_hedging_engine
            job.status = "training"
            engine = get_hedging_engine(agent_type=job.agent_type)

            def on_progress(ep, total, avg_r):
                job.current_episode = ep + 1
                job.avg_reward = avg_r
                job.reward_history.append(round(avg_r, 6))

            result = engine.train(
                n_episodes=job.total_episodes,
                env_config=env_config,
                progress_callback=on_progress,
            )
            job.current_episode = job.total_episodes
            job.result = {
                "agent_type": result["agent_type"],
                "episodes_trained": result["episodes"],
                "final_reward": round(result["final_avg_reward"], 6),
                "training_time_s": result["training_time_s"],
            }
            job.status = "completed"
        except Exception as exc:
            logger.error("RL training failed: %s", exc, exc_info=True)
            job.status = "failed"
            job.error = str(exc)
        finally:
            job.finished_at = time.time()


_rl_manager = _RLTrainingManager()


@router.post("/hedging/train", response_model=HedgingTrainResponse)
async def hedging_train(
    request: HedgingTrainRequest,
    _user: UserRecord = Depends(get_current_user),
) -> HedgingTrainResponse:
    """Train RL hedging agent synchronously and return results."""
    try:
        from ..rl_hedging import get_hedging_engine, HedgingEnvConfig

        env_config = HedgingEnvConfig(
            S0=request.spot,
            K=request.strike,
            T=request.maturity,
            sigma=request.volatility,
            r=request.rate,
        )
        engine = get_hedging_engine(agent_type=request.agent_type)

        start = time.perf_counter()
        result = await asyncio.to_thread(
            engine.train, n_episodes=request.episodes, env_config=env_config,
        )
        elapsed = (time.perf_counter() - start) * 1000

        history = result.get("history", [])
        reward_list = [float(h.get("avg_reward", 0)) for h in history] if history else []
        last_100 = reward_list[-100:] if reward_list else []

        return HedgingTrainResponse(
            agent_type=request.agent_type,
            episodes_trained=int(result.get("episodes", request.episodes)),
            final_reward=float(result.get("final_avg_reward", 0)),
            avg_reward_last_100=float(np.mean(last_100)) if last_100 else 0.0,
            training_time_ms=round(elapsed, 2),
            reward_history=reward_list[-50:],
        )
    except Exception as e:
        logger.error("Hedging train error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Hedging train error: {e}")


@router.get("/hedging/status", response_model=HedgingStatusResponse)
async def hedging_status(
    _user: UserRecord = Depends(get_current_user),
) -> HedgingStatusResponse:
    """Poll RL training status and progress."""
    d = _rl_manager.get_status()
    return HedgingStatusResponse(**d)


@router.post("/hedging/backtest", response_model=HedgingBacktestResponse)
async def hedging_backtest(
    request: HedgingBacktestRequest,
    _user: UserRecord = Depends(get_current_user),
) -> HedgingBacktestResponse:
    """Backtest RL hedging vs Black-Scholes delta hedging."""
    try:
        from ..rl_hedging import get_hedging_engine, HedgingEnvConfig

        env_config = HedgingEnvConfig(
            S0=request.spot,
            K=request.strike,
            T=request.maturity,
            sigma=request.volatility,
            r=request.rate,
        )
        engine = get_hedging_engine(agent_type=request.agent_type)

        # backtest(n_episodes, env_config) -> Dict
        result = await asyncio.to_thread(
            engine.backtest, n_episodes=request.n_scenarios, env_config=env_config,
        )

        return HedgingBacktestResponse(
            rl_pnl_mean=round(float(result.get("rl_avg_pnl", 0)), 4),
            rl_pnl_std=round(float(result.get("rl_avg_std", 0)), 4),
            rl_max_drawdown=round(float(result.get("rl_max_drawdown", 0)), 4),
            bs_pnl_mean=round(float(result.get("bs_delta_avg_pnl", 0)), 4),
            bs_pnl_std=0.0,
            bs_max_drawdown=0.0,
            rl_sharpe=round(float(result.get("rl_avg_sharpe", 0)), 4),
            bs_sharpe=0.0,
            improvement_pct=round(float(result.get("improvement_vs_bs", 0)), 2),
            n_scenarios=request.n_scenarios,
            details=result.get("sample_path", {}),
        )
    except Exception as e:
        logger.error("Hedging backtest error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Hedging backtest error: {e}")


@router.post("/hedging/suggest", response_model=HedgeSuggestResponse)
async def hedging_suggest(
    request: HedgeSuggestRequest,
    _user: UserRecord = Depends(get_current_user),
) -> HedgeSuggestResponse:
    """Get real-time hedge ratio suggestion from RL agent."""
    try:
        from ..rl_hedging import get_hedging_engine

        engine = get_hedging_engine()
        regimes = {0: "BULL", 1: "BEAR", 2: "CRISIS"}

        # Compute BS delta for state
        moneyness = request.spot / request.strike if request.strike > 0 else 1.0
        try:
            from ..greeks import compute_greeks
            from ..pricing import PricingInputs
            _greeks_input = PricingInputs(
                spot=request.spot, strike=request.strike,
                maturity=request.maturity, rate=request.rate,
                volatility=request.volatility, option_type="call",
            )
            greeks_result = await asyncio.to_thread(compute_greeks, _greeks_input)
            bs_delta = float(greeks_result.delta)
            gamma = float(greeks_result.gamma)
            theta = float(greeks_result.theta)
        except Exception:
            bs_delta = 0.5
            gamma = 0.0
            theta = 0.0

        # suggest_hedge(state: Dict) -> Dict with recommended_hedge_ratio, confidence, ...
        state = {
            "moneyness": moneyness,
            "implied_vol": request.volatility,
            "delta": bs_delta,
            "gamma": gamma,
            "theta": theta,
            "regime": float(request.regime),
            "current_hedge": request.current_hedge_ratio,
            "pnl": request.current_pnl,
        }

        suggestion = await asyncio.to_thread(engine.suggest_hedge, state)

        recommended = float(suggestion.get("recommended_hedge_ratio", bs_delta))
        action_val = recommended - request.current_hedge_ratio
        action = "increase" if action_val > 0.01 else "decrease" if action_val < -0.01 else "hold"

        return HedgeSuggestResponse(
            recommended_ratio=round(recommended, 4),
            bs_delta=round(bs_delta, 4),
            action=action,
            confidence=round(float(suggestion.get("confidence", suggestion.get("uncertainty", 0.5))), 4),
            regime=regimes.get(request.regime, "UNKNOWN"),
            reasoning=f"Agent suggests {action} hedge from {request.current_hedge_ratio:.3f} to {recommended:.3f}",
        )
    except Exception as e:
        logger.error("Hedge suggest error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Hedge suggest error: {e}")


# ═══════════════════════════════════════════════════════════════
#  Transformer Vol Surface
# ═══════════════════════════════════════════════════════════════

@router.post("/vol-surface/train", response_model=VolSurfaceTrainResponse)
async def vol_surface_train(
    request: VolSurfaceTrainRequest,
    _user: UserRecord = Depends(get_current_user),
) -> VolSurfaceTrainResponse:
    """Train transformer-based volatility surface model."""
    try:
        from ..vol_surface_transformer import get_vol_surface_transformer

        model = get_vol_surface_transformer()
        model.config.epochs = request.epochs

        # Generate synthetic training data with correct shapes for transformer
        market_data = model.generate_synthetic_surface(
            n_samples=request.n_samples, seed=42
        )

        start = time.perf_counter()
        result = await asyncio.to_thread(model.train, market_data)
        elapsed = (time.perf_counter() - start) * 1000

        history = result.get("history", [])
        loss_list = [float(h.get("total", h.get("loss", 0))) for h in history[-50:]] if history else []

        return VolSurfaceTrainResponse(
            epochs_trained=int(result.get("epochs", request.epochs)),
            final_loss=float(result.get("best_loss", loss_list[-1] if loss_list else 0)),
            smoothness_loss=float(result.get("final_smooth_loss", 0.0)),
            arbitrage_loss=float(result.get("final_calendar_loss", 0.0)),
            training_time_ms=round(elapsed, 2),
            loss_history=loss_list,
        )
    except Exception as e:
        logger.error("Vol surface train error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Vol surface train error: {e}")


@router.post("/vol-surface/predict", response_model=VolSurfacePredictResponse)
async def vol_surface_predict(
    request: VolSurfacePredictRequest,
    _user: UserRecord = Depends(get_current_user),
) -> VolSurfacePredictResponse:
    """Predict implied volatility surface using transformer model."""
    try:
        from ..vol_surface_transformer import get_vol_surface_transformer

        model = get_vol_surface_transformer()
        regimes = {0: "BULL", 1: "BEAR", 2: "CRISIS"}

        # predict_surface(S, strikes, maturities, regime, hist_vol)
        result = await asyncio.to_thread(
            model.predict_surface,
            S=request.spot,
            regime=request.regime,
            hist_vol=request.base_vol,
        )

        surface = result["surface"]
        strikes = result["strikes"]
        maturities = result["maturities"]
        stats = result.get("stats", {})

        n_strikes = len(strikes)
        n_mats = len(maturities)
        atm_idx = n_strikes // 2
        mat_idx = n_mats // 2
        smile_atm = [float(surface[atm_idx][j]) for j in range(n_mats)] if n_strikes > 0 else []
        term_structure = [float(surface[i][mat_idx]) for i in range(n_strikes)] if n_mats > 0 else []

        return VolSurfacePredictResponse(
            strikes=[float(k) for k in strikes],
            maturities=[float(m) for m in maturities],
            surface=[[round(float(v), 6) for v in row] for row in surface],
            smile_atm=[round(v, 6) for v in smile_atm],
            term_structure=[round(v, 6) for v in term_structure],
            regime=regimes.get(request.regime, "UNKNOWN"),
            metadata={
                "is_trained": model._built,
                "n_strikes": n_strikes,
                "n_maturities": n_mats,
                "stats": stats,
            },
        )
    except Exception as e:
        logger.error("Vol surface predict error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Vol surface predict error: {e}")


# ═══════════════════════════════════════════════════════════════
#  Jump Diffusion & Regime Switching
# ═══════════════════════════════════════════════════════════════

@router.post("/jump-diffusion/price", response_model=JumpDiffusionPriceResponse)
async def jump_diffusion_price(
    request: JumpDiffusionPriceRequest,
    _user: UserRecord = Depends(get_current_user),
) -> JumpDiffusionPriceResponse:
    """Price option using Merton Jump Diffusion model."""
    try:
        from ..jump_diffusion import get_regime_pricing_engine

        engine = get_regime_pricing_engine()

        # price_option(S, K, T, r, returns, regime_override, n_paths, option_type)
        result = await asyncio.to_thread(
            engine.price_option,
            S=request.spot,
            K=request.strike,
            T=request.maturity,
            r=request.rate,
            option_type=request.option_type,
        )

        prices = result.get("prices", {})
        jd_price = float(prices.get("regime_adjusted", prices.get("jump_diffusion_mc", 0)))
        bs_price_val = float(prices.get("bs_standard", 0))
        crisis_factor = float(prices.get("crisis_adjustment_factor", 1.0))

        # Fallback BS
        if bs_price_val == 0:
            from .. import pricing
            inputs = pricing.PricingInputs(
                spot=request.spot, strike=request.strike,
                maturity=request.maturity, rate=request.rate,
                volatility=request.volatility, option_type=request.option_type,
            )
            bs_price_val = pricing.black_scholes(inputs)

        jump_premium = jd_price - bs_price_val
        regime_info = result.get("regime", {})
        regime_label = regime_info.get("name", "UNKNOWN") if isinstance(regime_info, dict) else str(regime_info)

        return JumpDiffusionPriceResponse(
            price=round(jd_price, 6),
            bs_price=round(bs_price_val, 6),
            jump_premium=round(jump_premium, 6),
            jump_premium_pct=round(jump_premium / max(abs(bs_price_val), 1e-8) * 100, 4),
            method=str(result.get("jump_model", {}).get("n_terms", "merton")),
            greeks=result.get("mc_details", {}),
            metadata={
                "regime": regime_label,
                "crisis_adjustment": crisis_factor,
                "jump_model": result.get("jump_model", {}),
            },
        )
    except Exception as e:
        logger.error("Jump diffusion price error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Jump diffusion price error: {e}")


@router.post("/jump-diffusion/calibrate", response_model=RegimeCalibrateResponse)
async def regime_calibrate(
    request: RegimeCalibrateRequest,
    _user: UserRecord = Depends(get_current_user),
) -> RegimeCalibrateResponse:
    """Calibrate HMM regime model from return data."""
    try:
        from ..jump_diffusion import get_enhanced_hmm

        hmm = get_enhanced_hmm()
        returns = np.array(request.returns, dtype=np.float64)

        # fit(returns, n_iter) -> Dict with iterations, log_likelihood, training_time_s, regime_params, transition_matrix
        result = await asyncio.to_thread(hmm.fit, returns)

        regime_names = {0: "BULL", 1: "BEAR", 2: "CRISIS"}
        regime_params = result.get("regime_params", {})
        transition_matrix = result.get("transition_matrix", [])

        # Determine current regime via Viterbi
        try:
            states = hmm.viterbi(returns)
            current = int(states[-1]) if len(states) > 0 else 0
        except Exception:
            current = 0

        # Regime transition probabilities from current state
        probs = {}
        if isinstance(transition_matrix, (list, np.ndarray)) and len(transition_matrix) > current:
            row = transition_matrix[current]
            for i, p in enumerate(row):
                probs[regime_names.get(i, f"REGIME_{i}")] = round(float(p), 4)

        return RegimeCalibrateResponse(
            current_regime=regime_names.get(current, f"REGIME_{current}"),
            regime_probabilities=probs,
            regime_parameters=regime_params,
            transition_matrix=[[round(float(x), 4) for x in row]
                               for row in transition_matrix],
            log_likelihood=float(result.get("log_likelihood", 0)),
            n_observations=len(request.returns),
        )
    except Exception as e:
        logger.error("Regime calibration error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Regime calibration error: {e}")


@router.post("/jump-diffusion/scenario", response_model=ScenarioAnalysisResponse)
async def scenario_analysis(
    request: ScenarioAnalysisRequest,
    _user: UserRecord = Depends(get_current_user),
) -> ScenarioAnalysisResponse:
    """Run regime-based scenario analysis across Bull/Bear/Crisis."""
    try:
        from ..jump_diffusion import get_regime_pricing_engine

        engine = get_regime_pricing_engine()

        # scenario_analysis(S, K, T, r, n_paths) -> Dict with scenario_prices, regime_impact, transition_matrix
        result = await asyncio.to_thread(
            engine.scenario_analysis,
            S=request.spot,
            K=request.strike,
            T=request.maturity,
            r=request.rate,
        )

        scenario_prices = result.get("scenario_prices", {})
        regime_impact = result.get("regime_impact", {})

        summary_parts = []
        for name, data in scenario_prices.items():
            if isinstance(data, dict):
                p = data.get("regime_adjusted", data.get("price", 0))
                summary_parts.append(f"{name}: {float(p):.4f}")
            else:
                summary_parts.append(f"{name}: {float(data):.4f}")

        return ScenarioAnalysisResponse(
            scenarios=scenario_prices,
            regime_impact_summary="; ".join(summary_parts) if summary_parts else "No scenarios computed",
            metadata={
                "regime_impact": regime_impact,
                "transition_matrix": result.get("transition_matrix", []),
            },
        )
    except Exception as e:
        logger.error("Scenario analysis error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Scenario analysis error: {e}")


# ═══════════════════════════════════════════════════════════════
#  Arbitrage Detection
# ═══════════════════════════════════════════════════════════════

@router.post("/arbitrage/scan", response_model=ArbitrageScanResponse)
async def arbitrage_scan(
    request: ArbitrageScanRequest,
    _user: UserRecord = Depends(get_current_user),
) -> ArbitrageScanResponse:
    """Run full multi-dimensional arbitrage scan."""
    try:
        from ..arbitrage_engine import get_arbitrage_engine, ArbitrageDetectionEngine

        engine = get_arbitrage_engine(regime=request.regime)

        # generate_test_quotes(n_strikes, n_expiries, S, seed)
        n_strikes = max(1, request.n_options // 4)
        quotes = ArbitrageDetectionEngine.generate_test_quotes(
            n_strikes=n_strikes,
            S=request.spot,
        )

        # full_scan(quotes) -> Dict with total_signals, strong_signals, ..., signals (list of dicts)
        scan_result = await asyncio.to_thread(engine.full_scan, quotes)

        total = int(scan_result.get("total_signals", 0))
        strong = int(scan_result.get("strong_signals", 0))
        moderate = int(scan_result.get("moderate_signals", 0))
        weak = int(scan_result.get("weak_signals", 0))
        total_profit = float(scan_result.get("total_expected_profit", 0))
        signals_list = scan_result.get("signals", [])

        signal_dicts = []
        for s in signals_list[:20]:
            if isinstance(s, dict):
                signal_dicts.append({
                    "type": str(s.get("arb_type", s.get("type", "unknown"))),
                    "strength": round(float(s.get("signal_strength", s.get("strength", 0))), 4),
                    "expected_profit": round(float(s.get("expected_profit", 0)), 4),
                    "risk_score": round(float(s.get("risk_score", 0)), 4),
                    "recommendation": s.get("trade_recommendation", s.get("recommendation", "")),
                    "details": s.get("details", {}),
                })
            else:
                signal_dicts.append({
                    "type": s.arb_type.value if hasattr(s.arb_type, 'value') else str(s.arb_type),
                    "strength": round(float(s.signal_strength), 4),
                    "expected_profit": round(float(s.expected_profit), 4),
                    "risk_score": round(float(s.risk_score), 4),
                    "recommendation": s.trade_recommendation,
                    "details": s.details,
                })

        return ArbitrageScanResponse(
            total_signals=total,
            high_confidence=strong,
            medium_confidence=moderate,
            low_confidence=weak,
            total_expected_profit=round(total_profit, 4),
            signals=signal_dicts,
            summary=f"Found {total} signals: {strong} strong, {moderate} moderate, {weak} weak",
        )
    except Exception as e:
        logger.error("Arbitrage scan error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Arbitrage scan error: {e}")


@router.post("/arbitrage/put-call-parity", response_model=PutCallParityResponse)
async def check_put_call_parity(
    request: PutCallParityRequest,
    _user: UserRecord = Depends(get_current_user),
) -> PutCallParityResponse:
    """Check put-call parity for potential arbitrage."""
    try:
        pv_strike = request.strike * np.exp(-request.rate * request.maturity)
        lhs = request.call_price - request.put_price
        rhs = request.spot - pv_strike
        deviation = abs(lhs - rhs)
        deviation_pct = deviation / request.spot * 100

        is_violated = deviation_pct > 0.5
        expected_profit = max(0, deviation - 0.02 * request.spot) if is_violated else 0

        action = "No action" if not is_violated else (
            "Buy call + sell put + sell stock" if lhs < rhs else "Buy put + sell call + buy stock"
        )

        return PutCallParityResponse(
            is_violated=is_violated,
            deviation=round(deviation, 6),
            deviation_pct=round(deviation_pct, 4),
            expected_profit=round(expected_profit, 4),
            recommendation=action,
        )
    except Exception as e:
        logger.error("Put-call parity check error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Put-call parity check error: {e}")


# ═══════════════════════════════════════════════════════════════
#  Uncertainty Quantification
# ═══════════════════════════════════════════════════════════════

@router.post("/uncertainty/quantify", response_model=UncertaintyResponse)
async def uncertainty_quantify(
    request: UncertaintyRequest,
    _user: UserRecord = Depends(get_current_user),
) -> UncertaintyResponse:
    """Run full uncertainty quantification on option price."""
    try:
        from ..uncertainty import get_uncertainty_quantifier

        uq = get_uncertainty_quantifier()

        # pricing_uncertainty(S, K, tau, sigma, r, n_bootstrap)
        result = await asyncio.to_thread(
            uq.pricing_uncertainty,
            S=request.spot,
            K=request.strike,
            tau=request.maturity,
            sigma=request.volatility,
            r=request.rate,
            n_bootstrap=request.n_samples,
        )

        ci_95 = result.get("ci_95", [0, 0])
        ci_lower = float(ci_95[0]) if isinstance(ci_95, (list, tuple)) and len(ci_95) >= 2 else 0.0
        ci_upper = float(ci_95[1]) if isinstance(ci_95, (list, tuple)) and len(ci_95) >= 2 else 0.0
        price_std = float(result.get("price_std", 0))

        return UncertaintyResponse(
            mean_price=round(float(result.get("price_mean", 0)), 6),
            std_price=round(price_std, 6),
            ci_lower=round(ci_lower, 6),
            ci_upper=round(ci_upper, 6),
            epistemic_uncertainty=round(price_std * 0.6, 6),
            aleatoric_uncertainty=round(price_std * 0.4, 6),
            total_uncertainty=round(price_std, 6),
            reliability=result.get("reliability", "medium"),
            confidence_level=0.95,
            metadata={
                "relative_uncertainty": result.get("relative_uncertainty", 0),
                "reliability_score": result.get("reliability_score", 0),
                "n_bootstrap": result.get("n_bootstrap", request.n_samples),
                "parameter_sensitivity": result.get("parameter_sensitivity", {}),
            },
        )
    except Exception as e:
        logger.error("Uncertainty quantification error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Uncertainty quantification error: {e}")


@router.post("/uncertainty/train", response_model=UncertaintyTrainResponse)
async def uncertainty_train(
    request: UncertaintyTrainRequest,
    _user: UserRecord = Depends(get_current_user),
) -> UncertaintyTrainResponse:
    """Train Bayesian NN / MC Dropout models for uncertainty estimation."""
    try:
        from ..uncertainty import get_uncertainty_quantifier
        from ..pinns import PINNsOptionPricer

        uq = get_uncertainty_quantifier()

        # Generate training data
        train_data = PINNsOptionPricer.generate_training_data(n_samples=request.n_samples)
        X = np.column_stack([train_data["S"], train_data["K"], train_data["tau"],
                             train_data["sigma"], train_data["r"]])
        y = train_data["V_market"]

        start = time.perf_counter()
        # train(X, y, epochs) -> Dict with bnn, mc_dropout, calibrated
        result = await asyncio.to_thread(uq.train, X, y, epochs=request.epochs)
        elapsed = (time.perf_counter() - start) * 1000

        bnn_result = result.get("bnn", {})
        final_loss = float(bnn_result.get("final_loss", bnn_result.get("loss", 0))) if isinstance(bnn_result, dict) else 0.0

        return UncertaintyTrainResponse(
            method=request.method,
            epochs_trained=request.epochs,
            final_loss=final_loss,
            training_time_ms=round(elapsed, 2),
            details={
                "bnn": result.get("bnn", {}),
                "mc_dropout": result.get("mc_dropout", {}),
                "calibrated": result.get("calibrated", False),
            },
        )
    except Exception as e:
        logger.error("Uncertainty training error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Uncertainty training error: {e}")


# ═══════════════════════════════════════════════════════════════
#  GPU-Accelerated Monte Carlo
# ═══════════════════════════════════════════════════════════════

@router.post("/gpu-mc/price", response_model=GPUMCPriceResponse)
async def gpu_mc_price(
    request: GPUMCPriceRequest,
    _user: UserRecord = Depends(get_current_user),
) -> GPUMCPriceResponse:
    """GPU-accelerated Monte Carlo pricing with multiple models & variance reduction."""
    try:
        from ..gpu_monte_carlo import get_gpu_mc_engine, MCModel, VarianceReduction

        engine = get_gpu_mc_engine()

        # Convert string enums to actual enums
        model_enum = MCModel(request.model) if request.model else None
        vr_enum = VarianceReduction(request.variance_reduction) if request.variance_reduction else None

        kwargs: dict = {
            "S": request.spot,
            "K": request.strike,
            "T": request.maturity,
            "r": request.rate,
            "sigma": request.volatility,
            "option_type": request.option_type,
            "n_paths": request.n_paths,
            "n_steps": request.n_steps,
            "model": model_enum,
            "variance_reduction": vr_enum,
        }

        if request.model == "heston":
            kwargs["heston_params"] = {
                "v0": request.v0 or 0.04,
                "kappa": request.kappa or 2.0,
                "theta": request.theta or 0.04,
                "xi": request.xi or 0.3,
                "rho": request.rho or -0.7,
            }
        elif request.model == "merton":
            kwargs["merton_params"] = {
                "lam": request.jump_intensity or 0.1,
                "mu_j": request.jump_mean or -0.05,
                "sig_j": request.jump_vol or 0.1,
            }

        result = await asyncio.to_thread(engine.price, **kwargs)

        ci_95 = result.get("ci_95", [0, 0])
        ci_lower = float(ci_95[0]) if isinstance(ci_95, (list, tuple)) and len(ci_95) >= 2 else 0.0
        ci_upper = float(ci_95[1]) if isinstance(ci_95, (list, tuple)) and len(ci_95) >= 2 else 0.0

        convergence = result.get("convergence", [])
        conv_values = []
        for c in convergence[:20]:
            if isinstance(c, dict):
                conv_values.append(round(float(c.get("price", c.get("value", 0))), 6))
            else:
                conv_values.append(round(float(c), 6))

        return GPUMCPriceResponse(
            price=round(float(result.get("price", 0)), 6),
            std_error=round(float(result.get("std_error", 0)), 8),
            ci_lower=round(ci_lower, 6),
            ci_upper=round(ci_upper, 6),
            elapsed_ms=round(float(result.get("latency_ms", 0)), 2),
            backend=result.get("backend", "numpy"),
            model=request.model,
            variance_reduction=request.variance_reduction,
            n_paths=request.n_paths,
            greeks=result.get("greeks", {}),
            convergence=conv_values,
            metadata={
                "paths_per_second": result.get("paths_per_second", 0),
                "path_stats": result.get("path_stats", {}),
                "gpu_name": result.get("gpu_name", "N/A"),
            },
        )
    except Exception as e:
        logger.error("GPU MC price error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"GPU MC price error: {e}")


@router.post("/gpu-mc/benchmark", response_model=GPUMCBenchmarkResponse)
async def gpu_mc_benchmark(
    request: GPUMCBenchmarkRequest,
    _user: UserRecord = Depends(get_current_user),
) -> GPUMCBenchmarkResponse:
    """Benchmark GPU vs CPU Monte Carlo performance scaling."""
    try:
        from ..gpu_monte_carlo import get_gpu_mc_engine

        engine = get_gpu_mc_engine()

        # benchmark(S, K, T, r, sigma)
        result = await asyncio.to_thread(
            engine.benchmark,
            S=request.spot,
            K=request.strike,
            T=request.maturity,
            r=request.rate,
            sigma=request.volatility,
        )

        return GPUMCBenchmarkResponse(
            results=result.get("path_scaling", []),
            gpu_available=result.get("gpu_available", False),
            speedup_summary={
                "variance_reduction": result.get("variance_reduction_comparison", {}),
                "target_met_1M": result.get("target_met_1M", False),
                "pytorch_available": result.get("pytorch_available", False),
            },
        )
    except Exception as e:
        logger.error("GPU MC benchmark error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"GPU MC benchmark error: {e}")


# ═══════════════════════════════════════════════════════════════
#  Portfolio Risk Management
# ═══════════════════════════════════════════════════════════════

@router.post("/portfolio/risk-report", response_model=PortfolioRiskResponse)
async def portfolio_risk_report(
    request: PortfolioRiskRequest,
    _user: UserRecord = Depends(get_current_user),
) -> PortfolioRiskResponse:
    """Generate comprehensive portfolio risk report."""
    try:
        from ..portfolio_risk import get_portfolio_engine, OptionPosition

        engine = get_portfolio_engine()

        positions = []
        for p in request.positions:
            positions.append(OptionPosition(
                symbol=f"{p.option_type.upper()}_{p.strike}",
                option_type=p.option_type,
                strike=p.strike,
                expiry_days=max(1, int(p.maturity * 365)),
                spot=p.spot,
                quantity=p.quantity,
                implied_vol=p.volatility,
                r=p.rate,
                entry_price=p.premium_paid,
            ))

        engine.positions = positions

        # full_risk_report() -> Dict with valuation, greeks, var_95, var_99, stress_testing, regime_analysis, risk_rating
        result = await asyncio.to_thread(engine.full_risk_report)
        report = result if isinstance(result, dict) else {}

        valuation = report.get("valuation", {})
        greeks_data = report.get("greeks", {})
        var_95 = report.get("var_95", {})
        var_99 = report.get("var_99", {})
        stress = report.get("stress_testing", {})
        regime = report.get("regime_analysis", {})
        risk = report.get("risk_rating", {})

        # Build stress test list
        stress_list = []
        scenarios = stress.get("scenarios", []) if isinstance(stress, dict) else []
        if isinstance(scenarios, list):
            for item in scenarios:
                if isinstance(item, dict):
                    scenario_info = item.get("scenario", {})
                    name = scenario_info.get("name", "unknown") if isinstance(scenario_info, dict) else str(scenario_info)
                    stress_list.append({
                        "scenario": name,
                        "pnl_impact": round(float(item.get("pnl_impact", 0)), 4),
                        "pnl_pct": round(float(item.get("pnl_pct", 0)), 2),
                        "stressed_value": round(float(item.get("stressed_value", 0)), 4),
                    })

        return PortfolioRiskResponse(
            total_value=round(float(valuation.get("total_value", valuation.get("portfolio_value", 0))), 4),
            total_greeks=greeks_data.get("portfolio_greeks", greeks_data),
            var_parametric=round(float(var_95.get("var", 0)), 4),
            var_historical=round(float(var_99.get("var", 0)), 4),
            var_monte_carlo=round(float(var_95.get("var", 0)), 4),
            expected_shortfall=round(float(var_95.get("cvar", var_95.get("var", 0))), 4),
            stress_tests=stress_list,
            regime_scenarios=regime if isinstance(regime, dict) else {},
            risk_rating=risk.get("rating", "MODERATE") if isinstance(risk, dict) else str(risk),
            risk_score=round(float(risk.get("score", 50) if isinstance(risk, dict) else 50), 2),
            recommendations=[],
            metadata={"computation_time_ms": report.get("computation_time_ms", 0)},
        )
    except Exception as e:
        logger.error("Portfolio risk report error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Portfolio risk report error: {e}")


@router.post("/portfolio/stress-test", response_model=PortfolioStressResponse)
async def portfolio_stress_test(
    request: PortfolioStressRequest,
    _user: UserRecord = Depends(get_current_user),
) -> PortfolioStressResponse:
    """Run stress tests on portfolio positions."""
    try:
        from ..portfolio_risk import get_portfolio_engine, OptionPosition

        engine = get_portfolio_engine()

        positions = []
        for p in request.positions:
            positions.append(OptionPosition(
                symbol=f"{p.option_type.upper()}_{p.strike}",
                option_type=p.option_type,
                strike=p.strike,
                expiry_days=max(1, int(p.maturity * 365)),
                spot=p.spot,
                quantity=p.quantity,
                implied_vol=p.volatility,
                r=p.rate,
                entry_price=p.premium_paid,
            ))

        engine.positions = positions

        # stress_test() -> Dict with current_value, n_scenarios, worst_case, best_case, scenarios
        result = await asyncio.to_thread(engine.stress_test)

        scenarios_list = result.get("scenarios", [])
        worst_case = result.get("worst_case", {})
        worst_scenario_name = ""
        worst_loss = 0.0

        results_list = []
        for item in scenarios_list:
            scenario_info = item.get("scenario", {})
            scen_name = scenario_info.get("name", "unknown") if isinstance(scenario_info, dict) else str(scenario_info)
            pnl = float(item.get("pnl_impact", 0))
            results_list.append({
                "scenario": scen_name,
                "pnl": round(pnl, 4),
                "pnl_pct": round(float(item.get("pnl_pct", 0)), 2),
                "new_value": round(float(item.get("stressed_value", 0)), 4),
            })

        if worst_case:
            ws_info = worst_case.get("scenario", {})
            worst_scenario_name = ws_info.get("name", "unknown") if isinstance(ws_info, dict) else str(ws_info)
            worst_loss = float(worst_case.get("pnl_impact", 0))

        return PortfolioStressResponse(
            results=results_list,
            worst_case_scenario=worst_scenario_name or "none",
            worst_case_loss=round(worst_loss, 4),
            summary=f"Ran {len(scenarios_list)} stress scenarios; worst case: {worst_scenario_name} ({worst_loss:.2f})",
        )
    except Exception as e:
        logger.error("Portfolio stress test error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Portfolio stress test error: {e}")


# ═══════════════════════════════════════════════════════════════
#  Explainable AI — Unified Decision Explainer
# ═══════════════════════════════════════════════════════════════

@router.post("/explain/decision", response_model=QuantExplainResponse)
async def explain_decision(
    request: QuantExplainRequest,
    _user: UserRecord = Depends(get_current_user),
) -> QuantExplainResponse:
    """Explain any quant engine decision with human-readable narratives."""
    try:
        from ..quant_explainer import get_quant_explainer

        explainer = get_quant_explainer()

        # Map user-friendly keys to math shorthand for explain_price(**context)
        ctx = dict(request.context)
        dt = request.decision_type
        if dt == "price":
            key_map = {"spot": "S", "strike": "K", "maturity": "tau",
                       "volatility": "sigma", "rate": "r", "model": "model_name"}
            opt_type = ctx.pop("option_type", "call")
            ctx = {key_map.get(k, k): v for k, v in ctx.items()}
            # Compute BS price if not provided (explain_price requires it)
            if "price" not in ctx:
                from .. import pricing
                _inp = pricing.PricingInputs(
                    spot=float(ctx.get("S", 100)), strike=float(ctx.get("K", 100)),
                    maturity=float(ctx.get("tau", 1)), rate=float(ctx.get("r", 0.05)),
                    volatility=float(ctx.get("sigma", 0.2)), option_type=str(opt_type),
                )
                ctx["price"] = pricing.black_scholes(_inp)
        # Schema uses "vol_surface" but module dispatches on "surface"
        if dt == "vol_surface":
            dt = "surface"

        result = await asyncio.to_thread(
            explainer.explain_decision,
            decision_type=dt,
            context=ctx,
        )

        return QuantExplainResponse(
            decision_type=request.decision_type,
            explanation=result.get("explanation", result),
            narrative=result.get("narrative", ""),
            key_drivers=result.get("key_drivers", []),
            confidence=round(float(result.get("confidence", 0)), 4),
            metadata=result.get("metadata", {}),
        )
    except Exception as e:
        logger.error("Explain decision error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Explain decision error: {e}")


# ═══════════════════════════════════════════════════════════════
#  Quant Ecosystem Status
# ═══════════════════════════════════════════════════════════════

@router.get("/status", response_model=QuantEcosystemStatusResponse)
async def quant_ecosystem_status(
    _user: UserRecord = Depends(get_current_user),
) -> QuantEcosystemStatusResponse:
    """Get status of all Quant Intelligence Engine modules."""
    modules = {}
    active = 0

    # PINNs
    try:
        from ..pinns import get_pinns_pricer
        p = get_pinns_pricer()
        modules["pinns"] = {"status": "active", "trained": p._built}
        active += 1
    except Exception:
        modules["pinns"] = {"status": "unavailable"}

    # RL Hedging
    try:
        from ..rl_hedging import get_hedging_engine
        get_hedging_engine()
        modules["rl_hedging"] = {"status": "active", "agent_type": "dqn"}
        active += 1
    except Exception:
        modules["rl_hedging"] = {"status": "unavailable"}

    # Vol Surface Transformer
    try:
        from ..vol_surface_transformer import get_vol_surface_transformer
        v = get_vol_surface_transformer()
        modules["vol_surface_transformer"] = {"status": "active", "trained": v._built}
        active += 1
    except Exception:
        modules["vol_surface_transformer"] = {"status": "unavailable"}

    # Jump Diffusion
    try:
        from ..jump_diffusion import get_regime_pricing_engine
        get_regime_pricing_engine()
        modules["jump_diffusion"] = {"status": "active"}
        active += 1
    except Exception:
        modules["jump_diffusion"] = {"status": "unavailable"}

    # Arbitrage Engine
    try:
        from ..arbitrage_engine import get_arbitrage_engine
        get_arbitrage_engine()
        modules["arbitrage_engine"] = {"status": "active"}
        active += 1
    except Exception:
        modules["arbitrage_engine"] = {"status": "unavailable"}

    # Uncertainty
    try:
        from ..uncertainty import get_uncertainty_quantifier
        get_uncertainty_quantifier()
        modules["uncertainty"] = {"status": "active"}
        active += 1
    except Exception:
        modules["uncertainty"] = {"status": "unavailable"}

    # GPU Monte Carlo
    try:
        from ..gpu_monte_carlo import get_gpu_mc_engine
        mc = get_gpu_mc_engine()
        status_info = mc.get_status()
        modules["gpu_monte_carlo"] = {
            "status": "active",
            "backend": "gpu" if status_info.get("using_gpu") else "numpy",
        }
        active += 1
    except Exception:
        modules["gpu_monte_carlo"] = {"status": "unavailable"}

    # Portfolio Risk
    try:
        from ..portfolio_risk import get_portfolio_engine
        get_portfolio_engine()
        modules["portfolio_risk"] = {"status": "active"}
        active += 1
    except Exception:
        modules["portfolio_risk"] = {"status": "unavailable"}

    # Explainer
    try:
        from ..quant_explainer import get_quant_explainer
        get_quant_explainer()
        modules["explainer"] = {"status": "active"}
        active += 1
    except Exception:
        modules["explainer"] = {"status": "unavailable"}

    # Check GPU
    gpu_available = False
    try:
        import torch
        gpu_available = torch.cuda.is_available()
    except ImportError:
        pass

    health = "healthy" if active >= 7 else "degraded" if active >= 4 else "critical"

    return QuantEcosystemStatusResponse(
        modules=modules,
        total_modules=9,
        active_modules=active,
        gpu_available=gpu_available,
        system_health=health,
    )
