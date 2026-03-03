"""
OptionQuant — Market Intelligence API Routes
═══════════════════════════════════════════════
Endpoints:
  GET  /market/quote/{symbol}      — Real-time quote
  GET  /market/chain/{symbol}      — Full option chain
  GET  /market/snapshot/{symbol}   — Market snapshot
  GET  /market/health              — Pipeline health
  POST /market/mispricing/scan     — Full-chain mispricing scan
  POST /market/mispricing/detect   — Single contract mispricing
  POST /market/regime/detect       — Regime detection
  GET  /market/regime/adjustment   — Regime-based adjustments
  POST /market/risk/confidence     — Confidence estimation
  POST /market/risk/var            — VaR/CVaR computation
  POST /market/risk/reliability    — Model reliability
  POST /market/explain/shap        — SHAP-like explanation
  POST /market/benchmark           — Performance benchmark
  GET  /market/alerts              — Recent alerts
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import asdict

from fastapi import APIRouter, Depends, HTTPException, Query

from ..auth import UserRecord, get_current_user
from ..schemas import (
    MispricingDetectRequest,
    MispricingScanRequest,
    RegimeDetectRequest,
    ConfidenceRequest,
    VaRRequest,
    ReliabilityRequest,
    SHAPExplainRequest,
    BenchmarkRequest,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/market", tags=["market-intelligence"])


# ═══════════════════════════════════════════════════════════════
#  Market Data
# ═══════════════════════════════════════════════════════════════

@router.get("/quote/{symbol}")
async def get_quote(
    symbol: str = "SPY",
    _user: UserRecord = Depends(get_current_user),
):
    """Fetch latest market quote (simulated)."""
    from ..market_data import fetch_quote
    try:
        q = fetch_quote(symbol)
        return asdict(q)
    except Exception as e:
        logger.error("Quote fetch error: %s", e, exc_info=True)
        raise HTTPException(500, detail=str(e))


@router.get("/chain/{symbol}")
async def get_chain(
    symbol: str = "SPY",
    _user: UserRecord = Depends(get_current_user),
):
    """Fetch full option chain."""
    from ..market_data import fetch_option_chain
    try:
        chain = fetch_option_chain(symbol)
        return asdict(chain)
    except Exception as e:
        logger.error("Chain fetch error: %s", e, exc_info=True)
        raise HTTPException(500, detail=str(e))


@router.get("/snapshot/{symbol}")
async def get_snapshot(
    symbol: str = "SPY",
    _user: UserRecord = Depends(get_current_user),
):
    """Full market snapshot: quote + chain + VIX."""
    from ..market_data import fetch_snapshot
    try:
        snap = fetch_snapshot(symbol)
        return asdict(snap)
    except Exception as e:
        logger.error("Snapshot error: %s", e, exc_info=True)
        raise HTTPException(500, detail=str(e))


@router.get("/health")
async def market_health():
    """Market data pipeline health check."""
    from ..market_data import get_pipeline_health
    return get_pipeline_health()


# ═══════════════════════════════════════════════════════════════
#  Mispricing Detection
# ═══════════════════════════════════════════════════════════════

@router.post("/mispricing/detect")
async def detect_mispricing_endpoint(
    request: MispricingDetectRequest,
    _user: UserRecord = Depends(get_current_user),
):
    """Detect mispricing for a single option contract."""
    from ..mispricing import detect_mispricing
    try:
        signal = await asyncio.to_thread(
            detect_mispricing,
            spot=request.spot,
            strike=request.strike,
            maturity=request.maturity,
            rate=request.rate,
            volatility=request.volatility,
            option_type=request.option_type,
            market_price=request.market_price,
            bid=request.bid,
            ask=request.ask,
            model=request.pricing_model,
            significance_threshold=request.significance_threshold,
            min_deviation_pct=request.min_deviation_pct,
        )
        return asdict(signal)
    except Exception as e:
        logger.error("Mispricing detect error: %s", e, exc_info=True)
        raise HTTPException(500, detail=str(e))


@router.post("/mispricing/scan")
async def scan_chain_endpoint(
    request: MispricingScanRequest,
    _user: UserRecord = Depends(get_current_user),
):
    """Scan full option chain for mispricing and arbitrage."""
    from ..mispricing import scan_chain
    from ..market_data import fetch_snapshot
    try:
        snap = fetch_snapshot(request.symbol)
        chain_data = {
            "symbol": request.symbol,
            "spot": snap.quote.price,
            "vix": snap.vix,
            "risk_free_rate": snap.chain.risk_free_rate,
            "calls": [
                {
                    "strike": c.strike,
                    "expiry": c.expiry,
                    "option_type": c.option_type,
                    "bid": c.bid,
                    "ask": c.ask,
                    "mid": c.mid,
                    "implied_vol": c.implied_vol,
                    "volume": c.volume,
                    "open_interest": c.open_interest,
                }
                for c in snap.chain.calls
            ],
            "puts": [
                {
                    "strike": p.strike,
                    "expiry": p.expiry,
                    "option_type": p.option_type,
                    "bid": p.bid,
                    "ask": p.ask,
                    "mid": p.mid,
                    "implied_vol": p.implied_vol,
                    "volume": p.volume,
                    "open_interest": p.open_interest,
                }
                for p in snap.chain.puts
            ],
        }
        result = await asyncio.to_thread(
            scan_chain, chain_data,
            model=request.pricing_model,
            significance_threshold=request.significance_threshold,
            min_deviation_pct=request.min_deviation_pct,
        )
        return asdict(result)
    except Exception as e:
        logger.error("Chain scan error: %s", e, exc_info=True)
        raise HTTPException(500, detail=str(e))


# ═══════════════════════════════════════════════════════════════
#  Regime Detection
# ═══════════════════════════════════════════════════════════════

@router.post("/regime/detect")
async def detect_regime_endpoint(
    request: RegimeDetectRequest,
    _user: UserRecord = Depends(get_current_user),
):
    """Detect market regime from return series."""
    from ..regime import detect_regime
    try:
        state = await asyncio.to_thread(
            detect_regime,
            returns=request.returns,
            vix=request.vix,
        )
        return asdict(state)
    except Exception as e:
        logger.error("Regime detection error: %s", e, exc_info=True)
        raise HTTPException(500, detail=str(e))


@router.get("/regime/adjustment/{regime}")
async def get_adjustment(
    regime: str = "bull",
    _user: UserRecord = Depends(get_current_user),
):
    """Get model adjustments for a given regime."""
    from ..regime import get_regime_adjustment
    return get_regime_adjustment(regime)


# ═══════════════════════════════════════════════════════════════
#  Risk & Confidence
# ═══════════════════════════════════════════════════════════════

@router.post("/risk/confidence")
async def confidence_endpoint(
    request: ConfidenceRequest,
    _user: UserRecord = Depends(get_current_user),
):
    """Bootstrap confidence interval estimation."""
    from ..risk_engine import estimate_confidence
    try:
        result = await asyncio.to_thread(
            estimate_confidence,
            spot=request.spot,
            strike=request.strike,
            maturity=request.maturity,
            rate=request.rate,
            volatility=request.volatility,
            option_type=request.option_type,
            confidence_level=request.confidence_level,
            n_bootstrap=request.n_bootstrap,
        )
        return asdict(result)
    except Exception as e:
        logger.error("Confidence estimation error: %s", e, exc_info=True)
        raise HTTPException(500, detail=str(e))


@router.post("/risk/var")
async def var_endpoint(
    request: VaRRequest,
    _user: UserRecord = Depends(get_current_user),
):
    """Compute Value-at-Risk and CVaR."""
    from ..risk_engine import compute_var
    try:
        result = await asyncio.to_thread(
            compute_var,
            spot=request.spot,
            strike=request.strike,
            maturity=request.maturity,
            rate=request.rate,
            volatility=request.volatility,
            option_type=request.option_type,
            position_size=request.position_size,
            horizon_days=request.horizon_days,
        )
        return asdict(result)
    except Exception as e:
        logger.error("VaR error: %s", e, exc_info=True)
        raise HTTPException(500, detail=str(e))


@router.post("/risk/reliability")
async def reliability_endpoint(
    request: ReliabilityRequest,
    _user: UserRecord = Depends(get_current_user),
):
    """Assess model reliability for given inputs."""
    from ..risk_engine import assess_reliability
    try:
        result = await asyncio.to_thread(
            assess_reliability,
            spot=request.spot,
            strike=request.strike,
            maturity=request.maturity,
            rate=request.rate,
            volatility=request.volatility,
            option_type=request.option_type,
        )
        return asdict(result)
    except Exception as e:
        logger.error("Reliability error: %s", e, exc_info=True)
        raise HTTPException(500, detail=str(e))


# ═══════════════════════════════════════════════════════════════
#  SHAP Explainability
# ═══════════════════════════════════════════════════════════════

@router.post("/explain/shap")
async def shap_explain_endpoint(
    request: SHAPExplainRequest,
    _user: UserRecord = Depends(get_current_user),
):
    """SHAP-like explanation for option pricing."""
    from ..shap_explain import explain_pricing
    try:
        result = await asyncio.to_thread(
            explain_pricing,
            spot=request.spot,
            strike=request.strike,
            maturity=request.maturity,
            rate=request.rate,
            volatility=request.volatility,
            option_type=request.option_type,
            model=request.pricing_model,
        )
        return asdict(result)
    except Exception as e:
        logger.error("SHAP explain error: %s", e, exc_info=True)
        raise HTTPException(500, detail=str(e))


# ═══════════════════════════════════════════════════════════════
#  Performance Benchmark
# ═══════════════════════════════════════════════════════════════

@router.post("/benchmark")
async def benchmark_endpoint(
    request: BenchmarkRequest,
    _user: UserRecord = Depends(get_current_user),
):
    """Run comprehensive performance benchmark."""
    from ..performance import run_full_benchmark
    try:
        result = await asyncio.to_thread(
            run_full_benchmark,
            spot=request.spot,
            strike=request.strike,
            maturity=request.maturity,
            rate=request.rate,
            volatility=request.volatility,
            option_type=request.option_type,
        )
        return asdict(result)
    except Exception as e:
        logger.error("Benchmark error: %s", e, exc_info=True)
        raise HTTPException(500, detail=str(e))


# ═══════════════════════════════════════════════════════════════
#  Alerts
# ═══════════════════════════════════════════════════════════════

@router.get("/alerts")
async def get_alerts(
    limit: int = Query(50, ge=1, le=200),
    _user: UserRecord = Depends(get_current_user),
):
    """Get recent alerts from alert engine."""
    from ..websocket_manager import alert_engine
    history = alert_engine._alerts[-limit:]
    return {
        "alerts": [asdict(a) for a in reversed(history)],
        "total": len(alert_engine._alerts),
    }
