"""
vol_targets.py — Volatility Target Variable Construction
==========================================================
Constructs forward‑looking volatility targets for supervised learning.
All targets are properly forward‑shifted to prevent data leakage.

Supported targets:
  1. Realized Volatility (rolling std of future returns)
  2. Parkinson Volatility (range‑based, forward window)
  3. Garman‑Klass Volatility (OHLC‑based, forward window)
  4. EWMA Volatility (exponentially weighted)
"""
from __future__ import annotations

import numpy as np


def realized_vol_target(
    log_returns: np.ndarray,
    forward_window: int = 20,
    annualise: bool = True,
) -> np.ndarray:
    """
    Forward‑looking realised volatility.
    y[t] = std(r[t+1], ..., r[t+fw]) × sqrt(252)
    Last `forward_window` values are NaN (unknowable at prediction time).
    """
    n = len(log_returns)
    target = np.full(n, np.nan)
    scale = np.sqrt(252) if annualise else 1.0
    for i in range(n - forward_window):
        window = log_returns[i + 1: i + 1 + forward_window]
        target[i] = np.std(window, ddof=1) * scale
    return target


def parkinson_vol_target(
    high: np.ndarray,
    low: np.ndarray,
    forward_window: int = 20,
) -> np.ndarray:
    """
    Forward‑looking Parkinson estimator.
    Uses future high/low bars for the target.
    """
    n = len(high)
    target = np.full(n, np.nan)
    for i in range(n - forward_window):
        h = high[i + 1: i + 1 + forward_window]
        l = low[i + 1: i + 1 + forward_window]
        hl_sq = np.log(np.maximum(h, 1e-10) / np.maximum(l, 1e-10)) ** 2
        target[i] = np.sqrt(np.mean(hl_sq) / (4 * np.log(2)) * 252)
    return target


def garman_klass_vol_target(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    forward_window: int = 20,
) -> np.ndarray:
    """
    Forward‑looking Garman‑Klass estimator.
    """
    n = len(close)
    target = np.full(n, np.nan)
    for i in range(n - forward_window):
        s = slice(i + 1, i + 1 + forward_window)
        hl = np.log(np.maximum(high[s], 1e-10) / np.maximum(low[s], 1e-10))
        co = np.log(np.maximum(close[s], 1e-10) / np.maximum(open_[s], 1e-10))
        gk = 0.5 * hl**2 - (2 * np.log(2) - 1) * co**2
        target[i] = np.sqrt(max(np.mean(gk), 0) * 252)
    return target


def ewma_vol_target(
    log_returns: np.ndarray,
    span: int = 20,
    annualise: bool = True,
) -> np.ndarray:
    """
    Exponentially Weighted Moving Average volatility.
    This is a causal estimator (uses past data), used as a benchmark target.
    """
    scale = np.sqrt(252) if annualise else 1.0
    alpha = 2.0 / (span + 1)
    n = len(log_returns)
    ewma = np.full(n, np.nan)
    var_ = log_returns[0] ** 2
    ewma[0] = np.sqrt(var_) * scale

    for i in range(1, n):
        var_ = alpha * log_returns[i] ** 2 + (1 - alpha) * var_
        ewma[i] = np.sqrt(var_) * scale

    return ewma


# ─── Baselines for Benchmarking ─────────────────────────────────
def historical_vol_baseline(
    log_returns: np.ndarray,
    window: int = 20,
    forward_window: int = 20,
) -> np.ndarray:
    """
    Naive forecast: use trailing realised vol as prediction for future vol.
    This is the simplest baseline — "tomorrow's vol = today's vol".
    """
    n = len(log_returns)
    pred = np.full(n, np.nan)
    scale = np.sqrt(252)
    for i in range(window, n - forward_window):
        pred[i] = np.std(log_returns[i - window: i], ddof=1) * scale
    return pred


def garch11_baseline(
    log_returns: np.ndarray,
    forward_window: int = 20,
    omega: float = 0.000002,
    alpha: float = 0.10,
    beta: float = 0.85,
) -> np.ndarray:
    """
    Simple GARCH(1,1) forecast baseline.
    σ²[t] = ω + α·r²[t-1] + β·σ²[t-1]
    """
    n = len(log_returns)
    sigma2 = np.full(n, np.nan)
    sigma2[0] = np.var(log_returns[:min(20, n)])

    for i in range(1, n):
        sigma2[i] = omega + alpha * log_returns[i - 1]**2 + beta * sigma2[i - 1]

    # Forecast: use current conditional variance for forward window
    pred = np.sqrt(np.maximum(sigma2, 0)) * np.sqrt(252)
    # Null out positions where we don't have a valid target
    pred[-forward_window:] = np.nan
    return pred


def ewma_baseline(
    log_returns: np.ndarray,
    span: int = 20,
    forward_window: int = 20,
) -> np.ndarray:
    """EWMA baseline forecast."""
    pred = ewma_vol_target(log_returns, span, annualise=True)
    pred[-forward_window:] = np.nan
    return pred


def build_targets(
    log_returns: np.ndarray,
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    forward_window: int = 20,
) -> dict[str, np.ndarray]:
    """Build all target variables."""
    return {
        "realized_vol": realized_vol_target(log_returns, forward_window),
        "parkinson_vol": parkinson_vol_target(high, low, forward_window),
        "garman_klass_vol": garman_klass_vol_target(open_, high, low, close, forward_window),
    }


def build_baselines(
    log_returns: np.ndarray,
    forward_window: int = 20,
) -> dict[str, np.ndarray]:
    """Build all baseline predictions for comparison."""
    return {
        "historical_vol": historical_vol_baseline(log_returns, 20, forward_window),
        "garch11": garch11_baseline(log_returns, forward_window),
        "ewma": ewma_baseline(log_returns, 20, forward_window),
    }
