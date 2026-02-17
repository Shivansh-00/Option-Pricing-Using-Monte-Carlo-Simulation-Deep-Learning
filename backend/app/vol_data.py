"""
vol_data.py — Synthetic Market Data Generator & OHLCV Pipeline
================================================================
Generates realistic multi‑regime synthetic market data for training and
testing the ML volatility forecasting engine.  Supports:
  • GBM + regime‑switching stochastic vol
  • Realistic volume & VIX simulation
  • Parkinson / Garman‑Klass / Yang‑Zhang OHLC generation
  • Train/val/test splits with strict temporal ordering
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ── Data Structures ──────────────────────────────────────────────
@dataclass
class OHLCVBar:
    """Single OHLCV bar with auxiliary fields."""
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    vix: float = 20.0
    rate: float = 0.05


@dataclass
class MarketDataset:
    """Complete dataset with temporal split metadata."""
    bars: list[OHLCVBar]
    train_end: int = 0       # index boundary
    val_end: int = 0
    n_regimes: int = 4


# ── Regime Definitions ──────────────────────────────────────────
REGIMES = {
    "low_vol":   {"mu": 0.08, "sigma": 0.10, "vix_base": 12, "vol_of_vol": 0.05},
    "normal":    {"mu": 0.06, "sigma": 0.18, "vix_base": 18, "vol_of_vol": 0.12},
    "high_vol":  {"mu": -0.02, "sigma": 0.32, "vix_base": 28, "vol_of_vol": 0.25},
    "crisis":    {"mu": -0.15, "sigma": 0.55, "vix_base": 45, "vol_of_vol": 0.40},
}
REGIME_NAMES = list(REGIMES.keys())
TRANSITION_MATRIX = np.array([
    [0.92, 0.06, 0.015, 0.005],   # low_vol →
    [0.04, 0.88, 0.06,  0.02],    # normal →
    [0.01, 0.08, 0.85,  0.06],    # high_vol →
    [0.005, 0.03, 0.12, 0.845],   # crisis →
])


def _next_regime(current_idx: int, rng: np.random.Generator) -> int:
    return int(rng.choice(4, p=TRANSITION_MATRIX[current_idx]))


# ── Synthetic Data Generator ────────────────────────────────────
def generate_synthetic_market(
    n_days: int = 2520,      # ~10 years of trading days
    seed: int = 42,
    initial_price: float = 100.0,
    initial_rate: float = 0.05,
) -> MarketDataset:
    """Generate regime‑switching synthetic OHLCV with VIX & rate."""
    rng = np.random.default_rng(seed)
    bars: list[OHLCVBar] = []

    price = initial_price
    rate = initial_rate
    regime_idx = 1  # start in 'normal'
    dt = 1 / 252

    for i in range(n_days):
        # Regime transition (daily)
        if rng.random() < 0.03:  # 3% chance of checking transition
            regime_idx = _next_regime(regime_idx, rng)
        regime = REGIMES[REGIME_NAMES[regime_idx]]

        mu = regime["mu"]
        sigma = regime["sigma"]
        vix_base = regime["vix_base"]
        vol_of_vol = regime["vol_of_vol"]

        # Stochastic vol perturbation
        sigma_t = max(0.03, sigma + vol_of_vol * rng.standard_normal() * math.sqrt(dt))

        # Generate intraday-like OHLC from close-to-close
        z = rng.standard_normal()
        log_ret = (mu - 0.5 * sigma_t**2) * dt + sigma_t * math.sqrt(dt) * z
        new_price = price * math.exp(log_ret)

        # Realistic OHLC: high/low within daily range
        daily_range = sigma_t * math.sqrt(dt) * price
        intra_noise1 = abs(rng.standard_normal()) * 0.5
        intra_noise2 = abs(rng.standard_normal()) * 0.5
        high = max(price, new_price) + daily_range * intra_noise1
        low = min(price, new_price) - daily_range * intra_noise2
        low = max(low, 0.5)  # floor

        # Volume (mean-reverting with vol correlation)
        base_vol = 1_000_000
        vol_mult = 1.0 + 2.0 * (sigma_t / 0.30)  # higher vol → higher volume
        volume = max(100_000, base_vol * vol_mult * (0.7 + 0.6 * rng.random()))

        # VIX with mean-reversion + noise
        vix = max(8, vix_base + 5 * rng.standard_normal())

        # Rate random walk (slow)
        rate = max(0.0, rate + 0.001 * rng.standard_normal())

        # Date
        year = 2015 + i // 252
        day_in_year = (i % 252) + 1
        month = min(12, (day_in_year - 1) // 21 + 1)
        day = min(28, (day_in_year - 1) % 21 + 1)
        date_str = f"{year}-{month:02d}-{day:02d}"

        bars.append(OHLCVBar(
            date=date_str,
            open=round(price, 4),
            high=round(high, 4),
            low=round(low, 4),
            close=round(new_price, 4),
            volume=round(volume),
            vix=round(vix, 2),
            rate=round(rate, 5),
        ))
        price = new_price

    # Temporal split: 70% train, 15% val, 15% test
    train_end = int(n_days * 0.70)
    val_end = int(n_days * 0.85)

    return MarketDataset(
        bars=bars,
        train_end=train_end,
        val_end=val_end,
        n_regimes=4,
    )


def bars_to_arrays(bars: list[OHLCVBar]) -> dict[str, np.ndarray]:
    """Convert bars to numpy arrays for vectorized feature computation."""
    n = len(bars)
    return {
        "open": np.array([b.open for b in bars], dtype=np.float64),
        "high": np.array([b.high for b in bars], dtype=np.float64),
        "low": np.array([b.low for b in bars], dtype=np.float64),
        "close": np.array([b.close for b in bars], dtype=np.float64),
        "volume": np.array([b.volume for b in bars], dtype=np.float64),
        "vix": np.array([b.vix for b in bars], dtype=np.float64),
        "rate": np.array([b.rate for b in bars], dtype=np.float64),
    }
