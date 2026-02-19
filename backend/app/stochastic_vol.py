from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class HestonParams:
    spot: float
    strike: float
    maturity: float
    rate: float
    v0: float
    kappa: float
    theta: float
    xi: float
    rho: float
    steps: int = 252
    paths: int = 20000
    option_type: str = "call"


def heston_mc(params: HestonParams, seed: int | None = None) -> float:
    """Vectorized Heston stochastic volatility Monte Carlo simulation."""
    rng = np.random.default_rng(seed)
    dt = params.maturity / params.steps

    s = np.full(params.paths, params.spot)
    v = np.full(params.paths, max(params.v0, 1e-8))

    for _ in range(params.steps):
        z1 = rng.standard_normal(params.paths)
        z2 = rng.standard_normal(params.paths)
        w2 = params.rho * z1 + math.sqrt(1 - params.rho**2) * z2

        sqrt_v_dt = np.sqrt(np.maximum(v, 1e-8) * dt)
        # Use old v for stock price (Euler consistency)
        s = s * np.exp((params.rate - 0.5 * np.maximum(v, 1e-8)) * dt + sqrt_v_dt * z1)
        v = np.maximum(
            v + params.kappa * (params.theta - v) * dt + params.xi * sqrt_v_dt * w2,
            1e-8,
        )

    if params.option_type == "call":
        payoffs = np.maximum(s - params.strike, 0.0)
    else:
        payoffs = np.maximum(params.strike - s, 0.0)

    return float(math.exp(-params.rate * params.maturity) * np.mean(payoffs))
