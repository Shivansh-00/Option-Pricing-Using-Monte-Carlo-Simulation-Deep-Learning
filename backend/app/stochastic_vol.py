from __future__ import annotations

import math
import random
from dataclasses import dataclass


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
    if seed is not None:
        random.seed(seed)
    dt = params.maturity / params.steps
    payoffs = []
    for _ in range(params.paths):
        s = params.spot
        v = max(params.v0, 1e-8)
        for _ in range(params.steps):
            z1 = random.gauss(0, 1)
            z2 = random.gauss(0, 1)
            w2 = params.rho * z1 + math.sqrt(1 - params.rho**2) * z2
            v = max(v + params.kappa * (params.theta - v) * dt + params.xi * math.sqrt(v * dt) * w2, 1e-8)
            s *= math.exp((params.rate - 0.5 * v) * dt + math.sqrt(v * dt) * z1)
        if params.option_type == "call":
            payoffs.append(max(s - params.strike, 0.0))
        else:
            payoffs.append(max(params.strike - s, 0.0))
    return math.exp(-params.rate * params.maturity) * (sum(payoffs) / len(payoffs))
