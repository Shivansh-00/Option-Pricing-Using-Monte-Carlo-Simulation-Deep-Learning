from __future__ import annotations

import math
import random
from dataclasses import dataclass


@dataclass
class VarianceResult:
    price: float
    variance: float


def antithetic_payoff(payoff_fn, paths: int = 10000) -> VarianceResult:
    payoffs = []
    for _ in range(paths):
        z = random.gauss(0, 1)
        payoffs.append(payoff_fn(z))
        payoffs.append(payoff_fn(-z))
    mean_payoff = sum(payoffs) / len(payoffs)
    variance = sum((p - mean_payoff) ** 2 for p in payoffs) / (len(payoffs) - 1)
    return VarianceResult(price=mean_payoff, variance=variance)


def control_variate(mc_price: float, bs_price: float, beta: float = 0.7) -> float:
    return mc_price + beta * (bs_price - mc_price)
