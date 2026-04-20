from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class VarianceResult:
    price: float
    variance: float


def antithetic_payoff(payoff_fn, paths: int = 10000, seed: int | None = None) -> VarianceResult:
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(paths)
    payoffs_pos = np.array([payoff_fn(zi) for zi in z])
    payoffs_neg = np.array([payoff_fn(-zi) for zi in z])
    payoffs = 0.5 * (payoffs_pos + payoffs_neg)
    mean_payoff = float(np.mean(payoffs))
    variance = float(np.var(payoffs, ddof=1))
    return VarianceResult(price=mean_payoff, variance=variance)


def control_variate(mc_payoffs: np.ndarray, control_values: np.ndarray,
                    control_mean: float, discount: float = 1.0,
                    fallback_beta: float = 0.7) -> float:
    """Control variate with optimal beta = -Cov(payoff, CV) / Var(CV)."""
    if isinstance(mc_payoffs, (int, float)):
        # Backward-compatible: scalar mc_price + bs_price
        return mc_payoffs + fallback_beta * (control_values - mc_payoffs)
    cov = np.cov(mc_payoffs, control_values)[0, 1]
    var_cv = np.var(control_values)
    beta = cov / (var_cv + 1e-10) if var_cv > 1e-12 else fallback_beta
    adjusted = mc_payoffs - beta * (control_values - control_mean)
    return float(discount * np.mean(adjusted))
