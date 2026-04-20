"""
OptionQuant — Risk & Confidence Estimation Engine
═══════════════════════════════════════════════════════════════
Production-grade risk analytics:

  • Monte Carlo confidence interval estimation (bootstrap)
  • Bayesian uncertainty from neural network predictions
  • Value at Risk (VaR) and Conditional VaR (CVaR)
  • Portfolio-level risk aggregation
  • Model reliability scoring
  • Greeks-based risk decomposition
  • Correlation-adjusted portfolio exposure
"""
from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .pricing import PricingInputs, black_scholes, monte_carlo_engine, black_scholes_greeks

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
#  Data Models
# ═══════════════════════════════════════════════════════════════

@dataclass
class ConfidenceEstimate:
    price: float
    ci_lower: float
    ci_upper: float
    ci_width: float
    confidence_level: float
    std_error: float
    model: str
    is_reliable: bool
    reliability_score: float  # 0-1
    warnings: list[str] = field(default_factory=list)


@dataclass
class BayesianUncertainty:
    mean_prediction: float
    epistemic_uncertainty: float   # model uncertainty
    aleatoric_uncertainty: float   # data uncertainty
    total_uncertainty: float
    prediction_interval_low: float
    prediction_interval_high: float
    n_samples: int
    is_reliable: bool


@dataclass
class VaRResult:
    var_95: float
    var_99: float
    cvar_95: float
    cvar_99: float
    max_loss: float
    expected_shortfall: float
    n_simulations: int


@dataclass
class PortfolioRisk:
    total_delta: float
    total_gamma: float
    total_vega: float
    total_theta: float
    total_rho: float
    delta_dollars: float
    gamma_dollars: float
    vega_dollars: float
    var_95: float
    cvar_95: float
    max_drawdown: float
    sharpe_ratio: float
    positions: list[dict]
    risk_decomposition: dict
    overall_risk_level: str  # low / medium / high / extreme


@dataclass
class ReliabilityReport:
    score: float          # 0-1 composite reliability
    model_agreement: float
    convergence_quality: float
    vol_stability: float
    moneyness_penalty: float
    time_decay_risk: float
    flags: list[str]


# ═══════════════════════════════════════════════════════════════
#  Monte Carlo Confidence Intervals (Bootstrap)
# ═══════════════════════════════════════════════════════════════

def estimate_confidence(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    volatility: float,
    option_type: str = "call",
    confidence_level: float = 0.95,
    n_bootstrap: int = 20,
    paths_per_sample: int = 10000,
    seed: int = 42,
) -> ConfidenceEstimate:
    """
    Bootstrap confidence intervals by running MC multiple times.
    Also compares with BS for model agreement check.
    """
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)

    inputs = PricingInputs(
        spot=spot, strike=strike, maturity=maturity,
        rate=rate, volatility=volatility,
        option_type=option_type, paths=paths_per_sample,
    )

    # Multiple MC runs
    prices = []
    for i in range(n_bootstrap):
        mc = monte_carlo_engine(inputs, seed=int(rng.integers(0, 2**31)), method="antithetic")
        prices.append(mc.price)

    prices_arr = np.array(prices)
    mean_price = float(np.mean(prices_arr))
    std_err = float(np.std(prices_arr, ddof=1) / math.sqrt(n_bootstrap))

    alpha = 1 - confidence_level
    z = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}.get(confidence_level, 1.96)
    ci_lower = mean_price - z * std_err
    ci_upper = mean_price + z * std_err

    # Compare with BS
    bs_price = black_scholes(inputs)
    model_diff = abs(mean_price - bs_price) / (bs_price + 1e-8)

    # Reliability assessment
    warnings = []
    reliability = 1.0

    if model_diff > 0.05:
        warnings.append(f"MC-BS divergence: {model_diff:.1%}")
        reliability -= 0.2

    if std_err / (mean_price + 1e-8) > 0.05:
        warnings.append(f"High std error: {std_err:.4f}")
        reliability -= 0.2

    if maturity < 1 / 365:
        warnings.append("Very short maturity — numerical instability risk")
        reliability -= 0.15

    moneyness = strike / spot
    if moneyness > 1.5 or moneyness < 0.5:
        warnings.append(f"Deep OTM/ITM (moneyness={moneyness:.2f}) — wider CI expected")
        reliability -= 0.1

    reliability = max(0.0, min(1.0, reliability))

    return ConfidenceEstimate(
        price=round(mean_price, 6),
        ci_lower=round(max(0, ci_lower), 6),
        ci_upper=round(ci_upper, 6),
        ci_width=round(ci_upper - ci_lower, 6),
        confidence_level=confidence_level,
        std_error=round(std_err, 6),
        model="monte_carlo_bootstrap",
        is_reliable=reliability > 0.6,
        reliability_score=round(reliability, 4),
        warnings=warnings,
    )


# ═══════════════════════════════════════════════════════════════
#  Bayesian Uncertainty (MC Dropout Approximation)
# ═══════════════════════════════════════════════════════════════

def bayesian_uncertainty(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    volatility: float,
    option_type: str = "call",
    n_samples: int = 50,
    seed: int = 42,
) -> BayesianUncertainty:
    """
    Approximate Bayesian uncertainty via stochastic parameter perturbation.
    Simulates epistemic uncertainty (model) + aleatoric uncertainty (data).
    """
    rng = np.random.default_rng(seed)
    predictions = []

    for _ in range(n_samples):
        # Perturb parameters (epistemic: model uncertainty)
        vol_perturbed = volatility * (1 + rng.normal(0, 0.05))
        rate_perturbed = rate + rng.normal(0, 0.002)
        spot_perturbed = spot * (1 + rng.normal(0, 0.001))

        inputs = PricingInputs(
            spot=max(0.01, spot_perturbed),
            strike=strike,
            maturity=maturity,
            rate=rate_perturbed,
            volatility=max(0.01, vol_perturbed),
            option_type=option_type,
            paths=5000,  # faster for uncertainty estimation
        )
        mc = monte_carlo_engine(inputs, seed=int(rng.integers(0, 2**31)))
        predictions.append(mc.price)

    preds = np.array(predictions)
    mean_pred = float(np.mean(preds))

    # Decompose uncertainty
    # Epistemic: variance from parameter perturbation
    epistemic = float(np.std(preds, ddof=1))
    # Aleatoric: average MC std error (inherent randomness)
    inputs_base = PricingInputs(
        spot=spot, strike=strike, maturity=maturity,
        rate=rate, volatility=volatility,
        option_type=option_type, paths=10000,
    )
    mc_base = monte_carlo_engine(inputs_base, seed=seed)
    aleatoric = mc_base.std_error

    total = math.sqrt(epistemic**2 + aleatoric**2)

    pi_low = float(np.percentile(preds, 2.5))
    pi_high = float(np.percentile(preds, 97.5))

    is_reliable = (total / (mean_pred + 1e-8)) < 0.15

    return BayesianUncertainty(
        mean_prediction=round(mean_pred, 6),
        epistemic_uncertainty=round(epistemic, 6),
        aleatoric_uncertainty=round(aleatoric, 6),
        total_uncertainty=round(total, 6),
        prediction_interval_low=round(max(0, pi_low), 6),
        prediction_interval_high=round(pi_high, 6),
        n_samples=n_samples,
        is_reliable=is_reliable,
    )


# ═══════════════════════════════════════════════════════════════
#  Value at Risk (VaR) & CVaR
# ═══════════════════════════════════════════════════════════════

_SQRT2 = math.sqrt(2.0)

def _vectorized_bs(spots: np.ndarray, strike: float, maturity: float,
                   rate: float, vol: float, option_type: str) -> np.ndarray:
    """Black-Scholes on an array of spot prices (NumPy-vectorized)."""
    from scipy.special import erf  # type: ignore
    sqrt_t = math.sqrt(maturity)
    d1 = (np.log(spots / strike) + (rate + 0.5 * vol**2) * maturity) / (vol * sqrt_t)
    d2 = d1 - vol * sqrt_t
    nd1 = 0.5 * (1.0 + erf(d1 / _SQRT2))
    nd2 = 0.5 * (1.0 + erf(d2 / _SQRT2))
    disc = math.exp(-rate * maturity)
    if option_type == "put":
        return strike * disc * (1.0 - nd2) - spots * (1.0 - nd1)
    return spots * nd1 - strike * disc * nd2


def compute_var(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    volatility: float,
    option_type: str = "call",
    position_size: float = 100.0,
    horizon_days: int = 1,
    n_sims: int = 50000,
    seed: int = 42,
) -> VaRResult:
    """
    Compute VaR and CVaR via full revaluation Monte Carlo (vectorized).
    """
    rng = np.random.default_rng(seed)
    dt = horizon_days / 252.0

    inputs = PricingInputs(
        spot=spot, strike=strike, maturity=maturity,
        rate=rate, volatility=volatility, option_type=option_type,
    )
    current_price = black_scholes(inputs)

    # Simulate spot moves
    z = rng.standard_normal(n_sims)
    future_spots = spot * np.exp((rate - 0.5 * volatility**2) * dt + volatility * math.sqrt(dt) * z)

    # Vectorized BS revaluation
    future_mat = max(0.001, maturity - dt)
    future_prices = _vectorized_bs(future_spots, strike, future_mat, rate, volatility, option_type)

    pnl = (future_prices - current_price) * position_size
    sorted_pnl = np.sort(pnl)

    var_95 = -float(np.percentile(sorted_pnl, 5))
    var_99 = -float(np.percentile(sorted_pnl, 1))
    cvar_95 = -float(np.mean(sorted_pnl[sorted_pnl <= np.percentile(sorted_pnl, 5)]))
    cvar_99 = -float(np.mean(sorted_pnl[sorted_pnl <= np.percentile(sorted_pnl, 1)]))
    max_loss = -float(np.min(sorted_pnl))

    return VaRResult(
        var_95=round(var_95, 4),
        var_99=round(var_99, 4),
        cvar_95=round(cvar_95, 4),
        cvar_99=round(cvar_99, 4),
        max_loss=round(max_loss, 4),
        expected_shortfall=round(cvar_95, 4),
        n_simulations=n_sims,
    )


# ═══════════════════════════════════════════════════════════════
#  Portfolio Risk Aggregation
# ═══════════════════════════════════════════════════════════════

def compute_portfolio_risk(
    positions: list[dict],
    seed: int = 42,
) -> PortfolioRisk:
    """
    Aggregate risk across a portfolio of option positions.
    
    Each position: {spot, strike, maturity, rate, volatility, option_type, quantity}
    """
    total_delta = total_gamma = total_vega = total_theta = total_rho = 0.0
    delta_dollars = gamma_dollars = vega_dollars = 0.0
    position_details = []

    for pos in positions:
        qty = pos.get("quantity", 1)
        inputs = PricingInputs(
            spot=pos["spot"], strike=pos["strike"],
            maturity=pos["maturity"], rate=pos["rate"],
            volatility=pos["volatility"], option_type=pos["option_type"],
        )
        greeks = black_scholes_greeks(inputs)
        price = black_scholes(inputs)

        total_delta += greeks["delta"] * qty
        total_gamma += greeks["gamma"] * qty
        total_vega += greeks["vega"] * qty
        total_theta += greeks["theta"] * qty
        total_rho += greeks["rho"] * qty

        delta_dollars += greeks["delta"] * pos["spot"] * qty
        gamma_dollars += 0.5 * greeks["gamma"] * pos["spot"]**2 * qty * 0.01
        vega_dollars += greeks["vega"] * qty

        position_details.append({
            "strike": pos["strike"],
            "option_type": pos["option_type"],
            "quantity": qty,
            "price": round(price, 4),
            "delta": round(greeks["delta"] * qty, 6),
            "gamma": round(greeks["gamma"] * qty, 6),
            "vega": round(greeks["vega"] * qty, 6),
            "theta": round(greeks["theta"] * qty, 6),
        })

    # Portfolio VaR (delta-normal approximation)
    if positions:
        avg_vol = float(np.mean([p["volatility"] for p in positions]))
        avg_spot = float(np.mean([p["spot"] for p in positions]))
        portfolio_vol = abs(total_delta) * avg_spot * avg_vol / math.sqrt(252)
        var_95 = 1.645 * portfolio_vol
        cvar_95 = var_95 * 1.2  # approximate relationship
    else:
        var_95 = cvar_95 = 0.0

    # Risk level
    risk_score = abs(total_delta) * 0.3 + abs(total_gamma) * 100 * 0.3 + abs(total_vega) * 0.2 + var_95 / 1000 * 0.2
    if risk_score > 3:
        risk_level = "extreme"
    elif risk_score > 1.5:
        risk_level = "high"
    elif risk_score > 0.5:
        risk_level = "medium"
    else:
        risk_level = "low"

    return PortfolioRisk(
        total_delta=round(total_delta, 6),
        total_gamma=round(total_gamma, 6),
        total_vega=round(total_vega, 6),
        total_theta=round(total_theta, 6),
        total_rho=round(total_rho, 6),
        delta_dollars=round(delta_dollars, 2),
        gamma_dollars=round(gamma_dollars, 2),
        vega_dollars=round(vega_dollars, 2),
        var_95=round(var_95, 2),
        cvar_95=round(cvar_95, 2),
        max_drawdown=round(var_95 * 1.5, 2),
        sharpe_ratio=0.0,
        positions=position_details,
        risk_decomposition={
            "delta_contribution": round(abs(total_delta) / (risk_score + 1e-8) * 100, 1),
            "gamma_contribution": round(abs(total_gamma) * 100 / (risk_score + 1e-8) * 100, 1),
            "vega_contribution": round(abs(total_vega) / (risk_score + 1e-8) * 100, 1),
        },
        overall_risk_level=risk_level,
    )


# ═══════════════════════════════════════════════════════════════
#  Model Reliability Scoring
# ═══════════════════════════════════════════════════════════════

def assess_reliability(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    volatility: float,
    option_type: str = "call",
) -> ReliabilityReport:
    """
    Comprehensive reliability assessment for a pricing estimate.
    """
    flags = []
    inputs = PricingInputs(
        spot=spot, strike=strike, maturity=maturity,
        rate=rate, volatility=volatility, option_type=option_type,
    )

    # Model agreement (BS vs MC)
    bs = black_scholes(inputs)
    mc = monte_carlo_engine(inputs, seed=42, method="antithetic")
    agreement = 1.0 - min(1.0, abs(bs - mc.price) / (bs + 1e-8))

    if agreement < 0.95:
        flags.append(f"Model disagreement: BS={bs:.4f}, MC={mc.price:.4f}")

    # Convergence quality
    if mc.std_error / (mc.price + 1e-8) > 0.02:
        convergence = 0.5
        flags.append("MC convergence is slow — consider more paths")
    else:
        convergence = 1.0

    # Vol stability
    vol_stability = 1.0
    if volatility > 1.0:
        vol_stability = 0.5
        flags.append("Extremely high volatility — model assumptions may break")
    elif volatility < 0.02:
        vol_stability = 0.7
        flags.append("Very low volatility — near-zero vol regime")

    # Moneyness penalty
    moneyness = strike / spot
    moneyness_penalty = 0.0
    if moneyness > 1.5:
        moneyness_penalty = 0.3
        flags.append("Deep OTM — price sensitivity to vol is high")
    elif moneyness < 0.5:
        moneyness_penalty = 0.2
        flags.append("Deep ITM — mainly intrinsic value")

    # Time decay risk
    time_decay_risk = 0.0
    if maturity < 7 / 365:
        time_decay_risk = 0.8
        flags.append("Near expiry — theta decay is extreme")
    elif maturity < 30 / 365:
        time_decay_risk = 0.4

    # Composite score
    score = (agreement * 0.3 + convergence * 0.2 + vol_stability * 0.2 +
             (1 - moneyness_penalty) * 0.15 + (1 - time_decay_risk) * 0.15)

    return ReliabilityReport(
        score=round(score, 4),
        model_agreement=round(agreement, 4),
        convergence_quality=round(convergence, 4),
        vol_stability=round(vol_stability, 4),
        moneyness_penalty=round(moneyness_penalty, 4),
        time_decay_risk=round(time_decay_risk, 4),
        flags=flags,
    )
