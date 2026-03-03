"""
OptionQuant — SHAP-like Explainability Engine
═══════════════════════════════════════════════════════════════
Model-agnostic feature attribution for option pricing:

  • Permutation-based SHAP approximation (no external deps)
  • Marginal contribution analysis for each pricing input
  • Greeks-impact decomposition
  • Volatility sensitivity analysis
  • Deep learning prediction explanation
  • Factor contribution waterfall data
  • Human-readable explanation generation
"""
from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .pricing import PricingInputs, black_scholes, monte_carlo_engine, black_scholes_greeks

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
#  Data Models
# ═══════════════════════════════════════════════════════════════

@dataclass
class FeatureAttribution:
    feature: str
    shap_value: float
    contribution_pct: float
    direction: str  # positive / negative / neutral
    description: str


@dataclass
class PricingExplanation:
    base_price: float            # average prediction (baseline)
    predicted_price: float       # actual model price
    attributions: list[FeatureAttribution]
    greeks_impact: dict
    vol_sensitivity: dict        # price at different vol levels
    moneyness_analysis: dict
    time_decay_profile: dict
    model: str
    computation_time_ms: float
    narrative: str               # human-readable explanation


@dataclass
class DLExplanation:
    prediction: float
    feature_importance: list[dict]
    model_components: dict       # BS weight, MC weight, LSTM weight
    sentiment_impact: float
    confidence_factors: list[dict]
    narrative: str


# ═══════════════════════════════════════════════════════════════
#  Permutation SHAP Approximation
# ═══════════════════════════════════════════════════════════════

def _compute_shap_values(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    volatility: float,
    option_type: str = "call",
    n_samples: int = 200,
    seed: int = 42,
) -> list[FeatureAttribution]:
    """
    Compute approximate SHAP values using permutation importance.
    
    For each feature, measures the average marginal contribution
    to the prediction when added to random coalitions of other features.
    """
    rng = np.random.default_rng(seed)

    # Feature definitions with realistic perturbation ranges
    features = {
        "spot": {"value": spot, "range": (spot * 0.8, spot * 1.2)},
        "strike": {"value": strike, "range": (strike * 0.8, strike * 1.2)},
        "maturity": {"value": maturity, "range": (max(0.01, maturity * 0.5), maturity * 2)},
        "rate": {"value": rate, "range": (max(0.0, rate - 0.03), rate + 0.03)},
        "volatility": {"value": volatility, "range": (max(0.01, volatility * 0.5), volatility * 2)},
    }

    base_inputs = PricingInputs(
        spot=spot, strike=strike, maturity=maturity,
        rate=rate, volatility=volatility, option_type=option_type,
    )
    base_price = black_scholes(base_inputs)

    # Background dataset: random parameter combinations
    background = []
    for _ in range(n_samples):
        bg = {
            "spot": rng.uniform(*features["spot"]["range"]),
            "strike": rng.uniform(*features["strike"]["range"]),
            "maturity": rng.uniform(*features["maturity"]["range"]),
            "rate": rng.uniform(*features["rate"]["range"]),
            "volatility": rng.uniform(*features["volatility"]["range"]),
        }
        background.append(bg)

    # Compute marginal contributions
    shap_vals = {}
    for feat_name in features:
        marginals = []
        for bg in background[:50]:  # use subset for speed
            # With feature = actual value
            params_with = bg.copy()
            params_with[feat_name] = features[feat_name]["value"]
            inp_with = PricingInputs(
                spot=params_with["spot"], strike=params_with["strike"],
                maturity=params_with["maturity"], rate=params_with["rate"],
                volatility=params_with["volatility"], option_type=option_type,
            )
            price_with = black_scholes(inp_with)

            # Without feature = background value
            inp_without = PricingInputs(
                spot=bg["spot"], strike=bg["strike"],
                maturity=bg["maturity"], rate=bg["rate"],
                volatility=bg["volatility"], option_type=option_type,
            )
            price_without = black_scholes(inp_without)

            marginals.append(price_with - price_without)

        shap_vals[feat_name] = float(np.mean(marginals))

    # Normalize to contributions
    total_abs = sum(abs(v) for v in shap_vals.values()) or 1.0

    attributions = []
    for feat, val in sorted(shap_vals.items(), key=lambda x: abs(x[1]), reverse=True):
        pct = val / total_abs * 100
        direction = "positive" if val > 0.01 else "negative" if val < -0.01 else "neutral"

        descriptions = {
            "spot": f"Current price (${spot:.2f}) {'increases' if val > 0 else 'decreases'} option value",
            "strike": f"Strike (${strike:.2f}) {'increases' if val > 0 else 'decreases'} option value",
            "maturity": f"Time to expiry ({maturity:.3f}y) contributes {'positively' if val > 0 else 'negatively'}",
            "rate": f"Risk-free rate ({rate:.2%}) has {'positive' if val > 0 else 'negative'} effect",
            "volatility": f"Volatility ({volatility:.2%}) is the {'key driver' if abs(pct) > 30 else 'moderate factor'}",
        }

        attributions.append(FeatureAttribution(
            feature=feat,
            shap_value=round(val, 6),
            contribution_pct=round(abs(pct), 2),
            direction=direction,
            description=descriptions.get(feat, ""),
        ))

    return attributions


# ═══════════════════════════════════════════════════════════════
#  Sensitivity Analysis
# ═══════════════════════════════════════════════════════════════

def _vol_sensitivity(
    spot: float, strike: float, maturity: float,
    rate: float, volatility: float, option_type: str,
) -> dict:
    """Price at different volatility levels."""
    vol_levels = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.80]
    results = {}
    for v in vol_levels:
        inp = PricingInputs(spot=spot, strike=strike, maturity=maturity,
                            rate=rate, volatility=v, option_type=option_type)
        results[f"{v:.0%}"] = round(black_scholes(inp), 4)
    return results


def _time_decay_profile(
    spot: float, strike: float, maturity: float,
    rate: float, volatility: float, option_type: str,
) -> dict:
    """Price decay over time."""
    if maturity <= 0:
        return {}
    steps = min(10, max(2, int(maturity * 365 / 7)))
    profile = {}
    for i in range(steps + 1):
        t = maturity * (1 - i / steps)
        if t < 0.001:
            t = 0.001
        inp = PricingInputs(spot=spot, strike=strike, maturity=t,
                            rate=rate, volatility=volatility, option_type=option_type)
        days_left = int(t * 365)
        profile[f"{days_left}d"] = round(black_scholes(inp), 4)
    return profile


def _moneyness_analysis(
    spot: float, strike: float, maturity: float,
    rate: float, volatility: float, option_type: str,
) -> dict:
    """Analysis across moneyness levels."""
    moneyness = strike / spot
    if option_type == "call":
        status = "ITM" if moneyness < 0.98 else "ATM" if moneyness < 1.02 else "OTM"
    else:
        status = "OTM" if moneyness < 0.98 else "ATM" if moneyness < 1.02 else "ITM"

    intrinsic = max(0, (spot - strike) if option_type == "call" else (strike - spot))
    inp = PricingInputs(spot=spot, strike=strike, maturity=maturity,
                        rate=rate, volatility=volatility, option_type=option_type)
    total_price = black_scholes(inp)
    extrinsic = max(0, total_price - intrinsic)

    return {
        "moneyness": round(moneyness, 4),
        "status": status,
        "intrinsic_value": round(intrinsic, 4),
        "extrinsic_value": round(extrinsic, 4),
        "extrinsic_pct": round(extrinsic / (total_price + 1e-8) * 100, 1),
        "leverage": round(spot / (total_price + 1e-8), 2),
    }


# ═══════════════════════════════════════════════════════════════
#  Main Explanation Builder
# ═══════════════════════════════════════════════════════════════

def explain_pricing(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    volatility: float,
    option_type: str = "call",
    model: str = "black_scholes",
) -> PricingExplanation:
    """
    Generate comprehensive SHAP-based explanation for option pricing.
    """
    t0 = time.perf_counter()

    inputs = PricingInputs(
        spot=spot, strike=strike, maturity=maturity,
        rate=rate, volatility=volatility, option_type=option_type,
    )
    predicted = black_scholes(inputs)
    greeks = black_scholes_greeks(inputs)

    # SHAP values
    attributions = _compute_shap_values(
        spot, strike, maturity, rate, volatility, option_type,
    )

    # Sensitivity analyses
    vol_sens = _vol_sensitivity(spot, strike, maturity, rate, volatility, option_type)
    time_decay = _time_decay_profile(spot, strike, maturity, rate, volatility, option_type)
    moneyness = _moneyness_analysis(spot, strike, maturity, rate, volatility, option_type)

    # Greeks impact interpretation
    greeks_impact = {
        "delta": {
            "value": greeks["delta"],
            "interpretation": f"For $1 move in spot, option changes by ${greeks['delta']:.4f}",
        },
        "gamma": {
            "value": greeks["gamma"],
            "interpretation": f"Delta changes by {greeks['gamma']:.6f} per $1 spot move",
        },
        "vega": {
            "value": greeks["vega"],
            "interpretation": f"For 1% vol change, option changes by ${greeks['vega']:.4f}",
        },
        "theta": {
            "value": greeks["theta"],
            "interpretation": f"Option loses ${abs(greeks['theta']):.4f} per day from time decay",
        },
        "rho": {
            "value": greeks["rho"],
            "interpretation": f"For 1% rate change, option changes by ${greeks['rho']:.4f}",
        },
    }

    # Generate narrative
    top_driver = attributions[0] if attributions else None
    narrative = _generate_narrative(
        predicted, option_type, moneyness, greeks, top_driver, volatility,
    )

    elapsed = (time.perf_counter() - t0) * 1000

    return PricingExplanation(
        base_price=round(predicted, 6),
        predicted_price=round(predicted, 6),
        attributions=attributions,
        greeks_impact=greeks_impact,
        vol_sensitivity=vol_sens,
        moneyness_analysis=moneyness,
        time_decay_profile=time_decay,
        model=model,
        computation_time_ms=round(elapsed, 2),
        narrative=narrative,
    )


def _generate_narrative(
    price: float,
    option_type: str,
    moneyness: dict,
    greeks: dict,
    top_driver: FeatureAttribution | None,
    volatility: float,
) -> str:
    """Generate human-readable explanation."""
    parts = []
    parts.append(
        f"This {option_type} option is priced at ${price:.2f} and is currently "
        f"{moneyness['status']} (moneyness: {moneyness['moneyness']:.2f})."
    )

    if moneyness["intrinsic_value"] > 0:
        parts.append(
            f"Of this, ${moneyness['intrinsic_value']:.2f} is intrinsic value "
            f"and ${moneyness['extrinsic_value']:.2f} ({moneyness['extrinsic_pct']:.0f}%) "
            f"is time/volatility premium."
        )

    if top_driver:
        parts.append(
            f"The primary pricing driver is {top_driver.feature} "
            f"(contributing {top_driver.contribution_pct:.0f}% of the price). "
            f"{top_driver.description}."
        )

    delta = greeks.get("delta", 0)
    parts.append(
        f"With a delta of {delta:.3f}, the option has a "
        f"{abs(delta)*100:.0f}% sensitivity to underlying moves."
    )

    if volatility > 0.4:
        parts.append("High volatility significantly inflates the option premium.")
    elif volatility < 0.1:
        parts.append("Low volatility keeps the premium relatively tight.")

    return " ".join(parts)


# ═══════════════════════════════════════════════════════════════
#  Deep Learning Explanation
# ═══════════════════════════════════════════════════════════════

def explain_dl_prediction(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    volatility: float,
    option_type: str = "call",
    news_text: str = "",
) -> DLExplanation:
    """
    Explain a hybrid DL prediction with component attribution.
    """
    from .dl import get_predictor
    predictor = get_predictor()
    forecast = predictor.predict(
        spot=spot, strike=strike, maturity=maturity,
        rate=rate, volatility=volatility,
        option_type=option_type, news_text=news_text,
    )

    # Component weights (from the HybridDLPredictor)
    bs_weight = 0.45
    mc_weight = 0.25
    lstm_weight = 0.20
    residual_weight = 0.10

    bs_price = forecast.details.get("bs_price", 0)
    mc_price = forecast.details.get("mc_price", 0)
    lstm_pred = forecast.details.get("lstm_prediction", 0)

    # Feature importance via pricing SHAP
    pricing_shap = _compute_shap_values(
        spot, strike, maturity, rate, volatility, option_type, n_samples=50
    )

    feature_importance = [
        {
            "feature": attr.feature,
            "importance": attr.contribution_pct,
            "shap_value": attr.shap_value,
            "direction": attr.direction,
        }
        for attr in pricing_shap
    ]

    model_components = {
        "black_scholes": {"weight": bs_weight, "price": round(bs_price, 4)},
        "monte_carlo": {"weight": mc_weight, "price": round(mc_price, 4)},
        "lstm": {"weight": lstm_weight, "price": round(lstm_pred, 4)},
        "residual_correction": {"weight": residual_weight, "value": round(forecast.residual, 4)},
    }

    confidence_factors = [
        {"factor": "Model agreement", "score": round(forecast.confidence, 3),
         "status": "good" if forecast.confidence > 0.7 else "moderate"},
        {"factor": "BS-MC spread", "score": round(1 - abs(bs_price - mc_price) / (bs_price + 1e-8), 3),
         "status": "good" if abs(bs_price - mc_price) < bs_price * 0.02 else "caution"},
        {"factor": "LSTM trained", "score": 1.0 if forecast.details.get("lstm_trained") else 0.0,
         "status": "good" if forecast.details.get("lstm_trained") else "not_trained"},
    ]

    narrative = (
        f"The hybrid DL model prices this {option_type} at ${forecast.forecast_price:.2f}. "
        f"Black-Scholes (weight {bs_weight:.0%}) contributes ${bs_price:.2f}, "
        f"Monte Carlo (weight {mc_weight:.0%}) contributes ${mc_price:.2f}, "
        f"and LSTM (weight {lstm_weight:.0%}) predicts ${lstm_pred:.2f}. "
        f"Confidence: {forecast.confidence:.0%}. "
        f"Sentiment: {forecast.transformer_sentiment}."
    )

    return DLExplanation(
        prediction=forecast.forecast_price,
        feature_importance=feature_importance,
        model_components=model_components,
        sentiment_impact=forecast.details.get("sentiment_adjustment", 0),
        confidence_factors=confidence_factors,
        narrative=narrative,
    )
