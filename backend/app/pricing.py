from __future__ import annotations

import math
import random
from dataclasses import dataclass


@dataclass
class PricingInputs:
    spot: float
    strike: float
    maturity: float
    rate: float
    volatility: float
    option_type: str
    steps: int
    paths: int


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def black_scholes(inputs: PricingInputs) -> float:
    s = inputs.spot
    k = inputs.strike
    t = inputs.maturity
    r = inputs.rate
    sigma = inputs.volatility
    if t <= 0:
        return max(0.0, (s - k) if inputs.option_type == "call" else (k - s))
    d1 = (math.log(s / k) + (r + 0.5 * sigma**2) * t) / (sigma * math.sqrt(t))
    d2 = d1 - sigma * math.sqrt(t)
    if inputs.option_type == "call":
        return s * _norm_cdf(d1) - k * math.exp(-r * t) * _norm_cdf(d2)
    return k * math.exp(-r * t) * _norm_cdf(-d2) - s * _norm_cdf(-d1)


def monte_carlo_gbm(inputs: PricingInputs, seed: int | None = None) -> float:
    if seed is not None:
        random.seed(seed)
    dt = inputs.maturity / inputs.steps
    drift = (inputs.rate - 0.5 * inputs.volatility**2) * dt
    diffusion = inputs.volatility * math.sqrt(dt)
    payoffs = []
    for _ in range(inputs.paths):
        price = inputs.spot
        for _ in range(inputs.steps):
            z = random.gauss(0, 1)
            price *= math.exp(drift + diffusion * z)
        if inputs.option_type == "call":
            payoffs.append(max(price - inputs.strike, 0.0))
        else:
            payoffs.append(max(inputs.strike - price, 0.0))
    discounted = math.exp(-inputs.rate * inputs.maturity) * (sum(payoffs) / len(payoffs))
    return discounted


def greeks_fd(inputs: PricingInputs) -> dict[str, float]:
    epsilon = 1e-4 * inputs.spot
    base = black_scholes(inputs)
    up = PricingInputs(
        spot=inputs.spot + epsilon,
        strike=inputs.strike,
        maturity=inputs.maturity,
        rate=inputs.rate,
        volatility=inputs.volatility,
        option_type=inputs.option_type,
        steps=inputs.steps,
        paths=inputs.paths,
    )
    down = PricingInputs(
        spot=inputs.spot - epsilon,
        strike=inputs.strike,
        maturity=inputs.maturity,
        rate=inputs.rate,
        volatility=inputs.volatility,
        option_type=inputs.option_type,
        steps=inputs.steps,
        paths=inputs.paths,
    )
    price_up = black_scholes(up)
    price_down = black_scholes(down)
    delta = (price_up - price_down) / (2 * epsilon)
    gamma = (price_up - 2 * base + price_down) / (epsilon**2)

    vol_bump = 1e-4
    vol_up = PricingInputs(
        spot=inputs.spot,
        strike=inputs.strike,
        maturity=inputs.maturity,
        rate=inputs.rate,
        volatility=inputs.volatility + vol_bump,
        option_type=inputs.option_type,
        steps=inputs.steps,
        paths=inputs.paths,
    )
    vega = (black_scholes(vol_up) - base) / vol_bump

    time_bump = 1e-4
    t_up = PricingInputs(
        spot=inputs.spot,
        strike=inputs.strike,
        maturity=inputs.maturity + time_bump,
        rate=inputs.rate,
        volatility=inputs.volatility,
        option_type=inputs.option_type,
        steps=inputs.steps,
        paths=inputs.paths,
    )
    theta = (black_scholes(t_up) - base) / time_bump

    rate_bump = 1e-4
    r_up = PricingInputs(
        spot=inputs.spot,
        strike=inputs.strike,
        maturity=inputs.maturity,
        rate=inputs.rate + rate_bump,
        volatility=inputs.volatility,
        option_type=inputs.option_type,
        steps=inputs.steps,
        paths=inputs.paths,
    )
    rho = (black_scholes(r_up) - base) / rate_bump

    return {
        "delta": delta,
        "gamma": gamma,
        "vega": vega,
        "theta": theta,
        "rho": rho,
    }
