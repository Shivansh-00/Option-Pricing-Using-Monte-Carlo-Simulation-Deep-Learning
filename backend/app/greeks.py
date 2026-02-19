from __future__ import annotations

from dataclasses import dataclass, replace

from .pricing import PricingInputs, black_scholes


@dataclass
class Greeks:
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float


def compute_greeks(inputs: PricingInputs) -> Greeks:
    base = black_scholes(inputs)
    epsilon = 1e-4 * inputs.spot
    price_up = black_scholes(replace(inputs, spot=inputs.spot + epsilon))
    price_down = black_scholes(replace(inputs, spot=inputs.spot - epsilon))
    delta = (price_up - price_down) / (2 * epsilon)
    gamma = (price_up - 2 * base + price_down) / (epsilon**2)

    vol_bump = 1e-4
    vega = (black_scholes(replace(inputs, volatility=inputs.volatility + vol_bump)) - base) / vol_bump

    time_bump = 1e-4
    # Backward bump: theta = (price at shorter maturity - base) / bump  → negative for time decay
    theta = (black_scholes(replace(inputs, maturity=inputs.maturity - time_bump)) - base) / time_bump

    rate_bump = 1e-4
    rho = (black_scholes(replace(inputs, rate=inputs.rate + rate_bump)) - base) / rate_bump

    return Greeks(delta=delta, gamma=gamma, vega=vega, theta=theta, rho=rho)
