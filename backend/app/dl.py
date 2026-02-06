from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DLForecast:
    forecast_price: float
    forecast_vol: float
    model: str
    residual: float


def residual_learning(price: float, mc_price: float) -> DLForecast:
    residual = price - mc_price
    return DLForecast(
        forecast_price=price,
        forecast_vol=max(0.01, abs(residual) * 0.1),
        model="hybrid-residual",
        residual=residual,
    )
