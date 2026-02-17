from __future__ import annotations

import math

from .schemas import VolatilityRequest


REGIMES = (
    "low-vol",
    "risk-on",
    "risk-off",
    "high-vol",
)


def predict_iv(request: VolatilityRequest) -> tuple[float, str, dict]:
    vix_decimal = request.vix / 100          # VIX points → decimal (e.g. 20 → 0.20)
    base = 0.4 * request.realized_vol + 0.4 * vix_decimal + 0.2 * abs(request.skew)
    implied = max(0.05, min(2.5, base))
    score = math.tanh(request.vix / 25)      # scale so VIX≈25 → tanh≈0.76
    if score < 0.3:
        regime = REGIMES[0]
    elif score < 0.6:
        regime = REGIMES[1]
    elif score < 0.9:
        regime = REGIMES[2]
    else:
        regime = REGIMES[3]
    drivers = {
        "realized_vol_weight": 0.4,
        "vix_weight": 0.4,
        "skew_weight": 0.2,
    }
    return implied, regime, drivers
