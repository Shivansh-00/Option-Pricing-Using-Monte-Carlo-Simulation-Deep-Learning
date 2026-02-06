from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, pstdev

from .data_loader import MarketRow


@dataclass
class FeatureRow:
    date: str
    log_return: float
    realized_vol: float
    vix: float
    rate: float
    volume: float


def compute_features(rows: list[MarketRow], window: int = 20) -> list[FeatureRow]:
    features: list[FeatureRow] = []
    closes = [row.spot for row in rows]
    for idx in range(1, len(rows)):
        log_return = 0.0
        if closes[idx - 1] > 0:
            log_return = (closes[idx] / closes[idx - 1]) - 1
        start = max(0, idx - window)
        window_returns = []
        for j in range(start + 1, idx + 1):
            if closes[j - 1] > 0:
                window_returns.append((closes[j] / closes[j - 1]) - 1)
        realized_vol = pstdev(window_returns) if len(window_returns) > 1 else 0.0
        features.append(
            FeatureRow(
                date=rows[idx].date,
                log_return=log_return,
                realized_vol=realized_vol,
                vix=rows[idx].vix,
                rate=rows[idx].rate,
                volume=rows[idx].volume,
            )
        )
    return features


def summarize_features(rows: list[FeatureRow]) -> dict:
    if not rows:
        return {"mean_return": 0.0, "mean_vol": 0.0}
    return {
        "mean_return": mean([r.log_return for r in rows]),
        "mean_vol": mean([r.realized_vol for r in rows]),
    }
