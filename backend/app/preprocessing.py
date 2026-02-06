from __future__ import annotations

from dataclasses import dataclass

from .feature_engineering import FeatureRow


@dataclass
class NormalizedRow:
    date: str
    log_return: float
    realized_vol: float
    vix: float
    rate: float
    volume: float


def normalize(rows: list[FeatureRow]) -> list[NormalizedRow]:
    if not rows:
        return []
    max_vol = max(row.realized_vol for row in rows) or 1.0
    max_vix = max(row.vix for row in rows) or 1.0
    max_volume = max(row.volume for row in rows) or 1.0
    normalized: list[NormalizedRow] = []
    for row in rows:
        normalized.append(
            NormalizedRow(
                date=row.date,
                log_return=row.log_return,
                realized_vol=row.realized_vol / max_vol,
                vix=row.vix / max_vix,
                rate=row.rate,
                volume=row.volume / max_volume,
            )
        )
    return normalized


def sliding_windows(rows: list[NormalizedRow], window: int = 30) -> list[list[NormalizedRow]]:
    return [rows[i : i + window] for i in range(0, max(0, len(rows) - window + 1))]
