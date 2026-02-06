from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass
class MarketRow:
    date: str
    spot: float
    rate: float
    vix: float
    volume: float


def load_csv(path: str | Path) -> list[MarketRow]:
    rows: list[MarketRow] = []
    with Path(path).open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                MarketRow(
                    date=row.get("date", ""),
                    spot=float(row.get("spot", 0.0)),
                    rate=float(row.get("rate", 0.0)),
                    vix=float(row.get("vix", 0.0)),
                    volume=float(row.get("volume", 0.0)),
                )
            )
    return rows


def stream_directory(directory: str | Path) -> Iterable[MarketRow]:
    for csv_file in Path(directory).glob("*.csv"):
        yield from load_csv(csv_file)
