from __future__ import annotations

import math
from typing import Iterable


def mean_squared_error(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    errors = [(a - b) ** 2 for a, b in zip(y_true, y_pred, strict=False)]
    return sum(errors) / len(errors) if errors else 0.0


def root_mean_squared_error(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    return math.sqrt(mean_squared_error(y_true, y_pred))


def mean_absolute_error(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    errors = [abs(a - b) for a, b in zip(y_true, y_pred, strict=False)]
    return sum(errors) / len(errors) if errors else 0.0
