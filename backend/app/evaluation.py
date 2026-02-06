from __future__ import annotations

from dataclasses import dataclass

from .metrics import mean_absolute_error, root_mean_squared_error


@dataclass
class EvaluationReport:
    rmse: float
    mae: float


def evaluate(y_true: list[float], y_pred: list[float]) -> EvaluationReport:
    return EvaluationReport(
        rmse=root_mean_squared_error(y_true, y_pred),
        mae=mean_absolute_error(y_true, y_pred),
    )
