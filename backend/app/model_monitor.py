from __future__ import annotations

from dataclasses import dataclass

from .logging import log_event


@dataclass
class MonitorReport:
    model: str
    drift_score: float
    status: str


def track_prediction(model: str, y_true: float, y_pred: float) -> MonitorReport:
    error = abs(y_true - y_pred)
    drift_score = min(1.0, error)
    status = "stable" if drift_score < 0.2 else "watch"
    log_event("prediction", {"model": model, "error": error, "drift_score": drift_score})
    return MonitorReport(model=model, drift_score=drift_score, status=status)
