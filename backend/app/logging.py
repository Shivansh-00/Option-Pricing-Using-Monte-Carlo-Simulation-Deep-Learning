from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class LogEvent:
    event: str
    payload: dict
    timestamp: str


def log_event(event: str, payload: dict, log_path: str | Path = "backend/data/processed/events.log") -> None:
    record = LogEvent(event=event, payload=payload, timestamp=datetime.now(timezone.utc).isoformat())
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(record.__dict__) + "\n")
