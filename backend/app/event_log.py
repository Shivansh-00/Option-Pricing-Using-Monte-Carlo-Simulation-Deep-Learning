from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class LogEvent:
    event: str
    payload: dict
    timestamp: str


def log_event(event: str, payload: dict, log_path: str | Path | None = None) -> None:
    """Log an event to the Neon PostgreSQL database.

    Falls back to file-based logging if the database is unavailable
    (e.g. during initial setup before DATABASE_URL is configured).
    """
    try:
        from .database import get_cursor
        with get_cursor() as cur:
            cur.execute(
                "INSERT INTO event_log (event, payload) VALUES (%s, %s)",
                (event, json.dumps(payload)),
            )
        return
    except Exception as exc:
        logger.debug("DB event logging unavailable, falling back to file: %s", exc)

    # File-based fallback
    if log_path is None:
        log_path = Path(__file__).resolve().parents[1] / "data" / "processed" / "events.log"
    record = LogEvent(event=event, payload=payload, timestamp=datetime.now(timezone.utc).isoformat())
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(record.__dict__) + "\n")


def get_recent_events(limit: int = 50, event_type: str | None = None) -> list[dict[str, Any]]:
    """Retrieve recent events from the database."""
    from .database import get_cursor
    with get_cursor() as cur:
        if event_type:
            cur.execute(
                "SELECT id, event, payload, created_at FROM event_log "
                "WHERE event = %s ORDER BY created_at DESC LIMIT %s",
                (event_type, limit),
            )
        else:
            cur.execute(
                "SELECT id, event, payload, created_at FROM event_log "
                "ORDER BY created_at DESC LIMIT %s",
                (limit,),
            )
        rows = cur.fetchall()
    return [
        {
            "id": r["id"],
            "event": r["event"],
            "payload": r["payload"],
            "created_at": str(r["created_at"]),
        }
        for r in rows
    ]
