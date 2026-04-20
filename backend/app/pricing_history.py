"""
OptiQuant — Pricing History (Neon PostgreSQL)
==============================================
Stores every pricing computation for analytics, auditing, and
model performance tracking.
"""
from __future__ import annotations

import json
import logging
from typing import Any

from .database import get_cursor

logger = logging.getLogger(__name__)


def save_pricing_result(
    user_id: int | None,
    model: str,
    option_type: str,
    spot: float,
    strike: float,
    expiry: float,
    rate: float,
    volatility: float,
    computed_price: float,
    greeks: dict[str, Any] | None = None,
) -> None:
    """Insert a pricing result into the pricing_history table."""
    try:
        with get_cursor() as cur:
            cur.execute(
                "INSERT INTO pricing_history "
                "(user_id, model, option_type, spot, strike, expiry, rate, volatility, computed_price, greeks) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                (
                    user_id,
                    model,
                    option_type,
                    spot,
                    strike,
                    expiry,
                    rate,
                    volatility,
                    computed_price,
                    json.dumps(greeks or {}),
                ),
            )
    except Exception as exc:
        logger.warning("Failed to save pricing history: %s", exc)


def get_pricing_history(
    user_id: int | None = None,
    model: str | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Retrieve recent pricing history with optional filters."""
    conditions: list[str] = []
    params: list[Any] = []
    if user_id is not None:
        conditions.append("user_id = %s")
        params.append(user_id)
    if model is not None:
        conditions.append("model = %s")
        params.append(model)

    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    params.append(limit)

    with get_cursor() as cur:
        cur.execute(
            f"SELECT * FROM pricing_history {where} ORDER BY created_at DESC LIMIT %s",
            tuple(params),
        )
        rows = cur.fetchall()
    return [dict(r) for r in rows]
