"""
OptionQuant — WebSocket Manager & Alert System
═══════════════════════════════════════════════════════════════
Real-time streaming infrastructure:

  • WebSocket connection manager (multi-client)
  • Market data streaming with mispricing alerts
  • Auto-alert when deviation > configurable threshold
  • Portfolio-level exposure notifications
  • Color-coded risk level broadcasting
  • Heartbeat / keepalive
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Optional, Any

from fastapi import WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
#  Data Models
# ═══════════════════════════════════════════════════════════════

@dataclass
class Alert:
    alert_id: str
    alert_type: str        # mispricing / regime_change / risk_limit / parity_violation
    severity: str          # info / warning / critical
    title: str
    message: str
    data: dict = field(default_factory=dict)
    timestamp: str = ""
    acknowledged: bool = False

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()


@dataclass
class AlertConfig:
    mispricing_threshold_pct: float = 5.0
    risk_var_threshold: float = 10000.0
    regime_change_enabled: bool = True
    parity_violation_enabled: bool = True
    max_alerts_per_minute: int = 30


# ═══════════════════════════════════════════════════════════════
#  WebSocket Connection Manager
# ═══════════════════════════════════════════════════════════════

class ConnectionManager:
    """Manages WebSocket connections with channel support."""

    def __init__(self):
        self._connections: dict[str, list[WebSocket]] = defaultdict(list)
        self._lock = asyncio.Lock()
        self._stats = {
            "total_connections": 0,
            "total_messages": 0,
            "total_disconnections": 0,
        }

    async def connect(self, websocket: WebSocket, channel: str = "default"):
        await websocket.accept()
        async with self._lock:
            self._connections[channel].append(websocket)
            self._stats["total_connections"] += 1
        logger.info("WebSocket connected to channel: %s (total: %d)",
                     channel, len(self._connections[channel]))

    async def disconnect(self, websocket: WebSocket, channel: str = "default"):
        async with self._lock:
            if websocket in self._connections[channel]:
                self._connections[channel].remove(websocket)
                self._stats["total_disconnections"] += 1
        logger.info("WebSocket disconnected from channel: %s", channel)

    async def broadcast(self, message: dict, channel: str = "default"):
        """Send message to all clients on a channel."""
        async with self._lock:
            connections = list(self._connections.get(channel, []))

        dead = []
        for ws in connections:
            try:
                await ws.send_json(message)
                self._stats["total_messages"] += 1
            except Exception:
                dead.append(ws)

        # Clean dead connections
        if dead:
            async with self._lock:
                for ws in dead:
                    if ws in self._connections[channel]:
                        self._connections[channel].remove(ws)

    async def send_personal(self, websocket: WebSocket, message: dict):
        try:
            await websocket.send_json(message)
            self._stats["total_messages"] += 1
        except Exception:
            pass

    @property
    def stats(self) -> dict:
        return {
            **self._stats,
            "active_connections": sum(len(v) for v in self._connections.values()),
            "channels": {k: len(v) for k, v in self._connections.items()},
        }


# Global connection manager
ws_manager = ConnectionManager()


# ═══════════════════════════════════════════════════════════════
#  Alert Engine
# ═══════════════════════════════════════════════════════════════

class AlertEngine:
    """Generates and manages alerts based on market conditions."""

    def __init__(self, config: Optional[AlertConfig] = None):
        self.config = config or AlertConfig()
        self._alerts: list[Alert] = []
        self._alert_count_window: list[float] = []
        self._alert_counter = 0
        self._max_history = 500

    def _rate_ok(self) -> bool:
        now = time.time()
        self._alert_count_window = [t for t in self._alert_count_window if now - t < 60]
        return len(self._alert_count_window) < self.config.max_alerts_per_minute

    def check_mispricing(self, signal: dict) -> Optional[Alert]:
        """Check if mispricing signal warrants an alert."""
        dev = abs(signal.get("deviation_pct", 0))
        if dev < self.config.mispricing_threshold_pct:
            return None
        if not self._rate_ok():
            return None

        self._alert_counter += 1
        severity = "critical" if dev > 10 else "warning"
        direction = signal.get("direction", "unknown")

        alert = Alert(
            alert_id=f"MISPRICE-{self._alert_counter:06d}",
            alert_type="mispricing",
            severity=severity,
            title=f"Mispricing Detected: K={signal.get('strike', '?')} {signal.get('option_type', '?')}",
            message=(
                f"{direction.upper()} by {dev:.1f}% "
                f"(model=${signal.get('model_price', 0):.2f} vs "
                f"market=${signal.get('market_price', 0):.2f}). "
                f"Signal strength: {signal.get('signal_strength', 0):.2f}"
            ),
            data=signal,
        )
        self._alerts.append(alert)
        self._alert_count_window.append(time.time())

        if len(self._alerts) > self._max_history:
            self._alerts = self._alerts[-self._max_history:]

        return alert

    def check_regime_change(self, old_regime: str, new_regime: str) -> Optional[Alert]:
        """Alert on regime transition."""
        if not self.config.regime_change_enabled or old_regime == new_regime:
            return None
        if not self._rate_ok():
            return None

        self._alert_counter += 1
        severity = "critical" if new_regime in ("high_vol", "bear") else "info"

        alert = Alert(
            alert_id=f"REGIME-{self._alert_counter:06d}",
            alert_type="regime_change",
            severity=severity,
            title=f"Regime Change: {old_regime} → {new_regime}",
            message=f"Market regime shifted from {old_regime} to {new_regime}. Model parameters adjusted.",
            data={"old_regime": old_regime, "new_regime": new_regime},
        )
        self._alerts.append(alert)
        self._alert_count_window.append(time.time())
        return alert

    def check_parity_violation(self, arb: dict) -> Optional[Alert]:
        """Alert on arbitrage opportunity."""
        if not self.config.parity_violation_enabled:
            return None
        if not self._rate_ok():
            return None

        self._alert_counter += 1
        alert = Alert(
            alert_id=f"ARB-{self._alert_counter:06d}",
            alert_type="parity_violation",
            severity="critical",
            title=f"Arbitrage: {arb.get('arb_type', 'unknown')}",
            message=arb.get("description", "Arbitrage opportunity detected"),
            data=arb,
        )
        self._alerts.append(alert)
        self._alert_count_window.append(time.time())
        return alert

    def get_recent_alerts(self, limit: int = 50) -> list[dict]:
        return [asdict(a) for a in reversed(self._alerts[-limit:])]

    def acknowledge(self, alert_id: str) -> bool:
        for a in self._alerts:
            if a.alert_id == alert_id:
                a.acknowledged = True
                return True
        return False

    @property
    def stats(self) -> dict:
        return {
            "total_alerts": len(self._alerts),
            "unacknowledged": sum(1 for a in self._alerts if not a.acknowledged),
            "by_type": {
                "mispricing": sum(1 for a in self._alerts if a.alert_type == "mispricing"),
                "regime_change": sum(1 for a in self._alerts if a.alert_type == "regime_change"),
                "parity_violation": sum(1 for a in self._alerts if a.alert_type == "parity_violation"),
            },
            "by_severity": {
                "info": sum(1 for a in self._alerts if a.severity == "info"),
                "warning": sum(1 for a in self._alerts if a.severity == "warning"),
                "critical": sum(1 for a in self._alerts if a.severity == "critical"),
            },
        }


# Global alert engine
alert_engine = AlertEngine()


# ═══════════════════════════════════════════════════════════════
#  Streaming Orchestrator
# ═══════════════════════════════════════════════════════════════

async def run_market_stream(
    websocket: WebSocket,
    symbol: str = "SPY",
    interval_ms: int = 1000,
    include_mispricing: bool = True,
    include_regime: bool = True,
):
    """
    Full streaming pipeline: market data → pricing → mispricing → alerts.
    """
    from .market_data import get_generator
    from .mispricing import detect_mispricing
    from .regime import get_regime_detector

    await ws_manager.connect(websocket, channel=f"market:{symbol}")
    gen = get_generator(symbol)
    detector = get_regime_detector()
    prev_regime = "unknown"
    prev_price = None
    tick = 0

    try:
        while True:
            snap = gen.tick()
            tick += 1

            # Base market data message
            msg: dict[str, Any] = {
                "type": "market_update",
                "tick": tick,
                "symbol": symbol,
                "price": snap.quote.price,
                "bid": snap.quote.bid,
                "ask": snap.quote.ask,
                "volume": snap.quote.volume,
                "vix": snap.vix,
                "timestamp": snap.quote.timestamp,
            }

            # Regime detection using actual price changes
            if include_regime and tick > 1 and prev_price is not None and prev_price > 0:
                daily_return = (snap.quote.price - prev_price) / prev_price
                regime = detector.update(daily_return, snap.vix)
                msg["regime"] = {
                    "label": regime.label,
                    "probability": regime.probability,
                    "risk_level": regime.risk_level,
                    "recommended_model": regime.recommended_model,
                    "vol_adjustment": regime.vol_adjustment,
                }

                if regime.label != prev_regime and prev_regime != "unknown":
                    alert = alert_engine.check_regime_change(prev_regime, regime.label)
                    if alert:
                        msg["alert"] = asdict(alert)
                prev_regime = regime.label

            # Mispricing check on a sample ATM option
            if include_mispricing and snap.chain.calls:
                atm_call = min(snap.chain.calls,
                               key=lambda c: abs(c.strike - snap.quote.price))
                # Parse expiry for accurate maturity
                try:
                    from datetime import datetime as _dt
                    _exp = _dt.strptime(atm_call.expiry, "%Y-%m-%d")
                    _mat = max(1, (_exp - _dt.now()).days) / 365.0
                except (ValueError, TypeError, AttributeError):
                    _mat = 30 / 365.0
                sig = detect_mispricing(
                    spot=snap.quote.price,
                    strike=atm_call.strike,
                    maturity=_mat,
                    rate=0.05,
                    volatility=atm_call.implied_vol,
                    option_type="call",
                    market_price=atm_call.mid,
                    bid=atm_call.bid,
                    ask=atm_call.ask,
                )
                msg["mispricing"] = {
                    "strike": sig.strike,
                    "deviation_pct": sig.deviation_pct,
                    "direction": sig.direction,
                    "signal_strength": sig.signal_strength,
                    "z_score": sig.z_score,
                    "is_significant": sig.is_significant,
                }

                if sig.is_significant:
                    alert = alert_engine.check_mispricing({
                        "strike": sig.strike,
                        "option_type": sig.option_type,
                        "deviation_pct": sig.deviation_pct,
                        "direction": sig.direction,
                        "model_price": sig.model_price,
                        "market_price": sig.market_price,
                        "signal_strength": sig.signal_strength,
                    })
                    if alert:
                        msg["alert"] = asdict(alert)

            await websocket.send_json(msg)
            prev_price = snap.quote.price
            await asyncio.sleep(interval_ms / 1000.0)

    except WebSocketDisconnect:
        logger.info("Client disconnected from market stream: %s", symbol)
    except Exception as e:
        logger.error("Stream error: %s", e)
    finally:
        await ws_manager.disconnect(websocket, channel=f"market:{symbol}")


# Need this import at module level for the stream function
import numpy as np
