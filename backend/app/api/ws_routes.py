"""
OptionQuant — WebSocket Routes
═══════════════════════════════════════════════
Endpoints:
  WS /ws/market/{symbol}   — Real-time market data stream with alerts
  WS /ws/pricing           — Real-time pricing stream
  GET /ws/stats             — WebSocket connection stats
"""
from __future__ import annotations

import asyncio
import json
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query

from ..websocket_manager import ws_manager, run_market_stream

logger = logging.getLogger(__name__)
router = APIRouter(tags=["websocket"])


@router.websocket("/ws/market/{symbol}")
async def market_stream_ws(
    websocket: WebSocket,
    symbol: str = "SPY",
    interval: int = Query(1000, ge=200, le=10000),
    mispricing: bool = Query(True),
    regime: bool = Query(True),
):
    """
    Real-time market data stream via WebSocket.
    
    Sends JSON messages with:
    - Market quotes (price, bid, ask, volume)
    - VIX level
    - Mispricing signals (optional)
    - Regime detection (optional)
    - Alerts for significant events
    """
    try:
        await run_market_stream(
            websocket=websocket,
            symbol=symbol,
            interval_ms=interval,
            include_mispricing=mispricing,
            include_regime=regime,
        )
    except WebSocketDisconnect:
        logger.info("Market stream client disconnected: %s", symbol)
    except Exception as e:
        logger.error("Market stream error: %s", e, exc_info=True)


@router.websocket("/ws/pricing")
async def pricing_stream_ws(websocket: WebSocket):
    """
    Interactive pricing WebSocket.
    Client sends pricing parameters, server responds with real-time prices.
    
    Client message format:
    {
        "spot": 100, "strike": 100, "maturity": 1.0,
        "rate": 0.05, "volatility": 0.2, "option_type": "call"
    }
    """
    from ..pricing import PricingInputs, black_scholes, monte_carlo_engine, black_scholes_greeks
    from ..stochastic_vol import HestonParams, heston_mc

    await ws_manager.connect(websocket, channel="pricing")
    try:
        while True:
            data = await websocket.receive_json()

            spot = data.get("spot", 100)
            strike = data.get("strike", 100)
            maturity = data.get("maturity", 1.0)
            rate = data.get("rate", 0.05)
            volatility = data.get("volatility", 0.2)
            option_type = data.get("option_type", "call")

            inputs = PricingInputs(
                spot=spot, strike=strike, maturity=maturity,
                rate=rate, volatility=volatility, option_type=option_type,
            )

            # BS
            bs_price = black_scholes(inputs)
            greeks = black_scholes_greeks(inputs)

            # MC (light)
            mc = await asyncio.to_thread(
                monte_carlo_engine, inputs, 42, "antithetic"
            )

            # Heston
            hparams = HestonParams(
                spot=spot, strike=strike, maturity=maturity, rate=rate,
                v0=volatility**2, kappa=2.0, theta=volatility**2,
                xi=0.3, rho=-0.7, option_type=option_type, paths=10000,
            )
            heston_price = await asyncio.to_thread(heston_mc, hparams, 42)

            response = {
                "type": "pricing_update",
                "inputs": data,
                "black_scholes": round(bs_price, 6),
                "monte_carlo": round(mc.price, 6),
                "mc_std_error": round(mc.std_error, 6),
                "heston": round(heston_price, 6),
                "greeks": greeks,
                "model_agreement": round(1 - abs(bs_price - mc.price) / max(bs_price, 0.01), 4),
            }

            await websocket.send_json(response)

    except WebSocketDisconnect:
        logger.info("Pricing stream client disconnected")
    except Exception as e:
        logger.error("Pricing stream error: %s", e, exc_info=True)
    finally:
        await ws_manager.disconnect(websocket, channel="pricing")


@router.get("/api/v1/ws/stats")
async def websocket_stats():
    """WebSocket connection statistics."""
    return ws_manager.stats
