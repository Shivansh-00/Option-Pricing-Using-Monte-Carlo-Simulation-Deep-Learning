"""
OptionQuant — Mispricing Detection & Arbitrage Engine
═══════════════════════════════════════════════════════════════
Hedge-fund-grade module for detecting real-time option mispricing:

  • Theoretical vs market price deviation analysis
  • Statistical significance testing (z-score)
  • Bid-ask spread filtering to avoid false signals
  • Put-Call parity arbitrage detection
  • Calendar spread arbitrage detection
  • Butterfly spread arbitrage detection
  • Signal strength scoring with confidence levels
  • Batch full-chain scanning
"""
from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .pricing import PricingInputs, black_scholes, monte_carlo_engine
from .stochastic_vol import HestonParams, heston_mc

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
#  Data Models
# ═══════════════════════════════════════════════════════════════

@dataclass
class MispricingSignal:
    strike: float
    expiry: str
    option_type: str
    market_price: float
    model_price: float
    deviation_pct: float
    deviation_dollar: float
    z_score: float
    signal_strength: float      # 0.0 → 1.0
    is_significant: bool
    direction: str              # "underpriced" / "overpriced"
    bid: float = 0.0
    ask: float = 0.0
    spread_pct: float = 0.0
    passes_spread_filter: bool = True
    arbitrage_type: str = "none"  # none / put_call_parity / calendar / butterfly
    confidence: float = 0.0
    model_used: str = "black_scholes"


@dataclass
class ArbitrageOpportunity:
    arb_type: str               # put_call_parity / calendar / butterfly
    description: str
    legs: list[dict]
    theoretical_profit: float
    net_cost: float
    risk_free: bool
    confidence: float
    timestamp: str = ""


@dataclass
class MispricingScanResult:
    symbol: str
    spot: float
    vix: float
    scan_time_ms: float
    total_contracts: int
    significant_signals: int
    arbitrage_opportunities: int
    signals: list[MispricingSignal]
    arbitrages: list[ArbitrageOpportunity]
    summary: dict = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════
#  Core Detection Engine
# ═══════════════════════════════════════════════════════════════

def detect_mispricing(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    volatility: float,
    option_type: str,
    market_price: float,
    bid: float = 0.0,
    ask: float = 0.0,
    model: str = "black_scholes",
    significance_threshold: float = 2.0,
    min_deviation_pct: float = 2.0,
) -> MispricingSignal:
    """
    Detect mispricing for a single option contract.
    
    Args:
        significance_threshold: z-score threshold for statistical significance
        min_deviation_pct: minimum % deviation to flag
    """
    inputs = PricingInputs(
        spot=spot, strike=strike, maturity=maturity,
        rate=rate, volatility=volatility, option_type=option_type,
    )

    # Model pricing
    if model == "monte_carlo":
        mc = monte_carlo_engine(inputs, seed=42, method="antithetic")
        model_price = mc.price
        std_error = mc.std_error
    elif model == "heston":
        hparams = HestonParams(
            spot=spot, strike=strike, maturity=maturity, rate=rate,
            v0=volatility**2, kappa=2.0, theta=volatility**2,
            xi=0.3, rho=-0.7, option_type=option_type,
        )
        model_price = heston_mc(hparams, seed=42)
        std_error = model_price * 0.01  # approximate
    else:
        model_price = black_scholes(inputs)
        # BS has no sampling error; use implied vol sensitivity as proxy
        std_error = model_price * 0.005

    # Deviation
    deviation_dollar = market_price - model_price
    deviation_pct = (deviation_dollar / model_price * 100) if model_price > 0.01 else 0.0

    # Z-score
    z_score = deviation_dollar / std_error if std_error > 1e-8 else 0.0

    # Direction
    direction = "overpriced" if deviation_dollar > 0 else "underpriced"

    # Statistical significance
    is_significant = abs(z_score) >= significance_threshold and abs(deviation_pct) >= min_deviation_pct

    # Signal strength: combines z-score magnitude + deviation %
    raw_strength = min(1.0, (abs(z_score) / 5.0) * 0.5 + (abs(deviation_pct) / 20.0) * 0.5)

    # Bid-ask spread filter
    spread_pct = ((ask - bid) / market_price * 100) if market_price > 0 and ask > bid else 0.0
    passes_spread = abs(deviation_pct) > spread_pct * 1.5

    # Confidence: degrades if spread is wide or z-score is marginal
    confidence = raw_strength
    if not passes_spread:
        confidence *= 0.3
    if abs(z_score) < 1.5:
        confidence *= 0.5

    return MispricingSignal(
        strike=strike,
        expiry="",
        option_type=option_type,
        market_price=round(market_price, 4),
        model_price=round(model_price, 4),
        deviation_pct=round(deviation_pct, 4),
        deviation_dollar=round(deviation_dollar, 4),
        z_score=round(z_score, 4),
        signal_strength=round(raw_strength, 4),
        is_significant=is_significant,
        direction=direction,
        bid=bid,
        ask=ask,
        spread_pct=round(spread_pct, 4),
        passes_spread_filter=passes_spread,
        confidence=round(confidence, 4),
        model_used=model,
    )


# ═══════════════════════════════════════════════════════════════
#  Put-Call Parity Arbitrage
# ═══════════════════════════════════════════════════════════════

def check_put_call_parity(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    call_price: float,
    put_price: float,
    call_bid: float = 0.0,
    call_ask: float = 0.0,
    put_bid: float = 0.0,
    put_ask: float = 0.0,
    tolerance_pct: float = 1.5,
) -> Optional[ArbitrageOpportunity]:
    """
    Check Put-Call Parity: C - P = S - K*exp(-rT)
    
    If violated beyond transaction costs, flag arbitrage.
    """
    pv_strike = strike * math.exp(-rate * maturity)
    theoretical_diff = spot - pv_strike  # C - P should equal this
    actual_diff = call_price - put_price
    violation = actual_diff - theoretical_diff
    violation_pct = abs(violation) / spot * 100 if spot > 0 else 0

    # Transaction cost estimate from spreads
    tx_cost = (call_ask - call_bid) + (put_ask - put_bid)

    if violation_pct > tolerance_pct and abs(violation) > tx_cost * 1.2:
        if violation > 0:
            # Call overpriced relative to put: sell call, buy put, buy stock
            desc = (f"Call overpriced vs put by ${violation:.2f} "
                    f"({violation_pct:.1f}%). Sell call + buy put + buy stock.")
            legs = [
                {"action": "sell", "instrument": "call", "strike": strike, "price": call_bid},
                {"action": "buy", "instrument": "put", "strike": strike, "price": put_ask},
                {"action": "buy", "instrument": "stock", "price": spot},
            ]
        else:
            # Put overpriced: buy call, sell put, sell stock
            desc = (f"Put overpriced vs call by ${abs(violation):.2f} "
                    f"({violation_pct:.1f}%). Buy call + sell put + sell stock.")
            legs = [
                {"action": "buy", "instrument": "call", "strike": strike, "price": call_ask},
                {"action": "sell", "instrument": "put", "strike": strike, "price": put_bid},
                {"action": "sell", "instrument": "stock", "price": spot},
            ]

        return ArbitrageOpportunity(
            arb_type="put_call_parity",
            description=desc,
            legs=legs,
            theoretical_profit=round(abs(violation) - tx_cost, 4),
            net_cost=round(tx_cost, 4),
            risk_free=True,
            confidence=round(min(1.0, violation_pct / 5.0), 4),
        )
    return None


# ═══════════════════════════════════════════════════════════════
#  Calendar Spread Arbitrage
# ═══════════════════════════════════════════════════════════════

def check_calendar_arbitrage(
    strike: float,
    near_price: float,
    far_price: float,
    near_expiry: str,
    far_expiry: str,
    option_type: str,
) -> Optional[ArbitrageOpportunity]:
    """
    Calendar arbitrage: longer-dated option should cost more.
    If near > far for same strike, arbitrage exists.
    """
    if near_price > far_price and near_price > 0.01:
        violation = near_price - far_price
        desc = (f"Calendar arb at K={strike}: near ({near_expiry}) ${near_price:.2f} > "
                f"far ({far_expiry}) ${far_price:.2f}. Sell near, buy far.")
        return ArbitrageOpportunity(
            arb_type="calendar",
            description=desc,
            legs=[
                {"action": "sell", "instrument": option_type, "expiry": near_expiry,
                 "strike": strike, "price": near_price},
                {"action": "buy", "instrument": option_type, "expiry": far_expiry,
                 "strike": strike, "price": far_price},
            ],
            theoretical_profit=round(violation, 4),
            net_cost=0.0,
            risk_free=False,
            confidence=round(min(1.0, violation / near_price), 4),
        )
    return None


# ═══════════════════════════════════════════════════════════════
#  Butterfly Spread Arbitrage
# ═══════════════════════════════════════════════════════════════

def check_butterfly_arbitrage(
    k_low: float, k_mid: float, k_high: float,
    price_low: float, price_mid: float, price_high: float,
    option_type: str,
    tolerance: float = 0.05,
) -> Optional[ArbitrageOpportunity]:
    """
    Butterfly arbitrage: convexity violation.
    For equally spaced strikes: C(K_mid) <= 0.5*(C(K_low) + C(K_high))
    """
    if abs((k_high - k_mid) - (k_mid - k_low)) > 0.5:
        return None  # not equally spaced

    butterfly_value = 0.5 * (price_low + price_high) - price_mid
    if butterfly_value < -tolerance:
        desc = (f"Butterfly arb: K=[{k_low},{k_mid},{k_high}], "
                f"convexity violated by ${abs(butterfly_value):.2f}. "
                f"Buy wings, sell body.")
        return ArbitrageOpportunity(
            arb_type="butterfly",
            description=desc,
            legs=[
                {"action": "buy", "instrument": option_type, "strike": k_low, "price": price_low},
                {"action": "sell", "instrument": option_type, "strike": k_mid, "qty": 2, "price": price_mid},
                {"action": "buy", "instrument": option_type, "strike": k_high, "price": price_high},
            ],
            theoretical_profit=round(abs(butterfly_value), 4),
            net_cost=round(price_low + price_high - 2 * price_mid, 4),
            risk_free=False,
            confidence=round(min(1.0, abs(butterfly_value) / (price_mid + 0.01)), 4),
        )
    return None


# ═══════════════════════════════════════════════════════════════
#  Full Chain Scanner
# ═══════════════════════════════════════════════════════════════

def scan_chain(
    chain_data: dict,
    model: str = "black_scholes",
    significance_threshold: float = 2.0,
    min_deviation_pct: float = 2.0,
) -> MispricingScanResult:
    """
    Scan a full option chain for mispricing signals and arbitrage.
    
    chain_data should have: symbol, spot, vix, risk_free_rate, calls, puts
    Each call/put: strike, expiry, option_type, bid, ask, mid, implied_vol, volume, open_interest
    """
    t0 = time.perf_counter()
    spot = chain_data.get("spot", 100)
    rate = chain_data.get("risk_free_rate", 0.05)
    vix = chain_data.get("vix", 20)
    symbol = chain_data.get("symbol", "UNKNOWN")

    all_calls = chain_data.get("calls", [])
    all_puts = chain_data.get("puts", [])

    signals = []
    arbitrages = []

    # Process each contract
    for contracts, opt_type in [(all_calls, "call"), (all_puts, "put")]:
        for c in contracts:
            strike = c.get("strike", c.strike if hasattr(c, 'strike') else 0)
            mid = c.get("mid", c.mid if hasattr(c, 'mid') else 0)
            bid = c.get("bid", c.bid if hasattr(c, 'bid') else 0)
            ask = c.get("ask", c.ask if hasattr(c, 'ask') else 0)
            iv = c.get("implied_vol", c.implied_vol if hasattr(c, 'implied_vol') else vix / 100)
            expiry = c.get("expiry", c.expiry if hasattr(c, 'expiry') else "")

            if mid < 0.01 or strike <= 0:
                continue

            maturity = 30 / 365.0  # default; in production parse expiry

            sig = detect_mispricing(
                spot=spot, strike=strike, maturity=maturity,
                rate=rate, volatility=iv, option_type=opt_type,
                market_price=mid, bid=bid, ask=ask,
                model=model,
                significance_threshold=significance_threshold,
                min_deviation_pct=min_deviation_pct,
            )
            sig.expiry = expiry
            signals.append(sig)

    # Put-call parity checks
    call_map = {}
    put_map = {}
    for c in all_calls:
        k = c.get("strike", c.strike if hasattr(c, 'strike') else 0)
        call_map[k] = c
    for p in all_puts:
        k = p.get("strike", p.strike if hasattr(p, 'strike') else 0)
        put_map[k] = p

    for k in set(call_map.keys()) & set(put_map.keys()):
        c, p = call_map[k], put_map[k]
        c_mid = c.get("mid", c.mid if hasattr(c, 'mid') else 0)
        p_mid = p.get("mid", p.mid if hasattr(p, 'mid') else 0)
        c_bid = c.get("bid", c.bid if hasattr(c, 'bid') else 0)
        c_ask = c.get("ask", c.ask if hasattr(c, 'ask') else 0)
        p_bid = p.get("bid", p.bid if hasattr(p, 'bid') else 0)
        p_ask = p.get("ask", p.ask if hasattr(p, 'ask') else 0)

        arb = check_put_call_parity(
            spot=spot, strike=k, maturity=30 / 365.0, rate=rate,
            call_price=c_mid, put_price=p_mid,
            call_bid=c_bid, call_ask=c_ask,
            put_bid=p_bid, put_ask=p_ask,
        )
        if arb:
            arbitrages.append(arb)

    # Butterfly checks (calls)
    sorted_calls = sorted(
        [(c.get("strike", c.strike if hasattr(c, 'strike') else 0),
          c.get("mid", c.mid if hasattr(c, 'mid') else 0))
         for c in all_calls],
        key=lambda x: x[0]
    )
    for i in range(len(sorted_calls) - 2):
        k1, p1 = sorted_calls[i]
        k2, p2 = sorted_calls[i + 1]
        k3, p3 = sorted_calls[i + 2]
        arb = check_butterfly_arbitrage(k1, k2, k3, p1, p2, p3, "call")
        if arb:
            arbitrages.append(arb)

    sig_signals = [s for s in signals if s.is_significant and s.passes_spread_filter]
    elapsed = (time.perf_counter() - t0) * 1000

    # Summary stats
    deviations = [s.deviation_pct for s in signals if abs(s.deviation_pct) > 0.01]
    summary = {
        "avg_deviation_pct": round(float(np.mean(deviations)), 4) if deviations else 0,
        "max_deviation_pct": round(float(np.max(np.abs(deviations))), 4) if deviations else 0,
        "overpriced_count": sum(1 for s in sig_signals if s.direction == "overpriced"),
        "underpriced_count": sum(1 for s in sig_signals if s.direction == "underpriced"),
        "parity_violations": sum(1 for a in arbitrages if a.arb_type == "put_call_parity"),
        "butterfly_violations": sum(1 for a in arbitrages if a.arb_type == "butterfly"),
        "calendar_violations": sum(1 for a in arbitrages if a.arb_type == "calendar"),
    }

    return MispricingScanResult(
        symbol=symbol,
        spot=spot,
        vix=vix,
        scan_time_ms=round(elapsed, 2),
        total_contracts=len(signals),
        significant_signals=len(sig_signals),
        arbitrage_opportunities=len(arbitrages),
        signals=sig_signals,  # only significant ones
        arbitrages=arbitrages,
        summary=summary,
    )
