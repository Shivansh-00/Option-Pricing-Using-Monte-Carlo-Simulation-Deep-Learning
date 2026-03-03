"""
Arbitrage Detection Engine
============================
Automatically detects:
    - Put-Call parity violations
    - Calendar spread arbitrage
    - Butterfly spread mispricing
    - Volatility surface inconsistencies
    - Box spread arbitrage

Uses:
    - Statistical thresholds with confidence weighting
    - Integration with vol surface transformer for consistency checks
    - Regime-aware threshold adjustment

Output:
    - Signal strength (0-1)
    - Trade recommendation
    - Risk score
    - Confidence interval

Integrates with:
    - Vol surface model (surface consistency)
    - Risk engine (position sizing)
    - RL hedging (trade signals)
    - Portfolio dashboard (opportunity display)
"""

from __future__ import annotations
import numpy as np
import math
import time
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Any
from enum import Enum

logger = logging.getLogger(__name__)


class ArbitrageType(str, Enum):
    PUT_CALL_PARITY = "put_call_parity"
    CALENDAR_SPREAD = "calendar_spread"
    BUTTERFLY = "butterfly"
    SURFACE_INCONSISTENCY = "surface_inconsistency"
    BOX_SPREAD = "box_spread"
    VERTICAL_SPREAD = "vertical_spread"


class SignalStrength(str, Enum):
    STRONG = "strong"
    MODERATE = "moderate"
    WEAK = "weak"
    NOISE = "noise"


@dataclass
class ArbitrageSignal:
    arb_type: ArbitrageType
    signal_strength: float     # 0.0 to 1.0
    signal_class: SignalStrength
    expected_profit: float     # in price units
    risk_score: float          # 0.0 to 1.0 (0 = low risk)
    confidence: float          # statistical confidence
    description: str
    trade_recommendation: str
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.arb_type.value,
            "signal_strength": round(self.signal_strength, 4),
            "signal_class": self.signal_class.value,
            "expected_profit": round(self.expected_profit, 4),
            "risk_score": round(self.risk_score, 4),
            "confidence": round(self.confidence, 4),
            "description": self.description,
            "trade_recommendation": self.trade_recommendation,
            "details": self.details
        }


@dataclass
class OptionQuote:
    """Represents a single option quote."""
    strike: float
    expiry_days: int
    call_price: float
    put_price: float
    spot: float
    r: float = 0.05
    implied_vol: float = 0.20

    @property
    def tau(self) -> float:
        return self.expiry_days / 365.0


class ArbitrageDetectionEngine:
    """
    Comprehensive arbitrage detection across multiple dimensions.
    
    Configurable thresholds adapt to regime (wider in crisis, tighter in bull).
    """

    def __init__(self, regime: int = 0, transaction_cost_bps: float = 5.0):
        self.regime = regime  # 0=bull, 1=bear, 2=crisis
        self.tc_bps = transaction_cost_bps
        self._regime_multipliers = {0: 1.0, 1: 1.5, 2: 2.5}  # relax thresholds in volatile regimes
        self._signal_history: List[ArbitrageSignal] = []

    def _threshold(self, base: float) -> float:
        """Regime-adjusted threshold."""
        return base * self._regime_multipliers.get(self.regime, 1.0)

    def _tc_cost(self, notional: float) -> float:
        """Transaction cost in price units."""
        return notional * self.tc_bps / 10000

    def _signal_class(self, strength: float) -> SignalStrength:
        if strength >= 0.75:
            return SignalStrength.STRONG
        elif strength >= 0.5:
            return SignalStrength.MODERATE
        elif strength >= 0.25:
            return SignalStrength.WEAK
        return SignalStrength.NOISE

    # ── Put-Call Parity ──
    def check_put_call_parity(self, quotes: List[OptionQuote]) -> List[ArbitrageSignal]:
        """
        Put-Call Parity: C - P = S - K*exp(-rτ)
        Violation indicates riskless arbitrage opportunity.
        """
        signals = []
        threshold = self._threshold(0.50)

        for q in quotes:
            parity_lhs = q.call_price - q.put_price
            parity_rhs = q.spot - q.strike * math.exp(-q.r * q.tau)
            deviation = parity_lhs - parity_rhs
            abs_dev = abs(deviation)

            tc = self._tc_cost(q.spot) * 4  # 4 legs
            net_profit = abs_dev - tc

            if net_profit > threshold:
                strength = min(net_profit / (threshold * 4), 1.0)
                confidence = 1.0 - math.exp(-abs_dev / (q.spot * 0.01))

                if deviation > 0:
                    trade = f"Sell call, buy put, buy stock, borrow at K={q.strike}"
                else:
                    trade = f"Buy call, sell put, short stock, lend at K={q.strike}"

                signals.append(ArbitrageSignal(
                    arb_type=ArbitrageType.PUT_CALL_PARITY,
                    signal_strength=strength,
                    signal_class=self._signal_class(strength),
                    expected_profit=net_profit,
                    risk_score=0.1,  # near riskless
                    confidence=confidence,
                    description=f"Put-call parity violation: deviation={abs_dev:.4f}, "
                                f"net profit after costs={net_profit:.4f}",
                    trade_recommendation=trade,
                    details={
                        "strike": q.strike,
                        "expiry_days": q.expiry_days,
                        "call_price": q.call_price,
                        "put_price": q.put_price,
                        "parity_lhs": round(parity_lhs, 4),
                        "parity_rhs": round(parity_rhs, 4),
                        "deviation": round(deviation, 4),
                        "transaction_cost": round(tc, 4),
                    }
                ))

        return signals

    # ── Calendar Spread Arbitrage ──
    def check_calendar_spread(self, quotes: List[OptionQuote]) -> List[ArbitrageSignal]:
        """
        Calendar spread: option with longer maturity should be worth more
        (for same strike, same type). Violation = arbitrage.
        """
        signals = []
        threshold = self._threshold(0.30)

        # Group by strike
        by_strike: Dict[float, List[OptionQuote]] = {}
        for q in quotes:
            by_strike.setdefault(q.strike, []).append(q)

        for strike, strike_quotes in by_strike.items():
            sorted_q = sorted(strike_quotes, key=lambda x: x.expiry_days)
            for i in range(len(sorted_q) - 1):
                near = sorted_q[i]
                far = sorted_q[i + 1]

                # Call calendar
                if far.call_price < near.call_price:
                    dev = near.call_price - far.call_price
                    tc = self._tc_cost(near.spot) * 2
                    net = dev - tc
                    if net > threshold:
                        strength = min(net / (threshold * 3), 1.0)
                        signals.append(ArbitrageSignal(
                            arb_type=ArbitrageType.CALENDAR_SPREAD,
                            signal_strength=strength,
                            signal_class=self._signal_class(strength),
                            expected_profit=net,
                            risk_score=0.25,
                            confidence=min(0.95, strength),
                            description=f"Call calendar arbitrage at K={strike}: "
                                        f"near({near.expiry_days}d)={near.call_price:.2f} > "
                                        f"far({far.expiry_days}d)={far.call_price:.2f}",
                            trade_recommendation=f"Sell {near.expiry_days}d call, buy {far.expiry_days}d call at K={strike}",
                            details={
                                "strike": strike,
                                "near_expiry": near.expiry_days,
                                "far_expiry": far.expiry_days,
                                "near_call": near.call_price,
                                "far_call": far.call_price,
                                "deviation": round(dev, 4),
                            }
                        ))

                # Put calendar
                if far.put_price < near.put_price:
                    dev = near.put_price - far.put_price
                    tc = self._tc_cost(near.spot) * 2
                    net = dev - tc
                    if net > threshold:
                        strength = min(net / (threshold * 3), 1.0)
                        signals.append(ArbitrageSignal(
                            arb_type=ArbitrageType.CALENDAR_SPREAD,
                            signal_strength=strength,
                            signal_class=self._signal_class(strength),
                            expected_profit=net,
                            risk_score=0.25,
                            confidence=min(0.95, strength),
                            description=f"Put calendar arbitrage at K={strike}: near > far",
                            trade_recommendation=f"Sell {near.expiry_days}d put, buy {far.expiry_days}d put at K={strike}",
                            details={
                                "strike": strike,
                                "near_expiry": near.expiry_days,
                                "far_expiry": far.expiry_days,
                                "near_put": near.put_price,
                                "far_put": far.put_price,
                            }
                        ))

        return signals

    # ── Butterfly Spread Arbitrage ──
    def check_butterfly(self, quotes: List[OptionQuote]) -> List[ArbitrageSignal]:
        """
        Butterfly spread: C(K-ΔK) - 2C(K) + C(K+ΔK) >= 0
        Violation implies negative probability density (arbitrage).
        """
        signals = []
        threshold = self._threshold(0.20)

        # Group by expiry
        by_expiry: Dict[int, List[OptionQuote]] = {}
        for q in quotes:
            by_expiry.setdefault(q.expiry_days, []).append(q)

        for expiry, exp_quotes in by_expiry.items():
            sorted_q = sorted(exp_quotes, key=lambda x: x.strike)
            for i in range(1, len(sorted_q) - 1):
                low = sorted_q[i - 1]
                mid = sorted_q[i]
                high = sorted_q[i + 1]

                # Check uniform spacing (approximately)
                dk1 = mid.strike - low.strike
                dk2 = high.strike - mid.strike
                if abs(dk1 - dk2) / dk1 > 0.5:
                    continue

                # Butterfly value for calls
                bf_call = low.call_price - 2 * mid.call_price + high.call_price
                tc = self._tc_cost(mid.spot) * 4
                net = -bf_call - tc  # should be >= 0

                if bf_call < -threshold:
                    strength = min(abs(bf_call) / (threshold * 3), 1.0)
                    signals.append(ArbitrageSignal(
                        arb_type=ArbitrageType.BUTTERFLY,
                        signal_strength=strength,
                        signal_class=self._signal_class(strength),
                        expected_profit=abs(bf_call) - tc,
                        risk_score=0.15,
                        confidence=min(0.90, strength),
                        description=f"Butterfly violation: C({low.strike}) - 2C({mid.strike}) + C({high.strike}) = {bf_call:.4f} < 0",
                        trade_recommendation=f"Buy butterfly: buy C({low.strike}), sell 2×C({mid.strike}), buy C({high.strike})",
                        details={
                            "strikes": [low.strike, mid.strike, high.strike],
                            "call_prices": [low.call_price, mid.call_price, high.call_price],
                            "butterfly_value": round(bf_call, 4),
                            "expiry_days": expiry,
                        }
                    ))

        return signals

    # ── Volatility Surface Consistency ──
    def check_surface_consistency(self, vol_surface: Dict[str, Any]) -> List[ArbitrageSignal]:
        """
        Check implied volatility surface for:
            1. Negative local variance (calendar arbitrage)
            2. Negative butterfly (strike arbitrage)
            3. Extreme skew deviations
        """
        signals = []

        if "surface" not in vol_surface:
            return signals

        surface = np.array(vol_surface["surface"])
        strikes = np.array(vol_surface.get("strikes", []))
        maturities = np.array(vol_surface.get("maturities", []))

        if surface.size == 0 or len(strikes) < 3 or len(maturities) < 2:
            return signals

        # Check total variance monotonicity (calendar)
        total_var = surface ** 2 * maturities[np.newaxis, :]
        for i in range(surface.shape[0]):
            dvar = np.diff(total_var[i])
            violations = np.where(dvar < -self._threshold(0.001))[0]
            for v_idx in violations:
                signals.append(ArbitrageSignal(
                    arb_type=ArbitrageType.SURFACE_INCONSISTENCY,
                    signal_strength=0.7,
                    signal_class=SignalStrength.MODERATE,
                    expected_profit=0.0,
                    risk_score=0.5,
                    confidence=0.8,
                    description=f"Calendar arbitrage in vol surface: total variance decreasing "
                                f"at K={strikes[i]:.1f} between τ={maturities[v_idx]:.2f} and τ={maturities[v_idx+1]:.2f}",
                    trade_recommendation="Sell near-term variance, buy far-term variance",
                    details={
                        "strike": float(strikes[i]),
                        "maturities": [float(maturities[v_idx]), float(maturities[v_idx+1])],
                        "total_variances": [float(total_var[i, v_idx]), float(total_var[i, v_idx+1])],
                    }
                ))

        # Check butterfly condition (convexity in strike)
        for j in range(surface.shape[1]):
            d2_sigma = np.diff(surface[:, j], n=2)
            violations = np.where(d2_sigma < -self._threshold(0.005))[0]
            for v_idx in violations:
                signals.append(ArbitrageSignal(
                    arb_type=ArbitrageType.SURFACE_INCONSISTENCY,
                    signal_strength=0.6,
                    signal_class=SignalStrength.MODERATE,
                    expected_profit=0.0,
                    risk_score=0.4,
                    confidence=0.75,
                    description=f"Butterfly arbitrage in vol surface at τ={maturities[j]:.2f}: "
                                f"non-convex around K={strikes[v_idx+1]:.1f}",
                    trade_recommendation="Buy butterfly spread at flagged strikes",
                    details={
                        "maturity": float(maturities[j]),
                        "strikes": [float(strikes[v_idx]), float(strikes[v_idx+1]), float(strikes[v_idx+2])],
                        "vols": [float(surface[v_idx, j]), float(surface[v_idx+1, j]), float(surface[v_idx+2, j])],
                    }
                ))

        return signals

    # ── Box Spread Arbitrage ──
    def check_box_spread(self, quotes: List[OptionQuote]) -> List[ArbitrageSignal]:
        """
        Box spread: (C(K1)-C(K2)) - (P(K1)-P(K2)) should equal (K2-K1)*exp(-rτ)
        """
        signals = []
        threshold = self._threshold(0.40)

        by_expiry: Dict[int, List[OptionQuote]] = {}
        for q in quotes:
            by_expiry.setdefault(q.expiry_days, []).append(q)

        for expiry, exp_quotes in by_expiry.items():
            sorted_q = sorted(exp_quotes, key=lambda x: x.strike)
            for i in range(len(sorted_q)):
                for j in range(i + 1, len(sorted_q)):
                    q1, q2 = sorted_q[i], sorted_q[j]
                    box_value = (q1.call_price - q2.call_price) - (q1.put_price - q2.put_price)
                    theoretical = (q2.strike - q1.strike) * math.exp(-q1.r * q1.tau)
                    deviation = abs(box_value - theoretical)
                    tc = self._tc_cost(q1.spot) * 4
                    net = deviation - tc

                    if net > threshold:
                        strength = min(net / (threshold * 3), 1.0)
                        signals.append(ArbitrageSignal(
                            arb_type=ArbitrageType.BOX_SPREAD,
                            signal_strength=strength,
                            signal_class=self._signal_class(strength),
                            expected_profit=net,
                            risk_score=0.1,
                            confidence=min(0.95, strength),
                            description=f"Box spread arbitrage: K1={q1.strike}, K2={q2.strike}, "
                                        f"deviation={deviation:.4f}",
                            trade_recommendation=f"Execute box spread at K1={q1.strike}, K2={q2.strike}",
                            details={
                                "k1": q1.strike, "k2": q2.strike,
                                "box_value": round(box_value, 4),
                                "theoretical": round(theoretical, 4),
                                "deviation": round(deviation, 4),
                            }
                        ))

        return signals

    # ── Full Scan ──
    def full_scan(self, quotes: List[OptionQuote],
                  vol_surface: Optional[Dict] = None) -> Dict[str, Any]:
        """Run all arbitrage checks and return consolidated results."""
        t0 = time.time()

        all_signals: List[ArbitrageSignal] = []
        all_signals.extend(self.check_put_call_parity(quotes))
        all_signals.extend(self.check_calendar_spread(quotes))
        all_signals.extend(self.check_butterfly(quotes))
        all_signals.extend(self.check_box_spread(quotes))

        if vol_surface:
            all_signals.extend(self.check_surface_consistency(vol_surface))

        # Sort by signal strength
        all_signals.sort(key=lambda s: s.signal_strength, reverse=True)
        self._signal_history.extend(all_signals)

        # Aggregate statistics
        by_type = {}
        for sig in all_signals:
            t = sig.arb_type.value
            if t not in by_type:
                by_type[t] = {"count": 0, "avg_strength": 0, "total_profit": 0}
            by_type[t]["count"] += 1
            by_type[t]["avg_strength"] += sig.signal_strength
            by_type[t]["total_profit"] += sig.expected_profit
        for t in by_type:
            if by_type[t]["count"] > 0:
                by_type[t]["avg_strength"] = round(by_type[t]["avg_strength"] / by_type[t]["count"], 4)
                by_type[t]["total_profit"] = round(by_type[t]["total_profit"], 4)

        elapsed = time.time() - t0

        return {
            "total_signals": len(all_signals),
            "strong_signals": sum(1 for s in all_signals if s.signal_class == SignalStrength.STRONG),
            "moderate_signals": sum(1 for s in all_signals if s.signal_class == SignalStrength.MODERATE),
            "weak_signals": sum(1 for s in all_signals if s.signal_class == SignalStrength.WEAK),
            "total_expected_profit": round(sum(s.expected_profit for s in all_signals), 4),
            "by_type": by_type,
            "signals": [s.to_dict() for s in all_signals[:20]],  # top 20
            "scan_time_ms": round(elapsed * 1000, 1),
            "regime": self.regime,
            "regime_label": ["Bull", "Bear", "Crisis"][self.regime],
        }

    @staticmethod
    def generate_test_quotes(n_strikes: int = 10, n_expiries: int = 4,
                             S: float = 100, seed: int = 42) -> List[OptionQuote]:
        """Generate test option quotes with some artificial arbitrage opportunities."""
        rng = np.random.default_rng(seed)
        from scipy.stats import norm

        quotes = []
        r = 0.05
        sigma = 0.20
        strikes = np.linspace(80, 120, n_strikes)
        expiries = [30, 60, 90, 180][:n_expiries]

        for K in strikes:
            for exp in expiries:
                tau = exp / 365.0
                d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * tau) / (sigma * math.sqrt(tau))
                d2 = d1 - sigma * math.sqrt(tau)
                call = S * norm.cdf(d1) - K * math.exp(-r * tau) * norm.cdf(d2)
                put = K * math.exp(-r * tau) * norm.cdf(-d2) - S * norm.cdf(-d1)

                # Add noise and occasional mispricing
                noise_c = rng.normal(0, 0.1)
                noise_p = rng.normal(0, 0.1)

                # Inject occasional arbitrage
                if rng.random() < 0.15:
                    noise_c += rng.choice([-1.5, 1.5])  # large mispricing
                if rng.random() < 0.1:
                    noise_p += rng.choice([-1.0, 1.0])

                quotes.append(OptionQuote(
                    strike=float(K), expiry_days=exp,
                    call_price=max(float(call + noise_c), 0.01),
                    put_price=max(float(put + noise_p), 0.01),
                    spot=S, r=r,
                    implied_vol=sigma + rng.normal(0, 0.02)
                ))

        return quotes


# Singleton
_arb_engine: Optional[ArbitrageDetectionEngine] = None

def get_arbitrage_engine(regime: int = 0) -> ArbitrageDetectionEngine:
    global _arb_engine
    if _arb_engine is None or _arb_engine.regime != regime:
        _arb_engine = ArbitrageDetectionEngine(regime=regime)
    return _arb_engine
