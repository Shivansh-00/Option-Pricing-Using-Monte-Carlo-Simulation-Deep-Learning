"""
Portfolio-Level Risk Dashboard Engine
=======================================
Supports:
    - Multi-option portfolio management
    - Portfolio Greeks aggregation
    - Value at Risk (VaR) — Historical, Parametric, Monte Carlo
    - Conditional VaR (CVaR / Expected Shortfall)
    - Stress testing (vol shock, rate shock, jump, crisis)
    - Regime-based scenario analysis
    - Integration with RL hedging and arbitrage signals

Integrates with:
    - RL hedging (hedge recommendations per position)
    - Arbitrage engine (opportunity signals)
    - Regime detection (scenario conditioning)
    - Uncertainty quantification (confidence on risk metrics)
    - PINNs (position repricing)
"""

from __future__ import annotations
import numpy as np
import math
import time
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Any

logger = logging.getLogger(__name__)


@dataclass
class OptionPosition:
    """Single option position in a portfolio."""
    symbol: str
    option_type: str   # "call" or "put"
    strike: float
    expiry_days: int
    spot: float
    quantity: int      # positive = long, negative = short
    implied_vol: float = 0.20
    r: float = 0.05
    entry_price: float = 0.0

    @property
    def tau(self) -> float:
        return max(self.expiry_days / 365.0, 1e-6)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol, "type": self.option_type,
            "strike": self.strike, "expiry_days": self.expiry_days,
            "spot": self.spot, "quantity": self.quantity,
            "implied_vol": self.implied_vol, "r": self.r,
            "entry_price": self.entry_price
        }


@dataclass
class StressScenario:
    """Defines a stress test scenario."""
    name: str
    spot_shock: float = 0.0       # fractional change in spot
    vol_shock: float = 0.0        # absolute change in IV
    rate_shock: float = 0.0       # absolute change in r
    time_decay_days: int = 0      # days of theta decay
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name, "spot_shock": self.spot_shock,
            "vol_shock": self.vol_shock, "rate_shock": self.rate_shock,
            "time_decay_days": self.time_decay_days, "description": self.description
        }


# Predefined stress scenarios
STRESS_SCENARIOS = [
    StressScenario("Market Crash", spot_shock=-0.20, vol_shock=0.30, description="20% drop + vol spike"),
    StressScenario("Flash Crash", spot_shock=-0.10, vol_shock=0.50, description="10% drop + extreme vol"),
    StressScenario("Vol Explosion", vol_shock=0.25, description="Volatility doubles"),
    StressScenario("Rate Hike", rate_shock=0.02, description="200bps rate increase"),
    StressScenario("Bull Run", spot_shock=0.15, vol_shock=-0.05, description="15% rally + vol compression"),
    StressScenario("Time Decay (7d)", time_decay_days=7, description="1 week theta burn"),
    StressScenario("Time Decay (30d)", time_decay_days=30, description="1 month theta burn"),
    StressScenario("Crisis (2008-like)", spot_shock=-0.35, vol_shock=0.60, rate_shock=-0.01,
                   description="Severe crisis: 35% drop + 60% vol spike + rate cut"),
]


class PortfolioRiskEngine:
    """
    Comprehensive portfolio risk management engine.
    """

    def __init__(self, seed: int = 42):
        self.positions: List[OptionPosition] = []
        self.rng = np.random.default_rng(seed)

    # ── BS Pricing ──
    @staticmethod
    def _bs_price(S: float, K: float, tau: float, sigma: float, r: float,
                  opt_type: str = "call") -> float:
        if tau <= 0:
            return max(S - K, 0) if opt_type == "call" else max(K - S, 0)
        from scipy.stats import norm
        d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * tau) / (sigma * math.sqrt(tau))
        d2 = d1 - sigma * math.sqrt(tau)
        if opt_type == "call":
            return float(S * norm.cdf(d1) - K * math.exp(-r * tau) * norm.cdf(d2))
        return float(K * math.exp(-r * tau) * norm.cdf(-d2) - S * norm.cdf(-d1))

    @staticmethod
    def _bs_greeks(S: float, K: float, tau: float, sigma: float, r: float,
                   opt_type: str = "call") -> Dict[str, float]:
        if tau <= 0:
            return {"delta": 0, "gamma": 0, "theta": 0, "vega": 0, "rho": 0}
        from scipy.stats import norm
        d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * tau) / (sigma * math.sqrt(tau))
        d2 = d1 - sigma * math.sqrt(tau)
        nd1 = norm.pdf(d1)

        if opt_type == "call":
            delta = float(norm.cdf(d1))
            rho_val = float(K * tau * math.exp(-r * tau) * norm.cdf(d2))
        else:
            delta = float(norm.cdf(d1) - 1)
            rho_val = float(-K * tau * math.exp(-r * tau) * norm.cdf(-d2))

        gamma = float(nd1 / (S * sigma * math.sqrt(tau)))
        vega = float(S * nd1 * math.sqrt(tau))
        theta = float(-(S * nd1 * sigma) / (2 * math.sqrt(tau)) - r * K * math.exp(-r * tau) *
                       (norm.cdf(d2) if opt_type == "call" else norm.cdf(-d2)))

        return {"delta": round(delta, 6), "gamma": round(gamma, 6),
                "theta": round(theta, 6), "vega": round(vega, 6), "rho": round(rho_val, 6)}

    # ── Portfolio Management ──
    def add_position(self, position: OptionPosition):
        self.positions.append(position)

    def clear_positions(self):
        self.positions = []

    def set_positions(self, positions: List[OptionPosition]):
        self.positions = positions

    # ── Portfolio Valuation ──
    def portfolio_value(self, positions: Optional[List[OptionPosition]] = None) -> Dict[str, Any]:
        """Calculate current portfolio value and P&L."""
        pos = positions or self.positions
        total_value = 0.0
        total_pnl = 0.0
        details = []

        for p in pos:
            price = self._bs_price(p.spot, p.strike, p.tau, p.implied_vol, p.r, p.option_type)
            position_value = price * p.quantity * 100  # assume 100 multiplier
            pnl = (price - p.entry_price) * p.quantity * 100

            total_value += position_value
            total_pnl += pnl

            details.append({
                "symbol": p.symbol, "type": p.option_type,
                "strike": p.strike, "quantity": p.quantity,
                "current_price": round(price, 4),
                "position_value": round(position_value, 2),
                "pnl": round(pnl, 2),
            })

        return {
            "total_value": round(total_value, 2),
            "total_pnl": round(total_pnl, 2),
            "n_positions": len(pos),
            "positions": details
        }

    # ── Portfolio Greeks ──
    def portfolio_greeks(self, positions: Optional[List[OptionPosition]] = None) -> Dict[str, Any]:
        """Aggregate Greeks across all positions."""
        pos = positions or self.positions
        agg = {"delta": 0, "gamma": 0, "theta": 0, "vega": 0, "rho": 0}
        position_greeks = []

        for p in pos:
            g = self._bs_greeks(p.spot, p.strike, p.tau, p.implied_vol, p.r, p.option_type)
            multiplied = {k: v * p.quantity * 100 for k, v in g.items()}
            for k in agg:
                agg[k] += multiplied[k]
            position_greeks.append({
                "symbol": p.symbol, "strike": p.strike, "type": p.option_type,
                "quantity": p.quantity, **{f"pos_{k}": round(v, 4) for k, v in multiplied.items()}
            })

        return {
            "portfolio_greeks": {k: round(v, 4) for k, v in agg.items()},
            "position_greeks": position_greeks,
            "risk_summary": {
                "net_delta_exposure": round(agg["delta"], 2),
                "gamma_risk": "high" if abs(agg["gamma"]) > 100 else "moderate" if abs(agg["gamma"]) > 20 else "low",
                "theta_daily": round(agg["theta"] / 365, 2),
                "vega_exposure": round(agg["vega"], 2),
            }
        }

    # ── VaR Calculation ──
    def calculate_var(self, confidence: float = 0.95, horizon_days: int = 1,
                      n_simulations: int = 10000,
                      method: str = "monte_carlo") -> Dict[str, Any]:
        """
        Calculate Value at Risk using multiple methods.
        
        Methods: parametric, historical, monte_carlo
        """
        pos = self.positions
        if not pos:
            return {"error": "No positions in portfolio"}

        t0 = time.time()

        if method == "parametric":
            return self._parametric_var(pos, confidence, horizon_days)
        elif method == "historical":
            return self._historical_var(pos, confidence, horizon_days, n_simulations)
        else:
            return self._mc_var(pos, confidence, horizon_days, n_simulations)

    def _parametric_var(self, positions: List[OptionPosition], confidence: float,
                        horizon: int) -> Dict[str, float]:
        """Delta-normal VaR."""
        from scipy.stats import norm
        z = norm.ppf(confidence)
        total_var = 0.0

        for p in positions:
            g = self._bs_greeks(p.spot, p.strike, p.tau, p.implied_vol, p.r, p.option_type)
            daily_vol = p.implied_vol / math.sqrt(252) * math.sqrt(horizon)
            position_var = abs(g["delta"]) * p.spot * daily_vol * z * abs(p.quantity) * 100
            total_var += position_var

        return {
            "var": round(total_var, 2),
            "confidence": confidence,
            "horizon_days": horizon,
            "method": "parametric (delta-normal)",
        }

    def _mc_var(self, positions: List[OptionPosition], confidence: float,
                horizon: int, n_sims: int) -> Dict[str, Any]:
        """Monte Carlo VaR with full repricing (vectorized)."""
        from scipy.special import erf
        _SQRT2 = math.sqrt(2.0)

        t0 = time.time()
        dt = horizon / 365.0

        # Current portfolio value
        current_value = sum(
            self._bs_price(p.spot, p.strike, p.tau, p.implied_vol, p.r, p.option_type) * p.quantity * 100
            for p in positions
        )

        # Vectorized repricing per position
        future_values = np.zeros(n_sims)
        for p in positions:
            dW = self.rng.normal(0, math.sqrt(dt), n_sims)
            new_S = p.spot * np.exp((p.r - 0.5 * p.implied_vol**2) * dt + p.implied_vol * dW)
            new_tau = max(p.tau - dt, 1e-6)
            vol_perturb = p.implied_vol * (1 + 0.1 * self.rng.normal(size=n_sims))
            new_vol = np.clip(vol_perturb, 0.01, None)
            sqrt_tau = math.sqrt(new_tau)
            d1 = (np.log(new_S / p.strike) + (p.r + 0.5 * new_vol**2) * new_tau) / (new_vol * sqrt_tau)
            d2 = d1 - new_vol * sqrt_tau
            nd1 = 0.5 * (1.0 + erf(d1 / _SQRT2))
            nd2 = 0.5 * (1.0 + erf(d2 / _SQRT2))
            disc = math.exp(-p.r * new_tau)
            if p.option_type == "put":
                prices = p.strike * disc * (1.0 - nd2) - new_S * (1.0 - nd1)
            else:
                prices = new_S * nd1 - p.strike * disc * nd2
            future_values += prices * p.quantity * 100

        pnl_dist = future_values - current_value
        var = float(-np.percentile(pnl_dist, (1 - confidence) * 100))
        cvar = float(-np.mean(pnl_dist[pnl_dist <= -var])) if np.any(pnl_dist <= -var) else var

        elapsed = time.time() - t0

        return {
            "var": round(var, 2),
            "cvar": round(cvar, 2),
            "confidence": confidence,
            "horizon_days": horizon,
            "method": "monte_carlo",
            "n_simulations": n_sims,
            "current_value": round(current_value, 2),
            "pnl_stats": {
                "mean": round(float(pnl_dist.mean()), 2),
                "std": round(float(pnl_dist.std()), 2),
                "min": round(float(pnl_dist.min()), 2),
                "max": round(float(pnl_dist.max()), 2),
                "skewness": round(float(((pnl_dist - pnl_dist.mean())**3).mean() / (pnl_dist.std()**3 + 1e-8)), 4),
            },
            "pnl_percentiles": {
                f"p{int(p)}": round(float(np.percentile(pnl_dist, p)), 2)
                for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]
            },
            "latency_ms": round(elapsed * 1000, 1),
        }

    def _historical_var(self, positions: List[OptionPosition], confidence: float,
                        horizon: int, n_sims: int) -> Dict[str, Any]:
        """Historical simulation VaR using synthetic returns."""
        # Generate synthetic historical returns (GBM-based)
        returns = self.rng.normal(0.0005, 0.015, (500, 1))  # 500 historical days
        pnl_sim = np.zeros(len(returns) - horizon)

        current_value = sum(
            self._bs_price(p.spot, p.strike, p.tau, p.implied_vol, p.r, p.option_type) * p.quantity * 100
            for p in positions
        )

        for i in range(len(returns) - horizon):
            cumulative_return = float(returns[i:i+horizon].sum())
            sim_value = 0.0
            for p in positions:
                new_S = p.spot * math.exp(cumulative_return)
                new_tau = max(p.tau - horizon / 365.0, 1e-6)
                price = self._bs_price(new_S, p.strike, new_tau, p.implied_vol, p.r, p.option_type)
                sim_value += price * p.quantity * 100
            pnl_sim[i] = sim_value - current_value

        var = float(-np.percentile(pnl_sim, (1 - confidence) * 100))
        cvar = float(-np.mean(pnl_sim[pnl_sim <= -var])) if np.any(pnl_sim <= -var) else var

        return {
            "var": round(var, 2),
            "cvar": round(cvar, 2),
            "confidence": confidence,
            "horizon_days": horizon,
            "method": "historical_simulation",
            "n_scenarios": len(pnl_sim),
        }

    # ── Stress Testing ──
    def stress_test(self, scenarios: Optional[List[StressScenario]] = None) -> Dict[str, Any]:
        """Run stress tests across predefined + custom scenarios."""
        scenarios = scenarios or STRESS_SCENARIOS
        pos = self.positions
        if not pos:
            return {"error": "No positions in portfolio"}

        # Current value
        current = sum(
            self._bs_price(p.spot, p.strike, p.tau, p.implied_vol, p.r, p.option_type) * p.quantity * 100
            for p in pos
        )

        results = []
        for scenario in scenarios:
            stressed_value = 0.0
            for p in pos:
                new_S = p.spot * (1 + scenario.spot_shock)
                new_vol = max(p.implied_vol + scenario.vol_shock, 0.01)
                new_r = max(p.r + scenario.rate_shock, 0.001)
                new_tau = max(p.tau - scenario.time_decay_days / 365.0, 1e-6)

                price = self._bs_price(new_S, p.strike, new_tau, new_vol, new_r, p.option_type)
                stressed_value += price * p.quantity * 100

            pnl_impact = stressed_value - current
            results.append({
                "scenario": scenario.to_dict(),
                "stressed_value": round(stressed_value, 2),
                "pnl_impact": round(pnl_impact, 2),
                "pnl_pct": round(pnl_impact / (abs(current) + 1e-8) * 100, 2),
            })

        # Sort by worst impact
        results.sort(key=lambda x: x["pnl_impact"])

        return {
            "current_value": round(current, 2),
            "n_scenarios": len(results),
            "worst_case": results[0] if results else None,
            "best_case": results[-1] if results else None,
            "scenarios": results
        }

    # ── Regime Scenario Analysis ──
    def regime_scenario_analysis(self) -> Dict[str, Any]:
        """Analyse portfolio P&L under different market regimes."""
        regime_params = {
            "BULL": {"spot_change": 0.02, "vol_change": -0.02, "description": "Mild rally, vol decline"},
            "BEAR": {"spot_change": -0.05, "vol_change": 0.08, "description": "Moderate sell-off"},
            "CRISIS": {"spot_change": -0.15, "vol_change": 0.30, "description": "Severe stress"},
        }

        pos = self.positions
        current = sum(
            self._bs_price(p.spot, p.strike, p.tau, p.implied_vol, p.r, p.option_type) * p.quantity * 100
            for p in pos
        )

        results = {}
        for regime_name, params in regime_params.items():
            stressed = 0.0
            for p in pos:
                new_S = p.spot * (1 + params["spot_change"])
                new_vol = max(p.implied_vol + params["vol_change"], 0.01)
                price = self._bs_price(new_S, p.strike, p.tau, new_vol, p.r, p.option_type)
                stressed += price * p.quantity * 100

            pnl = stressed - current
            results[regime_name] = {
                "value": round(stressed, 2),
                "pnl": round(pnl, 2),
                "pnl_pct": round(pnl / (abs(current) + 1e-8) * 100, 2),
                "params": params
            }

        return {
            "current_value": round(current, 2),
            "regimes": results
        }

    # ── Full Risk Report ──
    def full_risk_report(self) -> Dict[str, Any]:
        """Comprehensive risk report combining all analytics."""
        t0 = time.time()

        valuation = self.portfolio_value()
        greeks = self.portfolio_greeks()
        var_95 = self.calculate_var(confidence=0.95, n_simulations=5000)
        var_99 = self.calculate_var(confidence=0.99, n_simulations=5000)
        stress = self.stress_test()
        regime_analysis = self.regime_scenario_analysis()

        elapsed = time.time() - t0

        return {
            "timestamp": time.time(),
            "valuation": valuation,
            "greeks": greeks,
            "var_95": var_95,
            "var_99": var_99,
            "stress_testing": stress,
            "regime_analysis": regime_analysis,
            "computation_time_ms": round(elapsed * 1000, 1),
            "risk_rating": self._overall_risk_rating(var_95, greeks, stress)
        }

    def _overall_risk_rating(self, var_result: Dict, greeks: Dict, stress: Dict) -> Dict[str, Any]:
        """Compute an overall portfolio risk rating."""
        var_val = var_result.get("var", 0)
        portfolio_val = var_result.get("current_value", 1)
        var_pct = abs(var_val) / (abs(portfolio_val) + 1e-8) * 100

        worst_stress = stress.get("worst_case", {}).get("pnl_pct", 0)
        gamma_risk = greeks.get("risk_summary", {}).get("gamma_risk", "low")

        # Simple scoring
        score = 0
        if var_pct > 5:
            score += 3
        elif var_pct > 2:
            score += 2
        else:
            score += 1

        if abs(worst_stress) > 20:
            score += 3
        elif abs(worst_stress) > 10:
            score += 2
        else:
            score += 1

        if gamma_risk == "high":
            score += 2
        elif gamma_risk == "moderate":
            score += 1

        rating = "LOW" if score <= 3 else "MODERATE" if score <= 5 else "HIGH" if score <= 7 else "CRITICAL"

        return {
            "rating": rating,
            "score": score,
            "max_score": 8,
            "var_pct": round(var_pct, 2),
            "worst_stress_pct": round(worst_stress, 2),
            "gamma_risk": gamma_risk
        }

    @staticmethod
    def create_sample_portfolio() -> List[OptionPosition]:
        """Create a sample multi-leg portfolio for testing."""
        return [
            OptionPosition("NIFTY", "call", 22000, 30, 21800, 10, 0.18, 0.065, 450),
            OptionPosition("NIFTY", "put", 21500, 30, 21800, 5, 0.22, 0.065, 320),
            OptionPosition("NIFTY", "call", 22500, 60, 21800, -5, 0.16, 0.065, 280),
            OptionPosition("NIFTY", "put", 21000, 60, 21800, -3, 0.25, 0.065, 400),
            OptionPosition("NIFTY", "call", 22000, 90, 21800, 8, 0.17, 0.065, 550),
            OptionPosition("BANKNIFTY", "call", 48000, 30, 47500, 5, 0.20, 0.065, 600),
            OptionPosition("BANKNIFTY", "put", 47000, 30, 47500, 5, 0.23, 0.065, 500),
        ]


# Singleton
_portfolio_engine: Optional[PortfolioRiskEngine] = None

def get_portfolio_engine() -> PortfolioRiskEngine:
    global _portfolio_engine
    if _portfolio_engine is None:
        _portfolio_engine = PortfolioRiskEngine()
    return _portfolio_engine
