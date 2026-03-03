"""
Jump Diffusion & Regime Switching Models
==========================================
Extends base pricing to include:
    - Merton Jump Diffusion model
    - Enhanced Hidden Markov Model regime detection
    - Regime-conditioned parameter calibration
    - Crisis-aware pricing adjustments

Mathematical Foundation:
    Merton Jump Diffusion:
        dS/S = (μ - λk)dt + σdW + J dN(λ)
        where J ~ N(μ_J, σ_J²), N(λ) is Poisson process

    Regime Switching:
        Parameters {μ_i, σ_i, λ_i} depend on regime state R_t
        R_t follows Hidden Markov Model with states {Bull, Bear, Crisis}

Integrates with:
    - Monte Carlo engine (regime-dependent parameters)
    - PINNs (regime-conditioned pricing)
    - RL hedging (regime-aware state)
    - Risk engine (regime scenario analysis)
"""

from __future__ import annotations
import numpy as np
import math
import time
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Any
from enum import IntEnum

logger = logging.getLogger(__name__)


class MarketRegime(IntEnum):
    BULL = 0
    BEAR = 1
    CRISIS = 2


@dataclass
class RegimeParameters:
    """Parameters for each market regime."""
    mu: float          # drift
    sigma: float       # volatility
    jump_intensity: float  # Poisson jump rate
    jump_mean: float   # mean jump size
    jump_std: float    # jump size volatility
    label: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mu": self.mu, "sigma": self.sigma,
            "jump_intensity": self.jump_intensity,
            "jump_mean": self.jump_mean, "jump_std": self.jump_std,
            "label": self.label
        }


# Default regime parameters (calibrated to typical equity markets)
DEFAULT_REGIME_PARAMS = {
    MarketRegime.BULL: RegimeParameters(
        mu=0.12, sigma=0.15, jump_intensity=0.5,
        jump_mean=0.02, jump_std=0.03, label="Bull"
    ),
    MarketRegime.BEAR: RegimeParameters(
        mu=-0.05, sigma=0.25, jump_intensity=1.5,
        jump_mean=-0.03, jump_std=0.05, label="Bear"
    ),
    MarketRegime.CRISIS: RegimeParameters(
        mu=-0.30, sigma=0.50, jump_intensity=5.0,
        jump_mean=-0.08, jump_std=0.12, label="Crisis"
    ),
}


# ═══════════════════════════════════════════════════════════════════════
#  Merton Jump Diffusion Pricing
# ═══════════════════════════════════════════════════════════════════════

class MertonJumpDiffusion:
    """
    Merton (1976) Jump Diffusion model for option pricing.

    Uses:
      1. Analytical series expansion (truncated at N terms)
      2. Monte Carlo simulation with jumps
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def analytical_call(self, S: float, K: float, T: float, r: float,
                        sigma: float, lam: float, mu_j: float, sig_j: float,
                        n_terms: int = 50) -> Dict[str, Any]:
        """
        Merton series expansion:
            C = Σ_{n=0}^{∞} (e^{-λ'T} (λ'T)^n / n!) * BS(S, K, T, r_n, σ_n)

        where:
            λ' = λ(1 + k), k = e^{μ_J + σ_J²/2} - 1
            r_n = r - λk + nγ/T
            σ_n² = σ² + nσ_J²/T
            γ = μ_J + σ_J²/2
        """
        from scipy.stats import norm

        k = math.exp(mu_j + 0.5 * sig_j**2) - 1
        lam_prime = lam * (1 + k)
        gamma = mu_j + 0.5 * sig_j**2

        price = 0.0
        term_contributions = []

        for n in range(n_terms):
            # Poisson weight
            log_weight = -lam_prime * T + n * math.log(lam_prime * T + 1e-30) - math.lgamma(n + 1)
            weight = math.exp(log_weight)

            r_n = r - lam * k + n * gamma / T
            sigma_n_sq = sigma**2 + n * sig_j**2 / T
            sigma_n = math.sqrt(max(sigma_n_sq, 1e-10))

            # BS formula
            d1 = (math.log(S / K) + (r_n + 0.5 * sigma_n**2) * T) / (sigma_n * math.sqrt(T))
            d2 = d1 - sigma_n * math.sqrt(T)
            bs_price = S * norm.cdf(d1) - K * math.exp(-r_n * T) * norm.cdf(d2)

            contribution = weight * bs_price
            price += contribution
            term_contributions.append({"n": n, "weight": round(weight, 8), "price": round(contribution, 6)})

            if weight < 1e-12 and n > 5:
                break

        return {
            "price": round(price, 6),
            "n_terms_used": len(term_contributions),
            "lambda_prime": round(lam_prime, 4),
            "jump_compensator": round(k, 6),
            "top_terms": term_contributions[:10]
        }

    def monte_carlo(self, S: float, K: float, T: float, r: float,
                    sigma: float, lam: float, mu_j: float, sig_j: float,
                    n_paths: int = 100000, n_steps: int = 252,
                    option_type: str = "call") -> Dict[str, Any]:
        """Monte Carlo with Merton jump diffusion paths."""
        dt = T / n_steps
        k = math.exp(mu_j + 0.5 * sig_j**2) - 1
        drift = (r - 0.5 * sigma**2 - lam * k) * dt

        S_paths = np.full(n_paths, S, dtype=np.float64)
        all_paths = [S_paths.copy()]

        t0 = time.time()
        for step in range(n_steps):
            dW = self.rng.normal(0, math.sqrt(dt), n_paths)
            # Jumps
            n_jumps = self.rng.poisson(lam * dt, n_paths)
            jump_sizes = np.zeros(n_paths)
            mask = n_jumps > 0
            if mask.any():
                for idx in np.where(mask)[0]:
                    jump_sizes[idx] = sum(
                        self.rng.normal(mu_j, sig_j)
                        for _ in range(n_jumps[idx])
                    )

            S_paths = S_paths * np.exp(drift + sigma * dW + jump_sizes)
            all_paths.append(S_paths.copy())

        elapsed = time.time() - t0

        # Payoff
        if option_type == "call":
            payoff = np.maximum(S_paths - K, 0)
        else:
            payoff = np.maximum(K - S_paths, 0)

        discounted = np.exp(-r * T) * payoff
        price = float(np.mean(discounted))
        se = float(np.std(discounted) / math.sqrt(n_paths))

        # Path statistics
        final_prices = S_paths
        jumps_detected = np.abs(np.diff(np.log(np.array(all_paths[-10:])), axis=0)) > 3 * sigma * math.sqrt(dt)

        return {
            "price": round(price, 6),
            "std_error": round(se, 6),
            "ci_95": [round(price - 1.96 * se, 6), round(price + 1.96 * se, 6)],
            "n_paths": n_paths,
            "n_steps": n_steps,
            "latency_ms": round(elapsed * 1000, 1),
            "path_stats": {
                "mean_final": round(float(final_prices.mean()), 2),
                "std_final": round(float(final_prices.std()), 2),
                "min_final": round(float(final_prices.min()), 2),
                "max_final": round(float(final_prices.max()), 2),
            },
            "sample_paths": [all_paths[i][::max(1, n_steps//50)].tolist()[:50] for i in range(min(5, len(all_paths)))]
        }


# ═══════════════════════════════════════════════════════════════════════
#  Enhanced Hidden Markov Model
# ═══════════════════════════════════════════════════════════════════════

class EnhancedHMM:
    """
    Hidden Markov Model for market regime detection with:
        - Baum-Welch training (EM algorithm)
        - Viterbi decoding
        - Online regime prediction
        - Transition probability matrix estimation
        - Regime persistence and switching statistics
    """

    def __init__(self, n_states: int = 3, seed: int = 42):
        self.n_states = n_states
        self.rng = np.random.default_rng(seed)
        self._fitted = False

        # Initialise parameters
        self.pi = np.ones(n_states) / n_states  # initial state probs
        self.A = np.array([  # transition matrix (bull, bear, crisis)
            [0.92, 0.06, 0.02],
            [0.05, 0.88, 0.07],
            [0.03, 0.12, 0.85]
        ])
        self.means = np.array([0.001, -0.0005, -0.003])  # return means
        self.stds = np.array([0.01, 0.018, 0.035])        # return stds

    def _emission_prob(self, x: np.ndarray, state: int) -> np.ndarray:
        """Gaussian emission probability."""
        mu = self.means[state]
        sigma = self.stds[state]
        return np.exp(-0.5 * ((x - mu) / sigma)**2) / (sigma * math.sqrt(2 * math.pi))

    def fit(self, returns: np.ndarray, n_iter: int = 50) -> Dict[str, Any]:
        """Baum-Welch (EM) training."""
        T = len(returns)
        K = self.n_states
        t0 = time.time()

        for iteration in range(n_iter):
            # ── E-step: Forward-Backward ──
            alpha = np.zeros((T, K))
            beta = np.zeros((T, K))
            scale = np.zeros(T)

            # Forward
            for k in range(K):
                alpha[0, k] = self.pi[k] * self._emission_prob(returns[0:1], k)[0]
            scale[0] = alpha[0].sum()
            alpha[0] /= scale[0] + 1e-300

            for t in range(1, T):
                for k in range(K):
                    alpha[t, k] = sum(alpha[t-1, j] * self.A[j, k] for j in range(K)) * \
                                  self._emission_prob(returns[t:t+1], k)[0]
                scale[t] = alpha[t].sum()
                alpha[t] /= scale[t] + 1e-300

            # Backward
            beta[-1] = 1.0
            for t in range(T - 2, -1, -1):
                for k in range(K):
                    beta[t, k] = sum(self.A[k, j] * self._emission_prob(returns[t+1:t+2], j)[0] * beta[t+1, j]
                                     for j in range(K))
                beta[t] /= scale[t+1] + 1e-300

            # Gamma and Xi
            gamma = alpha * beta
            gamma /= gamma.sum(axis=1, keepdims=True) + 1e-300

            # ── M-step ──
            self.pi = gamma[0] / (gamma[0].sum() + 1e-300)

            for k in range(K):
                w = gamma[:, k]
                w_sum = w.sum() + 1e-300
                self.means[k] = (w * returns).sum() / w_sum
                self.stds[k] = math.sqrt((w * (returns - self.means[k])**2).sum() / w_sum + 1e-8)

                for j in range(K):
                    xi_sum = 0
                    for t in range(T - 1):
                        xi_sum += alpha[t, k] * self.A[k, j] * \
                                  self._emission_prob(returns[t+1:t+2], j)[0] * beta[t+1, j]
                    self.A[k, j] = xi_sum / (w[:-1].sum() + 1e-300)

            # Normalise A
            self.A /= self.A.sum(axis=1, keepdims=True) + 1e-300

            ll = np.sum(np.log(scale + 1e-300))
            if iteration % 10 == 0:
                logger.info(f"HMM iter {iteration}: log-likelihood={ll:.4f}")

        self._fitted = True
        elapsed = time.time() - t0

        # Sort states by volatility (ascending)
        order = np.argsort(self.stds)
        self.means = self.means[order]
        self.stds = self.stds[order]
        self.A = self.A[order][:, order]
        self.pi = self.pi[order]

        return {
            "iterations": n_iter,
            "log_likelihood": round(float(ll), 4),
            "training_time_s": round(elapsed, 2),
            "regime_params": {
                MarketRegime(i).name: {
                    "mean_return": round(float(self.means[i]), 6),
                    "volatility": round(float(self.stds[i]), 6),
                    "initial_prob": round(float(self.pi[i]), 4),
                }
                for i in range(K)
            },
            "transition_matrix": self.A.tolist()
        }

    def viterbi(self, returns: np.ndarray) -> np.ndarray:
        """Viterbi decoding for most likely state sequence."""
        T = len(returns)
        K = self.n_states
        delta = np.zeros((T, K))
        psi = np.zeros((T, K), dtype=int)

        for k in range(K):
            delta[0, k] = math.log(self.pi[k] + 1e-300) + \
                          math.log(self._emission_prob(returns[0:1], k)[0] + 1e-300)

        for t in range(1, T):
            for k in range(K):
                probs = delta[t-1] + np.log(self.A[:, k] + 1e-300)
                psi[t, k] = int(np.argmax(probs))
                delta[t, k] = probs[psi[t, k]] + \
                              math.log(self._emission_prob(returns[t:t+1], k)[0] + 1e-300)

        # Backtrack
        path = np.zeros(T, dtype=int)
        path[-1] = int(np.argmax(delta[-1]))
        for t in range(T - 2, -1, -1):
            path[t] = psi[t + 1, path[t + 1]]

        return path

    def predict_regime(self, returns: np.ndarray) -> Dict[str, Any]:
        """Predict current regime and forward probabilities."""
        if len(returns) < 2:
            return {"current_regime": "BULL", "regime_id": 0, "probabilities": self.pi.tolist()}

        path = self.viterbi(returns)
        current = int(path[-1])

        # Forward regime probabilities (next step)
        forward_probs = self.A[current]

        # Regime persistence
        regime_changes = np.diff(path)
        n_switches = int(np.count_nonzero(regime_changes))
        regime_counts = {MarketRegime(i).name: int(np.sum(path == i)) for i in range(self.n_states)}

        # Average regime duration
        durations = {i: [] for i in range(self.n_states)}
        current_run = 1
        for t in range(1, len(path)):
            if path[t] == path[t-1]:
                current_run += 1
            else:
                durations[path[t-1]].append(current_run)
                current_run = 1
        durations[path[-1]].append(current_run)

        avg_durations = {
            MarketRegime(i).name: round(float(np.mean(durations[i])) if durations[i] else 0, 1)
            for i in range(self.n_states)
        }

        return {
            "current_regime": MarketRegime(current).name,
            "regime_id": current,
            "state_probabilities": {
                MarketRegime(i).name: round(float(forward_probs[i]), 4)
                for i in range(self.n_states)
            },
            "regime_distribution": regime_counts,
            "n_regime_switches": n_switches,
            "avg_regime_duration": avg_durations,
            "transition_matrix": self.A.tolist(),
            "regime_path": path[-50:].tolist()  # last 50 states
        }


# ═══════════════════════════════════════════════════════════════════════
#  Regime-Aware Pricing Engine
# ═══════════════════════════════════════════════════════════════════════

class RegimeAwarePricingEngine:
    """
    Combines regime detection, jump diffusion, and standard pricing
    into a unified regime-aware option pricing framework.
    """

    def __init__(self, seed: int = 42):
        self.hmm = EnhancedHMM(seed=seed)
        self.jump_model = MertonJumpDiffusion(seed=seed)
        self.regime_params = DEFAULT_REGIME_PARAMS.copy()

    def calibrate(self, returns: np.ndarray) -> Dict[str, Any]:
        """Calibrate HMM and regime-specific parameters."""
        result = self.hmm.fit(returns)

        # Update regime parameters from fitted HMM
        for i in range(self.hmm.n_states):
            regime = MarketRegime(i)
            self.regime_params[regime] = RegimeParameters(
                mu=float(self.hmm.means[i]) * 252,  # annualise
                sigma=float(self.hmm.stds[i]) * math.sqrt(252),
                jump_intensity=DEFAULT_REGIME_PARAMS[regime].jump_intensity,
                jump_mean=DEFAULT_REGIME_PARAMS[regime].jump_mean,
                jump_std=DEFAULT_REGIME_PARAMS[regime].jump_std,
                label=regime.name
            )

        result["calibrated_params"] = {
            r.name: self.regime_params[r].to_dict() for r in MarketRegime
        }
        return result

    def price_option(self, S: float, K: float, T: float, r: float,
                     returns: Optional[np.ndarray] = None,
                     regime_override: Optional[int] = None,
                     n_paths: int = 50000, option_type: str = "call") -> Dict[str, Any]:
        """
        Price option with regime-aware jump diffusion.

        If returns provided, detect regime automatically.
        Otherwise use regime_override or default to Bull.
        """
        # Detect regime
        if returns is not None and len(returns) > 10:
            regime_info = self.hmm.predict_regime(returns)
            regime = MarketRegime(regime_info["regime_id"])
        elif regime_override is not None:
            regime = MarketRegime(regime_override)
            regime_info = {"current_regime": regime.name, "regime_id": int(regime)}
        else:
            regime = MarketRegime.BULL
            regime_info = {"current_regime": "BULL", "regime_id": 0}

        params = self.regime_params[regime]

        # Jump diffusion pricing
        jd_result = self.jump_model.analytical_call(
            S, K, T, r, params.sigma, params.jump_intensity,
            params.jump_mean, params.jump_std
        )

        # MC with jumps
        mc_result = self.jump_model.monte_carlo(
            S, K, T, r, params.sigma, params.jump_intensity,
            params.jump_mean, params.jump_std,
            n_paths=n_paths, option_type=option_type
        )

        # Standard BS (no jumps) for comparison
        from scipy.stats import norm
        d1 = (math.log(S / K) + (r + 0.5 * params.sigma**2) * T) / (params.sigma * math.sqrt(T))
        d2 = d1 - params.sigma * math.sqrt(T)
        if option_type == "call":
            bs_price = S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
        else:
            bs_price = K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

        # Crisis adjustment factor
        crisis_adj = 1.0
        if regime == MarketRegime.CRISIS:
            crisis_adj = 1.15  # 15% premium in crisis
        elif regime == MarketRegime.BEAR:
            crisis_adj = 1.05  # 5% premium in bear

        adjusted_price = mc_result["price"] * crisis_adj

        return {
            "regime": regime_info,
            "regime_params": params.to_dict(),
            "prices": {
                "bs_standard": round(bs_price, 6),
                "jump_diffusion_analytical": jd_result["price"],
                "jump_diffusion_mc": mc_result["price"],
                "regime_adjusted": round(adjusted_price, 6),
                "crisis_adjustment_factor": crisis_adj,
            },
            "mc_details": {
                "std_error": mc_result["std_error"],
                "ci_95": mc_result["ci_95"],
                "n_paths": mc_result["n_paths"],
                "latency_ms": mc_result["latency_ms"],
            },
            "jump_model": {
                "lambda": params.jump_intensity,
                "jump_mean": params.jump_mean,
                "jump_std": params.jump_std,
                "n_terms": jd_result["n_terms_used"],
            }
        }

    def scenario_analysis(self, S: float, K: float, T: float, r: float,
                          n_paths: int = 20000) -> Dict[str, Any]:
        """Price under all regime scenarios for stress testing."""
        results = {}
        for regime in MarketRegime:
            params = self.regime_params[regime]
            mc = self.jump_model.monte_carlo(
                S, K, T, r, params.sigma, params.jump_intensity,
                params.jump_mean, params.jump_std, n_paths=n_paths
            )
            results[regime.name] = {
                "price": mc["price"],
                "std_error": mc["std_error"],
                "volatility": params.sigma,
                "jump_intensity": params.jump_intensity,
            }

        # Price impact of regime transition
        bull_price = results["BULL"]["price"]
        bear_price = results["BEAR"]["price"]
        crisis_price = results["CRISIS"]["price"]

        return {
            "scenario_prices": results,
            "regime_impact": {
                "bull_to_bear": round((bear_price - bull_price) / bull_price * 100, 2),
                "bull_to_crisis": round((crisis_price - bull_price) / bull_price * 100, 2),
                "bear_to_crisis": round((crisis_price - bear_price) / bear_price * 100, 2) if bear_price > 0 else 0,
            },
            "transition_matrix": self.hmm.A.tolist()
        }


# Singletons
_hmm_instance: Optional[EnhancedHMM] = None
_regime_pricing: Optional[RegimeAwarePricingEngine] = None

def get_enhanced_hmm() -> EnhancedHMM:
    global _hmm_instance
    if _hmm_instance is None:
        _hmm_instance = EnhancedHMM()
    return _hmm_instance

def get_regime_pricing_engine() -> RegimeAwarePricingEngine:
    global _regime_pricing
    if _regime_pricing is None:
        _regime_pricing = RegimeAwarePricingEngine()
    return _regime_pricing
