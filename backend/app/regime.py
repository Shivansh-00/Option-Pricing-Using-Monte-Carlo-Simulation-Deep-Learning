"""
OptionQuant — Regime Detection Engine
═══════════════════════════════════════════════════════════════
Market regime identification for dynamic model selection:

  • Hidden Markov Model (Gaussian HMM) — NumPy implementation
  • LSTM-based regime classifier
  • Regime states: Bull / Bear / High-Volatility / Low-Volatility
  • Dynamic parameter adjustment per regime
  • Real-time regime probability streaming
  • Transition probability matrix estimation
"""
from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
#  Data Models
# ═══════════════════════════════════════════════════════════════

REGIME_LABELS = ["bull", "bear", "high_vol", "low_vol"]

@dataclass
class RegimeState:
    label: str              # bull / bear / high_vol / low_vol
    probability: float      # 0-1
    confidence: float
    duration_days: int      # how long current regime has lasted
    transition_probs: dict  # probability of transitioning to each state
    recommended_model: str  # pricing model to use
    vol_adjustment: float   # multiplier for volatility
    risk_level: str         # low / medium / high / extreme


@dataclass
class RegimeHistory:
    states: list[str]
    probabilities: list[list[float]]  # per-timestep regime probabilities
    transitions: int
    avg_duration: float
    current: RegimeState


@dataclass
class HMMParams:
    n_states: int = 4
    means: Optional[np.ndarray] = None
    variances: Optional[np.ndarray] = None
    transition_matrix: Optional[np.ndarray] = None
    initial_probs: Optional[np.ndarray] = None


# ═══════════════════════════════════════════════════════════════
#  Gaussian HMM (NumPy-only)
# ═══════════════════════════════════════════════════════════════

class GaussianHMM:
    """
    Hidden Markov Model with Gaussian emissions.
    Implements Baum-Welch (EM) for training and Viterbi for decoding.
    
    States:
      0 = bull (positive returns, low vol)
      1 = bear (negative returns, moderate vol)
      2 = high_vol (high volatility, mixed returns)
      3 = low_vol (low volatility, near-zero returns)
    """

    def __init__(self, n_states: int = 4, seed: int = 42):
        self.n_states = n_states
        self.rng = np.random.default_rng(seed)
        self._fitted = False

        # Initialize with financial priors
        self.means = np.array([0.0008, -0.0006, 0.0001, 0.0002])  # daily returns
        self.variances = np.array([0.0001, 0.0003, 0.0008, 0.00005])

        # Transition matrix: regimes tend to persist
        self.trans = np.array([
            [0.92, 0.03, 0.03, 0.02],  # bull stays bull
            [0.03, 0.90, 0.05, 0.02],  # bear stays bear
            [0.05, 0.05, 0.85, 0.05],  # high_vol
            [0.04, 0.02, 0.02, 0.92],  # low_vol stays
        ])

        self.pi = np.array([0.35, 0.20, 0.15, 0.30])  # initial state probs

    def _gaussian_pdf(self, x: float, mean: float, var: float) -> float:
        if var < 1e-12:
            var = 1e-12
        return (1.0 / math.sqrt(2 * math.pi * var)) * math.exp(-0.5 * (x - mean)**2 / var)

    def _emission_probs(self, x: float) -> np.ndarray:
        return np.array([self._gaussian_pdf(x, self.means[j], self.variances[j])
                         for j in range(self.n_states)])

    def fit(self, observations: np.ndarray, max_iter: int = 50, tol: float = 1e-4) -> dict:
        """
        Baum-Welch (EM) algorithm for HMM parameter estimation.
        """
        t0 = time.perf_counter()
        T = len(observations)
        if T < 10:
            raise ValueError("Need at least 10 observations")

        prev_ll = -np.inf
        for iteration in range(max_iter):
            # ── E-step: Forward-backward ──
            alpha = np.zeros((T, self.n_states))
            beta = np.zeros((T, self.n_states))
            scale = np.zeros(T)

            # Forward pass
            b0 = self._emission_probs(observations[0])
            alpha[0] = self.pi * b0
            scale[0] = alpha[0].sum()
            if scale[0] > 0:
                alpha[0] /= scale[0]

            for t in range(1, T):
                bt = self._emission_probs(observations[t])
                for j in range(self.n_states):
                    alpha[t, j] = bt[j] * np.sum(alpha[t-1] * self.trans[:, j])
                scale[t] = alpha[t].sum()
                if scale[t] > 0:
                    alpha[t] /= scale[t]

            # Backward pass
            beta[T-1] = 1.0
            for t in range(T-2, -1, -1):
                bt1 = self._emission_probs(observations[t+1])
                for i in range(self.n_states):
                    beta[t, i] = np.sum(self.trans[i] * bt1 * beta[t+1])
                if scale[t+1] > 0:
                    beta[t] /= scale[t+1]

            # Gamma and Xi
            gamma = alpha * beta
            gamma_sum = gamma.sum(axis=1, keepdims=True)
            gamma_sum[gamma_sum < 1e-300] = 1e-300
            gamma /= gamma_sum

            # Log-likelihood
            ll = np.sum(np.log(scale[scale > 0]))
            if abs(ll - prev_ll) < tol:
                break
            prev_ll = ll

            # ── M-step ──
            # Initial probs
            self.pi = gamma[0] / gamma[0].sum()

            # Transition matrix
            for i in range(self.n_states):
                for j in range(self.n_states):
                    num = 0.0
                    den = 0.0
                    for t in range(T-1):
                        bt1 = self._emission_probs(observations[t+1])
                        xi_ij = alpha[t, i] * self.trans[i, j] * bt1[j] * beta[t+1, j]
                        num += xi_ij
                        den += gamma[t, i]
                    self.trans[i, j] = num / (den + 1e-300)
                # Normalize row
                row_sum = self.trans[i].sum()
                if row_sum > 0:
                    self.trans[i] /= row_sum

            # Emission parameters
            for j in range(self.n_states):
                w = gamma[:, j]
                w_sum = w.sum()
                if w_sum > 1e-300:
                    self.means[j] = np.sum(w * observations) / w_sum
                    self.variances[j] = np.sum(w * (observations - self.means[j])**2) / w_sum
                    self.variances[j] = max(self.variances[j], 1e-8)

        self._fitted = True
        elapsed = (time.perf_counter() - t0) * 1000

        return {
            "iterations": iteration + 1,
            "log_likelihood": float(ll),
            "elapsed_ms": round(elapsed, 2),
            "means": self.means.tolist(),
            "variances": self.variances.tolist(),
        }

    def decode(self, observations: np.ndarray) -> list[int]:
        """Viterbi algorithm: find most likely state sequence."""
        T = len(observations)
        delta = np.zeros((T, self.n_states))
        psi = np.zeros((T, self.n_states), dtype=int)

        b0 = self._emission_probs(observations[0])
        delta[0] = np.log(self.pi + 1e-300) + np.log(b0 + 1e-300)

        for t in range(1, T):
            bt = self._emission_probs(observations[t])
            for j in range(self.n_states):
                candidates = delta[t-1] + np.log(self.trans[:, j] + 1e-300)
                psi[t, j] = int(np.argmax(candidates))
                delta[t, j] = candidates[psi[t, j]] + np.log(bt[j] + 1e-300)

        # Backtrack
        states = [0] * T
        states[T-1] = int(np.argmax(delta[T-1]))
        for t in range(T-2, -1, -1):
            states[t] = psi[t+1, states[t+1]]

        return states

    def predict_proba(self, observations: np.ndarray) -> np.ndarray:
        """Forward algorithm: return per-timestep state probabilities."""
        T = len(observations)
        alpha = np.zeros((T, self.n_states))

        b0 = self._emission_probs(observations[0])
        alpha[0] = self.pi * b0
        s = alpha[0].sum()
        if s > 0:
            alpha[0] /= s

        for t in range(1, T):
            bt = self._emission_probs(observations[t])
            for j in range(self.n_states):
                alpha[t, j] = bt[j] * np.sum(alpha[t-1] * self.trans[:, j])
            s = alpha[t].sum()
            if s > 0:
                alpha[t] /= s

        return alpha


# ═══════════════════════════════════════════════════════════════
#  Regime Detection Service
# ═══════════════════════════════════════════════════════════════

class RegimeDetector:
    """
    High-level regime detection using HMM + heuristic rules.
    Dynamically adjusts pricing model params based on detected regime.
    """

    def __init__(self, lookback: int = 252, seed: int = 42):
        self.lookback = lookback
        self.hmm = GaussianHMM(n_states=4, seed=seed)
        self._history: list[float] = []
        self._regime_history: list[str] = []
        self._current_regime: Optional[RegimeState] = None

    def update(self, daily_return: float, vix: float = 20.0) -> RegimeState:
        """
        Update with new observation and return current regime.
        """
        self._history.append(daily_return)

        # Keep rolling window
        if len(self._history) > self.lookback * 2:
            self._history = self._history[-self.lookback:]

        obs = np.array(self._history[-self.lookback:])

        if len(obs) >= 30:
            # Fit HMM on recent data
            try:
                self.hmm.fit(obs, max_iter=20)
                # Sanitize HMM parameters after fit (replace NaN/Inf)
                self.hmm.trans = np.nan_to_num(self.hmm.trans, nan=0.0, posinf=1.0, neginf=0.0)
                row_sums = self.hmm.trans.sum(axis=1, keepdims=True)
                row_sums[row_sums == 0] = 1.0
                self.hmm.trans /= row_sums
                self.hmm.means = np.nan_to_num(self.hmm.means, nan=0.0, posinf=0.01, neginf=-0.01)
                self.hmm.variances = np.nan_to_num(self.hmm.variances, nan=1e-4, posinf=1e-2, neginf=1e-8)
                self.hmm.variances = np.clip(self.hmm.variances, 1e-8, None)
                states = self.hmm.decode(obs)
                probs = self.hmm.predict_proba(obs)
                # Sanitize probabilities
                probs = np.nan_to_num(probs, nan=0.0, posinf=1.0, neginf=0.0)
                current_state = states[-1]
                current_probs = probs[-1]
            except Exception as e:
                logger.warning("HMM fit failed: %s, using heuristic", e)
                current_state, current_probs = self._heuristic_regime(obs, vix)
        else:
            current_state, current_probs = self._heuristic_regime(obs, vix)

        label = REGIME_LABELS[current_state]
        self._regime_history.append(label)

        # Compute duration
        duration = 1
        for i in range(len(self._regime_history) - 2, -1, -1):
            if self._regime_history[i] == label:
                duration += 1
            else:
                break

        # Transition probabilities from current state
        trans_probs = {}
        for j in range(4):
            v = float(self.hmm.trans[current_state, j])
            if not math.isfinite(v):
                v = 0.0
            trans_probs[REGIME_LABELS[j]] = round(v, 4)

        # Sanitize probability/confidence
        prob_val = float(current_probs[current_state])
        conf_val = float(np.max(current_probs))
        if not math.isfinite(prob_val):
            prob_val = 0.0
        if not math.isfinite(conf_val):
            conf_val = 0.0

        # Model recommendation based on regime
        model_map = {
            "bull": "black_scholes",
            "bear": "heston",
            "high_vol": "heston",
            "low_vol": "black_scholes",
        }

        vol_adj_map = {
            "bull": 0.95,
            "bear": 1.15,
            "high_vol": 1.35,
            "low_vol": 0.85,
        }

        risk_map = {
            "bull": "low",
            "bear": "high",
            "high_vol": "extreme",
            "low_vol": "low",
        }

        self._current_regime = RegimeState(
            label=label,
            probability=round(prob_val, 4),
            confidence=round(conf_val, 4),
            duration_days=duration,
            transition_probs=trans_probs,
            recommended_model=model_map[label],
            vol_adjustment=vol_adj_map[label],
            risk_level=risk_map[label],
        )

        return self._current_regime

    def _heuristic_regime(self, obs: np.ndarray, vix: float) -> tuple[int, np.ndarray]:
        """Fallback heuristic when HMM can't fit."""
        recent_mean = float(np.mean(obs[-20:])) if len(obs) >= 20 else float(np.mean(obs))
        recent_vol = float(np.std(obs[-20:])) if len(obs) >= 20 else float(np.std(obs))

        probs = np.array([0.25, 0.25, 0.25, 0.25])
        if recent_mean > 0.0005 and recent_vol < 0.015:
            state = 0  # bull
            probs = np.array([0.6, 0.1, 0.1, 0.2])
        elif recent_mean < -0.0005:
            state = 1  # bear
            probs = np.array([0.1, 0.6, 0.2, 0.1])
        elif recent_vol > 0.02 or vix > 30:
            state = 2  # high_vol
            probs = np.array([0.1, 0.2, 0.6, 0.1])
        else:
            state = 3  # low_vol
            probs = np.array([0.15, 0.1, 0.1, 0.65])

        return state, probs

    def get_history(self) -> RegimeHistory:
        """Return full regime analysis history."""
        if not self._regime_history:
            return RegimeHistory(
                states=[], probabilities=[], transitions=0,
                avg_duration=0, current=self._current_regime or RegimeState(
                    label="unknown", probability=0, confidence=0,
                    duration_days=0, transition_probs={},
                    recommended_model="black_scholes",
                    vol_adjustment=1.0, risk_level="medium",
                ),
            )

        # Count transitions
        transitions = sum(1 for i in range(1, len(self._regime_history))
                          if self._regime_history[i] != self._regime_history[i-1])

        # Average duration
        durations = []
        current_dur = 1
        for i in range(1, len(self._regime_history)):
            if self._regime_history[i] == self._regime_history[i-1]:
                current_dur += 1
            else:
                durations.append(current_dur)
                current_dur = 1
        durations.append(current_dur)

        return RegimeHistory(
            states=self._regime_history[-100:],
            probabilities=[],
            transitions=transitions,
            avg_duration=round(float(np.mean(durations)), 1) if durations else 0,
            current=self._current_regime,
        )


# ═══════════════════════════════════════════════════════════════
#  Singleton Service
# ═══════════════════════════════════════════════════════════════

_detector: Optional[RegimeDetector] = None


def get_regime_detector() -> RegimeDetector:
    global _detector
    if _detector is None:
        _detector = RegimeDetector()
    return _detector


def detect_regime(returns: list[float], vix: float = 20.0) -> RegimeState:
    """Convenience: detect regime from a series of returns."""
    detector = get_regime_detector()
    state = None
    for r in returns:
        state = detector.update(r, vix)
    return state


def get_regime_adjustment(regime_label: str) -> dict:
    """Get model parameters adjusted for current regime."""
    adjustments = {
        "bull": {
            "vol_multiplier": 0.95,
            "paths_multiplier": 1.0,
            "preferred_model": "black_scholes",
            "heston_kappa": 2.0,
            "heston_xi": 0.2,
        },
        "bear": {
            "vol_multiplier": 1.15,
            "paths_multiplier": 1.5,
            "preferred_model": "heston",
            "heston_kappa": 3.0,
            "heston_xi": 0.4,
        },
        "high_vol": {
            "vol_multiplier": 1.35,
            "paths_multiplier": 2.0,
            "preferred_model": "heston",
            "heston_kappa": 4.0,
            "heston_xi": 0.6,
        },
        "low_vol": {
            "vol_multiplier": 0.85,
            "paths_multiplier": 0.8,
            "preferred_model": "black_scholes",
            "heston_kappa": 1.5,
            "heston_xi": 0.15,
        },
    }
    return adjustments.get(regime_label, adjustments["bull"])
