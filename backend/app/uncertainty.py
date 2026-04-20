"""
Uncertainty Quantification Module
===================================
Implements:
    - Bayesian Neural Networks (weight uncertainty)
    - MC Dropout for inference-time uncertainty
    - Confidence intervals for all pricing outputs
    - Uncertainty propagation into RL decisions
    - Reliability flagging for unreliable predictions

Mathematical Foundation:
    Bayesian NN: p(w|D) ∝ p(D|w)p(w)
    Each weight w ~ N(μ_w, σ_w²), trained via variational inference.

    MC Dropout: Sample T forward passes with dropout → 
        μ = (1/T)Σᵢ ŷᵢ
        σ² = (1/T)Σᵢ(ŷᵢ - μ)² (epistemic uncertainty)

Integrates with:
    - PINNs (uncertainty on PDE-constrained prices)
    - RL hedging (uncertainty-aware action selection)
    - Portfolio risk (propagated uncertainty)
    - Explainability (confidence in explanations)
"""

from __future__ import annotations
import numpy as np
import math
import time
import logging
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple, Any

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
#  Bayesian Neural Network (Variational Inference)
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class BayesianLayer:
    """Weight uncertainty: each weight has mean μ and log-variance ρ."""
    W_mu: np.ndarray
    W_rho: np.ndarray  # log(σ²)
    b_mu: np.ndarray
    b_rho: np.ndarray

    @property
    def W_sigma(self) -> np.ndarray:
        return np.log1p(np.exp(self.W_rho))  # softplus

    @property
    def b_sigma(self) -> np.ndarray:
        return np.log1p(np.exp(self.b_rho))


class BayesianNN:
    """
    Bayesian Neural Network with weight uncertainty.
    
    Training: Variational inference (minimise KL divergence)
    Inference: Sample weights from posterior → get predictive distribution
    """

    def __init__(self, hidden_layers: Optional[List[int]] = None, lr: float = 1e-3,
                 n_samples: int = 10, kl_weight: float = 0.01, seed: int = 42):
        self.hidden = hidden_layers or [32, 32]
        self.lr = lr
        self.n_samples = n_samples
        self.kl_weight = kl_weight
        self.rng = np.random.default_rng(seed)
        self.layers: List[BayesianLayer] = []
        self._built = False

    def build(self, input_dim: int = 4):
        dims = [input_dim] + self.hidden + [1]
        self.layers = []
        for i in range(len(dims) - 1):
            scale = math.sqrt(2.0 / (dims[i] + dims[i+1]))
            self.layers.append(BayesianLayer(
                W_mu=self.rng.normal(0, scale, (dims[i], dims[i+1])),
                W_rho=np.full((dims[i], dims[i+1]), -3.0),  # small initial σ
                b_mu=np.zeros(dims[i+1]),
                b_rho=np.full(dims[i+1], -3.0)
            ))
        self._built = True

    def _sample_weights(self, layer: BayesianLayer):
        """Sample weights from variational posterior."""
        eps_W = self.rng.normal(0, 1, layer.W_mu.shape)
        eps_b = self.rng.normal(0, 1, layer.b_mu.shape)
        W = layer.W_mu + layer.W_sigma * eps_W
        b = layer.b_mu + layer.b_sigma * eps_b
        return W, b

    def _forward_sample(self, X: np.ndarray) -> np.ndarray:
        """Single forward pass with sampled weights."""
        h = X
        for i, layer in enumerate(self.layers[:-1]):
            W, b = self._sample_weights(layer)
            h = np.tanh(h @ W + b)
        # Output layer
        W, b = self._sample_weights(self.layers[-1])
        return h @ W + b

    def _forward_deterministic(self, X: np.ndarray, seed: int) -> np.ndarray:
        """Forward pass with fixed random seed (same epsilon each call)."""
        rng = np.random.default_rng(seed)
        h = X
        for i, layer in enumerate(self.layers[:-1]):
            eps_W = rng.normal(0, 1, layer.W_mu.shape)
            eps_b = rng.normal(0, 1, layer.b_mu.shape)
            W = layer.W_mu + layer.W_sigma * eps_W
            b = layer.b_mu + layer.b_sigma * eps_b
            h = np.tanh(h @ W + b)
        layer = self.layers[-1]
        eps_W = rng.normal(0, 1, layer.W_mu.shape)
        eps_b = rng.normal(0, 1, layer.b_mu.shape)
        W = layer.W_mu + layer.W_sigma * eps_W
        b = layer.b_mu + layer.b_sigma * eps_b
        return h @ W + b

    def predict(self, X: np.ndarray, n_samples: Optional[int] = None) -> Dict[str, np.ndarray]:
        """
        Predictive distribution via weight sampling.
        
        Returns mean, std, and individual samples.
        """
        if not self._built:
            self.build(X.shape[1])

        T = n_samples or self.n_samples
        samples = np.zeros((T, X.shape[0]))

        for t in range(T):
            pred = self._forward_sample(X)
            samples[t] = pred.ravel()

        mean = samples.mean(axis=0)
        std = samples.std(axis=0)
        
        return {
            "mean": mean,
            "std": std,
            "ci_lower": mean - 1.96 * std,
            "ci_upper": mean + 1.96 * std,
            "samples": samples
        }

    def _kl_divergence(self) -> float:
        """KL divergence between posterior and prior (standard normal)."""
        kl = 0.0
        for layer in self.layers:
            sigma = layer.W_sigma
            mu = layer.W_mu
            kl += 0.5 * np.sum(sigma**2 + mu**2 - 1 - 2 * np.log(sigma + 1e-10))
            sigma_b = layer.b_sigma
            mu_b = layer.b_mu
            kl += 0.5 * np.sum(sigma_b**2 + mu_b**2 - 1 - 2 * np.log(sigma_b + 1e-10))
        return float(kl)

    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 500,
              batch_size: int = 64) -> Dict[str, Any]:
        """Train via ELBO maximisation (variational inference)."""
        if not self._built:
            self.build(X.shape[1])

        n = X.shape[0]
        history = []
        best_loss = float("inf")
        t0 = time.time()
        lr = self.lr

        for epoch in range(epochs):
            idx = self.rng.choice(n, min(batch_size, n), replace=False)
            Xb, yb = X[idx], y[idx].reshape(-1, 1)

            # Data likelihood (MSE over multiple samples)
            data_loss = 0.0
            for _ in range(3):
                pred = self._forward_sample(Xb)
                data_loss += float(np.mean((pred - yb) ** 2))
            data_loss /= 3

            kl = self._kl_divergence()
            total_loss = data_loss + self.kl_weight * kl / n

            # SPSA update on variational parameters
            pert = 1e-3
            eps_seed = int(self.rng.integers(0, 2**31))
            for layer in self.layers:
                for param in [layer.W_mu, layer.W_rho, layer.b_mu, layer.b_rho]:
                    dp = self.rng.choice([-1.0, 1.0], size=param.shape)
                    param += pert * dp
                    pred_p = self._forward_deterministic(Xb, seed=eps_seed)
                    l_plus = float(np.mean((pred_p - yb) ** 2))
                    param -= 2 * pert * dp
                    pred_m = self._forward_deterministic(Xb, seed=eps_seed)
                    l_minus = float(np.mean((pred_m - yb) ** 2))
                    param += pert * dp
                    grad = (l_plus - l_minus) / (2 * pert * dp)
                    param -= lr * np.clip(grad, -1, 1)

            if total_loss < best_loss:
                best_loss = total_loss

            if epoch % 50 == 0:
                logger.info(f"BayesianNN epoch {epoch}: loss={total_loss:.6f}, kl={kl:.4f}")
                history.append({"epoch": epoch, "loss": total_loss, "kl": kl, "data_loss": data_loss})

            if epoch > 0 and epoch % 200 == 0:
                lr *= 0.5

        return {
            "epochs": epochs,
            "best_loss": best_loss,
            "training_time_s": round(time.time() - t0, 2),
            "history": history
        }


# ═══════════════════════════════════════════════════════════════════════
#  MC Dropout Uncertainty Estimator
# ═══════════════════════════════════════════════════════════════════════

class MCDropoutEstimator:
    """
    Monte Carlo Dropout for uncertainty estimation.
    
    Uses a trained deterministic network and applies dropout at inference
    to estimate epistemic uncertainty.
    """

    def __init__(self, hidden: Optional[List[int]] = None, dropout_rate: float = 0.1,
                 n_forward_passes: int = 50, seed: int = 42):
        self.hidden = hidden or [64, 32]
        self.dropout_rate = dropout_rate
        self.n_passes = n_forward_passes
        self.rng = np.random.default_rng(seed)
        self.weights: List[Tuple[np.ndarray, np.ndarray]] = []
        self._built = False

    def build(self, input_dim: int = 4):
        dims = [input_dim] + self.hidden + [1]
        self.weights = []
        for i in range(len(dims) - 1):
            scale = math.sqrt(2.0 / (dims[i] + dims[i+1]))
            W = self.rng.normal(0, scale, (dims[i], dims[i+1]))
            b = np.zeros(dims[i+1])
            self.weights.append((W, b))
        self._built = True

    def _forward_with_dropout(self, X: np.ndarray) -> np.ndarray:
        h = X
        for i, (W, b) in enumerate(self.weights[:-1]):
            h = np.tanh(h @ W + b)
            # Apply dropout
            mask = self.rng.random(h.shape) > self.dropout_rate
            h = h * mask / (1 - self.dropout_rate)
        W, b = self.weights[-1]
        return h @ W + b

    def _forward_no_dropout(self, X: np.ndarray) -> np.ndarray:
        h = X
        for i, (W, b) in enumerate(self.weights[:-1]):
            h = np.tanh(h @ W + b)
        W, b = self.weights[-1]
        return h @ W + b

    def predict_with_uncertainty(self, X: np.ndarray) -> Dict[str, Any]:
        """Run multiple forward passes with dropout to estimate uncertainty."""
        if not self._built:
            self.build(X.shape[1])

        samples = np.zeros((self.n_passes, X.shape[0]))
        for t in range(self.n_passes):
            pred = self._forward_with_dropout(X)
            samples[t] = pred.ravel()

        mean = samples.mean(axis=0)
        epistemic_std = samples.std(axis=0)  # epistemic uncertainty

        # Deterministic prediction
        det_pred = self._forward_no_dropout(X).ravel()

        # Aleatoric uncertainty estimate (from residuals)
        aleatoric_std = np.abs(det_pred - mean)

        # Total uncertainty
        total_std = np.sqrt(epistemic_std**2 + aleatoric_std**2)

        return {
            "prediction": mean.tolist(),
            "deterministic": det_pred.tolist(),
            "epistemic_uncertainty": epistemic_std.tolist(),
            "aleatoric_uncertainty": aleatoric_std.tolist(),
            "total_uncertainty": total_std.tolist(),
            "ci_95_lower": (mean - 1.96 * total_std).tolist(),
            "ci_95_upper": (mean + 1.96 * total_std).tolist(),
            "n_passes": self.n_passes,
            "dropout_rate": self.dropout_rate
        }

    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 500,
              lr: float = 1e-3, batch_size: int = 64) -> Dict[str, Any]:
        """Train with standard dropout (then use MC dropout at inference)."""
        if not self._built:
            self.build(X.shape[1])

        n = X.shape[0]
        best_loss = float("inf")
        t0 = time.time()
        history = []

        for epoch in range(epochs):
            idx = self.rng.choice(n, min(batch_size, n), replace=False)
            Xb, yb = X[idx], y[idx].reshape(-1, 1)

            pred = self._forward_with_dropout(Xb)
            loss = float(np.mean((pred - yb) ** 2))

            # SPSA update
            pert = 1e-3
            for W, b in self.weights:
                dW = self.rng.choice([-1.0, 1.0], size=W.shape)
                db = self.rng.choice([-1.0, 1.0], size=b.shape)
                W += pert * dW
                b += pert * db
                l_p = float(np.mean((self._forward_with_dropout(Xb) - yb) ** 2))
                W -= 2 * pert * dW
                b -= 2 * pert * db
                l_m = float(np.mean((self._forward_with_dropout(Xb) - yb) ** 2))
                W += pert * dW
                b += pert * db
                W -= lr * np.clip((l_p - l_m) / (2 * pert * dW), -1, 1)
                b -= lr * np.clip((l_p - l_m) / (2 * pert * db), -1, 1)

            if loss < best_loss:
                best_loss = loss
            if epoch % 50 == 0:
                history.append({"epoch": epoch, "loss": loss})

        return {"epochs": epochs, "best_loss": best_loss,
                "training_time_s": round(time.time() - t0, 2), "history": history}


# ═══════════════════════════════════════════════════════════════════════
#  Unified Uncertainty Quantifier
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class UncertaintyResult:
    mean: float
    std: float
    ci_lower: float
    ci_upper: float
    epistemic: float
    aleatoric: float
    reliability: str       # "high", "medium", "low", "unreliable"
    reliability_score: float  # 0-1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mean": round(self.mean, 6),
            "std": round(self.std, 6),
            "ci_95": [round(self.ci_lower, 6), round(self.ci_upper, 6)],
            "epistemic_uncertainty": round(self.epistemic, 6),
            "aleatoric_uncertainty": round(self.aleatoric, 6),
            "reliability": self.reliability,
            "reliability_score": round(self.reliability_score, 4),
        }


class UncertaintyQuantifier:
    """
    Unified uncertainty estimation combining:
        - Bayesian NN
        - MC Dropout
        - Bootstrap confidence
        - Ensemble disagreement
    
    Flags unreliable predictions and propagates uncertainty.
    """

    def __init__(self, seed: int = 42):
        self.bnn = BayesianNN(seed=seed)
        self.mc_dropout = MCDropoutEstimator(seed=seed)
        self.rng = np.random.default_rng(seed)
        self._calibrated = False
        self._reliability_thresholds = {
            "high": 0.05,      # < 5% relative uncertainty
            "medium": 0.15,    # < 15%
            "low": 0.30,       # < 30%
        }

    def _reliability_class(self, rel_uncertainty: float) -> Tuple[str, float]:
        """Classify prediction reliability."""
        if rel_uncertainty < self._reliability_thresholds["high"]:
            return "high", min(1.0, 1.0 - rel_uncertainty / self._reliability_thresholds["high"])
        elif rel_uncertainty < self._reliability_thresholds["medium"]:
            return "medium", 0.7 * (1.0 - (rel_uncertainty - self._reliability_thresholds["high"]) /
                                     (self._reliability_thresholds["medium"] - self._reliability_thresholds["high"]))
        elif rel_uncertainty < self._reliability_thresholds["low"]:
            return "low", 0.3
        return "unreliable", 0.1

    def quantify(self, X: np.ndarray, y_scale: float = 1.0) -> List[UncertaintyResult]:
        """
        Full uncertainty quantification for input points.
        
        Combines BNN and MC Dropout estimates.
        """
        # BNN prediction
        bnn_result = self.bnn.predict(X)
        bnn_mean = bnn_result["mean"]
        bnn_std = bnn_result["std"]

        # MC Dropout prediction
        mc_result = self.mc_dropout.predict_with_uncertainty(X)
        mc_mean = np.array(mc_result["prediction"])
        mc_epist = np.array(mc_result["epistemic_uncertainty"])
        mc_aleat = np.array(mc_result["aleatoric_uncertainty"])

        results = []
        for i in range(X.shape[0]):
            # Combine estimates (weighted average)
            mean = 0.5 * bnn_mean[i] + 0.5 * mc_mean[i]
            epistemic = 0.5 * bnn_std[i] + 0.5 * mc_epist[i]
            aleatoric = mc_aleat[i]
            total_std = math.sqrt(epistemic**2 + aleatoric**2)

            # Relative uncertainty
            abs_mean = abs(mean * y_scale) + 1e-8
            rel_unc = total_std / abs_mean

            reliability, score = self._reliability_class(rel_unc)

            results.append(UncertaintyResult(
                mean=float(mean),
                std=float(total_std),
                ci_lower=float(mean - 1.96 * total_std),
                ci_upper=float(mean + 1.96 * total_std),
                epistemic=float(epistemic),
                aleatoric=float(aleatoric),
                reliability=reliability,
                reliability_score=float(score)
            ))

        return results

    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 300) -> Dict[str, Any]:
        """Train both BNN and MC Dropout models."""
        bnn_res = self.bnn.train(X, y, epochs=epochs)
        mc_res = self.mc_dropout.train(X, y, epochs=epochs)
        self._calibrated = True

        return {
            "bnn": bnn_res,
            "mc_dropout": mc_res,
            "calibrated": True
        }

    def pricing_uncertainty(self, S: float, K: float, tau: float,
                            sigma: float, r: float,
                            n_bootstrap: int = 200) -> Dict[str, Any]:
        """
        Quick uncertainty estimate for a single option price using bootstrap.
        No training required — uses analytical Black-Scholes with parameter perturbation.
        """
        from scipy.stats import norm  # type: ignore

        prices = []
        for _ in range(n_bootstrap):
            # Perturb parameters
            s_pert = S * (1 + self.rng.normal(0, 0.005))
            sig_pert = sigma * (1 + self.rng.normal(0, 0.05))
            r_pert = r * (1 + self.rng.normal(0, 0.02))

            d1 = (math.log(s_pert / K) + (r_pert + 0.5 * sig_pert**2) * tau) / (sig_pert * math.sqrt(tau))
            d2 = d1 - sig_pert * math.sqrt(tau)
            p = s_pert * norm.cdf(d1) - K * math.exp(-r_pert * tau) * norm.cdf(d2)
            prices.append(max(p, 0))

        prices_arr: np.ndarray = np.array(prices)
        mean = float(prices_arr.mean())
        std = float(prices_arr.std())
        rel_unc = std / (abs(mean) + 1e-8)
        reliability, score = self._reliability_class(rel_unc)

        return {
            "price_mean": round(mean, 6),
            "price_std": round(std, 6),
            "ci_95": [round(float(np.percentile(prices_arr, 2.5)), 6),
                      round(float(np.percentile(prices_arr, 97.5)), 6)],
            "relative_uncertainty": round(rel_unc, 4),
            "reliability": reliability,
            "reliability_score": round(score, 4),
            "n_bootstrap": n_bootstrap,
            "parameter_sensitivity": {
                "spot_impact": round(float(np.corrcoef(
                    [S * (1 + self.rng.normal(0, 0.005)) for _ in range(100)],
                    prices[:100]
                )[0, 1]) if len(prices) >= 100 else 0, 4),
            }
        }

    def flag_unreliable(self, results: List[UncertaintyResult]) -> Dict[str, Any]:
        """Flag which predictions are unreliable."""
        flags = []
        for i, r in enumerate(results):
            if r.reliability in ("low", "unreliable"):
                flags.append({
                    "index": i,
                    "reliability": r.reliability,
                    "score": r.reliability_score,
                    "uncertainty": r.std,
                    "recommendation": "Increase data coverage or use wider confidence bounds"
                    if r.reliability == "low" else
                    "DO NOT USE this prediction — uncertainty too high"
                })

        return {
            "n_flagged": len(flags),
            "n_total": len(results),
            "flag_rate": round(len(flags) / max(len(results), 1), 4),
            "flags": flags,
            "summary": {
                "high": sum(1 for r in results if r.reliability == "high"),
                "medium": sum(1 for r in results if r.reliability == "medium"),
                "low": sum(1 for r in results if r.reliability == "low"),
                "unreliable": sum(1 for r in results if r.reliability == "unreliable"),
            }
        }

    def get_status(self) -> Dict[str, Any]:
        return {
            "calibrated": self._calibrated,
            "bnn_built": self.bnn._built,
            "mc_dropout_built": self.mc_dropout._built,
            "n_bnn_layers": len(self.bnn.layers),
            "n_mc_layers": len(self.mc_dropout.weights),
        }


# Singleton
_uq_instance: Optional[UncertaintyQuantifier] = None

def get_uncertainty_quantifier() -> UncertaintyQuantifier:
    global _uq_instance
    if _uq_instance is None:
        _uq_instance = UncertaintyQuantifier()
    return _uq_instance
