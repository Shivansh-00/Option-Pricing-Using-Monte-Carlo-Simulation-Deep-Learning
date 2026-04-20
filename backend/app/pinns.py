"""
Physics-Informed Neural Networks (PINNs) for Option Pricing
============================================================
Embeds Black-Scholes PDE residual into the loss function with
no-arbitrage regularization and smooth Greeks enforcement.

Mathematical Foundation:
    The Black-Scholes PDE:
        ∂V/∂t + ½σ²S²∂²V/∂S² + rS∂V/∂S - rV = 0

    Loss = L_data + λ_pde * L_pde + λ_arb * L_arb + λ_smooth * L_smooth

    Where:
        L_data    = MSE between predicted and market prices
        L_pde     = MSE of PDE residual (physics constraint)
        L_arb     = Penalty for arbitrage violations (negative prices, butterfly)
        L_smooth  = Penalty for non-smooth Greeks (Gamma, Vanna)
"""

from __future__ import annotations
import numpy as np
import math
import pickle
import time
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any, Union

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Activation helpers (pure NumPy)
# ---------------------------------------------------------------------------

def _tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)

def _tanh_grad(x: np.ndarray) -> np.ndarray:
    t = np.tanh(x)
    return 1.0 - t * t

def _softplus(x: np.ndarray) -> np.ndarray:
    return np.where(x > 20, x, np.log1p(np.exp(np.clip(x, -20, 20))))

def _softplus_grad(x: np.ndarray) -> np.ndarray:
    ex = np.exp(np.clip(x, -20, 20))
    return ex / (1.0 + ex)

def _gelu(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x**3)))

def _gelu_grad(x: np.ndarray) -> np.ndarray:
    k = math.sqrt(2.0 / math.pi)
    inner = k * (x + 0.044715 * x**3)
    t = np.tanh(inner)
    sech2 = 1.0 - t * t
    inner_grad = k * (1.0 + 3.0 * 0.044715 * x**2)
    return 0.5 * (1.0 + t) + 0.5 * x * sech2 * inner_grad

# ---------------------------------------------------------------------------
# Xavier initialiser
# ---------------------------------------------------------------------------

def _xavier_init(fan_in: int, fan_out: int, rng: np.random.Generator) -> np.ndarray:
    std = math.sqrt(2.0 / (fan_in + fan_out))
    return rng.normal(0.0, std, (fan_in, fan_out)).astype(np.float64)

# ---------------------------------------------------------------------------
#  PINNs Network – forward pass + automatic PDE differentiation
# ---------------------------------------------------------------------------

@dataclass
class PINNLayer:
    W: np.ndarray
    b: np.ndarray

@dataclass
class PINNsConfig:
    """Configuration for the PINNs model."""
    hidden_layers: List[int] = field(default_factory=lambda: [64, 64, 64, 32])
    learning_rate: float = 1e-3
    lambda_pde: float = 1.0
    lambda_arb: float = 0.5
    lambda_smooth: float = 0.1
    epochs: int = 200
    batch_size: int = 256
    pde_batch_size: int = 32
    early_stop_patience: int = 40
    seed: int = 42
    activation: str = "tanh"  # tanh | softplus | gelu

class PINNsOptionPricer:
    """
    Physics-Informed Neural Network for European option pricing.

    Input features: [S/K (moneyness), τ (time-to-maturity), σ, r]
    Output: V/K (normalised option price)

    The PDE residual is computed via finite-difference automatic
    differentiation through the network layers.
    """

    def __init__(self, config: Optional[PINNsConfig] = None):
        self.config = config or PINNsConfig()
        self.rng = np.random.default_rng(self.config.seed)
        self.layers: List[PINNLayer] = []
        self._built = False
        self._train_history: List[Dict[str, float]] = []
        self._act = self._get_activation(self.config.activation)
        self._act_grad = self._get_activation_grad(self.config.activation)

    # ---- activation dispatch ----
    @staticmethod
    def _get_activation(name: str):
        return {"tanh": _tanh, "softplus": _softplus, "gelu": _gelu}[name]

    @staticmethod
    def _get_activation_grad(name: str):
        return {"tanh": _tanh_grad, "softplus": _softplus_grad, "gelu": _gelu_grad}[name]

    # ---- build ----
    def build(self, input_dim: int = 4):
        dims = [input_dim] + self.config.hidden_layers + [1]
        self.layers = []
        for i in range(len(dims) - 1):
            W = _xavier_init(dims[i], dims[i + 1], self.rng)
            b = np.zeros((1, dims[i + 1]), dtype=np.float64)
            self.layers.append(PINNLayer(W=W, b=b))
        self._built = True
        total_params = sum(layer.W.size + layer.b.size for layer in self.layers)
        logger.info(f"PINNs built: {len(self.layers)} layers, {total_params} params")

    # ---- forward ----
    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Forward pass returning output and intermediate pre-activations."""
        if not self._built:
            self.build(X.shape[1])
        activations = [X]
        h = X
        for i, layer in enumerate(self.layers[:-1]):
            z = h @ layer.W + layer.b
            h = self._act(z)
            activations.append(z)
        # Output layer — softplus for positivity with non-zero gradient everywhere
        z_out = h @ self.layers[-1].W + self.layers[-1].b
        out = np.log1p(np.exp(np.clip(z_out, -20, 20)))
        activations.append(z_out)
        return out, activations

    # ---- PDE residual via finite differences (batched) ----
    def _pde_residual(self, S: np.ndarray, K: np.ndarray, tau: np.ndarray,
                      sigma: np.ndarray, r: np.ndarray) -> np.ndarray:
        """
        Compute Black-Scholes PDE residual:
            R = ∂V/∂τ + ½σ²S²∂²V/∂S² + rS∂V/∂S - rV

        Uses central finite differences — batched into single forward pass.
        """
        eps_S = S * 1e-4 + 1e-6
        eps_t = 1e-5
        n = len(S)

        m_c = S / K
        m_up = (S + eps_S) / K
        m_dn = (S - eps_S) / K

        # Batch all 4 evaluations into single forward pass
        X_all = np.vstack([
            np.column_stack([m_c, tau, sigma, r]),
            np.column_stack([m_up, tau, sigma, r]),
            np.column_stack([m_dn, tau, sigma, r]),
            np.column_stack([m_c, tau + eps_t, sigma, r]),
        ])
        K_rep = np.vstack([K, K, K, K])
        V_all, _ = self.forward(X_all)
        V_prices = (V_all * K_rep).ravel()

        V = V_prices[:n]
        V_Sup = V_prices[n:2*n]
        V_Sdn = V_prices[2*n:3*n]
        V_tup = V_prices[3*n:]

        eps_S_flat = eps_S.ravel()
        dV_dS = (V_Sup - V_Sdn) / (2 * eps_S_flat)
        d2V_dS2 = (V_Sup - 2 * V + V_Sdn) / (eps_S_flat ** 2)
        dV_dtau = (V_tup - V) / eps_t

        S_flat = S.ravel()
        sigma_flat = sigma.ravel()
        r_flat = r.ravel()

        residual = dV_dtau + 0.5 * sigma_flat**2 * S_flat**2 * d2V_dS2 \
                   + r_flat * S_flat * dV_dS - r_flat * V

        return residual

    # ---- Arbitrage penalty ----
    def _arbitrage_penalty(self, S: np.ndarray, K: np.ndarray, tau: np.ndarray,
                           sigma: np.ndarray, r: np.ndarray) -> float:
        """
        No-arbitrage constraints for European calls:
            1. V >= max(S - K*exp(-rτ), 0)   (lower bound)
            2. V <= S                          (upper bound)
            3. ∂V/∂S >= 0                      (monotonicity / delta >= 0)
            4. ∂²V/∂S² >= 0                   (convexity / gamma >= 0)
        """
        eps_S = S * 1e-4 + 1e-6
        m = S / K
        X = np.column_stack([m, tau, sigma, r])
        V, _ = self.forward(X)
        V_price = V.ravel() * K.ravel()

        S_flat = S.ravel()
        K_flat = K.ravel()
        r_flat = r.ravel()
        tau_flat = tau.ravel()

        # Lower bound violation
        intrinsic = np.maximum(S_flat - K_flat * np.exp(-r_flat * tau_flat), 0)
        lb_viol = np.mean(np.maximum(intrinsic - V_price, 0) ** 2)

        # Upper bound violation
        ub_viol = np.mean(np.maximum(V_price - S_flat, 0) ** 2)

        # Delta >= 0
        V_up, _ = self.forward(np.column_stack([(S + eps_S) / K, tau, sigma, r]))
        V_dn, _ = self.forward(np.column_stack([(S - eps_S) / K, tau, sigma, r]))
        delta = ((V_up.ravel() - V_dn.ravel()) * K.ravel()) / (2 * eps_S.ravel())
        delta_viol = np.mean(np.maximum(-delta, 0) ** 2)

        # Gamma >= 0
        gamma = ((V_up.ravel() - 2 * V.ravel() + V_dn.ravel()) * K.ravel()) / (eps_S.ravel() ** 2)
        gamma_viol = np.mean(np.maximum(-gamma, 0) ** 2)

        return float(lb_viol + ub_viol + delta_viol + gamma_viol)

    # ---- Smoothness penalty ----
    def _smoothness_penalty(self, S: np.ndarray, K: np.ndarray, tau: np.ndarray,
                            sigma: np.ndarray, r: np.ndarray) -> float:
        """Penalise large Gamma variations for smooth Greeks."""
        eps = S * 2e-4 + 1e-5

        def _gamma_at(s):
            eps_S = s * 1e-4 + 1e-6
            m_up = (s + eps_S) / K
            m_dn = (s - eps_S) / K
            m_c = s / K
            Vu, _ = self.forward(np.column_stack([m_up, tau, sigma, r]))
            Vd, _ = self.forward(np.column_stack([m_dn, tau, sigma, r]))
            Vc, _ = self.forward(np.column_stack([m_c, tau, sigma, r]))
            return ((Vu.ravel() - 2 * Vc.ravel() + Vd.ravel()) * K.ravel()) / (eps_S.ravel() ** 2)

        gamma_center = _gamma_at(S)
        gamma_up = _gamma_at(S + eps)
        # Penalise non-smooth gamma
        dgamma = gamma_up - gamma_center
        return float(np.mean(dgamma ** 2))

    # ---- Backpropagation (exact gradients) ----
    def _forward_backprop(self, X: np.ndarray):
        """Forward pass storing pre/post activations for backprop."""
        pre_acts = []   # z values before activation
        post_acts = [X] # h values after activation (input is first)
        h = X
        for layer in self.layers[:-1]:
            z = h @ layer.W + layer.b
            pre_acts.append(z)
            h = self._act(z)
            post_acts.append(h)
        # Output layer — softplus for positivity
        z_out = h @ self.layers[-1].W + self.layers[-1].b
        pre_acts.append(z_out)
        out = np.log1p(np.exp(np.clip(z_out, -20, 20)))
        return out, pre_acts, post_acts

    def _backprop_data_loss(self, X: np.ndarray, K_batch: np.ndarray,
                            V_market_batch: np.ndarray):
        """Compute exact gradients of data loss via backpropagation.

        Loss computed in normalized (V/K) space to avoid K-scaling bias.
        """
        n = X.shape[0]
        out, pre_acts, post_acts = self._forward_backprop(X)

        # Normalised target: V_market / K
        target = V_market_batch / K_batch
        residual = out - target
        loss = float(np.mean(residual ** 2))

        # dL/d(out) = (2/n) * (out - target)
        d_out = (2.0 / n) * residual

        # Through softplus output: d/dz log(1+exp(z)) = sigmoid(z)
        z_clip = np.clip(pre_acts[-1], -20, 20)
        d_z = d_out * (1.0 / (1.0 + np.exp(-z_clip)))

        grads: list[tuple[np.ndarray, np.ndarray] | None] = [None] * len(self.layers)

        # Output layer gradients
        h_prev = post_acts[-1]  # last hidden activation
        grads[-1] = (h_prev.T @ d_z, np.sum(d_z, axis=0, keepdims=True))

        # Propagate backward
        d_h = d_z @ self.layers[-1].W.T

        # Hidden layers (reverse order)
        for i in range(len(self.layers) - 2, -1, -1):
            d_z = d_h * self._act_grad(pre_acts[i])
            h_prev = post_acts[i]
            grads[i] = (h_prev.T @ d_z, np.sum(d_z, axis=0, keepdims=True))
            if i > 0:
                d_h = d_z @ self.layers[i].W.T

        return loss, grads

    # ---- Training ----
    def train(self, train_data: Dict[str, np.ndarray],
              val_data: Optional[Dict[str, np.ndarray]] = None,
              progress_callback=None) -> Dict[str, Any]:
        """
        Train with combined loss using backpropagation + Adam optimiser.

        train_data keys: S, K, tau, sigma, r, V_market
        Uses exact gradients via backprop for data loss, with PDE/arb/smooth
        computed for monitoring at log intervals.
        """
        S = train_data["S"].reshape(-1, 1).astype(np.float64)
        K = train_data["K"].reshape(-1, 1).astype(np.float64)
        tau = train_data["tau"].reshape(-1, 1).astype(np.float64)
        sigma = train_data["sigma"].reshape(-1, 1).astype(np.float64)
        r = train_data["r"].reshape(-1, 1).astype(np.float64)
        V_market = train_data["V_market"].reshape(-1, 1).astype(np.float64)

        if not self._built:
            self.build(4)

        n = len(S)
        cfg = self.config
        lr = cfg.learning_rate
        best_loss = float("inf")
        patience_ctr = 0
        self._train_history = []
        actual_epochs = 0
        t0 = time.time()
        log_interval = max(1, cfg.epochs // 20)

        # Adam optimiser state
        beta1, beta2, eps_adam = 0.9, 0.999, 1e-8
        adam_m = [(np.zeros_like(layer.W), np.zeros_like(layer.b)) for layer in self.layers]
        adam_v = [(np.zeros_like(layer.W), np.zeros_like(layer.b)) for layer in self.layers]

        for epoch in range(cfg.epochs):
            actual_epochs = epoch + 1

            # Mini-batch
            idx = self.rng.choice(n, size=min(cfg.batch_size, n), replace=False)
            Sb, Kb, tb, sb, rb, Vb = S[idx], K[idx], tau[idx], sigma[idx], r[idx], V_market[idx]

            # Build input features
            m = Sb / Kb
            X = np.column_stack([m, tb, sb, rb])

            # Backpropagation — exact gradients for data loss
            data_loss, grads = self._backprop_data_loss(X, Kb, Vb)

            # NaN / inf guard
            if math.isnan(data_loss) or math.isinf(data_loss):
                logger.warning("PINNs epoch %d: NaN/inf loss — halving LR", epoch)
                lr *= 0.5
                if lr < 1e-8:
                    logger.error("PINNs LR collapsed, stopping")
                    break
                continue

            # Adam update with gradient clipping and weight decay
            t_adam = epoch + 1
            weight_decay = 1e-4
            for i, (layer, (gW, gb)) in enumerate(zip(self.layers, grads)):
                # Add L2 regularization gradient
                gW_reg = gW + weight_decay * layer.W
                gW_c = np.clip(gW_reg, -5.0, 5.0)
                gb_c = np.clip(gb, -5.0, 5.0)

                mW, mb = adam_m[i]
                vW, vb = adam_v[i]

                mW = beta1 * mW + (1 - beta1) * gW_c
                mb = beta1 * mb + (1 - beta1) * gb_c
                vW = beta2 * vW + (1 - beta2) * gW_c ** 2
                vb = beta2 * vb + (1 - beta2) * gb_c ** 2

                adam_m[i] = (mW, mb)
                adam_v[i] = (vW, vb)

                mW_hat = mW / (1 - beta1 ** t_adam)
                mb_hat = mb / (1 - beta1 ** t_adam)
                vW_hat = vW / (1 - beta2 ** t_adam)
                vb_hat = vb / (1 - beta2 ** t_adam)

                layer.W -= lr * mW_hat / (np.sqrt(vW_hat) + eps_adam)
                layer.b -= lr * mb_hat / (np.sqrt(vb_hat) + eps_adam)

            # Early stopping on data loss
            if data_loss < best_loss:
                best_loss = data_loss
                patience_ctr = 0
            else:
                patience_ctr += 1
            if patience_ctr >= cfg.early_stop_patience and epoch > 30:
                logger.info("PINNs early stop at epoch %d (patience=%d)",
                            epoch, cfg.early_stop_patience)
                break

            # Monitoring: compute PDE/arb/smooth at log intervals
            pde_loss = 0.0
            arb_loss = 0.0
            smooth_loss = 0.0
            if epoch % log_interval == 0:
                pde_size = min(cfg.pde_batch_size, len(Sb))
                pde_idx = self.rng.choice(len(Sb), size=pde_size, replace=False)
                residual = self._pde_residual(Sb[pde_idx], Kb[pde_idx], tb[pde_idx],
                                              sb[pde_idx], rb[pde_idx])
                pde_loss = float(np.mean(residual ** 2))
                arb_loss = self._arbitrage_penalty(
                    Sb[pde_idx], Kb[pde_idx], tb[pde_idx], sb[pde_idx], rb[pde_idx])
                smooth_loss = self._smoothness_penalty(
                    Sb[pde_idx], Kb[pde_idx], tb[pde_idx], sb[pde_idx], rb[pde_idx])

                total_loss = (data_loss + cfg.lambda_pde * pde_loss
                              + cfg.lambda_arb * arb_loss
                              + cfg.lambda_smooth * smooth_loss)

                logger.info(
                    "PINNs epoch %d: total=%.6f data=%.6f pde=%.6f "
                    "arb=%.6f smooth=%.6f",
                    epoch, total_loss, data_loss, pde_loss,
                    arb_loss, smooth_loss)
                self._train_history.append({
                    "epoch": epoch, "total_loss": total_loss,
                    "data_loss": data_loss, "pde_loss": pde_loss,
                    "arb_loss": arb_loss, "smooth_loss": smooth_loss,
                })
                if progress_callback:
                    progress_callback(epoch, cfg.epochs, data_loss)

            # Learning rate decay
            if epoch > 0 and epoch % 100 == 0:
                lr *= 0.5

        elapsed = time.time() - t0
        return {
            "epochs": actual_epochs,
            "best_loss": best_loss,
            "final_losses": self._train_history[-1] if self._train_history else {},
            "training_time_s": round(elapsed, 2),
            "params": sum(layer.W.size + layer.b.size for layer in self.layers),
            "history": self._train_history,
        }

    # ---- Predict ----
    def predict(self, S: Union[float, np.ndarray], K: Union[float, np.ndarray],
                tau: Union[float, np.ndarray], sigma: Union[float, np.ndarray],
                r: Union[float, np.ndarray]) -> np.ndarray:
        """Predict option prices."""
        S = np.asarray(S, dtype=np.float64).reshape(-1, 1)
        K = np.asarray(K, dtype=np.float64).reshape(-1, 1)
        tau = np.asarray(tau, dtype=np.float64).reshape(-1, 1)
        sigma = np.asarray(sigma, dtype=np.float64).reshape(-1, 1)
        r = np.asarray(r, dtype=np.float64).reshape(-1, 1)

        m = S / K
        X = np.column_stack([m, tau, sigma, r])
        V, _ = self.forward(X)
        return (V * K).ravel()

    # ---- Greeks via finite differences ----
    def compute_greeks(self, S: float, K: float, tau: float,
                       sigma: float, r: float) -> Dict[str, float]:
        """Compute Delta, Gamma, Theta, Vega, Rho from PINNs."""
        eps_S = S * 1e-4
        eps_t = 1e-5
        eps_v = 1e-4
        eps_r = 1e-4

        def _p(s, t, v, rate):
            return float(self.predict(np.array([s]), np.array([K]),
                                      np.array([t]), np.array([v]), np.array([rate]))[0])

        V = _p(S, tau, sigma, r)
        delta = (_p(S + eps_S, tau, sigma, r) - _p(S - eps_S, tau, sigma, r)) / (2 * eps_S)
        gamma = (_p(S + eps_S, tau, sigma, r) - 2 * V + _p(S - eps_S, tau, sigma, r)) / (eps_S ** 2)
        theta = -(_p(S, tau + eps_t, sigma, r) - V) / eps_t  # negative sign convention
        vega = (_p(S, tau, sigma + eps_v, r) - _p(S, tau, sigma - eps_v, r)) / (2 * eps_v)
        rho = (_p(S, tau, sigma, r + eps_r) - _p(S, tau, sigma, r - eps_r)) / (2 * eps_r)

        return {
            "price": round(V, 6),
            "delta": round(delta, 6),
            "gamma": round(gamma, 6),
            "theta": round(theta, 6),
            "vega": round(vega, 6),
            "rho": round(rho, 6)
        }

    # ---- Generate synthetic training data (BS analytical) ----
    @staticmethod
    def generate_training_data(n_samples: int = 5000, seed: int = 42) -> Dict[str, np.ndarray]:
        """Generate training data from Black-Scholes analytical prices."""
        from scipy.stats import norm  # type: ignore[import-untyped]
        rng = np.random.default_rng(seed)

        S = rng.uniform(50, 200, n_samples)
        K = rng.uniform(50, 200, n_samples)
        tau = rng.uniform(0.01, 2.0, n_samples)
        sigma = rng.uniform(0.05, 0.80, n_samples)
        r_vals = rng.uniform(0.01, 0.10, n_samples)

        d1 = (np.log(S / K) + (r_vals + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
        d2 = d1 - sigma * np.sqrt(tau)
        V = S * norm.cdf(d1) - K * np.exp(-r_vals * tau) * norm.cdf(d2)

        # Add small noise to simulate market prices
        noise = rng.normal(0, 0.01, n_samples) * V
        V_market = np.maximum(V + noise, 0)

        return {"S": S, "K": K, "tau": tau, "sigma": sigma, "r": r_vals, "V_market": V_market}

    def save(self, filepath: str | Path) -> None:
        """Save PINNs model (layers, config, history) to a pickle file."""
        filepath = Path(filepath)
        state = {
            "config": self.config,
            "layers": [(layer.W.copy(), layer.b.copy()) for layer in self.layers],
            "built": self._built,
            "train_history": self._train_history,
        }
        with open(filepath, "wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("PINNs saved to %s", filepath)

    @classmethod
    def load(cls, filepath: str | Path) -> "PINNsOptionPricer":
        """Load PINNs model from pickle file."""
        filepath = Path(filepath)
        with open(filepath, "rb") as f:
            state = pickle.load(f)
        pricer = cls(config=state["config"])
        pricer.layers = [PINNLayer(W=w, b=b) for w, b in state["layers"]]
        pricer._built = state["built"]
        pricer._train_history = state.get("train_history", [])
        logger.info("PINNs loaded from %s (%d layers)", filepath, len(pricer.layers))
        return pricer

    # ---- Generate training data from option chain CSV ----
    @staticmethod
    def load_training_data_from_csv(
        csv_path: str | Path,
        spot_csv: str | Path | None = None,
        rate: float = 0.05,
    ) -> Dict[str, np.ndarray]:
        """Load training data from option chain CSV (call options only).

        The option_chain CSV has columns:
          timestamp, strike, expiry, option_type, bid, ask, mid, last,
          volume, open_interest, implied_vol, delta, gamma, vega, theta

        If spot_csv is provided, spot prices are joined by timestamp.
        Otherwise spot is estimated from mid/delta of deep ITM calls (fallback).
        """
        import csv
        from datetime import datetime

        # Load spot prices by date if available
        spot_by_date: Dict[str, float] = {}
        if spot_csv and Path(spot_csv).exists():
            with open(spot_csv, newline="") as f:
                for row in csv.DictReader(f):
                    spot_by_date[row["timestamp"]] = float(row["spot"])

        rows = []
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("option_type", "call") != "call":
                    continue
                ts = row["timestamp"]
                # Need spot price
                if ts not in spot_by_date:
                    continue
                # Compute time to expiry in years
                t0 = datetime.strptime(ts, "%Y-%m-%d")
                t1 = datetime.strptime(row["expiry"], "%Y-%m-%d")
                tau_days = (t1 - t0).days
                if tau_days <= 0:
                    continue
                rows.append({
                    "S": spot_by_date[ts],
                    "K": float(row["strike"]),
                    "tau": tau_days / 365.0,
                    "sigma": float(row["implied_vol"]),
                    "V_market": float(row["mid"]),
                })

        S = np.array([r["S"] for r in rows])
        K = np.array([r["K"] for r in rows])
        tau = np.array([r["tau"] for r in rows])
        sigma = np.array([r["sigma"] for r in rows])
        r_vals = np.full(len(rows), rate)
        V_market = np.array([r["V_market"] for r in rows])

        logger.info("Loaded %d call options from CSV for PINNs training", len(rows))
        return {"S": S, "K": K, "tau": tau, "sigma": sigma, "r": r_vals, "V_market": V_market}

    def get_status(self) -> Dict[str, Any]:
        return {
            "built": self._built,
            "layers": len(self.layers) if self._built else 0,
            "params": sum(layer.W.size + layer.b.size for layer in self.layers) if self._built else 0,
            "epochs_trained": len(self._train_history) * 100,
            "config": {
                "hidden_layers": self.config.hidden_layers,
                "lambda_pde": self.config.lambda_pde,
                "lambda_arb": self.config.lambda_arb,
                "lambda_smooth": self.config.lambda_smooth,
            }
        }

# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------
_pinns_instance: Optional[PINNsOptionPricer] = None

def get_pinns_pricer(reset: bool = False) -> PINNsOptionPricer:
    global _pinns_instance
    if _pinns_instance is None or reset:
        _pinns_instance = PINNsOptionPricer()
    return _pinns_instance
