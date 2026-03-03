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
import time
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Any

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
    epochs: int = 2000
    batch_size: int = 256
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
        return {"tanh": _tanh_grad, "softplus": _softplus_grad, "gelu": _tanh_grad}[name]

    # ---- build ----
    def build(self, input_dim: int = 4):
        dims = [input_dim] + self.config.hidden_layers + [1]
        self.layers = []
        for i in range(len(dims) - 1):
            W = _xavier_init(dims[i], dims[i + 1], self.rng)
            b = np.zeros((1, dims[i + 1]), dtype=np.float64)
            self.layers.append(PINNLayer(W=W, b=b))
        self._built = True
        total_params = sum(l.W.size + l.b.size for l in self.layers)
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
        # Output layer — softplus to ensure positivity
        z_out = h @ self.layers[-1].W + self.layers[-1].b
        out = _softplus(z_out)
        activations.append(z_out)
        return out, activations

    # ---- PDE residual via finite differences ----
    def _pde_residual(self, S: np.ndarray, K: np.ndarray, tau: np.ndarray,
                      sigma: np.ndarray, r: np.ndarray) -> np.ndarray:
        """
        Compute Black-Scholes PDE residual:
            R = ∂V/∂τ + ½σ²S²∂²V/∂S² + rS∂V/∂S - rV

        Uses central finite differences through the network.
        """
        eps_S = S * 1e-4 + 1e-6
        eps_t = 1e-5

        def _price(s, t):
            m = s / K
            X = np.column_stack([m, t, sigma, r])
            V, _ = self.forward(X)
            return V.ravel() * K.ravel()

        V = _price(S, tau)
        V_Sup = _price(S + eps_S, tau)
        V_Sdn = _price(S - eps_S, tau)
        V_tup = _price(S, tau + eps_t)

        dV_dS = (V_Sup - V_Sdn) / (2 * eps_S.ravel())
        d2V_dS2 = (V_Sup - 2 * V + V_Sdn) / (eps_S.ravel() ** 2)
        dV_dtau = (V_tup - V) / eps_t  # ∂V/∂τ (positive τ direction)

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

    # ---- SPSA gradient estimation ----
    def _spsa_gradient(self, loss_fn, perturbation: float = 1e-3):
        """Simultaneous Perturbation Stochastic Approximation."""
        grads = []
        for layer in self.layers:
            dW = self.rng.choice([-1.0, 1.0], size=layer.W.shape)
            db = self.rng.choice([-1.0, 1.0], size=layer.b.shape)

            # Perturb +
            layer.W += perturbation * dW
            layer.b += perturbation * db
            loss_plus = loss_fn()

            # Perturb -
            layer.W -= 2 * perturbation * dW
            layer.b -= 2 * perturbation * db
            loss_minus = loss_fn()

            # Restore
            layer.W += perturbation * dW
            layer.b += perturbation * db

            grad_W = (loss_plus - loss_minus) / (2 * perturbation * dW)
            grad_b = (loss_plus - loss_minus) / (2 * perturbation * db)
            grads.append((grad_W, grad_b))
        return grads

    # ---- Training ----
    def train(self, train_data: Dict[str, np.ndarray],
              val_data: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, Any]:
        """
        Train with combined loss.

        train_data keys: S, K, tau, sigma, r, V_market
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
        t0 = time.time()

        for epoch in range(cfg.epochs):
            # Mini-batch
            idx = self.rng.choice(n, size=min(cfg.batch_size, n), replace=False)
            Sb, Kb, tb, sb, rb, Vb = S[idx], K[idx], tau[idx], sigma[idx], r[idx], V_market[idx]

            # Forward
            m = Sb / Kb
            X = np.column_stack([m, tb, sb, rb])
            V_pred, _ = self.forward(X)
            V_pred_price = V_pred * Kb

            # Data loss
            data_loss = float(np.mean((V_pred_price - Vb) ** 2))

            # PDE residual loss (subsample for speed)
            pde_idx = self.rng.choice(len(Sb), size=min(64, len(Sb)), replace=False)
            residual = self._pde_residual(Sb[pde_idx], Kb[pde_idx], tb[pde_idx],
                                          sb[pde_idx], rb[pde_idx])
            pde_loss = float(np.mean(residual ** 2))

            # Arbitrage loss
            arb_loss = self._arbitrage_penalty(Sb[pde_idx], Kb[pde_idx], tb[pde_idx],
                                               sb[pde_idx], rb[pde_idx])

            # Smoothness loss
            smooth_loss = self._smoothness_penalty(Sb[pde_idx], Kb[pde_idx], tb[pde_idx],
                                                   sb[pde_idx], rb[pde_idx])

            total_loss = data_loss + cfg.lambda_pde * pde_loss + \
                         cfg.lambda_arb * arb_loss + cfg.lambda_smooth * smooth_loss

            # SPSA update
            def _loss_fn():
                Vp, _ = self.forward(X)
                dl = float(np.mean((Vp * Kb - Vb) ** 2))
                res = self._pde_residual(Sb[pde_idx], Kb[pde_idx], tb[pde_idx],
                                         sb[pde_idx], rb[pde_idx])
                return dl + cfg.lambda_pde * float(np.mean(res ** 2))

            grads = self._spsa_gradient(_loss_fn)
            for layer, (gW, gb) in zip(self.layers, grads):
                layer.W -= lr * np.clip(gW, -1.0, 1.0)
                layer.b -= lr * np.clip(gb, -1.0, 1.0)

            if total_loss < best_loss:
                best_loss = total_loss

            if epoch % 100 == 0:
                logger.info(f"PINNs epoch {epoch}: total={total_loss:.6f} "
                            f"data={data_loss:.6f} pde={pde_loss:.6f} "
                            f"arb={arb_loss:.6f} smooth={smooth_loss:.6f}")
                self._train_history.append({
                    "epoch": epoch, "total_loss": total_loss,
                    "data_loss": data_loss, "pde_loss": pde_loss,
                    "arb_loss": arb_loss, "smooth_loss": smooth_loss
                })

            # Learning rate decay
            if epoch > 0 and epoch % 500 == 0:
                lr *= 0.5

        elapsed = time.time() - t0
        return {
            "epochs": cfg.epochs,
            "best_loss": best_loss,
            "final_losses": self._train_history[-1] if self._train_history else {},
            "training_time_s": round(elapsed, 2),
            "params": sum(l.W.size + l.b.size for l in self.layers),
            "history": self._train_history
        }

    # ---- Predict ----
    def predict(self, S: np.ndarray, K: np.ndarray, tau: np.ndarray,
                sigma: np.ndarray, r: np.ndarray) -> np.ndarray:
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
        from scipy.stats import norm
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

    def get_status(self) -> Dict[str, Any]:
        return {
            "built": self._built,
            "layers": len(self.layers) if self._built else 0,
            "params": sum(l.W.size + l.b.size for l in self.layers) if self._built else 0,
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

def get_pinns_pricer() -> PINNsOptionPricer:
    global _pinns_instance
    if _pinns_instance is None:
        _pinns_instance = PINNsOptionPricer()
    return _pinns_instance
