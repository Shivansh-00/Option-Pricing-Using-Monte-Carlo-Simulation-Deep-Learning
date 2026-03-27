"""
Transformer-Based Volatility Surface Model
============================================
Multi-head attention model for predicting the full implied volatility surface.

Architecture:
    - Positional encoding for maturity dimension
    - Cross-attention between strike and time-to-maturity
    - Surface smoothness constraints in loss function
    - Regime-conditioned generation

Inputs:  [strike/spot moneyness, τ, regime, historical_vol_features]
Outputs: Implied volatility at each (K, τ) grid point

Feeds into:
    - Monte Carlo simulations (local vol surface)
    - PINNs (implied vol input)
    - Arbitrage detection (surface consistency)
"""

from __future__ import annotations
import numpy as np
import math
import time
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Any

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════
#  Positional Encoding
# ═══════════════════════════════════════════════════════════════════════

def sinusoidal_encoding(positions: np.ndarray, d_model: int) -> np.ndarray:
    """
    Sinusoidal positional encoding for the maturity dimension.
    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    pe = np.zeros((len(positions), d_model))
    for i in range(0, d_model, 2):
        div = 10000.0 ** (i / d_model)
        pe[:, i] = np.sin(positions / div)
        if i + 1 < d_model:
            pe[:, i + 1] = np.cos(positions / div)
    return pe


# ═══════════════════════════════════════════════════════════════════════
#  Attention Mechanism (pure NumPy)
# ═══════════════════════════════════════════════════════════════════════

def scaled_dot_product_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                                  mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Attention(Q, K, V) = softmax(QK^T / √d_k) V
    """
    d_k = Q.shape[-1]
    scores = (Q @ K.transpose(0, 2, 1)) / math.sqrt(d_k)
    if mask is not None:
        scores = np.where(mask, scores, -1e9)
    # Stable softmax
    scores_max = scores.max(axis=-1, keepdims=True)
    exp_scores = np.exp(scores - scores_max)
    attn_weights = exp_scores / (exp_scores.sum(axis=-1, keepdims=True) + 1e-10)
    return attn_weights @ V


class MultiHeadAttention:
    """Multi-head attention with learnable projections."""

    def __init__(self, d_model: int, n_heads: int, rng: np.random.Generator):
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        scale = math.sqrt(2.0 / (d_model + self.d_k))

        self.W_Q = rng.normal(0, scale, (d_model, d_model))
        self.W_K = rng.normal(0, scale, (d_model, d_model))
        self.W_V = rng.normal(0, scale, (d_model, d_model))
        self.W_O = rng.normal(0, scale, (d_model, d_model))
        self.b_O = np.zeros(d_model)

    def forward(self, query: np.ndarray, key: np.ndarray, value: np.ndarray) -> np.ndarray:
        batch = query.shape[0]
        seq_q = query.shape[1]
        seq_k = key.shape[1]

        Q = (query @ self.W_Q).reshape(batch, seq_q, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        K = (key @ self.W_K).reshape(batch, seq_k, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        V = (value @ self.W_V).reshape(batch, seq_k, self.n_heads, self.d_k).transpose(0, 2, 1, 3)

        # Attention per head
        attn_out = np.zeros_like(Q)
        for h in range(self.n_heads):
            attn_out[:, h] = scaled_dot_product_attention(
                Q[:, h:h+1].reshape(batch, seq_q, self.d_k),
                K[:, h:h+1].reshape(batch, seq_k, self.d_k),
                V[:, h:h+1].reshape(batch, seq_k, self.d_k)
            )

        # Concat heads
        concat = attn_out.transpose(0, 2, 1, 3).reshape(batch, seq_q, self.d_model)
        return concat @ self.W_O + self.b_O


class FeedForward:
    """Position-wise feed-forward network."""

    def __init__(self, d_model: int, d_ff: int, rng: np.random.Generator):
        scale1 = math.sqrt(2.0 / (d_model + d_ff))
        scale2 = math.sqrt(2.0 / (d_ff + d_model))
        self.W1 = rng.normal(0, scale1, (d_model, d_ff))
        self.b1 = np.zeros(d_ff)
        self.W2 = rng.normal(0, scale2, (d_ff, d_model))
        self.b2 = np.zeros(d_model)

    def forward(self, x: np.ndarray) -> np.ndarray:
        h = np.maximum(0, x @ self.W1 + self.b1)  # GELU approx as ReLU
        return h @ self.W2 + self.b2


class LayerNorm:
    """Layer normalisation."""

    def __init__(self, d_model: int):
        self.gamma = np.ones(d_model)
        self.beta = np.zeros(d_model)

    def forward(self, x: np.ndarray) -> np.ndarray:
        mean = x.mean(axis=-1, keepdims=True)
        std = x.std(axis=-1, keepdims=True) + 1e-6
        return self.gamma * (x - mean) / std + self.beta


class TransformerBlock:
    """Single Transformer encoder block."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, rng: np.random.Generator):
        self.self_attn = MultiHeadAttention(d_model, n_heads, rng)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, rng)
        self.ff = FeedForward(d_model, d_ff, rng)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)

    def forward(self, x: np.ndarray, cross_input: Optional[np.ndarray] = None) -> np.ndarray:
        # Self-attention + residual
        attn_out = self.self_attn.forward(x, x, x)
        x = self.norm1.forward(x + attn_out)

        # Cross-attention (strike ↔ maturity)
        if cross_input is not None:
            cross_out = self.cross_attn.forward(x, cross_input, cross_input)
            x = self.norm2.forward(x + cross_out)

        # Feed-forward + residual
        ff_out = self.ff.forward(x)
        x = self.norm3.forward(x + ff_out)
        return x


# ═══════════════════════════════════════════════════════════════════════
#  Vol Surface Transformer
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class VolSurfaceConfig:
    d_model: int = 32
    n_heads: int = 4
    n_layers: int = 2
    d_ff: int = 64
    n_strikes: int = 15      # strike grid points
    n_maturities: int = 10   # maturity grid points
    learning_rate: float = 1e-3
    epochs: int = 500
    batch_size: int = 32
    lambda_smooth: float = 0.1   # surface smoothness penalty
    lambda_calendar: float = 0.05  # calendar arbitrage penalty
    seed: int = 42


class VolSurfaceTransformer:
    """
    Transformer model for implied volatility surface prediction.

    Input features per grid point:
        [moneyness, τ, regime_embedding, historical_vol_features]

    Output: σ_impl for each (K, τ) grid point
    """

    def __init__(self, config: Optional[VolSurfaceConfig] = None):
        self.config = config or VolSurfaceConfig()
        self.rng = np.random.default_rng(self.config.seed)
        self._built = False
        self._train_history: List[Dict] = []

    def build(self, input_dim: int = 6):
        cfg = self.config
        d = cfg.d_model
        scale = math.sqrt(2.0 / (input_dim + d))

        # Input projection
        self.W_input = self.rng.normal(0, scale, (input_dim, d))
        self.b_input = np.zeros(d)

        # Regime embedding (3 regimes → d_model)
        self.regime_embed = self.rng.normal(0, 0.1, (3, d))

        # Transformer blocks
        self.blocks: List[TransformerBlock] = []
        for _ in range(cfg.n_layers):
            self.blocks.append(TransformerBlock(d, cfg.n_heads, cfg.d_ff, self.rng))

        # Output head: d_model → 1 (implied vol)
        scale_out = math.sqrt(2.0 / (d + 1))
        self.W_out = self.rng.normal(0, scale_out, (d, 1))
        self.b_out = np.array([0.2])  # initialise near typical vol

        self._built = True
        total = sum(
            sum(a.size for a in [
                block.self_attn.W_Q, block.self_attn.W_K, block.self_attn.W_V, block.self_attn.W_O,
                block.cross_attn.W_Q, block.cross_attn.W_K, block.cross_attn.W_V, block.cross_attn.W_O,
                block.ff.W1, block.ff.W2
            ])
            for block in self.blocks
        )
        total += self.W_input.size + self.W_out.size + self.regime_embed.size
        logger.info(f"VolSurfaceTransformer built: {cfg.n_layers} layers, ~{total} params")

    def _forward(self, moneyness: np.ndarray, tau: np.ndarray,
                 regime: np.ndarray, hist_vol: np.ndarray) -> np.ndarray:
        """
        Forward pass.
        All inputs: (batch, n_points)
        Returns: (batch, n_points) implied vols
        """
        if not self._built:
            self.build()

        batch = moneyness.shape[0]
        n_pts = moneyness.shape[1]

        # Build feature matrix: (batch, n_points, input_dim)
        pe = sinusoidal_encoding(tau[0], self.config.d_model)  # (n_points, d_model)

        # Input features: [moneyness, tau, hist_vol_features...]
        features = np.stack([moneyness, tau] + [hist_vol[..., i] for i in range(hist_vol.shape[-1])], axis=-1)

        # Project to d_model
        x = features @ self.W_input[:features.shape[-1]] + self.b_input

        # Add positional encoding (maturity)
        x = x + pe[np.newaxis, :n_pts, :]

        # Add regime embedding
        regime_int = regime.astype(int).reshape(batch, 1)
        r_embed = self.regime_embed[regime_int[:, 0]]  # (batch, d_model)
        x = x + r_embed[:, np.newaxis, :]

        # Transformer blocks
        for block in self.blocks:
            x = block.forward(x)

        # Output head → implied vol (softplus for positivity)
        vol_raw = (x @ self.W_out + self.b_out).squeeze(-1)
        vol = np.log1p(np.exp(np.clip(vol_raw, -5, 5)))  # softplus
        return vol

    def predict_surface(self, S: float = 100.0, strikes: Optional[np.ndarray] = None,
                        maturities: Optional[np.ndarray] = None,
                        regime: int = 0, hist_vol: float = 0.2) -> Dict[str, Any]:
        """
        Generate a full implied volatility surface.

        Returns:
            Grid of (strikes × maturities) implied vols
        """
        if strikes is None:
            strikes = np.linspace(0.7, 1.3, self.config.n_strikes) * S
        if maturities is None:
            maturities = np.linspace(0.05, 2.0, self.config.n_maturities)

        n_k = len(strikes)
        n_t = len(maturities)
        moneyness_grid = np.zeros((1, n_k * n_t))
        tau_grid = np.zeros((1, n_k * n_t))

        idx = 0
        for k in strikes:
            for t in maturities:
                moneyness_grid[0, idx] = k / S
                tau_grid[0, idx] = t
                idx += 1

        hist_features = np.full((1, n_k * n_t, 4), hist_vol)
        regime_arr = np.array([[regime]])

        vol_flat = self._forward(moneyness_grid, tau_grid, regime_arr, hist_features)
        vol_surface = vol_flat[0].reshape(n_k, n_t)

        # Clip to reasonable bounds
        vol_surface = np.clip(vol_surface, 0.01, 2.0)

        return {
            "strikes": strikes.tolist(),
            "maturities": maturities.tolist(),
            "surface": vol_surface.tolist(),
            "regime": regime,
            "spot": S,
            "stats": {
                "mean_vol": round(float(vol_surface.mean()), 4),
                "min_vol": round(float(vol_surface.min()), 4),
                "max_vol": round(float(vol_surface.max()), 4),
                "atm_vol": round(float(vol_surface[n_k // 2, n_t // 2]), 4),
                "skew": round(float(vol_surface[0, n_t // 2] - vol_surface[-1, n_t // 2]), 4),
                "term_structure": round(float(vol_surface[n_k // 2, -1] - vol_surface[n_k // 2, 0]), 4),
            }
        }

    def _smoothness_loss(self, vol_surface: np.ndarray) -> float:
        """Penalise non-smooth surface (high second derivatives)."""
        if vol_surface.ndim == 1:
            return 0.0
        # Strike direction smoothness (d²σ/dK²)
        if vol_surface.shape[0] > 2:
            d2_k = np.diff(vol_surface, n=2, axis=0)
            k_smooth = float(np.mean(d2_k ** 2))
        else:
            k_smooth = 0.0
        # Maturity direction smoothness (d²σ/dτ²)
        if vol_surface.shape[1] > 2:
            d2_t = np.diff(vol_surface, n=2, axis=1)
            t_smooth = float(np.mean(d2_t ** 2))
        else:
            t_smooth = 0.0
        return k_smooth + t_smooth

    def _calendar_arb_loss(self, vol_surface: np.ndarray, maturities: np.ndarray) -> float:
        """
        Calendar arbitrage: total implied variance w(τ) = σ²τ must be non-decreasing.
        Penalise violations.
        """
        if vol_surface.shape[1] < 2:
            return 0.0
        total_var = vol_surface ** 2 * maturities[np.newaxis, :]
        diffs = np.diff(total_var, axis=1)
        violations = np.maximum(-diffs, 0)
        return float(np.mean(violations ** 2))

    def train(self, market_data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Train on market implied volatility data.

        market_data keys: moneyness, tau, regime, hist_vol, iv_target
        """
        if not self._built:
            self.build()

        moneyness = market_data["moneyness"]
        tau = market_data["tau"]
        regime = market_data["regime"]
        hist_vol = market_data["hist_vol"]
        iv_target = market_data["iv_target"]

        cfg = self.config
        n = moneyness.shape[0]
        lr = cfg.learning_rate
        best_loss = float("inf")
        self._train_history = []  # Clear history for fresh training run
        t0 = time.time()

        # Evaluation grid for smoothness/calendar arbitrage regularization
        n_k_grid, n_t_grid = 5, 3
        eval_m = np.linspace(0.8, 1.2, n_k_grid)
        eval_t = np.linspace(0.1, 1.5, n_t_grid)
        eval_mm, eval_tt = np.meshgrid(eval_m, eval_t)
        eval_m_flat = eval_mm.ravel()                         # (15,)
        eval_t_flat = eval_tt.ravel()                         # (15,)
        n_eval = len(eval_m_flat)
        eval_r_flat = np.zeros((1,), dtype=int)               # batch=1, regime=0
        eval_h_flat = np.full((1, n_eval, 4), 0.2)            # (batch=1, n_pts=15, 4)

        def _compute_reg_losses():
            """Compute smoothness and calendar arb losses on eval grid."""
            iv_grid = self._forward(
                eval_m_flat.reshape(1, -1),
                eval_t_flat.reshape(1, -1),
                eval_r_flat,
                eval_h_flat,
            )
            iv_surface = iv_grid.reshape(n_t_grid, n_k_grid)
            sl = self._smoothness_loss(iv_surface.T)
            cl = self._calendar_arb_loss(iv_surface.T, eval_t)
            return sl, cl

        for epoch in range(cfg.epochs):
            idx = self.rng.choice(n, min(cfg.batch_size, n), replace=False)
            m_b = moneyness[idx]
            t_b = tau[idx]
            r_b = regime[idx]
            h_b = hist_vol[idx]
            iv_b = iv_target[idx]

            # Forward
            iv_pred = self._forward(m_b, t_b, r_b, h_b)
            data_loss = float(np.mean((iv_pred - iv_b) ** 2))
            smooth_loss, calendar_loss = _compute_reg_losses()
            total_loss = data_loss + cfg.lambda_smooth * smooth_loss + cfg.lambda_calendar * calendar_loss

            # SPSA update
            pert = 1e-3
            all_params = [self.W_input, self.b_input, self.W_out, self.b_out, self.regime_embed]
            for block in self.blocks:
                all_params.extend([block.self_attn.W_Q, block.self_attn.W_K,
                                   block.self_attn.W_V, block.self_attn.W_O,
                                   block.ff.W1, block.ff.W2])

            def _total_loss():
                dl = float(np.mean((self._forward(m_b, t_b, r_b, h_b) - iv_b) ** 2))
                sl, cl = _compute_reg_losses()
                return dl + cfg.lambda_smooth * sl + cfg.lambda_calendar * cl

            for param in all_params:
                dp = self.rng.choice([-1.0, 1.0], size=param.shape)
                param += pert * dp
                l_plus = _total_loss()
                param -= 2 * pert * dp
                l_minus = _total_loss()
                param += pert * dp
                grad = (l_plus - l_minus) / (2 * pert * dp)
                param -= lr * np.clip(grad, -1, 1)

            if total_loss < best_loss:
                best_loss = total_loss

            if epoch % 50 == 0:
                logger.info(f"VolTransformer epoch {epoch}: total={total_loss:.6f} "
                            f"data={data_loss:.6f} smooth={smooth_loss:.6f} calendar={calendar_loss:.6f}")
                self._train_history.append({
                    "epoch": epoch, "loss": total_loss,
                    "data_loss": data_loss, "smooth_loss": smooth_loss,
                    "calendar_loss": calendar_loss,
                })

            if epoch > 0 and epoch % 200 == 0:
                lr *= 0.5

        elapsed = time.time() - t0
        return {
            "epochs": cfg.epochs,
            "best_loss": best_loss,
            "final_smooth_loss": smooth_loss,
            "final_calendar_loss": calendar_loss,
            "training_time_s": round(elapsed, 2),
            "history": self._train_history
        }

    @staticmethod
    def generate_synthetic_surface(n_samples: int = 500, seed: int = 42) -> Dict[str, np.ndarray]:
        """
        Generate synthetic vol surface data using SABR-like parametric model.
        """
        rng = np.random.default_rng(seed)
        n_points = 20  # grid points per sample

        all_m, all_t, all_r, all_h, all_iv = [], [], [], [], []

        for i in range(n_samples):
            moneyness = rng.uniform(0.7, 1.3, n_points)
            tau = rng.uniform(0.05, 2.0, n_points)
            regime = rng.choice([0, 1, 2])

            # SABR-like vol smile
            atm_vol = 0.15 + 0.1 * regime + rng.normal(0, 0.02)
            skew = -0.1 * (1 + 0.5 * regime) + rng.normal(0, 0.01)
            smile = 0.05 * (1 + 0.3 * regime) + rng.normal(0, 0.005)
            term = 0.02 * rng.normal(0, 0.01)

            iv = atm_vol + skew * (moneyness - 1) + smile * (moneyness - 1)**2 + term * np.sqrt(tau)
            iv = np.clip(iv + rng.normal(0, 0.005, n_points), 0.02, 1.5)

            hist_vol_features = np.full((n_points, 4), atm_vol) + rng.normal(0, 0.01, (n_points, 4))

            all_m.append(moneyness.reshape(1, -1))
            all_t.append(tau.reshape(1, -1))
            all_r.append(np.array([[regime]]))
            all_h.append(hist_vol_features.reshape(1, n_points, 4))
            all_iv.append(iv.reshape(1, -1))

        return {
            "moneyness": np.concatenate(all_m, axis=0),
            "tau": np.concatenate(all_t, axis=0),
            "regime": np.concatenate(all_r, axis=0),
            "hist_vol": np.concatenate(all_h, axis=0),
            "iv_target": np.concatenate(all_iv, axis=0)
        }

    def get_status(self) -> Dict[str, Any]:
        return {
            "built": self._built,
            "config": {
                "d_model": self.config.d_model,
                "n_heads": self.config.n_heads,
                "n_layers": self.config.n_layers,
                "n_strikes": self.config.n_strikes,
                "n_maturities": self.config.n_maturities,
            },
            "epochs_trained": len(self._train_history) * 50,
        }


# Singleton
_vol_transformer: Optional[VolSurfaceTransformer] = None

def get_vol_surface_transformer() -> VolSurfaceTransformer:
    global _vol_transformer
    if _vol_transformer is None:
        _vol_transformer = VolSurfaceTransformer()
    return _vol_transformer
