"""
vol_models.py — ML Volatility Models (Baseline + Advanced)
============================================================
Production ML models for volatility forecasting.
All models conform to a unified interface for training, prediction,
and serialization.

Implemented:
  1. Ridge / Lasso regression
  2. Random Forest
  3. Gradient Boosting (XGBoost‑style via sklearn)
  4. Ensemble (stacking)
  5. LSTM (PyTorch‑free, NumPy‑based for portability)
  6. Temporal CNN (NumPy‑based)
"""
from __future__ import annotations

import json
import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional

import numpy as np

# sklearn imports — available via requirements.txt
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    StackingRegressor,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# ─── Model Interface ───────────────────────────────────────────
@dataclass
class ModelResult:
    """Prediction result with metadata."""
    predictions: np.ndarray
    model_name: str
    train_time_ms: float = 0.0
    inference_time_ms: float = 0.0
    feature_importance: dict[str, float] = field(default_factory=dict)
    hyperparams: dict[str, Any] = field(default_factory=dict)


class VolModel(ABC):
    """Base class for all volatility models."""

    name: str = "base"

    @abstractmethod
    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray | None = None, y_val: np.ndarray | None = None) -> dict:
        """Train the model. Returns training metadata."""
        ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions."""
        ...

    @abstractmethod
    def get_feature_importance(self, feature_names: list[str]) -> dict[str, float]:
        """Return feature importance scores."""
        ...

    def predict_with_meta(self, X: np.ndarray, feature_names: list[str] | None = None) -> ModelResult:
        t0 = time.perf_counter()
        preds = self.predict(X)
        inf_time = (time.perf_counter() - t0) * 1000
        fi = self.get_feature_importance(feature_names or []) if feature_names else {}
        return ModelResult(
            predictions=preds,
            model_name=self.name,
            inference_time_ms=inf_time,
            feature_importance=fi,
        )


# ─── 1. Ridge Regression ───────────────────────────────────────
class RidgeVolModel(VolModel):
    name = "ridge"

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=alpha)),
        ])
        self._fitted = False

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        t0 = time.perf_counter()
        self.pipeline.fit(X_train, y_train)
        self._fitted = True
        return {"train_time_ms": (time.perf_counter() - t0) * 1000, "alpha": self.alpha}

    def predict(self, X):
        return self.pipeline.predict(X)

    def get_feature_importance(self, feature_names):
        if not self._fitted:
            return {}
        coefs = np.abs(self.pipeline.named_steps["model"].coef_)
        if len(feature_names) != len(coefs):
            return {}
        total = coefs.sum() + 1e-10
        return {name: round(float(c / total), 4) for name, c in zip(feature_names, coefs)}


# ─── 2. Lasso Regression ───────────────────────────────────────
class LassoVolModel(VolModel):
    name = "lasso"

    def __init__(self, alpha: float = 0.001):
        self.alpha = alpha
        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", Lasso(alpha=alpha, max_iter=5000)),
        ])
        self._fitted = False

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        t0 = time.perf_counter()
        self.pipeline.fit(X_train, y_train)
        self._fitted = True
        return {"train_time_ms": (time.perf_counter() - t0) * 1000, "alpha": self.alpha}

    def predict(self, X):
        return self.pipeline.predict(X)

    def get_feature_importance(self, feature_names):
        if not self._fitted:
            return {}
        coefs = np.abs(self.pipeline.named_steps["model"].coef_)
        if len(feature_names) != len(coefs):
            return {}
        total = coefs.sum() + 1e-10
        return {name: round(float(c / total), 4) for name, c in zip(feature_names, coefs)}


# ─── 3. Random Forest ──────────────────────────────────────────
class RandomForestVolModel(VolModel):
    name = "random_forest"

    def __init__(self, n_estimators: int = 200, max_depth: int = 12, min_samples_leaf: int = 10):
        self.params = {"n_estimators": n_estimators, "max_depth": max_depth, "min_samples_leaf": min_samples_leaf}
        self.scaler = StandardScaler()
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=42,
            n_jobs=-1,
        )
        self._fitted = False

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        t0 = time.perf_counter()
        X_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_scaled, y_train)
        self._fitted = True
        return {"train_time_ms": (time.perf_counter() - t0) * 1000, **self.params}

    def predict(self, X):
        return self.model.predict(self.scaler.transform(X))

    def get_feature_importance(self, feature_names):
        if not self._fitted or len(feature_names) != len(self.model.feature_importances_):
            return {}
        fi = self.model.feature_importances_
        return {name: round(float(v), 4) for name, v in zip(feature_names, fi)}


# ─── 4. Gradient Boosting ──────────────────────────────────────
class GradientBoostingVolModel(VolModel):
    name = "gradient_boosting"

    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: int = 5,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        min_samples_leaf: int = 15,
    ):
        self.params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "subsample": subsample,
            "min_samples_leaf": min_samples_leaf,
        }
        self.scaler = StandardScaler()
        self.model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            min_samples_leaf=min_samples_leaf,
            random_state=42,
            validation_fraction=0.15,
            n_iter_no_change=20,
            tol=1e-5,
        )
        self._fitted = False

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        t0 = time.perf_counter()
        X_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_scaled, y_train)
        self._fitted = True
        n_used = self.model.n_estimators_
        return {
            "train_time_ms": (time.perf_counter() - t0) * 1000,
            "n_estimators_used": n_used,
            **self.params,
        }

    def predict(self, X):
        return self.model.predict(self.scaler.transform(X))

    def get_feature_importance(self, feature_names):
        if not self._fitted or len(feature_names) != len(self.model.feature_importances_):
            return {}
        fi = self.model.feature_importances_
        return {name: round(float(v), 4) for name, v in zip(feature_names, fi)}


# ─── 5. Ensemble (Stacking) ────────────────────────────────────
class EnsembleVolModel(VolModel):
    name = "ensemble_stack"

    def __init__(self):
        estimators = [
            ("ridge", Ridge(alpha=1.0)),
            ("rf", RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)),
            ("gb", GradientBoostingRegressor(n_estimators=150, max_depth=4, learning_rate=0.05, random_state=42)),
        ]
        self.scaler = StandardScaler()
        self.model = StackingRegressor(
            estimators=estimators,
            final_estimator=Ridge(alpha=0.5),
            cv=3,
            n_jobs=-1,
        )
        self._fitted = False

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        t0 = time.perf_counter()
        X_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_scaled, y_train)
        self._fitted = True
        return {"train_time_ms": (time.perf_counter() - t0) * 1000}

    def predict(self, X):
        return self.model.predict(self.scaler.transform(X))

    def get_feature_importance(self, feature_names):
        if not self._fitted:
            return {}
        # Use the final estimator's coefficients as proxy
        try:
            coefs = np.abs(self.model.final_estimator_.coef_)
            base_names = [name for name, _ in self.model.estimators]
            total = coefs.sum() + 1e-10
            return {name: round(float(c / total), 4) for name, c in zip(base_names, coefs)}
        except Exception:
            return {}


# ─── 6. Simple LSTM (NumPy‑based, no PyTorch required) ─────────
class SimpleLSTMVolModel(VolModel):
    """
    Lightweight LSTM implemented purely in NumPy.
    For production use, this provides a genuine recurrent architecture
    without requiring PyTorch/TensorFlow installations.
    """
    name = "lstm_numpy"

    def __init__(
        self,
        hidden_size: int = 32,
        seq_len: int = 20,
        epochs: int = 100,
        lr: float = 0.005,
        dropout: float = 0.0,
    ):
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.epochs = epochs
        self.lr = lr
        self.dropout = dropout
        self.scaler = StandardScaler()
        self._fitted = False
        # Weights initialized in fit()
        self.Wf = self.Wi = self.Wc = self.Wo = None
        self.bf = self.bi = self.bc = self.bo = None
        self.Wy = None
        self.by = None

    def _init_weights(self, input_size: int):
        rng = np.random.default_rng(42)
        hs = self.hidden_size
        scale = 1.0 / math.sqrt(hs)
        total = input_size + hs

        self.Wf = rng.normal(0, scale, (total, hs)).astype(np.float32)
        self.Wi = rng.normal(0, scale, (total, hs)).astype(np.float32)
        self.Wc = rng.normal(0, scale, (total, hs)).astype(np.float32)
        self.Wo = rng.normal(0, scale, (total, hs)).astype(np.float32)
        self.bf = np.zeros(hs, dtype=np.float32)
        self.bi = np.zeros(hs, dtype=np.float32)
        self.bc = np.zeros(hs, dtype=np.float32)
        self.bo = np.zeros(hs, dtype=np.float32)
        self.Wy = rng.normal(0, scale, (hs, 1)).astype(np.float32)
        self.by = np.zeros(1, dtype=np.float32)

    @staticmethod
    def _sigmoid(x):
        x = np.clip(x, -30, 30)
        return 1.0 / (1.0 + np.exp(-x))

    def _forward_sequence(self, seq: np.ndarray) -> float:
        """Forward pass through one sequence. Returns scalar prediction."""
        hs = self.hidden_size
        h = np.zeros(hs, dtype=np.float32)
        c = np.zeros(hs, dtype=np.float32)

        for t in range(seq.shape[0]):
            x = seq[t]
            xh = np.concatenate([x, h])

            f = self._sigmoid(xh @ self.Wf + self.bf)
            i = self._sigmoid(xh @ self.Wi + self.bi)
            c_hat = np.tanh(xh @ self.Wc + self.bc)
            o = self._sigmoid(xh @ self.Wo + self.bo)

            c = f * c + i * c_hat
            h = o * np.tanh(c)

        y = float(h @ self.Wy + self.by)
        return y

    def _create_sequences(self, X: np.ndarray) -> list[np.ndarray]:
        """Create overlapping sequences for LSTM input."""
        sequences = []
        for i in range(len(X) - self.seq_len + 1):
            sequences.append(X[i:i + self.seq_len])
        return sequences

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        t0 = time.perf_counter()
        X_scaled = self.scaler.fit_transform(X_train)
        self._init_weights(X_scaled.shape[1])

        seqs = self._create_sequences(X_scaled)
        targets = y_train[self.seq_len - 1:]

        if len(seqs) != len(targets):
            targets = targets[:len(seqs)]

        best_loss = float('inf')
        patience_count = 0
        max_patience = 15

        for epoch in range(self.epochs):
            total_loss = 0.0
            n = 0
            # Mini-batch SGD with numerical gradient approximation
            indices = np.arange(len(seqs))
            np.random.shuffle(indices)

            # Sample a subset for efficiency
            batch_size = min(64, len(seqs))
            batch_idx = indices[:batch_size]

            for idx in batch_idx:
                seq = seqs[idx].astype(np.float32)
                target = targets[idx]
                pred = self._forward_sequence(seq)
                error = pred - target
                total_loss += error ** 2
                n += 1

                # Stochastic gradient update via SPSA (fast approximation)
                delta = 0.01
                for W in [self.Wf, self.Wi, self.Wc, self.Wo, self.Wy]:
                    perturbation = np.random.choice([-1, 1], size=W.shape).astype(np.float32)
                    W += delta * perturbation
                    pred_plus = self._forward_sequence(seq)
                    W -= 2 * delta * perturbation
                    pred_minus = self._forward_sequence(seq)
                    W += delta * perturbation  # restore
                    grad_approx = (pred_plus - pred_minus) / (2 * delta)
                    W -= self.lr * error * grad_approx * perturbation / (np.abs(perturbation) + 1e-8)

            avg_loss = total_loss / max(n, 1)
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_count = 0
            else:
                patience_count += 1
                if patience_count >= max_patience:
                    break

        self._fitted = True
        return {
            "train_time_ms": (time.perf_counter() - t0) * 1000,
            "epochs_run": epoch + 1,
            "final_loss": float(best_loss),
        }

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        seqs = self._create_sequences(X_scaled)
        preds = np.array([self._forward_sequence(s.astype(np.float32)) for s in seqs])
        # Pad front to match input length
        pad = np.full(self.seq_len - 1, preds[0] if len(preds) else 0)
        return np.concatenate([pad, preds])

    def get_feature_importance(self, feature_names):
        if not self._fitted or self.Wf is None:
            return {}
        # Use input gate weight magnitudes as feature importance proxy
        n_features = len(feature_names)
        if self.Wi.shape[0] < n_features:
            return {}
        importance = np.mean(np.abs(self.Wi[:n_features, :]), axis=1)
        total = importance.sum() + 1e-10
        return {name: round(float(v / total), 4) for name, v in zip(feature_names, importance)}


# ─── 7. Temporal CNN (NumPy‑based) ─────────────────────────────
class TemporalCNNVolModel(VolModel):
    """
    1D Convolutional model for temporal patterns.
    NumPy‑only implementation for portability.
    """
    name = "temporal_cnn"

    def __init__(self, kernel_size: int = 5, n_filters: int = 16, seq_len: int = 20, epochs: int = 80, lr: float = 0.003):
        self.kernel_size = kernel_size
        self.n_filters = n_filters
        self.seq_len = seq_len
        self.epochs = epochs
        self.lr = lr
        self.scaler = StandardScaler()
        self._fitted = False
        self.kernels = None
        self.W_out = None
        self.b_out = None

    def _init_weights(self, input_size: int):
        rng = np.random.default_rng(42)
        scale = 1.0 / math.sqrt(self.kernel_size * input_size)
        self.kernels = rng.normal(0, scale, (self.n_filters, self.kernel_size, input_size)).astype(np.float32)
        out_len = self.seq_len - self.kernel_size + 1
        fc_in = self.n_filters  # after global average pooling
        self.W_out = rng.normal(0, 1.0 / math.sqrt(fc_in), (fc_in, 1)).astype(np.float32)
        self.b_out = np.zeros(1, dtype=np.float32)

    def _forward(self, seq: np.ndarray) -> float:
        """Forward pass: Conv1D → ReLU → GlobalAvgPool → Linear."""
        # seq: (seq_len, n_features)
        out_len = seq.shape[0] - self.kernel_size + 1
        conv_out = np.zeros((out_len, self.n_filters), dtype=np.float32)
        for f in range(self.n_filters):
            for i in range(out_len):
                patch = seq[i:i + self.kernel_size]
                conv_out[i, f] = np.sum(patch * self.kernels[f])
        # ReLU
        conv_out = np.maximum(conv_out, 0)
        # Global average pooling
        pooled = np.mean(conv_out, axis=0)
        return float(pooled @ self.W_out + self.b_out)

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        t0 = time.perf_counter()
        X_scaled = self.scaler.fit_transform(X_train)
        self._init_weights(X_scaled.shape[1])

        # Create sequences
        seqs = [X_scaled[i:i + self.seq_len] for i in range(len(X_scaled) - self.seq_len + 1)]
        targets = y_train[self.seq_len - 1:][:len(seqs)]

        best_loss = float('inf')
        patience_count = 0

        for epoch in range(self.epochs):
            total_loss = 0.0
            indices = np.arange(len(seqs))
            np.random.shuffle(indices)
            batch_idx = indices[:min(48, len(seqs))]

            for idx in batch_idx:
                seq = seqs[idx].astype(np.float32)
                target = targets[idx]
                pred = self._forward(seq)
                error = pred - target
                total_loss += error ** 2

                # SPSA gradient for kernels and output weights
                delta = 0.01
                for W in [self.kernels, self.W_out]:
                    p = np.random.choice([-1, 1], size=W.shape).astype(np.float32)
                    W += delta * p
                    pp = self._forward(seq)
                    W -= 2 * delta * p
                    pm = self._forward(seq)
                    W += delta * p
                    grad = (pp - pm) / (2 * delta)
                    W -= self.lr * error * grad * p / (np.abs(p) + 1e-8)

            avg_loss = total_loss / max(len(batch_idx), 1)
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_count = 0
            else:
                patience_count += 1
                if patience_count >= 15:
                    break

        self._fitted = True
        return {"train_time_ms": (time.perf_counter() - t0) * 1000, "epochs_run": epoch + 1}

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        seqs = [X_scaled[i:i + self.seq_len] for i in range(len(X_scaled) - self.seq_len + 1)]
        preds = np.array([self._forward(s.astype(np.float32)) for s in seqs])
        pad = np.full(self.seq_len - 1, preds[0] if len(preds) else 0)
        return np.concatenate([pad, preds])

    def get_feature_importance(self, feature_names):
        if not self._fitted or self.kernels is None:
            return {}
        # Mean absolute kernel weight per feature across all filters and positions
        importance = np.mean(np.abs(self.kernels), axis=(0, 1))
        if len(feature_names) != len(importance):
            return {}
        total = importance.sum() + 1e-10
        return {name: round(float(v / total), 4) for name, v in zip(feature_names, importance)}


# ─── Model Registry ────────────────────────────────────────────
MODEL_REGISTRY: dict[str, type[VolModel]] = {
    "ridge": RidgeVolModel,
    "lasso": LassoVolModel,
    "random_forest": RandomForestVolModel,
    "gradient_boosting": GradientBoostingVolModel,
    "ensemble_stack": EnsembleVolModel,
    "lstm": SimpleLSTMVolModel,
    "temporal_cnn": TemporalCNNVolModel,
}


def get_model(name: str, **kwargs) -> VolModel:
    """Factory function to create a model by name."""
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name](**kwargs)


def get_all_models() -> dict[str, VolModel]:
    """Create one instance of each model with default hyperparameters."""
    return {name: cls() for name, cls in MODEL_REGISTRY.items()}
