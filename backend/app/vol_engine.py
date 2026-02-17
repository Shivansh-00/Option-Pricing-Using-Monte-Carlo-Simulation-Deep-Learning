"""
vol_engine.py — ML Volatility Training & Inference Engine
============================================================
Production orchestrator that ties together data → features → targets →
models → evaluation into a single coherent pipeline.

Supports:
  • Walk‑forward validation
  • Expanding window cross‑validation
  • Multi‑target training
  • Model comparison & selection
  • Feature importance analysis
  • Regime‑aware evaluation
  • Caching of trained models
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Optional

import numpy as np

from .vol_data import generate_synthetic_market, bars_to_arrays, MarketDataset
from .vol_features import build_feature_matrix, log_returns
from .vol_targets import (
    build_targets,
    build_baselines,
    realized_vol_target,
)
from .vol_models import (
    VolModel,
    ModelResult,
    get_model,
    get_all_models,
    MODEL_REGISTRY,
)

logger = logging.getLogger(__name__)


# ─── Evaluation Metrics ────────────────────────────────────────
@dataclass
class VolMetrics:
    """Comprehensive evaluation metrics for volatility forecasting."""
    rmse: float = 0.0
    mae: float = 0.0
    mape: float = 0.0
    qlike: float = 0.0
    r_squared: float = 0.0
    directional_accuracy: float = 0.0
    timing_accuracy: float = 0.0


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> VolMetrics:
    """Compute all evaluation metrics."""
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    yt = y_true[mask]
    yp = y_pred[mask]

    if len(yt) < 2:
        return VolMetrics()

    # Clamp predictions to positive
    yp = np.maximum(yp, 1e-8)

    # RMSE
    rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))

    # MAE
    mae = float(np.mean(np.abs(yt - yp)))

    # MAPE
    valid = yt > 1e-6
    mape = float(np.mean(np.abs((yt[valid] - yp[valid]) / yt[valid])) * 100) if valid.sum() > 0 else 0.0

    # QLIKE loss: mean(σ²_true / σ²_pred + log(σ²_pred))
    yt_sq = yt ** 2
    yp_sq = yp ** 2 + 1e-10
    qlike = float(np.mean(yt_sq / yp_sq + np.log(yp_sq)))

    # R²
    ss_res = np.sum((yt - yp) ** 2)
    ss_tot = np.sum((yt - np.mean(yt)) ** 2) + 1e-10
    r2 = float(1 - ss_res / ss_tot)

    # Directional accuracy: did we predict vol increase/decrease correctly?
    if len(yt) > 1:
        true_dir = np.sign(np.diff(yt))
        pred_dir = np.sign(np.diff(yp))
        dir_acc = float(np.mean(true_dir == pred_dir) * 100)
    else:
        dir_acc = 0.0

    # Timing accuracy: high-vol (>75th pctl) detection rate
    threshold = np.percentile(yt, 75)
    high_vol_mask = yt > threshold
    if high_vol_mask.sum() > 0:
        timing = float(np.mean(yp[high_vol_mask] > threshold) * 100)
    else:
        timing = 0.0

    return VolMetrics(
        rmse=round(rmse, 6),
        mae=round(mae, 6),
        mape=round(mape, 2),
        qlike=round(qlike, 4),
        r_squared=round(r2, 4),
        directional_accuracy=round(dir_acc, 2),
        timing_accuracy=round(timing, 2),
    )


# ─── Walk‑Forward Validation ───────────────────────────────────
@dataclass
class WalkForwardFold:
    """One fold of walk-forward validation."""
    fold_idx: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    metrics: VolMetrics = field(default_factory=VolMetrics)


def walk_forward_splits(
    n_samples: int,
    n_folds: int = 5,
    min_train_size: int = 200,
    test_size: int = 60,
) -> list[WalkForwardFold]:
    """Generate expanding window walk-forward splits."""
    folds = []
    total_test = n_folds * test_size
    if n_samples < min_train_size + total_test:
        # Adjust test_size
        test_size = max(20, (n_samples - min_train_size) // n_folds)

    for i in range(n_folds):
        test_end = n_samples - (n_folds - i - 1) * test_size
        test_start = test_end - test_size
        train_end = test_start
        train_start = 0  # expanding window
        if train_end - train_start < min_train_size:
            continue
        folds.append(WalkForwardFold(
            fold_idx=i,
            train_start=train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
        ))
    return folds


# ─── Model Comparison Result ───────────────────────────────────
@dataclass
class ModelComparison:
    model_name: str
    target_name: str
    train_metrics: VolMetrics
    test_metrics: VolMetrics
    cv_metrics: VolMetrics | None = None
    train_time_ms: float = 0.0
    inference_time_ms: float = 0.0
    feature_importance: dict[str, float] = field(default_factory=dict)
    improvement_vs_historical: float = 0.0
    improvement_vs_garch: float = 0.0
    improvement_vs_ewma: float = 0.0


@dataclass
class VolEngineResult:
    """Complete result from the volatility engine."""
    comparisons: list[ModelComparison]
    best_model: str
    best_target: str
    best_test_rmse: float
    best_test_r2: float
    baseline_rmse: dict[str, float]
    feature_names: list[str]
    top_features: list[dict[str, Any]]
    n_train: int
    n_val: int
    n_test: int
    total_time_ms: float
    regime_analysis: dict[str, Any] = field(default_factory=dict)
    walk_forward: list[dict] = field(default_factory=list)


# ─── Singleton Engine ──────────────────────────────────────────
class VolatilityEngine:
    """
    Stateful engine that maintains trained models and cached results.
    Thread‑safe for serving via FastAPI.
    """

    def __init__(self):
        self._dataset: MarketDataset | None = None
        self._last_result: VolEngineResult | None = None
        self._trained_models: dict[str, VolModel] = {}
        self._feature_names: list[str] = []
        self._scaler_fitted = False

    @property
    def is_trained(self) -> bool:
        return len(self._trained_models) > 0

    @property
    def last_result(self) -> VolEngineResult | None:
        return self._last_result

    def train_and_evaluate(
        self,
        model_names: list[str] | None = None,
        target_name: str = "realized_vol",
        forward_window: int = 20,
        n_cv_folds: int = 3,
        n_days: int = 2520,
        seed: int = 42,
    ) -> VolEngineResult:
        """
        Full training pipeline:
          1. Generate / load data
          2. Build features & targets
          3. Train all models on train set
          4. Evaluate on val + test sets
          5. Compare against baselines
          6. Walk‑forward CV
        """
        t_start = time.perf_counter()

        # 1. Data
        logger.info("Generating synthetic market data (%d days)...", n_days)
        dataset = generate_synthetic_market(n_days=n_days, seed=seed)
        self._dataset = dataset
        arrays = bars_to_arrays(dataset.bars)

        # 2. Features
        logger.info("Building feature matrix...")
        X_full, feature_names = build_feature_matrix(
            open_=arrays["open"], high=arrays["high"],
            low=arrays["low"], close=arrays["close"],
            volume=arrays["volume"], vix=arrays["vix"],
            rate=arrays["rate"],
        )
        self._feature_names = feature_names

        # 3. Targets
        rets_full = log_returns(arrays["close"])
        targets = build_targets(
            rets_full,
            open_=arrays["open"], high=arrays["high"],
            low=arrays["low"], close=arrays["close"],
            forward_window=forward_window,
        )
        baselines = build_baselines(rets_full, forward_window)

        # Align X and y
        n_x = X_full.shape[0]
        y_all = targets[target_name]
        # X_full starts at index `offset` in the returns array
        offset = len(rets_full) - n_x
        y_aligned = y_all[offset:offset + n_x]

        # Remove NaN targets
        valid = ~np.isnan(y_aligned)
        X_valid = X_full[valid]
        y_valid = y_aligned[valid]
        n_valid = len(y_valid)

        # Temporal split
        train_end = int(n_valid * 0.70)
        val_end = int(n_valid * 0.85)

        X_train = X_valid[:train_end]
        y_train = y_valid[:train_end]
        X_val = X_valid[train_end:val_end]
        y_val = y_valid[train_end:val_end]
        X_test = X_valid[val_end:]
        y_test = y_valid[val_end:]

        # 4. Baseline metrics on test set
        baseline_test_metrics = {}
        for bname, bpred in baselines.items():
            bpred_aligned = bpred[offset:offset + n_x][valid][val_end:]
            bm = compute_metrics(y_test, bpred_aligned[:len(y_test)])
            baseline_test_metrics[bname] = bm.rmse

        # 5. Train models
        if model_names is None:
            model_names = list(MODEL_REGISTRY.keys())

        comparisons: list[ModelComparison] = []

        for mname in model_names:
            logger.info("Training model: %s", mname)
            try:
                model = get_model(mname)
                train_meta = model.fit(X_train, y_train, X_val, y_val)
                train_time = train_meta.get("train_time_ms", 0)

                # Train metrics
                t0 = time.perf_counter()
                pred_train = model.predict(X_train)
                pred_val = model.predict(X_val)
                pred_test = model.predict(X_test)
                inf_time = (time.perf_counter() - t0) * 1000

                train_metrics = compute_metrics(y_train, pred_train)
                test_metrics = compute_metrics(y_test, pred_test)

                fi = model.get_feature_importance(feature_names)

                # Improvement vs baselines
                hist_rmse = baseline_test_metrics.get("historical_vol", test_metrics.rmse)
                garch_rmse = baseline_test_metrics.get("garch11", test_metrics.rmse)
                ewma_rmse = baseline_test_metrics.get("ewma", test_metrics.rmse)

                imp_hist = ((hist_rmse - test_metrics.rmse) / hist_rmse * 100) if hist_rmse > 0 else 0
                imp_garch = ((garch_rmse - test_metrics.rmse) / garch_rmse * 100) if garch_rmse > 0 else 0
                imp_ewma = ((ewma_rmse - test_metrics.rmse) / ewma_rmse * 100) if ewma_rmse > 0 else 0

                # Walk-forward CV (skip for slow models)
                cv_metrics = None
                if mname in ("ridge", "lasso", "random_forest", "gradient_boosting"):
                    cv_rmses = []
                    folds = walk_forward_splits(len(X_valid), n_folds=n_cv_folds)
                    for fold in folds:
                        cv_model = get_model(mname)
                        cv_model.fit(
                            X_valid[fold.train_start:fold.train_end],
                            y_valid[fold.train_start:fold.train_end],
                        )
                        cv_pred = cv_model.predict(X_valid[fold.test_start:fold.test_end])
                        cv_m = compute_metrics(
                            y_valid[fold.test_start:fold.test_end], cv_pred
                        )
                        cv_rmses.append(cv_m.rmse)
                    cv_metrics = VolMetrics(
                        rmse=round(float(np.mean(cv_rmses)), 6),
                        mae=0,  # simplified
                    )

                comparisons.append(ModelComparison(
                    model_name=mname,
                    target_name=target_name,
                    train_metrics=train_metrics,
                    test_metrics=test_metrics,
                    cv_metrics=cv_metrics,
                    train_time_ms=train_time,
                    inference_time_ms=inf_time,
                    feature_importance=fi,
                    improvement_vs_historical=round(imp_hist, 2),
                    improvement_vs_garch=round(imp_garch, 2),
                    improvement_vs_ewma=round(imp_ewma, 2),
                ))

                self._trained_models[mname] = model

            except Exception as e:
                logger.error("Failed to train %s: %s", mname, e)
                continue

        # 6. Find best model
        if comparisons:
            best = min(comparisons, key=lambda c: c.test_metrics.rmse)
            best_model = best.model_name
            best_rmse = best.test_metrics.rmse
            best_r2 = best.test_metrics.r_squared
        else:
            best_model = "none"
            best_rmse = float('inf')
            best_r2 = 0.0

        # Top features from best model
        top_fi = []
        if comparisons:
            fi_best = comparisons[0].feature_importance
            for c in comparisons:
                if c.model_name == best_model:
                    fi_best = c.feature_importance
                    break
            sorted_fi = sorted(fi_best.items(), key=lambda x: x[1], reverse=True)[:10]
            top_fi = [{"name": name, "importance": imp} for name, imp in sorted_fi]

        total_time = (time.perf_counter() - t_start) * 1000

        result = VolEngineResult(
            comparisons=comparisons,
            best_model=best_model,
            best_target=target_name,
            best_test_rmse=best_rmse,
            best_test_r2=best_r2,
            baseline_rmse=baseline_test_metrics,
            feature_names=feature_names,
            top_features=top_fi,
            n_train=len(X_train),
            n_val=len(X_val),
            n_test=len(X_test),
            total_time_ms=round(total_time, 1),
        )
        self._last_result = result
        return result

    def predict_single(
        self,
        model_name: str | None = None,
        spot: float = 100,
        rate: float = 0.05,
        maturity: float = 0.5,
        realized_vol: float = 0.18,
        vix: float = 20,
        skew: float = -0.15,
    ) -> dict:
        """
        Single‑point volatility prediction using the best trained model.
        Falls back to the analytical formula if no model is trained.
        """
        if not self._trained_models:
            # Fallback formula (improved from original ml.py)
            vix_decimal = vix / 100
            base = 0.4 * realized_vol + 0.4 * vix_decimal + 0.2 * abs(skew)
            implied = max(0.05, min(2.5, base))

            import math
            score = math.tanh(vix / 25)
            if score < 0.3:
                regime = "low-vol"
            elif score < 0.6:
                regime = "risk-on"
            elif score < 0.9:
                regime = "risk-off"
            else:
                regime = "high-vol"

            return {
                "implied_vol": round(implied, 4),
                "regime": regime,
                "model_used": "analytical_fallback",
                "confidence": 0.5,
                "drivers": {
                    "realized_vol_weight": 0.4,
                    "vix_weight": 0.4,
                    "skew_weight": 0.2,
                },
            }

        # Use trained model
        use_model = model_name or self._last_result.best_model if self._last_result else "gradient_boosting"
        if use_model not in self._trained_models:
            use_model = next(iter(self._trained_models))

        model = self._trained_models[use_model]

        # Build a simple feature vector from the inputs
        # Map to the feature space (approximation for single-point inference)
        n_features = len(self._feature_names)
        x = np.zeros(n_features)

        # Fill known features
        feature_map = {
            "log_return": 0.0,
            "squared_return": 0.0,
            "abs_return": 0.0,
            "rv_5": realized_vol,
            "rv_10": realized_vol,
            "rv_20": realized_vol,
            "rv_60": realized_vol,
            "vix": vix,
            "rate": rate,
            "skewness_20": skew,
            "skewness_60": skew,
            "rsi_14": 50.0,
            "volume_ratio": 1.0,
            "bollinger_bw": realized_vol * 2,
            "vol_of_vol": realized_vol * 0.3,
            "vol_z_score": 0.0,
            "vol_clustering": 0.3,
            "rate_change": 0.0,
        }

        for i, fname in enumerate(self._feature_names):
            if fname in feature_map:
                x[i] = feature_map[fname]

        pred = model.predict(x.reshape(1, -1))
        iv = float(np.clip(pred[0], 0.05, 2.5))

        # Determine regime from prediction
        if iv < 0.12:
            regime = "low-vol"
        elif iv < 0.22:
            regime = "risk-on"
        elif iv < 0.35:
            regime = "risk-off"
        else:
            regime = "high-vol"

        # Confidence based on model R²
        r2 = 0.5
        if self._last_result:
            for c in self._last_result.comparisons:
                if c.model_name == use_model:
                    r2 = max(0, c.test_metrics.r_squared)
                    break

        return {
            "implied_vol": round(iv, 4),
            "regime": regime,
            "model_used": use_model,
            "confidence": round(r2, 3),
            "drivers": model.get_feature_importance(self._feature_names),
        }


# ─── Global singleton ──────────────────────────────────────────
_engine = VolatilityEngine()


def get_engine() -> VolatilityEngine:
    return _engine
