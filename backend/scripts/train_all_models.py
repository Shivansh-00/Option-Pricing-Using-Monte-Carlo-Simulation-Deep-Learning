#!/usr/bin/env python3
"""
train_all_models.py — Master Training Pipeline
================================================
Trains ALL models (ML vol engine, DL LSTM, PINNs) using the
generated CSV datasets and saves artifacts to backend/models/.

Usage:
    python backend/scripts/train_all_models.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

# Force UTF-8 output on Windows
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
if sys.stdout.encoding != "utf-8":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# Ensure backend/ is on the Python path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

DATA_DIR = ROOT / "backend" / "data" / "raw"
MODEL_DIR = ROOT / "backend" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ── CSV paths ──
SPOT_CSV = DATA_DIR / "spot_prices.csv"
INDICATORS_CSV = DATA_DIR / "market_indicators.csv"
OPTION_CHAIN_CSV = DATA_DIR / "option_chain.csv"
MARKET_DATA_CSV = DATA_DIR / "market_data.csv"


def train_vol_engine() -> dict:
    """Train all 6 ML volatility models on real CSV data."""
    print("\n" + "=" * 60)
    print("  [1/3] TRAINING ML VOLATILITY ENGINE (6 models)")
    print("=" * 60)

    from backend.app.vol_engine import VolatilityEngine

    engine = VolatilityEngine()
    result = engine.train_from_csv(
        spot_csv=str(SPOT_CSV),
        indicators_csv=str(INDICATORS_CSV),
        target_name="realized_vol",
        forward_window=20,
        n_cv_folds=3,
    )

    saved = engine.save(MODEL_DIR)
    print(f"\n  Best model: {result.best_model}")
    print(f"  Best test RMSE: {result.best_test_rmse:.6f}")
    print(f"  Best test R²:   {result.best_test_r2:.4f}")
    print(f"  Models trained: {len(result.comparisons)}")
    print(f"  Train/Val/Test: {result.n_train}/{result.n_val}/{result.n_test}")
    print(f"  Time: {result.total_time_ms:.0f} ms")

    summary = {
        "best_model": result.best_model,
        "best_test_rmse": result.best_test_rmse,
        "best_test_r2": result.best_test_r2,
        "n_models": len(result.comparisons),
        "time_ms": result.total_time_ms,
        "models": {},
    }
    for c in result.comparisons:
        summary["models"][c.model_name] = {
            "test_rmse": c.test_metrics.rmse,
            "test_r2": c.test_metrics.r_squared,
            "test_mae": c.test_metrics.mae,
            "test_mape": c.test_metrics.mape,
            "train_time_ms": c.train_time_ms,
            "improvement_vs_historical": c.improvement_vs_historical,
        }
        print(f"    {c.model_name:25s} RMSE={c.test_metrics.rmse:.6f}  R²={c.test_metrics.r_squared:.4f}")

    return summary


def train_dl_lstm() -> dict:
    """Train LSTM on real spot price data."""
    print("\n" + "=" * 60)
    print("  [2/3] TRAINING DEEP LEARNING LSTM")
    print("=" * 60)

    from backend.app.dl import HybridDLPredictor

    predictor = HybridDLPredictor()

    def progress(epoch, total, train_loss, val_loss):
        if epoch % 10 == 0 or epoch == total:
            print(f"    Epoch {epoch:3d}/{total} — train_loss={train_loss:.6f}  val_loss={val_loss:.6f}")

    result = predictor.train_on_csv(
        spot_csv=str(SPOT_CSV),
        lookback=30,
        epochs=50,
        lr=0.002,
        patience=8,
        progress_callback=progress,
    )

    predictor.save(MODEL_DIR)
    print(f"\n  Epochs trained: {result.epochs_trained}")
    print(f"  Final RMSE:     {result.final_rmse:.6f}")
    print(f"  Final MAE:      {result.final_mae:.6f}")
    print(f"  R²:             {result.r_squared:.4f}")
    print(f"  Time:           {result.elapsed_ms:.0f} ms")

    return {
        "epochs_trained": result.epochs_trained,
        "final_rmse": result.final_rmse,
        "final_mae": result.final_mae,
        "r_squared": result.r_squared,
        "time_ms": result.elapsed_ms,
    }


def train_pinns() -> dict:
    """Train PINNs on option chain data (calls only)."""
    print("\n" + "=" * 60)
    print("  [3/3] TRAINING PHYSICS-INFORMED NEURAL NETWORK (PINNs)")
    print("=" * 60)

    from backend.app.pinns import PINNsOptionPricer, PINNsConfig

    config = PINNsConfig(
        hidden_layers=[64, 64, 64, 32],
        learning_rate=1e-3,
        lambda_pde=1.0,
        lambda_arb=0.5,
        lambda_smooth=0.1,
        epochs=200,
        batch_size=256,
        early_stop_patience=40,
    )
    pricer = PINNsOptionPricer(config=config)

    # Load training data from option chain CSV
    print("  Loading option chain data...")
    train_data = PINNsOptionPricer.load_training_data_from_csv(
        str(OPTION_CHAIN_CSV),
        spot_csv=str(SPOT_CSV),
    )
    n_samples = len(train_data["S"])
    print(f"  Training samples: {n_samples} call options")

    def progress(epoch, total, loss):
        if epoch % 20 == 0:
            print(f"    Epoch {epoch:3d}/{total} — data_loss={loss:.6f}")

    result = pricer.train(train_data, progress_callback=progress)

    pricer.save(MODEL_DIR / "pinns_model.pkl")

    # Quick validation: price a few options
    print(f"\n  Epochs completed: {result['epochs']}")
    print(f"  Best loss:        {result['best_loss']:.6f}")
    print(f"  Training time:    {result['training_time_s']:.1f}s")
    print(f"  Parameters:       {result['params']}")

    # Spot-check pricing
    greeks = pricer.compute_greeks(S=450.0, K=450.0, tau=0.25, sigma=0.18, r=0.05)
    print(f"\n  Spot-check (ATM, S=K=450, τ=0.25, σ=0.18):")
    print(f"    Price={greeks['price']:.4f}  Delta={greeks['delta']:.4f}  Gamma={greeks['gamma']:.6f}")

    return {
        "epochs": result["epochs"],
        "best_loss": result["best_loss"],
        "training_time_s": result["training_time_s"],
        "params": result["params"],
        "spot_check": greeks,
    }


def main():
    print("╔══════════════════════════════════════════════════════╗")
    print("║   OptiQuant — Master Model Training Pipeline        ║")
    print("╚══════════════════════════════════════════════════════╝")
    print(f"  Data dir:  {DATA_DIR}")
    print(f"  Model dir: {MODEL_DIR}")

    # Verify CSVs exist
    for csv_path in [SPOT_CSV, INDICATORS_CSV, OPTION_CHAIN_CSV]:
        if not csv_path.exists():
            print(f"  ERROR: Missing {csv_path}")
            sys.exit(1)
    print("  All CSV datasets found ✓")

    t_start = time.time()
    report = {}

    # 1. ML Volatility Models
    report["vol_engine"] = train_vol_engine()

    # 2. DL LSTM
    report["dl_lstm"] = train_dl_lstm()

    # 3. PINNs
    report["pinns"] = train_pinns()

    total_time = time.time() - t_start

    # Save training report
    report["total_time_s"] = round(total_time, 2)
    report_path = MODEL_DIR / "training_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Total time: {total_time:.1f}s")
    print(f"  Report saved: {report_path}")
    print(f"  Model artifacts in: {MODEL_DIR}")

    # List saved files
    print("\n  Saved files:")
    for f in sorted(MODEL_DIR.iterdir()):
        if f.name.startswith("."):
            continue
        size = f.stat().st_size
        unit = "KB" if size > 1024 else "B"
        size_str = f"{size / 1024:.1f} {unit}" if size > 1024 else f"{size} {unit}"
        print(f"    {f.name:35s} {size_str:>10s}")


if __name__ == "__main__":
    main()
