#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from backend.app.dl import HybridDLPredictor  # noqa: E402
from backend.app.pinns import PINNsOptionPricer, PINNsConfig  # noqa: E402


def rebuild_dl(spot_csv: Path, model_dir: Path, epochs: int) -> None:
    predictor = HybridDLPredictor()
    predictor.train_on_csv(
        spot_csv=spot_csv,
        lookback=30,
        epochs=epochs,
        lr=0.002,
        patience=max(3, epochs // 6),
    )
    predictor.save(model_dir)


def rebuild_pinns(option_chain_csv: Path, spot_csv: Path, model_dir: Path, epochs: int) -> None:
    config = PINNsConfig(
        hidden_layers=[64, 64, 64, 32],
        learning_rate=1e-3,
        lambda_pde=1.0,
        lambda_arb=0.5,
        lambda_smooth=0.1,
        epochs=epochs,
        batch_size=256,
        early_stop_patience=max(10, epochs // 4),
    )

    pricer = PINNsOptionPricer(config=config)
    train_data = PINNsOptionPricer.load_training_data_from_csv(
        str(option_chain_csv),
        spot_csv=str(spot_csv),
    )
    pricer.train(train_data)
    pricer.save(model_dir / "pinns_model.pkl")


def main() -> int:
    parser = argparse.ArgumentParser(description="Targeted compatibility rebuild for DL and PINNs artifacts")
    parser.add_argument("--dl-epochs", type=int, default=20)
    parser.add_argument("--pinns-epochs", type=int, default=40)
    args = parser.parse_args()

    data_dir = ROOT / "backend" / "data" / "raw"
    model_dir = ROOT / "backend" / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    spot_csv = data_dir / "spot_prices.csv"
    option_chain_csv = data_dir / "option_chain.csv"

    if not spot_csv.exists() or not option_chain_csv.exists():
        print("Required dataset files are missing for rebuild.")
        return 2

    print("Rebuilding DL artifact...")
    rebuild_dl(spot_csv, model_dir, args.dl_epochs)
    print("Rebuilding PINNs artifact...")
    rebuild_pinns(option_chain_csv, spot_csv, model_dir, args.pinns_epochs)
    print("Compatibility rebuild complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
