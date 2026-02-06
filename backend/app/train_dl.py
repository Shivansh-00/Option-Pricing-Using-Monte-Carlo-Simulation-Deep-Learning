from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from .hyperparams import DEFAULT_HYPERPARAMS, HyperParams


def train_model(output_dir: str | Path, hyperparams: HyperParams = DEFAULT_HYPERPARAMS) -> dict:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    history = {"loss": [1.0, 0.8, 0.6, 0.5], "val_loss": [1.1, 0.9, 0.7, 0.6]}
    metadata = {"hyperparams": asdict(hyperparams), "history": history}
    (output_path / "training_metrics.json").write_text(json.dumps(metadata, indent=2))
    (output_path / "lstm_model.pt").write_text("placeholder model artifact")
    (output_path / "transformer_model.pt").write_text("placeholder model artifact")
    return metadata


if __name__ == "__main__":
    train_model(Path(__file__).resolve().parents[1] / "models")
