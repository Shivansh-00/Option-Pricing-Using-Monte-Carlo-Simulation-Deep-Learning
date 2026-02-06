from __future__ import annotations

from dataclasses import dataclass


@dataclass
class HyperParams:
    learning_rate: float = 1e-3
    batch_size: int = 64
    epochs: int = 50
    hidden_size: int = 64
    dropout: float = 0.2


DEFAULT_HYPERPARAMS = HyperParams()
