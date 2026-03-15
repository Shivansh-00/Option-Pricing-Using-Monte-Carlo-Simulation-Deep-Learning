from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
	import torch
	from torch import nn
except ImportError:  # pragma: no cover - runtime dependency guard
	torch = None  # type: ignore[assignment]

	class _NNStub:
		class Module:
			pass

	nn = _NNStub()  # type: ignore[assignment]


def _require_torch() -> None:
	if torch is None:
		raise RuntimeError("PyTorch is required for Neural SDE module. Install torch>=2.0.0.")


@dataclass
class NeuralSDEConfig:
	hidden_dim: int = 96
	num_layers: int = 3
	dropout: float = 0.05
	drift_scale: float = 0.35
	sigma_floor: float = 1e-4
	sigma_cap: float = 4.0
	market_feature_dim: int = 0


@dataclass
class LossWeights:
	distribution: float = 1.0
	path_regularization: float = 0.1
	stability: float = 0.05
	wasserstein: float = 0.3


@dataclass
class TrainingConfig:
	epochs: int = 20
	learning_rate: float = 1e-3
	weight_decay: float = 1e-5
	grad_clip: float = 1.0
	seed: int = 42
	mu_clip: float = 3.0
	sigma_clip: float = 3.0
	weights: LossWeights = field(default_factory=LossWeights)


class _MLP(nn.Module):
	def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, num_layers: int, dropout: float):
		super().__init__()
		layers: list[nn.Module] = []
		current = in_dim
		for _ in range(max(1, num_layers - 1)):
			layers.extend([nn.Linear(current, hidden_dim), nn.GELU()])
			if dropout > 0:
				layers.append(nn.Dropout(dropout))
			current = hidden_dim
		layers.append(nn.Linear(current, out_dim))
		self.model = nn.Sequential(*layers)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.model(x)


class NeuralSDE(nn.Module):
	"""
	Neural SDE parameterization under log-Euler discretization:

	  dS_t = mu_theta(S_t, t, x_t) S_t dt + sigma_theta(S_t, t, x_t) S_t dW_t

	Discretized in log-space for positivity:

	  log S_{t+dt} - log S_t = (mu - 0.5 sigma^2) dt + sigma * sqrt(dt) * z
	"""

	def __init__(self, config: NeuralSDEConfig):
		_require_torch()
		super().__init__()
		self.config = config
		self.input_dim = 3 + config.market_feature_dim
		self.drift_net = _MLP(
			in_dim=self.input_dim,
			hidden_dim=config.hidden_dim,
			out_dim=1,
			num_layers=config.num_layers,
			dropout=config.dropout,
		)
		self.diffusion_net = _MLP(
			in_dim=self.input_dim,
			hidden_dim=config.hidden_dim,
			out_dim=1,
			num_layers=config.num_layers,
			dropout=config.dropout,
		)

	def _features(self, s_t: torch.Tensor, t_frac: torch.Tensor, market_features: torch.Tensor | None) -> torch.Tensor:
		log_s = torch.log(torch.clamp(s_t, min=1e-8)).unsqueeze(-1)
		t_col = t_frac.unsqueeze(-1)
		periodic = torch.cat([
			torch.sin(2.0 * math.pi * t_col),
			torch.cos(2.0 * math.pi * t_col),
		], dim=-1)
		base = torch.cat([log_s, periodic], dim=-1)
		if market_features is None:
			return base
		return torch.cat([base, market_features], dim=-1)

	def forward(
		self,
		s_t: torch.Tensor,
		t_frac: torch.Tensor,
		market_features: torch.Tensor | None = None,
	) -> tuple[torch.Tensor, torch.Tensor]:
		x = self._features(s_t, t_frac, market_features)
		mu = self.config.drift_scale * torch.tanh(self.drift_net(x)).squeeze(-1)
		sigma = torch.nn.functional.softplus(self.diffusion_net(x)).squeeze(-1) + self.config.sigma_floor
		sigma = torch.clamp(sigma, max=self.config.sigma_cap)
		return mu, sigma

	def simulate_paths(
		self,
		s0: float,
		maturity: float,
		steps: int,
		n_paths: int,
		market_features: torch.Tensor | None = None,
		seed: int = 42,
		risk_free_rate: float | None = None,
		max_batch_paths: int = 20_000,
		device: torch.device | None = None,
	) -> torch.Tensor:
		_require_torch()
		self.eval()
		dev = device or next(self.parameters()).device
		dt = maturity / max(1, steps)
		sqrtdt = math.sqrt(dt)
		gen = torch.Generator(device=dev)
		gen.manual_seed(seed)

		batches: list[torch.Tensor] = []
		with torch.no_grad():
			remaining = n_paths
			while remaining > 0:
				bsz = min(max_batch_paths, remaining)
				remaining -= bsz

				s = torch.full((bsz,), float(s0), device=dev)
				path = torch.empty((bsz, steps + 1), device=dev)
				path[:, 0] = s

				z = torch.randn((bsz, steps), generator=gen, device=dev)
				for step in range(steps):
					t_frac = torch.full((bsz,), float(step / max(1, steps)), device=dev)
					feat_t = None
					if market_features is not None:
						feat_t = market_features[:, step, :] if market_features.dim() == 3 else market_features[step, :].unsqueeze(0).repeat(bsz, 1)
					mu, sigma = self.forward(s, t_frac, feat_t)
					drift = torch.full_like(mu, risk_free_rate) if risk_free_rate is not None else mu
					log_inc = (drift - 0.5 * sigma * sigma) * dt + sigma * sqrtdt * z[:, step]
					s = s * torch.exp(log_inc)
					path[:, step + 1] = s

				batches.append(path)

		return torch.cat(batches, dim=0)

	@staticmethod
	def _wasserstein_like(z: torch.Tensor) -> torch.Tensor:
		zs = torch.sort(z).values
		ref = torch.sort(torch.randn_like(z)).values
		return torch.mean(torch.abs(zs - ref))

	def _batch_loss(self, batch: dict[str, torch.Tensor], cfg: TrainingConfig) -> tuple[torch.Tensor, dict[str, float]]:
		spots = batch["spots"]
		market = batch.get("market_features")
		dt = batch["dt"]

		s_t = spots[:, :-1]
		s_next = spots[:, 1:]
		bsz, steps = s_t.shape

		time_grid = torch.linspace(0.0, 1.0, steps, device=spots.device).unsqueeze(0).repeat(bsz, 1)

		flat_s = s_t.reshape(-1)
		flat_t = time_grid.reshape(-1)
		flat_market = market.reshape(-1, market.size(-1)) if market is not None else None
		mu, sigma = self.forward(flat_s, flat_t, flat_market)

		dt_flat = dt.unsqueeze(1).repeat(1, steps).reshape(-1)
		log_returns = torch.log(torch.clamp(s_next, min=1e-8) / torch.clamp(s_t, min=1e-8)).reshape(-1)

		mean = (mu - 0.5 * sigma * sigma) * dt_flat
		var = sigma * sigma * dt_flat + 1e-10
		nll = 0.5 * torch.mean(torch.log(2.0 * math.pi * var) + ((log_returns - mean) ** 2) / var)

		z = (log_returns - mean) / torch.sqrt(var)
		wloss = self._wasserstein_like(z)
		distribution_loss = nll + cfg.weights.wasserstein * wloss

		mu2d = mu.reshape(bsz, steps)
		sigma2d = sigma.reshape(bsz, steps)
		d_mu = mu2d[:, 1:] - mu2d[:, :-1]
		d_sigma = sigma2d[:, 1:] - sigma2d[:, :-1]
		path_reg = torch.mean(d_mu * d_mu) + torch.mean(d_sigma * d_sigma)

		stability = torch.mean(torch.relu(torch.abs(mu) - cfg.mu_clip) ** 2)
		stability = stability + torch.mean(torch.relu(sigma - cfg.sigma_clip) ** 2)

		total = (
			cfg.weights.distribution * distribution_loss
			+ cfg.weights.path_regularization * path_reg
			+ cfg.weights.stability * stability
		)
		return total, {
			"distribution": float(distribution_loss.detach().cpu().item()),
			"path_regularization": float(path_reg.detach().cpu().item()),
			"stability": float(stability.detach().cpu().item()),
			"nll": float(nll.detach().cpu().item()),
			"wasserstein": float(wloss.detach().cpu().item()),
		}

	def fit(
		self,
		train_loader: Any,
		val_loader: Any,
		config: TrainingConfig,
		device: torch.device | None = None,
	) -> dict[str, Any]:
		_require_torch()
		dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
		torch.manual_seed(config.seed)
		if torch.cuda.is_available():
			torch.cuda.manual_seed_all(config.seed)

		self.to(dev)
		optimizer = torch.optim.AdamW(self.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

		history: dict[str, Any] = {
			"train_loss": [],
			"val_loss": [],
			"components": [],
			"best_val_loss": float("inf"),
		}
		best_state: dict[str, Any] | None = None

		for _ in range(config.epochs):
			self.train()
			epoch_loss = 0.0
			count = 0
			comp_sum = {"distribution": 0.0, "path_regularization": 0.0, "stability": 0.0, "nll": 0.0, "wasserstein": 0.0}

			for batch in train_loader:
				batch = {k: v.to(dev) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
				optimizer.zero_grad(set_to_none=True)
				loss, comps = self._batch_loss(batch, config)
				loss.backward()
				nn.utils.clip_grad_norm_(self.parameters(), max_norm=config.grad_clip)
				optimizer.step()

				val = float(loss.detach().cpu().item())
				epoch_loss += val
				count += 1
				for key in comp_sum:
					comp_sum[key] += comps[key]

			train_avg = epoch_loss / max(1, count)
			self.eval()
			val_total = 0.0
			val_count = 0
			with torch.no_grad():
				for batch in val_loader:
					batch = {k: v.to(dev) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
					loss, _ = self._batch_loss(batch, config)
					val_total += float(loss.detach().cpu().item())
					val_count += 1

			val_avg = val_total / max(1, val_count)
			history["train_loss"].append(train_avg)
			history["val_loss"].append(val_avg)
			history["components"].append({k: v / max(1, count) for k, v in comp_sum.items()})

			if val_avg < history["best_val_loss"]:
				history["best_val_loss"] = val_avg
				best_state = {k: v.detach().cpu().clone() for k, v in self.state_dict().items()}

		if best_state is not None:
			self.load_state_dict(best_state)
		return history

	def save(self, path: str | Path, extra: dict[str, Any] | None = None) -> None:
		_require_torch()
		payload = {
			"state_dict": self.state_dict(),
			"config": self.config.__dict__,
			"extra": extra or {},
		}
		torch.save(payload, str(path))

	@classmethod
	def load(cls, path: str | Path, device: torch.device | None = None) -> tuple["NeuralSDE", dict[str, Any]]:
		_require_torch()
		dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
		payload = torch.load(str(path), map_location=dev)
		config = NeuralSDEConfig(**payload["config"])
		model = cls(config)
		model.load_state_dict(payload["state_dict"])
		model.to(dev)
		model.eval()
		return model, payload.get("extra", {})
