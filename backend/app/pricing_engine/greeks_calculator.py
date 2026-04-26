from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module, util
from typing import Any

from .neural_sde_model import NeuralSDE

torch = import_module("torch") if util.find_spec("torch") is not None else None


@dataclass
class NeuralGreeks:
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float


class NeuralSDEGreeksCalculator:
    def __init__(self, model: NeuralSDE, n_paths: int = 4096, steps: int = 128):
        self.model = model
        self.n_paths = n_paths
        self.steps = steps

    def _price(
        self,
        spot: Any,
        strike: float,
        maturity: float,
        rate: Any,
        option_type: str,
        seed: int,
        diffusion_scale: float = 1.0,
    ) -> Any:
        if torch is None:
            raise RuntimeError("PyTorch is required for Neural SDE Greeks.")
        dev = spot.device
        dt = maturity / max(1, self.steps)
        sqrtdt = dt ** 0.5

        s = spot.repeat(self.n_paths)
        gen = torch.Generator(device=dev)
        gen.manual_seed(seed)

        z = torch.randn((self.n_paths, self.steps), generator=gen, device=dev)

        for step in range(self.steps):
            t_frac = torch.full((self.n_paths,), float(step / max(1, self.steps)), device=dev)
            _, sigma = self.model.forward(s, t_frac, None)
            sigma = sigma * diffusion_scale
            drift = rate.repeat(self.n_paths)
            log_inc = (drift - 0.5 * sigma * sigma) * dt + sigma * sqrtdt * z[:, step]
            s = s * torch.exp(log_inc)

        if option_type == "call":
            payoff = torch.relu(s - strike)
        else:
            payoff = torch.relu(strike - s)

        discount = torch.exp(-rate * maturity)
        return discount * payoff.mean()

    def compute(
        self,
        spot: float,
        strike: float,
        maturity: float,
        rate: float,
        option_type: str = "call",
        seed: int = 123,
    ) -> NeuralGreeks:
        if torch is None:
            raise RuntimeError("PyTorch is required for Neural SDE Greeks.")
        dev = next(self.model.parameters()).device
        self.model.eval()

        s = torch.tensor(float(spot), device=dev, dtype=torch.float32, requires_grad=True)
        r = torch.tensor(float(rate), device=dev, dtype=torch.float32, requires_grad=True)

        price = self._price(s, strike, maturity, r, option_type, seed)
        d_price_ds = torch.autograd.grad(price, s, create_graph=True, retain_graph=True)[0]
        gamma = torch.autograd.grad(d_price_ds, s, retain_graph=True)[0]
        rho = torch.autograd.grad(price, r, retain_graph=True)[0]

        vol_bump = 0.01
        price_up = self._price(s.detach(), strike, maturity, r.detach(), option_type, seed, diffusion_scale=1.0 + vol_bump)
        price_dn = self._price(s.detach(), strike, maturity, r.detach(), option_type, seed, diffusion_scale=max(1e-4, 1.0 - vol_bump))
        vega = (price_up - price_dn) / (2.0 * vol_bump)

        t_bump = max(1e-4, maturity * 1e-3)
        theta_price = self._price(s.detach(), strike, max(1e-4, maturity - t_bump), r.detach(), option_type, seed + 2)
        theta = (theta_price - price.detach()) / t_bump

        return NeuralGreeks(
            delta=float(d_price_ds.detach().cpu().item()),
            gamma=float(gamma.detach().cpu().item()),
            vega=float(vega.detach().cpu().item()),
            theta=float(theta.detach().cpu().item()),
            rho=float(rho.detach().cpu().item()),
        )
