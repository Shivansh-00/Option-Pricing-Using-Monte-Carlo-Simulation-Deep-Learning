from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .. import pricing
from ..stochastic_vol import HestonParams, heston_mc
from .neural_sde_model import NeuralSDE


@dataclass
class OptionContract:
    spot: float
    strike: float
    maturity: float
    rate: float
    option_type: str = "call"
    dividend_yield: float = 0.0


@dataclass
class MCNeuralResult:
    price: float
    std_error: float
    ci_lower: float
    ci_upper: float
    discount_factor: float
    paths: int
    steps: int


class NeuralSDEMonteCarloPricer:
    def __init__(self, model: NeuralSDE, max_batch_paths: int = 20_000):
        self.model = model
        self.max_batch_paths = max_batch_paths

    @staticmethod
    def _payoff(terminal: np.ndarray, strike: float, option_type: str) -> np.ndarray:
        if option_type == "call":
            return np.maximum(terminal - strike, 0.0)
        return np.maximum(strike - terminal, 0.0)

    def simulate_paths(
        self,
        contract: OptionContract,
        n_paths: int,
        steps: int,
        seed: int = 42,
        risk_neutral: bool = True,
    ) -> np.ndarray:
        rate = contract.rate - contract.dividend_yield if risk_neutral else None
        paths = self.model.simulate_paths(
            s0=contract.spot,
            maturity=contract.maturity,
            steps=steps,
            n_paths=n_paths,
            seed=seed,
            risk_free_rate=rate,
            max_batch_paths=self.max_batch_paths,
        )
        return paths.detach().cpu().numpy()

    def price_european(
        self,
        contract: OptionContract,
        n_paths: int = 100_000,
        steps: int = 252,
        seed: int = 42,
    ) -> MCNeuralResult:
        paths = self.simulate_paths(contract=contract, n_paths=n_paths, steps=steps, seed=seed, risk_neutral=True)
        terminal = paths[:, -1]
        payoff = self._payoff(terminal, contract.strike, contract.option_type)
        discount = math.exp(-contract.rate * contract.maturity)

        price = discount * float(np.mean(payoff))
        std_error = discount * float(np.std(payoff)) / math.sqrt(max(1, n_paths))
        ci_lower = price - 1.96 * std_error
        ci_upper = price + 1.96 * std_error

        return MCNeuralResult(
            price=price,
            std_error=std_error,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            discount_factor=discount,
            paths=n_paths,
            steps=steps,
        )


def benchmark_vs_baselines(
    contract: OptionContract,
    neural_result: MCNeuralResult,
    implied_vol: float,
    n_paths: int,
    steps: int,
) -> dict[str, dict[str, float]]:
    inputs = pricing.PricingInputs(
        spot=contract.spot,
        strike=contract.strike,
        maturity=contract.maturity,
        rate=contract.rate,
        volatility=implied_vol,
        option_type=contract.option_type,
        paths=n_paths,
        steps=steps,
    )
    bs = pricing.black_scholes(inputs)
    mc = pricing.monte_carlo_engine(inputs, seed=42, method="antithetic")

    heston_params = HestonParams(
        spot=contract.spot,
        strike=contract.strike,
        maturity=contract.maturity,
        rate=contract.rate,
        v0=max(implied_vol * implied_vol, 1e-6),
        kappa=2.0,
        theta=max(implied_vol * implied_vol, 1e-6),
        xi=0.35,
        rho=-0.5,
        option_type=contract.option_type,
        paths=n_paths,
        steps=steps,
    )
    heston = heston_mc(heston_params, seed=42)

    return {
        "neural_sde": {
            "price": neural_result.price,
            "std_error": neural_result.std_error,
            "ci_lower": neural_result.ci_lower,
            "ci_upper": neural_result.ci_upper,
        },
        "black_scholes": {
            "price": bs,
            "abs_diff_vs_neural": abs(bs - neural_result.price),
        },
        "gbm_monte_carlo": {
            "price": mc.price,
            "std_error": mc.std_error,
            "elapsed_ms": mc.elapsed_ms,
            "abs_diff_vs_neural": abs(mc.price - neural_result.price),
        },
        "heston": {
            "price": heston,
            "abs_diff_vs_neural": abs(heston - neural_result.price),
        },
    }
