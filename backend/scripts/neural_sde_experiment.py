from __future__ import annotations

from pathlib import Path

from app.pricing_engine.monte_carlo_simulator import (
    NeuralSDEMonteCarloPricer,
    OptionContract,
    benchmark_vs_baselines,
)
from app.pricing_engine.neural_sde_model import NeuralSDE
from app.visualization.path_visualizer import NeuralSDEPathVisualizer


def main() -> None:
    checkpoint = Path(__file__).resolve().parents[1] / "models" / "neural_sde_default.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    model, _ = NeuralSDE.load(checkpoint)
    pricer = NeuralSDEMonteCarloPricer(model)

    contract = OptionContract(
        spot=100.0,
        strike=105.0,
        maturity=0.75,
        rate=0.04,
        option_type="call",
    )

    result = pricer.price_european(contract, n_paths=100_000, steps=252, seed=42)
    print("Neural SDE price:", round(result.price, 6))
    print("StdErr:", round(result.std_error, 6))
    print("95% CI:", (round(result.ci_lower, 6), round(result.ci_upper, 6)))

    benchmark = benchmark_vs_baselines(contract, result, implied_vol=0.2, n_paths=50_000, steps=252)
    print("Benchmark summary:")
    for model_name, payload in benchmark.items():
        print(model_name, payload)

    paths = pricer.simulate_paths(contract, n_paths=5000, steps=252, seed=42)
    fig1 = NeuralSDEPathVisualizer.plot_paths(paths)
    fig2 = NeuralSDEPathVisualizer.plot_terminal_distribution(paths)
    fig3 = NeuralSDEPathVisualizer.plot_learned_functions(model, s_min=60.0, s_max=160.0, t_frac=0.5)

    output_dir = Path(__file__).resolve().parents[1] / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    fig1.savefig(output_dir / "neural_sde_paths.png", dpi=150)
    fig2.savefig(output_dir / "neural_sde_terminal_distribution.png", dpi=150)
    fig3.savefig(output_dir / "neural_sde_learned_functions.png", dpi=150)
    print("Saved plots to", output_dir)


if __name__ == "__main__":
    main()
