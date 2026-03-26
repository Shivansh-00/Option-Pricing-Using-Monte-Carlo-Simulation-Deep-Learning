from __future__ import annotations

from pathlib import Path
import sys

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.pricing_engine.monte_carlo_simulator import (
    NeuralSDEMonteCarloPricer,
    OptionContract,
    benchmark_vs_baselines,
)
from app.pricing_engine.neural_sde_model import NeuralSDE, NeuralSDEConfig
from app.visualization.path_visualizer import NeuralSDEPathVisualizer


def main() -> None:
    candidates = [
        BACKEND_DIR / "models" / "neural_sde_default.pt",
        BACKEND_DIR.parent / "models" / "neural_sde_default.pt",
    ]
    checkpoint = next((p for p in candidates if p.exists()), None)

    if checkpoint is not None:
        model, _ = NeuralSDE.load(checkpoint)
        print(f"Loaded checkpoint: {checkpoint}")
    else:
        print("Checkpoint neural_sde_default.pt not found; using untrained default model for smoke validation")
        try:
            model = NeuralSDE(NeuralSDEConfig())
        except RuntimeError as exc:
            print(f"Neural SDE smoke run skipped: {exc}")
            return

    pricer = NeuralSDEMonteCarloPricer(model)

    contract = OptionContract(
        spot=100.0,
        strike=105.0,
        maturity=0.75,
        rate=0.04,
        option_type="call",
    )

    # Keep defaults light enough for local smoke tests while still exercising the full pipeline.
    result = pricer.price_european(contract, n_paths=10_000, steps=126, seed=42)
    print("Neural SDE price:", round(result.price, 6))
    print("StdErr:", round(result.std_error, 6))
    print("95% CI:", (round(result.ci_lower, 6), round(result.ci_upper, 6)))

    benchmark = benchmark_vs_baselines(contract, result, implied_vol=0.2, n_paths=5_000, steps=126)
    print("Benchmark summary:")
    for model_name, payload in benchmark.items():
        print(model_name, payload)

    paths = pricer.simulate_paths(contract, n_paths=1000, steps=126, seed=42)
    fig1 = NeuralSDEPathVisualizer.plot_paths(paths)
    fig2 = NeuralSDEPathVisualizer.plot_terminal_distribution(paths)
    fig3 = NeuralSDEPathVisualizer.plot_learned_functions(model, s_min=60.0, s_max=160.0, t_frac=0.5)

    output_dir = BACKEND_DIR / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    fig1.savefig(output_dir / "neural_sde_paths.png", dpi=150)
    fig2.savefig(output_dir / "neural_sde_terminal_distribution.png", dpi=150)
    fig3.savefig(output_dir / "neural_sde_learned_functions.png", dpi=150)
    print("Saved plots to", output_dir)


if __name__ == "__main__":
    main()
