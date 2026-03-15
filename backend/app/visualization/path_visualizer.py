from __future__ import annotations

import numpy as np


class NeuralSDEPathVisualizer:
    @staticmethod
    def plot_paths(paths: np.ndarray, max_paths: int = 200):
        import matplotlib.pyplot as plt

        n = min(max_paths, paths.shape[0])
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(paths[:n].T, alpha=0.2, linewidth=0.9)
        ax.set_title("Neural SDE Simulated Asset Paths")
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Asset Price")
        ax.grid(alpha=0.2)
        fig.tight_layout()
        return fig

    @staticmethod
    def plot_terminal_distribution(paths: np.ndarray, bins: int = 60):
        import matplotlib.pyplot as plt

        terminal = paths[:, -1]
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.hist(terminal, bins=bins, density=True, alpha=0.7)
        ax.set_title("Terminal Price Distribution")
        ax.set_xlabel("S(T)")
        ax.set_ylabel("Density")
        ax.grid(alpha=0.2)
        fig.tight_layout()
        return fig

    @staticmethod
    def plot_learned_functions(model, s_min: float, s_max: float, t_frac: float = 0.5, n_points: int = 150):
        import matplotlib.pyplot as plt
        import torch

        device = next(model.parameters()).device
        s_grid = torch.linspace(s_min, s_max, n_points, device=device)
        t_grid = torch.full((n_points,), float(t_frac), device=device)

        with torch.no_grad():
            mu, sigma = model.forward(s_grid, t_grid, None)

        s_np = s_grid.detach().cpu().numpy()
        mu_np = mu.detach().cpu().numpy()
        sigma_np = sigma.detach().cpu().numpy()

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        axes[0].plot(s_np, mu_np, color="#1f77b4")
        axes[0].set_title("Learned Drift mu(S,t)")
        axes[0].set_xlabel("S")
        axes[0].set_ylabel("Drift")
        axes[0].grid(alpha=0.2)

        axes[1].plot(s_np, sigma_np, color="#d62728")
        axes[1].set_title("Learned Diffusion sigma(S,t)")
        axes[1].set_xlabel("S")
        axes[1].set_ylabel("Volatility")
        axes[1].grid(alpha=0.2)

        fig.tight_layout()
        return fig
