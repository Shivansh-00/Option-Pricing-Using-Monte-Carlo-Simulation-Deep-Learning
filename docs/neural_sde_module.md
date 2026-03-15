# Neural SDE Option Pricing Module

## 1) Mathematical Formulation

The Neural SDE module models spot dynamics as:

\[
\mathrm{d}S_t = \mu_\theta(S_t, t, x_t) S_t \,\mathrm{d}t + \sigma_\theta(S_t, t, x_t) S_t \,\mathrm{d}W_t
\]

where:
- \(\mu_\theta\): neural drift network
- \(\sigma_\theta\): neural diffusion network
- \(x_t\): market feature vector (optional)

For numerical stability and positivity of \(S_t\), training and simulation use log-Euler discretization:

\[
\log S_{t+\Delta t} - \log S_t = \left(\mu_\theta - \tfrac{1}{2}\sigma_\theta^2\right)\Delta t + \sigma_\theta\sqrt{\Delta t}Z_t
\]

with \(Z_t \sim \mathcal{N}(0,1)\).

## 2) Training Objective

Loss combines three parts:

\[
\mathcal{L} = w_d \mathcal{L}_{dist} + w_p \mathcal{L}_{path} + w_s \mathcal{L}_{stab}
\]

- Distribution loss \(\mathcal{L}_{dist}\): Gaussian negative log-likelihood of observed log-returns under predicted local moments + Wasserstein-like moment matching of standardized residuals.
- Path regularization \(\mathcal{L}_{path}\): temporal smoothness penalty on drift and diffusion outputs.
- Stability penalty \(\mathcal{L}_{stab}\): soft constraints to avoid explosive drift/volatility outputs.

This combination improves calibration to empirical return dynamics while maintaining numerically robust Monte Carlo simulation.

## 3) Architecture Diagram (Text)

1. Data ingestion:
- historical prices
- implied vol data
- option chain features
- market indicators

2. Data pipeline:
- timestamp alignment (asfreq + interpolation)
- cleaning/imputation
- feature engineering: return, log-return, realized volatility
- train/validation/test split
- z-score normalization from train split
- PyTorch sequence dataloaders

3. Neural SDE core:
- Drift NN: input (log-spot, periodic time features, market features)
- Diffusion NN: input (log-spot, periodic time features, market features)
- Euler-Maruyama in log-space with vectorized tensors
- GPU acceleration when CUDA is available

4. Pricing engine:
- large-scale path simulation
- European payoff mapping
- discounting and confidence intervals

5. Risk/benchmarking:
- Greeks via autograd + common-random-number finite differences
- benchmarking against Black-Scholes, GBM Monte Carlo, Heston MC

6. API integration:
- train, price, greeks, simulate, benchmark endpoints under /api/v1/neural-sde

## 4) Implemented Components

- backend/app/pricing_engine/neural_sde_model.py
- backend/app/pricing_engine/monte_carlo_simulator.py
- backend/app/pricing_engine/greeks_calculator.py
- backend/app/data/market_data_loader.py
- backend/app/visualization/path_visualizer.py
- backend/app/api/pricing_api.py

## 5) Integration Notes

- New router added in backend/app/main.py.
- New schemas appended in backend/app/schemas.py.
- New dependencies added in backend/requirements.txt.

## 6) Example Experiment Flow

1. Train model:

```bash
POST /api/v1/neural-sde/train
{
  "prices_csv": "backend/data/raw/prices.csv",
  "implied_vol_csv": "backend/data/raw/implied_vol.csv",
  "option_chain_csv": "backend/data/raw/options.csv",
  "indicators_csv": "backend/data/raw/indicators.csv",
  "timestamp_col": "timestamp",
  "price_col": "spot",
  "freq": "1D",
  "lookback": 30,
  "epochs": 30,
  "learning_rate": 0.001,
  "model_tag": "spx_neural_sde"
}
```

2. Price option:

```bash
POST /api/v1/neural-sde/price
{
  "model_tag": "spx_neural_sde",
  "spot": 5050,
  "strike": 5100,
  "maturity": 0.5,
  "rate": 0.045,
  "option_type": "call",
  "paths": 100000,
  "steps": 252
}
```

3. Compute Greeks:

```bash
POST /api/v1/neural-sde/greeks
{
  "model_tag": "spx_neural_sde",
  "spot": 5050,
  "strike": 5100,
  "maturity": 0.5,
  "rate": 0.045,
  "option_type": "call"
}
```

4. Benchmark vs classical models:

```bash
POST /api/v1/neural-sde/benchmark
{
  "model_tag": "spx_neural_sde",
  "spot": 5050,
  "strike": 5100,
  "maturity": 0.5,
  "rate": 0.045,
  "implied_vol": 0.21,
  "option_type": "call",
  "paths": 100000,
  "steps": 252
}
```

## 7) Performance Guidance

- Use CUDA-enabled PyTorch for high-throughput simulation.
- Increase max_batch_paths and paths based on GPU memory.
- 100,000 paths is supported through batch simulation; larger runs can scale by raising paths while tuning batch size.
