# Model Comparison

| Model | Strength | Weakness |
|------|----------|----------|
| Black-Scholes | Fast, closed-form | Assumes constant vol |
| Monte Carlo (GBM) | Flexible, path-aware | Computationally expensive |
| Heston MC | Captures stochastic vol | More parameters to calibrate |
| ML (RF/XGB) | Learns nonlinear patterns | Requires clean features |
| DL (LSTM/Transformer) | Captures temporal regimes | Needs more data |

Metrics used: RMSE, MAE, latency per request.
