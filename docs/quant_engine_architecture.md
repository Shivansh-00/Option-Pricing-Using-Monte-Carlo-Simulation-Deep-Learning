# Quant Intelligence Engine — System Architecture

## Overview

The Quant Intelligence Engine is a fully integrated, institutional-grade AI-powered platform embedded within the OptionQuant ecosystem. It extends the existing Black-Scholes, Monte Carlo, and Deep Learning pricing capabilities with 9 advanced quantitative modules that interact cohesively.

```
┌─────────────────────────────────────────────────────────────────────┐
│                    QUANT INTELLIGENCE ENGINE                        │
│                                                                     │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐       │
│  │  PINNs   │  │RL Hedging│  │Vol Surface│  │ Jump Diffusion│      │
│  │ Pricing  │──│ DQN/PPO  │──│Transformer│──│ + HMM Regime │      │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └──────┬───────┘      │
│       │              │              │               │               │
│  ┌────▼──────────────▼──────────────▼───────────────▼──────┐       │
│  │              REGIME-AWARE PRICING CORE                    │      │
│  │   Regime detection feeds into ALL pricing & risk modules  │      │
│  └────┬──────────────┬──────────────┬───────────────┬──────┘       │
│       │              │              │               │               │
│  ┌────▼─────┐  ┌────▼─────┐  ┌────▼──────┐  ┌────▼──────┐       │
│  │Arbitrage │  │Uncertainty│  │  GPU MC   │  │ Portfolio │       │
│  │ Engine   │  │   Quant   │  │  Engine   │  │   Risk    │       │
│  └────┬─────┘  └────┬─────┘  └────┬──────┘  └────┬──────┘       │
│       │              │              │               │               │
│  ┌────▼──────────────▼──────────────▼───────────────▼──────┐       │
│  │              EXPLAINABLE AI LAYER                        │       │
│  │   Unified narratives for ALL decisions across modules    │       │
│  └──────────────────────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────────────────┘
```

## Module Catalog

### 1. PINNs Option Pricing (`pinns.py`)
- **Purpose**: Physics-Informed Neural Networks embedding Black-Scholes PDE directly into the loss function
- **Architecture**: Multi-layer feedforward NN with tanh activations
- **Loss**: `L = L_data + λ_pde·L_PDE + λ_arb·L_arbitrage + λ_smooth·L_smoothness`
- **PDE Residual**: `∂V/∂t + ½σ²S²·∂²V/∂S² + rS·∂V/∂S - rV = 0`
- **Arbitrage Constraints**: Lower/upper bounds, non-negative delta & gamma
- **Training**: SPSA gradient estimation (consistent with existing codebase)
- **API**: `POST /api/v1/quant/pinns/{train,predict,greeks}`

### 2. RL Dynamic Hedging (`rl_hedging.py`)  
- **Purpose**: Reinforcement learning agents for optimal delta hedging
- **Agents**: DQN (11 discrete actions) + PPO (continuous policy gradient)
- **State Space**: [moneyness, σ, Δ, Γ, Θ, regime, hedge_ratio, P&L]
- **Environment**: GBM + jumps + regime switching dynamics
- **Reward**: -|P&L| with transaction cost penalty & drawdown penalty
- **Benchmarks**: Backtested against Black-Scholes delta hedging
- **API**: `POST /api/v1/quant/hedging/{train,backtest,suggest}`

### 3. Transformer Vol Surface (`vol_surface_transformer.py`)
- **Purpose**: Self-attention model for predicting full implied volatility surfaces
- **Architecture**: Multi-head attention + cross-attention + sinusoidal positional encoding
- **Config**: d_model=32, n_heads=4, n_layers=2, 15 strikes × 10 maturities
- **Loss**: MSE + smoothness penalty + calendar arbitrage penalty
- **Training Data**: SABR-like synthetic surfaces with regime conditioning
- **API**: `POST /api/v1/quant/vol-surface/{train,predict}`

### 4. Jump Diffusion & Regime Switching (`jump_diffusion.py`)
- **Purpose**: Merton Jump Diffusion pricing + Enhanced HMM regime detection
- **Merton JD**: Analytical series expansion + Monte Carlo with Poisson jumps
- **HMM**: Full Baum-Welch EM training + Viterbi decoding + online prediction
- **Regimes**: BULL (σ≈12%), BEAR (σ≈25%), CRISIS (σ≈40%)
- **Crisis Adjustment**: +15% premium in crisis, +5% in bear
- **API**: `POST /api/v1/quant/jump-diffusion/{price,calibrate,scenario}`

### 5. Arbitrage Detection Engine (`arbitrage_engine.py`)
- **Purpose**: Multi-dimensional arbitrage scanning
- **Checks**: Put-Call Parity, Calendar Spread, Butterfly, Box Spread, Vol Surface Consistency
- **Regime Awareness**: Wider thresholds in crisis (2.5× multiplier)
- **Output**: Signal strength (0-1), expected profit, risk score, trade recommendation
- **API**: `POST /api/v1/quant/arbitrage/{scan,put-call-parity}`

### 6. Uncertainty Quantification (`uncertainty.py`)
- **Purpose**: Bayesian NN + MC Dropout for uncertainty decomposition
- **Bayesian NN**: Variational inference with weight uncertainty (μ, ρ parameterization)
- **MC Dropout**: Multiple forward passes with dropout at inference time
- **Output**: Epistemic + aleatoric uncertainty, reliability classification
- **API**: `POST /api/v1/quant/uncertainty/{quantify,train}`

### 7. GPU-Accelerated Monte Carlo (`gpu_monte_carlo.py`)
- **Purpose**: PyTorch CUDA-accelerated MC simulation
- **Models**: GBM, Heston (stochastic vol), Merton (jump diffusion)
- **Variance Reduction**: Antithetic, Control Variate, Stratified, Importance Sampling
- **Backend**: Auto-detects PyTorch/CUDA with graceful NumPy CPU fallback
- **Target**: <200ms for 1M paths on GPU
- **API**: `POST /api/v1/quant/gpu-mc/{price,benchmark}`

### 8. Portfolio Risk Dashboard (`portfolio_risk.py`)
- **Purpose**: Multi-option portfolio risk management
- **VaR**: Parametric (delta-normal), Historical simulation, Monte Carlo
- **Stress Tests**: 8 predefined scenarios (Market Crash, Flash Crash, 2008-like, etc.)
- **Risk Rating**: LOW / MODERATE / HIGH / CRITICAL
- **API**: `POST /api/v1/quant/portfolio/{risk-report,stress-test}`

### 9. Explainable AI Layer (`quant_explainer.py`)
- **Purpose**: Unified "Why?" interface for all quant decisions
- **Covers**: Price, Hedge, Arbitrage, Regime, Vol Surface explanations
- **Output**: Human-readable narratives + key drivers + confidence
- **API**: `POST /api/v1/quant/explain/decision`

## API Surface

All endpoints are under `/api/v1/quant/` with JWT authentication.

| Module | Endpoint | Method | Description |
|--------|----------|--------|-------------|
| PINNs | `/pinns/train` | POST | Train PINNs model |
| PINNs | `/pinns/predict` | POST | Price with PINNs |
| PINNs | `/pinns/greeks` | POST | PDE-informed Greeks |
| RL Hedging | `/hedging/train` | POST | Train RL agent |
| RL Hedging | `/hedging/backtest` | POST | Backtest vs BS delta |
| RL Hedging | `/hedging/suggest` | POST | Real-time hedge suggestion |
| Vol Surface | `/vol-surface/train` | POST | Train transformer |
| Vol Surface | `/vol-surface/predict` | POST | Predict vol surface |
| Jump Diffusion | `/jump-diffusion/price` | POST | Merton JD pricing |
| Jump Diffusion | `/jump-diffusion/calibrate` | POST | HMM regime calibration |
| Jump Diffusion | `/jump-diffusion/scenario` | POST | Regime scenario analysis |
| Arbitrage | `/arbitrage/scan` | POST | Full arbitrage scan |
| Arbitrage | `/arbitrage/put-call-parity` | POST | PCP check |
| Uncertainty | `/uncertainty/quantify` | POST | Full UQ analysis |
| Uncertainty | `/uncertainty/train` | POST | Train BNN/Dropout |
| GPU MC | `/gpu-mc/price` | POST | GPU-accelerated pricing |
| GPU MC | `/gpu-mc/benchmark` | POST | CPU vs GPU benchmark |
| Portfolio | `/portfolio/risk-report` | POST | Full risk report |
| Portfolio | `/portfolio/stress-test` | POST | Stress testing |
| Explainer | `/explain/decision` | POST | Explain any decision |
| Status | `/status` | GET | Ecosystem health |

## Cross-Module Integration

The system is designed as ONE cohesive ecosystem where modules interact:

1. **Regime → Everything**: HMM regime detection feeds into PINNs, Hedging, Arbitrage, Portfolio Risk
2. **Uncertainty → Pricing**: All pricing outputs can be uncertainty-quantified
3. **Explainer → All**: Unified explanation layer covers all module decisions
4. **Vol Surface → Arbitrage**: Surface predictions feed consistency checks
5. **GPU MC → Portfolio**: GPU engine powers portfolio VaR Monte Carlo
6. **Jump Diffusion → Hedging**: Jump dynamics in hedging environment

## Design Patterns

- **Singleton Pattern**: All modules use `get_*()` factory functions for single instances
- **Pure NumPy Core**: All implementations use NumPy with optional PyTorch for GPU
- **SPSA Gradients**: Simultaneous Perturbation Stochastic Approximation throughout
- **Async Integration**: `asyncio.to_thread()` for CPU-intensive operations
- **Graceful Degradation**: Optional dependencies (PyTorch, CUDA) with fallbacks

## File Structure

```
backend/app/
├── pinns.py                    # PINNs Option Pricer
├── rl_hedging.py               # RL Dynamic Hedging (DQN + PPO)
├── vol_surface_transformer.py  # Transformer Vol Surface
├── jump_diffusion.py           # Merton JD + HMM Regime
├── arbitrage_engine.py         # Arbitrage Detection
├── uncertainty.py              # Uncertainty Quantification
├── gpu_monte_carlo.py          # GPU Monte Carlo Engine
├── portfolio_risk.py           # Portfolio Risk Dashboard
├── quant_explainer.py          # Explainable AI Layer
├── quant_schemas.py            # Pydantic schemas for quant APIs
└── api/
    └── quant_routes.py         # FastAPI route handlers
```
