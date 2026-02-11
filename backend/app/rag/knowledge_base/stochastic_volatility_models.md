# Stochastic Volatility Models

## Overview
Stochastic volatility (SV) models allow volatility to evolve randomly over time, capturing the empirical observation that market volatility is not constant. Unlike the Black-Scholes constant volatility assumption, SV models generate realistic volatility smiles, skews, and term structures that match observed option market prices.

## The Heston Model

### Dynamics
The Heston (1993) model specifies:

dS = μS dt + √v S dW₁
dv = κ(θ - v) dt + ξ√v dW₂

where:
- v(t) is the instantaneous variance process
- κ is the mean reversion speed (how fast variance returns to θ)
- θ is the long-run variance (the equilibrium level)
- ξ is the volatility of volatility (vol-of-vol)
- ρ = Corr(dW₁, dW₂) is the correlation between price and variance shocks

### Key Parameters
- **κ (mean reversion)**: Higher κ means variance returns to θ faster. Typical values: 1-5.
- **θ (long-run variance)**: The steady-state variance level. θ = σ²_long where σ_long is the long-run volatility.
- **ξ (vol of vol)**: Controls the curvature of the volatility smile. Higher ξ produces more pronounced smiles.
- **ρ (correlation)**: Typically negative for equity markets (ρ ≈ -0.7 to -0.3), generating the volatility skew. Negative ρ means falling prices increase volatility (leverage effect).
- **v₀ (initial variance)**: Current instantaneous variance level.

### Feller Condition
If 2κθ > ξ², the variance process never reaches zero. This is the Feller condition and is important for ensuring well-defined dynamics.

### Semi-Analytical Pricing
The Heston model has a semi-analytical solution using the characteristic function:

C(S,v,t) = S × P₁ - K × e^(-rτ) × P₂

where P₁ and P₂ are computed via inverse Fourier transforms of the characteristic function. This is much faster than Monte Carlo for European options.

### Calibration
Heston parameters are calibrated by minimizing the difference between model-implied and market-observed implied volatilities:

min Σ [IV_model(Kᵢ, Tᵢ; κ,θ,ξ,ρ,v₀) - IV_market(Kᵢ, Tᵢ)]²

Common optimization methods: Levenberg-Marquardt, differential evolution, particle swarm.

## The SABR Model

### Dynamics
The SABR (Stochastic Alpha Beta Rho) model by Hagan et al. (2002):

dF = σ × F^β × dW₁
dσ = α × σ × dW₂

where:
- F is the forward price
- σ is the stochastic volatility
- β ∈ [0,1] controls the backbone (β=1 is lognormal, β=0 is normal)
- α is the volatility of volatility
- ρ = Corr(dW₁, dW₂)

### Implied Volatility Approximation
SABR provides an analytical approximation for the implied volatility:

σ_BS(K) ≈ σ_ATM × f(K/F; α, β, ρ)

This allows very fast calibration and computation, making SABR the industry standard for interest rate options.

### Use Cases
- Swaption pricing and hedging
- Cap/floor pricing
- FX option markets
- Any market where a simple, fast volatility model is needed

## Local Volatility (Dupire Model)

### Concept
Local volatility σ(S,t) is a deterministic function of price and time, calibrated to perfectly reproduce the entire implied volatility surface:

dS = μS dt + σ(S,t) S dW

### Dupire's Formula
The local volatility can be extracted from market option prices:

σ²(K,T) = [∂C/∂T + rK × ∂C/∂K] / [½K² × ∂²C/∂K²]

### Advantages
- Exact calibration to all observed option prices
- Consistent pricing across strikes and maturities
- Useful as a benchmark model

### Limitations
- Dynamics are unrealistic (predicts volatility moves in wrong direction after calibration)
- Forward smiles flatten unrealistically
- Not suitable for path-dependent exotic pricing without adjustment

## Local Stochastic Volatility (LSV)

### Hybrid Approach
LSV combines local and stochastic volatility:

dS = μS dt + L(S,t) × √v × S dW₁
dv = κ(θ - v) dt + ξ√v dW₂

The leverage function L(S,t) is calibrated so that the model reproduces the entire vanilla surface while maintaining realistic SV dynamics.

### Benefits
- Exact calibration to vanilla surface (like local vol)
- Realistic dynamics and Greeks (like stochastic vol)
- Industry standard for exotic option pricing at major banks

## GARCH Models

### GARCH(1,1)
σ²(t) = ω + α × ε²(t-1) + β × σ²(t-1)

where:
- ω is the constant (long-run variance contribution)
- α captures the impact of recent shocks (news coefficient)
- β captures persistence (memory coefficient)
- Stationarity requires α + β < 1

### Properties
- **Volatility clustering**: Large shocks lead to high volatility periods
- **Mean reversion**: Volatility reverts to ω/(1-α-β)
- **Symmetric response**: Positive and negative shocks have equal impact (unlike asymmetric variants)

### Asymmetric GARCH Variants
- **EGARCH**: Allows asymmetric response to positive and negative returns
- **GJR-GARCH**: Adds a leverage term for negative returns
- **TGARCH**: Threshold GARCH with regime-dependent parameters

### GARCH Option Pricing
Duan (1995) developed the GARCH option pricing model:
- Uses the locally risk-neutral valuation relationship (LRNVR)
- Monte Carlo simulation under the GARCH dynamics
- Captures volatility clustering in option prices
- More realistic than constant volatility but computationally intensive

## Model Comparison

| Model | Smile | Skew | Dynamics | Speed | Calibration |
|-------|-------|------|----------|-------|-------------|
| Black-Scholes | No | No | Constant vol | Instant | N/A |
| Heston | Yes | Yes | Realistic | Fast (semi-analytical) | Moderate |
| SABR | Yes | Yes | Approximate | Very fast | Easy |
| Local Vol | Yes | Yes | Unrealistic | Moderate | Exact |
| LSV | Yes | Yes | Realistic | Slow | Exact |
| GARCH | Yes | Partial | Realistic | Slow (MC) | Moderate |

## Practical Considerations

### Model Selection
- **Vanilla options**: Heston or SABR (fast, calibratable)
- **Barrier/exotic options**: LSV (accurate Greeks, exact calibration)
- **Interest rate options**: SABR (industry standard)
- **Risk management**: GARCH (captures clustering, discrete-time)
- **Academic research**: Heston (tractable, well-studied)

### Implementation Tips
1. Use the Heston characteristic function with the Gauss-Laguerre quadrature for speed
2. For Monte Carlo under Heston, use the QE (Quadratic Exponential) scheme for variance simulation
3. Calibrate SABR to ATM vol and 25-delta risk reversal/butterfly
4. Always check Feller condition when calibrating Heston
5. Use regularization in calibration to avoid overfitting to noisy market data
