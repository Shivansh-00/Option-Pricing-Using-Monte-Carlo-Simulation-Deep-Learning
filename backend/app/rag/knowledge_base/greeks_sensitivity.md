# Option Greeks and Sensitivity Analysis

## Overview
The Greeks are fundamental risk measures that quantify the sensitivity of an option's price to changes in underlying parameters. They are essential tools for risk management, hedging, and trading. Each Greek measures the partial derivative of the option price with respect to a specific input variable.

## The Five Major Greeks

### Delta (Δ)
**Definition**: Delta measures the rate of change of the option price with respect to changes in the underlying asset price.

- **Formula**: Δ = ∂V/∂S
- **Call Delta**: N(d₁), ranges from 0 to 1
- **Put Delta**: N(d₁) - 1, ranges from -1 to 0
- **At-the-Money (ATM)**: Delta is approximately 0.5 for calls and -0.5 for puts
- **Deep In-the-Money (ITM)**: Delta approaches 1 for calls, -1 for puts
- **Deep Out-of-the-Money (OTM)**: Delta approaches 0

**Interpretation**: A delta of 0.60 means the option price increases by approximately $0.60 for every $1 increase in the underlying stock price. Delta also approximates the probability that the option expires in-the-money under the risk-neutral measure.

**Hedging**: Delta hedging involves holding Δ shares of stock for each option sold to create a delta-neutral position. This eliminates first-order price risk but must be rebalanced as delta changes.

### Gamma (Γ)
**Definition**: Gamma measures the rate of change of delta with respect to changes in the underlying price. It is the second derivative of the option price with respect to the stock price.

- **Formula**: Γ = ∂²V/∂S² = ∂Δ/∂S
- **Always Positive**: For both calls and puts (long positions)
- **Maximum at ATM**: Gamma is highest when the option is at-the-money
- **Increases near Expiration**: Short-dated ATM options have very high gamma

**Interpretation**: High gamma means delta is changing rapidly, requiring frequent rebalancing. A gamma of 0.05 means delta increases by 0.05 for every $1 move in the stock. Gamma risk is the risk that large price moves will cause significant hedging losses.

**Gamma Scalping**: Traders with long gamma positions profit from large price moves in either direction by continuously rebalancing their delta hedge.

### Vega (ν)
**Definition**: Vega measures the sensitivity of the option price to changes in implied volatility. It is not a Greek letter, but the name is universally used in finance.

- **Formula**: ν = ∂V/∂σ
- **Always Positive**: For long options (both calls and puts)
- **Maximum at ATM**: Vega is highest for at-the-money options
- **Increases with Time**: Longer-dated options have higher vega

**Interpretation**: A vega of 0.15 means the option price increases by $0.15 for a 1 percentage point increase in implied volatility (e.g., from 20% to 21%). Vega is critical during earnings announcements and market events when volatility shifts dramatically.

**Vega Hedging**: Traders hedge vega exposure by taking offsetting positions in other options. Unlike delta, vega cannot be hedged with the underlying stock alone.

### Theta (Θ)
**Definition**: Theta measures the rate of time decay — how much the option price decreases as time passes, all else being equal.

- **Formula**: Θ = ∂V/∂t (often expressed as daily decay)
- **Usually Negative**: For long options (time decay erodes value)
- **Accelerates near Expiration**: Time decay is fastest in the final weeks before expiration
- **ATM options** have the highest absolute theta

**Interpretation**: A theta of -0.05 means the option loses $0.05 in value each day due to the passage of time. Theta and gamma are related: long gamma positions (which profit from large moves) pay for that benefit through negative theta (time decay).

**Time Decay Strategies**: Sellers of options (short positions) benefit from theta decay. Strategies like iron condors, credit spreads, and covered calls generate income from theta.

### Rho (ρ)
**Definition**: Rho measures the sensitivity of the option price to changes in the risk-free interest rate.

- **Formula**: ρ = ∂V/∂r
- **Calls have Positive Rho**: Higher interest rates increase call values because the present value of the strike price decreases
- **Puts have Negative Rho**: Higher interest rates decrease put values
- **Larger for Long-Dated Options**: Short-dated options have minimal rho exposure

**Interpretation**: Rho is generally the least significant Greek for short-dated equity options, but becomes important for long-dated options (LEAPS) and interest rate derivatives.

## Higher-Order Greeks

### Vanna
∂Δ/∂σ = ∂ν/∂S — Sensitivity of delta to volatility changes or equivalently sensitivity of vega to price changes. Important for managing the interaction between directional and volatility risk.

### Volga (Vomma)
∂²V/∂σ² = ∂ν/∂σ — Sensitivity of vega to volatility changes. Captures the convexity of the option price with respect to volatility.

### Charm (Delta Decay)
∂Δ/∂t — Rate of change of delta over time. Measures how the hedge ratio changes as time passes.

### Speed
∂Γ/∂S = ∂³V/∂S³ — Rate of change of gamma with respect to the underlying price.

## Finite Difference Method for Greeks

When analytical formulas are not available (e.g., for exotic options or Monte Carlo pricing), Greeks are computed numerically using finite differences:

- **Central Difference**: Δ ≈ [V(S+ε) - V(S-ε)] / (2ε)
- **Gamma**: Γ ≈ [V(S+ε) - 2V(S) + V(S-ε)] / ε²

The finite difference method is versatile and works with any pricing model, though it requires careful choice of the perturbation size ε to balance truncation error and numerical noise.

## Greeks in Risk Management

### Portfolio Greeks
Portfolio-level Greeks are the sum of individual position Greeks, weighted by position size. A trading desk monitors total delta, gamma, vega, and theta to manage aggregate risk exposure.

### Dynamic Hedging
Continuous rebalancing of the hedge based on changing Greeks. In practice, hedging is done at discrete intervals, introducing hedging error proportional to gamma and the square of the price move.

### Scenario Analysis
Greeks enable rapid estimation of P&L under different market scenarios without full repricing. For example: ΔP&L ≈ Δ×ΔS + ½Γ×(ΔS)² + ν×Δσ + Θ×Δt
