# Portfolio Construction and Hedging Strategies

## Overview
Portfolio construction in options trading involves combining different options and underlying assets to achieve specific risk-return profiles. Effective hedging transforms unacceptable risks into manageable exposures while preserving upside potential. Understanding the interaction between Greeks across multiple positions is essential for building robust portfolios.

## Delta Hedging

### Concept
Delta hedging creates a portfolio that is insensitive to small changes in the underlying price by holding Δ shares of stock for each option:

Portfolio = Option + Δ × Stock

The delta-neutral portfolio has zero first-order price sensitivity. As the stock price moves, the delta changes and the hedge must be rebalanced.

### Rebalancing Frequency
- Continuous rebalancing is theoretically ideal but practically impossible
- Daily rebalancing is standard practice
- Hedging error per rebalance ≈ ½ × Γ × (ΔS)²
- More frequent rebalancing reduces tracking error but increases transaction costs

### Hedging Cost
The expected cost of delta hedging is approximately:
Cost ≈ ½ × Γ × σ² × S² × Δt × N_rebalances

This equals approximately the option's theta over time, reflecting the theta-gamma tradeoff.

## Delta-Gamma Hedging

### Two-Instrument Hedge
To neutralize both delta and gamma, use two hedging instruments (the stock and another option):
- Stock hedge amount: ΔS = -(Δ_portfolio + Δ₂ × n₂)
- Option hedge amount: n₂ = -Γ_portfolio / Γ₂

This creates a portfolio with both zero delta and zero gamma, providing protection against larger price moves.

### Higher-Order Hedging
For maximum precision, hedge speed (∂Γ/∂S) and vanna (∂Δ/∂σ) using additional instruments. Each additional Greek requires one more hedging instrument.

## Vega Hedging

### Volatility Exposure
Options traders are exposed to changes in implied volatility. Vega hedging requires offsetting positions in other options since the underlying stock has zero vega.

### Cross-Expiry Hedging
Options with different expirations have different vega profiles. A long vega position in short-dated options can be hedged with a short vega position in longer-dated options, but this introduces term structure risk.

### Vega-Gamma Interaction
Vega and gamma are related: both are highest for ATM options. Hedging one often partially hedges the other, but the relationship is not perfect across strikes and maturities.

## Common Option Strategies

### Covered Call
- Hold stock + Sell OTM call
- Greeks: Positive delta, negative gamma, negative vega, positive theta
- Income generation in flat to mildly bullish markets
- Caps upside at strike price, provides limited downside protection

### Protective Put
- Hold stock + Buy OTM put
- Greeks: Positive delta (reduced), long gamma, long vega, negative theta
- Insurance against catastrophic decline
- Cost is the put premium (theta drag)

### Collar
- Hold stock + Buy put + Sell call
- Zero or low cost (call premium finances put)
- Limits both upside and downside
- Common for concentrated stock positions

### Bull Call Spread
- Buy lower strike call + Sell higher strike call
- Limited profit (capped at spread width) and limited loss (net premium paid)
- Lower cost than outright call purchase
- Defined risk-reward ratio

### Iron Condor
- Sell OTM put + Buy further OTM put + Sell OTM call + Buy further OTM call
- Profits from range-bound markets (high theta)
- Maximum profit = net premium received
- Risk = width of wider spread minus premium

### Butterfly Spread
- Buy 1 lower call + Sell 2 middle calls + Buy 1 upper call
- Maximum profit at middle strike, limited risk
- Profits from low realized volatility
- Low cost but narrow profit range

### Straddle
- Buy ATM call + Buy ATM put (same strike, same expiry)
- Profits from large moves in either direction
- Long gamma, long vega, negative theta
- Break-even = Strike ± total premium paid

### Strangle
- Buy OTM put + Buy OTM call (different strikes)
- Similar to straddle but cheaper (lower premium)
- Requires larger move to profit
- Common pre-earnings or pre-event strategy

## Portfolio-Level Risk Management

### Greek Aggregation
Portfolio Greeks = Σ (position_i × Greek_i) for all positions
- Total Delta: Net directional exposure
- Total Gamma: Net convexity exposure
- Total Vega: Net volatility exposure
- Total Theta: Daily time decay P&L

### Risk Decomposition
Break down portfolio risk by:
1. **Underlying**: Delta/gamma per underlying asset
2. **Expiration**: Vega/theta by expiration bucket
3. **Strategy**: Risk contribution by strategy type
4. **Scenario**: P&L under stress scenarios

### Correlation Risk
- In normal markets, correlations between underlyings are moderate
- In crises, correlations spike to near 1.0 (all assets decline together)
- Diversification benefits evaporate exactly when they're most needed
- Stress test with correlation = 1.0 to see worst-case portfolio behavior

## Dynamic Hedging vs Static Hedging

### Dynamic Hedging
- Continuous (or frequent) rebalancing based on current Greeks
- Hedges any path-dependent risk
- Incurs transaction costs proportional to gamma and realized volatility
- Standard approach for delta hedging

### Static Hedging
- Construct a portfolio of vanilla options that replicates an exotic payoff
- No rebalancing needed (hedge-and-forget)
- Lower transaction costs but only works for specific exotic payoff profiles
- Used extensively for barrier options (Carr-Chou-1998)

## Hedging Under Stochastic Volatility

### Model Risk in Hedging
- Black-Scholes delta is biased under stochastic volatility
- Heston delta includes a correction term for variance beta
- Minimum variance hedge ratio differs from the model delta
- Using the wrong model for hedging can create P&L leakage

### Minimum Variance Hedging
Choose the hedge ratio that minimizes the variance of the hedged portfolio:
h* = -Cov(ΔV, ΔS) / Var(ΔS)

This differs from the model delta when the model is misspecified. Empirically estimated hedge ratios can outperform model-based ratios.

## Performance Metrics

### Hedging Effectiveness
- **Tracking Error**: Standard deviation of the hedged portfolio returns
- **Maximum Drawdown**: Worst cumulative loss of the hedged portfolio
- **P&L Volatility**: Variability of daily hedging P&L
- **Hedge Ratio Stability**: How much the hedge ratio changes between rebalances

### Transaction Cost Analysis
- **Turnover**: Total notional traded per rebalance
- **Spread Cost**: Bid-ask spread paid on each trade
- **Market Impact**: Price movement caused by the hedging trade
- **Total Cost**: Sum of spread + impact + commission per rebalance cycle
