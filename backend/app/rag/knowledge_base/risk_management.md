# Risk Management in Options Trading

## Overview
Risk management is the systematic process of identifying, measuring, and mitigating financial risks in options portfolios. Effective risk management requires understanding the Greeks, portfolio construction principles, and stress testing methodologies. Options are leveraged instruments where small changes in underlying parameters can cause large P&L swings.

## Portfolio Risk Metrics

### Value at Risk (VaR)
VaR estimates the maximum loss over a given time horizon at a specified confidence level:
- **1-day 95% VaR of $1M**: There is a 5% probability that the portfolio will lose more than $1M in one day
- Methods: Historical simulation, parametric (variance-covariance), Monte Carlo VaR
- Limitations: Does not capture tail risk beyond the confidence level

### Conditional VaR (Expected Shortfall)
CVaR measures the expected loss given that the loss exceeds VaR:
- More coherent risk measure than VaR (satisfies subadditivity)
- Better captures tail risk
- Required by Basel III/IV for market risk capital

### Maximum Drawdown
The largest peak-to-trough decline in portfolio value. Important for assessing the worst-case scenario over a historical period.

## Hedging Strategies

### Delta Hedging
- Maintain a delta-neutral position by holding Δ shares per option
- Must be rebalanced as delta changes (due to gamma)
- Hedging cost ≈ ½ × Γ × σ² × S² × Δt per rebalance
- Discrete rebalancing introduces hedging error proportional to gamma

### Delta-Gamma Hedging
- Add a second option to neutralize both delta and gamma
- Requires solving a system of two equations (two instruments for two Greeks)
- More robust than delta-only hedging for large price moves

### Vega Hedging
- Use other options to offset volatility exposure
- Important during earnings, economic releases, and volatility regime shifts
- Often combined with delta hedging in practice

### Portfolio Insurance
- Protective puts: Buy puts to limit downside risk
- Collars: Buy put + sell call to finance protection
- Dynamic hedging with replication of put payoff

## Stress Testing

### Historical Scenarios
- 2008 Financial Crisis: VIX peaked at 80, correlations spiked to near 1
- 2020 COVID Crash: Fastest 30% decline in history, extreme dislocations
- Flash Crash 2010: S&P dropped 9% in minutes, testing market microstructure
- Volmageddon 2018: VIX tripled overnight, inverse VIX ETFs collapsed

### Hypothetical Scenarios
- Spot price shock: ±10%, ±20%, ±30%
- Volatility shock: ±5%, ±10%, ±20% in IV
- Rate shock: ±50bps, ±100bps
- Combined scenarios: Spot down 20% + Vol up 50% (common in crashes)
- Correlation breakdown: All assets move together

### Reverse Stress Testing
Start with a loss threshold and work backward to find scenarios that could cause it. Useful for identifying hidden risk concentrations.

## Risk Limits and Controls

### Position Limits
- Maximum notional exposure per underlying
- Maximum number of contracts per strategy
- Maximum portfolio delta, gamma, vega

### Greeks Limits
- Delta: Total portfolio delta within ±$X per 1% move
- Gamma: Limit maximum gamma to control hedging costs
- Vega: Limit vega to control volatility exposure
- Theta: Monitor daily time decay P&L

### Loss Limits
- Daily loss limit: Stop trading if daily loss exceeds threshold
- Weekly/Monthly loss limit: Review and reduce positions
- Concentration limit: No single position exceeds X% of portfolio

## Common Risk Scenarios for Options

### Pin Risk
When the underlying trades very close to a strike at expiration, creating uncertainty about whether the option will be exercised. Can lead to unexpected overnight positions.

### Assignment Risk
Early assignment of American-style options, particularly:
- Deep ITM calls before ex-dividend dates
- Deep ITM puts when interest rates are high
- Short options in spreads can be assigned individually, creating temporary naked exposure

### Liquidity Risk
- Wide bid-ask spreads in illiquid options
- Difficulty closing large positions without price impact
- Stale quotes that don't reflect current conditions
- Gap risk overnight when markets are closed

### Model Risk
- Incorrect volatility assumptions leading to mispricing
- Wrong choice of pricing model for exotic options
- Calibration errors in stochastic volatility models
- Over-fitting in machine learning pricing models

## Best Practices

1. **Diversify across Greeks**: Don't concentrate risk in a single Greek
2. **Monitor correlation**: Diversification benefits disappear in crises
3. **Stress test regularly**: Test portfolio against extreme but plausible scenarios
4. **Maintain liquidity buffers**: Keep cash reserves for margin calls
5. **Use multiple models**: Compare prices across different pricing models
6. **Review limits daily**: Adjust risk limits based on market regime
7. **Document everything**: Keep records of risk decisions and model assumptions
