# Volatility Modeling and Regime Detection

## Overview
Volatility is the most critical input in option pricing and risk management. Unlike price, which is directly observable, volatility must be estimated or implied from market data. Understanding volatility dynamics is essential for accurate option pricing, hedging, and trading strategies.

## Types of Volatility

### Historical (Realized) Volatility
Computed from past price data as the annualized standard deviation of log returns:

σ_realized = √(252/n × Σ(r_i - r̄)²)

where r_i = ln(S_i/S_{i-1}) are daily log returns and 252 is the number of trading days per year.

Realized volatility is backward-looking and tells you what volatility was, not what it will be. Different lookback windows (20-day, 60-day, 252-day) give different estimates.

### Implied Volatility (IV)
Extracted from observed option prices by inverting the Black-Scholes formula. If the market price of a call is C_market, then IV is the value of σ that satisfies:

C_BS(S, K, T, r, σ) = C_market

Implied volatility is forward-looking — it represents the market's consensus expectation of future volatility over the option's remaining life. It incorporates all available information and market sentiment.

### VIX Index
The CBOE Volatility Index (VIX) measures 30-day expected volatility of the S&P 500, calculated from a portfolio of out-of-the-money options. It is often called the "fear gauge" because it spikes during market turmoil. VIX above 30 indicates high fear; below 15 indicates complacency.

## The Volatility Surface

### Volatility Smile
A plot of implied volatility against strike price for a fixed expiration typically shows a U-shaped curve (smile) for equity index options. Deep out-of-the-money puts and calls have higher IV than at-the-money options.

The smile exists because:
- Real asset returns have fat tails (higher probability of extreme moves than the normal distribution predicts)
- There is demand for downside protection (put buying drives up IV for low strikes)
- Jump risk and crash fear are priced into options

### Volatility Skew
For equity index options, the smile is asymmetric — it is steeper on the downside (low strikes). This "skew" reflects:
- Leverage effect: Falling prices increase financial leverage, raising volatility
- Crash risk premium: Investors pay more for put protection
- Correlation effect: In market declines, correlations between stocks increase

### Term Structure of Volatility
Implied volatility varies by expiration date. Typically:
- Short-term IV is more sensitive to current market conditions
- Long-term IV tends to be more stable, closer to the long-run average
- During market stress, short-term IV spikes above long-term IV (inverted term structure)

## Volatility Models

### GARCH (Generalized Autoregressive Conditional Heteroskedasticity)
Models time-varying volatility as a function of past returns and past variance:

σ²(t) = ω + α × r²(t-1) + β × σ²(t-1)

Key features: captures volatility clustering (large moves follow large moves), mean reversion of volatility, and leverage effects (with asymmetric variants like EGARCH and GJR-GARCH).

### Heston Stochastic Volatility Model
Volatility itself follows a random process:

dS = μSdt + √v × S × dW₁
dv = κ(θ - v)dt + ξ√v × dW₂

where v is the variance process, κ is the mean reversion speed, θ is the long-run variance, and ξ is the volatility of volatility. The two Brownian motions W₁ and W₂ are correlated with parameter ρ.

The Heston model can generate volatility smiles and skews, unlike Black-Scholes.

### Local Volatility (Dupire)
Volatility is a deterministic function of price and time: σ(S, t). Calibrated to match the entire implied volatility surface. Provides consistent pricing across all strikes and maturities but may produce unrealistic dynamics.

### SABR Model
Stochastic Alpha Beta Rho model, widely used in interest rate markets:
dF = σ × F^β × dW₁
dσ = α × σ × dW₂

Parameters: α (volatility of volatility), β (backbone parameter), ρ (correlation). Provides an analytical approximation for implied volatility.

## Volatility Regimes

### Low Volatility Regime
- VIX below 15, IV below historical average
- Steady uptrend in markets, low put demand
- Options are cheap, good for buying protection
- Carry trades and short volatility strategies perform well

### Normal/Risk-On Regime
- VIX between 15-20, moderate IV levels
- Normal market functioning, balanced risk appetite
- Standard hedging ratios work well
- Both directional and volatility strategies viable

### Risk-Off/Elevated Regime
- VIX between 20-30, IV above historical average
- Increased market uncertainty, flight to quality
- Correlation spikes, diversification breaks down
- Delta hedging costs increase significantly

### High Volatility/Crisis Regime
- VIX above 30, IV at extreme levels
- Market panic, liquidity dries up
- Gap risk increases, models break down
- Extreme moves become more frequent
- Portfolio insurance becomes very expensive

## Regime Detection Methods

### Statistical Methods
- Hidden Markov Models (HMM): Identify latent volatility states
- Change-point detection: Find structural breaks in volatility
- Rolling window analysis: Compare short vs long-term realized vol

### Machine Learning Approaches
- Clustering (K-means, GMM) on volatility features
- Classification models using VIX, skew, term structure as inputs
- Neural networks for pattern recognition in volatility surfaces

### Volatility Risk Premium
The difference between implied and realized volatility (VRP = IV - RV) is a key indicator:
- Large positive VRP: Market overpricing risk, good for selling options
- Near zero VRP: Fair pricing
- Negative VRP: Rare, indicates extreme complacency or market dislocation
