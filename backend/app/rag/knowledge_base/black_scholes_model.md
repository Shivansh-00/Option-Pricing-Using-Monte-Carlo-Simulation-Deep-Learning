# Black-Scholes Option Pricing Model

## Overview
The Black-Scholes model (also known as the Black-Scholes-Merton model) is a mathematical framework for pricing European-style options. Developed by Fischer Black, Myron Scholes, and Robert Merton in 1973, it revolutionized quantitative finance and earned Scholes and Merton the Nobel Prize in Economics in 1997.

## The Black-Scholes Formula

### Call Option Price
C = S₀ × N(d₁) - K × e^(-rT) × N(d₂)

### Put Option Price
P = K × e^(-rT) × N(-d₂) - S₀ × N(-d₁)

Where:
- d₁ = [ln(S₀/K) + (r + σ²/2) × T] / (σ × √T)
- d₂ = d₁ - σ × √T
- S₀ = Current stock price (spot price)
- K = Strike price of the option
- T = Time to expiration in years (maturity)
- r = Risk-free interest rate (continuously compounded)
- σ = Volatility of the underlying asset (annualized standard deviation)
- N(x) = Cumulative standard normal distribution function
- e = Euler's number (approximately 2.71828)

## Key Assumptions

1. **Geometric Brownian Motion**: The underlying asset price follows a geometric Brownian motion with constant drift and volatility: dS = μSdt + σSdW, where W is a Wiener process.

2. **No Dividends**: The stock does not pay dividends during the option's life. Extensions exist for dividend-paying stocks using continuous dividend yield.

3. **Constant Volatility**: The volatility σ remains constant over the life of the option. In practice, volatility changes, leading to the volatility smile and skew phenomena.

4. **Constant Risk-Free Rate**: The risk-free interest rate r is known and constant. This is approximated using Treasury bill rates.

5. **European Exercise**: Options can only be exercised at expiration, not before. American options require different pricing methods.

6. **No Transaction Costs**: There are no fees for buying or selling the option or the underlying asset. Real markets have bid-ask spreads and commissions.

7. **Log-Normal Distribution**: The continuously compounded returns of the stock are normally distributed, meaning the stock price itself follows a log-normal distribution.

8. **Continuous Trading**: Trading occurs continuously without jumps. In reality, prices can gap overnight or during market events.

9. **No Arbitrage**: There are no risk-free arbitrage opportunities. This is the fundamental pricing principle.

10. **Frictionless Markets**: Short selling is allowed without restrictions, and assets are perfectly divisible.

## Understanding the Components

### N(d₁) - Delta
N(d₁) represents the option's delta for a call option. It measures the probability-weighted expected value of receiving the stock at expiration, adjusted for risk. In the risk-neutral world, it is the hedge ratio — how many shares of stock to hold to hedge one option.

### N(d₂) - Exercise Probability
N(d₂) represents the risk-neutral probability that the option will be exercised (i.e., finish in-the-money). For a call option, this is the probability that S_T > K under the risk-neutral measure.

### The Term Structure
- S₀ × N(d₁): Present value of receiving the stock conditional on the option being exercised
- K × e^(-rT) × N(d₂): Present value of paying the strike price conditional on exercise

## Practical Applications

### Implied Volatility
Since all other inputs to the Black-Scholes formula are observable, the model can be inverted to extract the market's implied volatility from observed option prices. This implied volatility is widely used as a gauge of market sentiment and expected future price fluctuations.

### Risk Management
The Black-Scholes model provides analytical solutions for option Greeks, enabling precise risk management. Portfolio managers use these Greeks to construct delta-neutral, gamma-neutral, or vega-neutral portfolios.

### Benchmarking
Even when more sophisticated models are used, Black-Scholes serves as a benchmark. Traders often quote option prices in terms of Black-Scholes implied volatility rather than dollar prices.

## Limitations

1. **Constant Volatility Assumption**: Markets exhibit volatility clustering, mean-reversion, and stochastic behavior that violates this assumption. The volatility smile demonstrates this failure.

2. **Fat Tails**: Real return distributions have fatter tails than the normal distribution assumes, meaning extreme events occur more frequently than predicted.

3. **Jump Risk**: The model does not account for sudden price jumps caused by earnings announcements, geopolitical events, or market crashes.

4. **Discrete Hedging**: Continuous rebalancing is impossible in practice. Transaction costs and discrete trading intervals introduce hedging errors.

5. **Early Exercise**: The model cannot handle American options, which allow early exercise. Techniques like the binomial tree or finite difference methods are needed.

## Extensions and Modifications

- **Merton's Jump-Diffusion Model**: Adds Poisson jumps to the GBM process to capture sudden price movements.
- **Heston Model**: Introduces stochastic volatility where volatility itself follows a mean-reverting process.
- **SABR Model**: A stochastic volatility model widely used in interest rate markets.
- **Local Volatility Models**: Allow volatility to be a function of both price and time, calibrated to match the entire implied volatility surface.
