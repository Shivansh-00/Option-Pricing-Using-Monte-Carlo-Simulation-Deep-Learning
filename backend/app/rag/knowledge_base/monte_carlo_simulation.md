# Monte Carlo Simulation for Option Pricing

## Overview
Monte Carlo simulation is a computational technique that uses random sampling to approximate the value of financial derivatives. Unlike closed-form solutions like Black-Scholes, Monte Carlo methods can handle complex payoffs, path-dependent options, and multi-asset derivatives. The method was pioneered in finance by Phelim Boyle in 1977.

## The Monte Carlo Method

### Basic Principle
The fair price of a European option under risk-neutral pricing is the discounted expected payoff:

V = e^(-rT) × E[payoff(S_T)]

Monte Carlo approximates this expectation by:
1. Simulating many possible price paths for the underlying asset
2. Computing the payoff for each path
3. Averaging all payoffs
4. Discounting the average to present value

### Geometric Brownian Motion (GBM)
The standard model for asset price dynamics is GBM:

dS = μSdt + σSdW

Under the risk-neutral measure, the drift μ is replaced by the risk-free rate r:

S(t + Δt) = S(t) × exp[(r - σ²/2)Δt + σ√(Δt) × Z]

where Z ~ N(0,1) is a standard normal random variable.

### Algorithm Steps

1. **Initialize Parameters**: Set spot price S₀, strike K, maturity T, risk-free rate r, volatility σ, number of paths N, number of time steps M.

2. **Discretize Time**: Divide [0, T] into M steps of size Δt = T/M.

3. **Generate Random Paths**: For each path i = 1,...,N:
   - Start at S₀
   - At each time step, generate Z ~ N(0,1)
   - Update: S(t+Δt) = S(t) × exp[(r - σ²/2)Δt + σ√(Δt) × Z]

4. **Compute Payoffs**: For each path, compute the option payoff at maturity:
   - Call: max(S_T - K, 0)
   - Put: max(K - S_T, 0)

5. **Average and Discount**: V = e^(-rT) × (1/N) × Σ payoff_i

## Variance Reduction Techniques

### Antithetic Variates
For every random path generated with Z, also generate the mirror path with -Z. This creates negatively correlated pairs that reduce variance by exploiting symmetry. The variance reduction can be up to 50% for near-at-the-money options.

### Control Variates
Use a related quantity with a known expected value (like the Black-Scholes price) as a control. Adjust the Monte Carlo estimate by the difference between the simulated and known values of the control variable.

### Importance Sampling
Shift the probability distribution to sample more from regions that contribute most to the option value. This is especially useful for deep out-of-the-money options where most paths have zero payoff.

### Stratified Sampling
Divide the probability space into strata and sample uniformly from each stratum. This ensures better coverage of the distribution and reduces clustering of random samples.

### Quasi-Random Sequences (Quasi-Monte Carlo)
Replace pseudorandom numbers with low-discrepancy sequences (Sobol, Halton) that fill the space more uniformly. Quasi-Monte Carlo achieves O(1/N) convergence vs O(1/√N) for standard Monte Carlo.

## Convergence Properties

- **Standard Error**: SE = σ_payoff / √N, where σ_payoff is the standard deviation of discounted payoffs
- **Convergence Rate**: O(1/√N) — doubling accuracy requires 4× more paths
- **Confidence Interval**: The 95% CI is approximately: V ± 1.96 × SE
- **Bias**: The Monte Carlo estimator is unbiased for European options with exact simulation

## Path-Dependent Options

Monte Carlo excels at pricing path-dependent options:

### Asian Options
Payoff depends on the average price over the option's life: max(Ā - K, 0) where Ā is the arithmetic average. No closed-form solution exists for arithmetic Asian options, making Monte Carlo the standard approach.

### Barrier Options
Payoff depends on whether the price crosses a barrier level during the option's life. Knock-in options activate when a barrier is hit; knock-out options expire worthless.

### Lookback Options
Payoff depends on the maximum or minimum price achieved during the option's life. For example, a floating strike lookback call pays S_T - S_min.

## Advantages of Monte Carlo

1. **Flexibility**: Can handle any payoff structure, path dependency, and multiple underlying assets
2. **Scalability**: Adding complexity to the model (stochastic volatility, jumps) requires only changing the simulation, not re-deriving formulas
3. **Dimensionality**: Performance does not deteriorate with the number of underlying assets (unlike PDE methods)
4. **Error Estimation**: Standard error provides a natural confidence interval for the price estimate
5. **Parallelization**: Each path is independent, making Monte Carlo highly parallelizable on modern hardware

## Limitations

1. **Computational Cost**: Requires thousands to millions of paths for accurate pricing, especially for exotic options
2. **Slow Convergence**: O(1/√N) convergence is slow compared to analytical or tree-based methods
3. **American Options**: Requires special algorithms (Longstaff-Schwartz least squares regression) for early exercise features
4. **Greeks Computation**: Finite difference Greeks from Monte Carlo have high noise; pathwise derivatives or likelihood ratio methods are needed
