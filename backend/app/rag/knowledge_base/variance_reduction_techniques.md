# Variance Reduction Techniques in Monte Carlo Simulation

## Overview
Variance reduction techniques are methods used to improve the efficiency of Monte Carlo simulations by reducing the statistical error (variance) of the estimate without increasing the number of simulation paths. These techniques can dramatically speed up convergence, sometimes achieving the equivalent accuracy of millions of additional paths.

## Antithetic Variates

### Principle
For every random path generated using standard normal draws Z₁, Z₂, ..., Zₙ, generate a mirror path using -Z₁, -Z₂, ..., -Zₙ. The final estimate is the average of the two correlated estimates.

### Implementation
V_antithetic = ½ × [V(Z) + V(-Z)]

Since the two estimates are negatively correlated, var(V_antithetic) < var(V_standard) / 2. The variance reduction is most effective when the payoff function is monotonic in the underlying price, which is the case for vanilla calls and puts.

### Effectiveness
- Reduces variance by 50-75% for at-the-money European options
- Less effective for deep out-of-the-money options where both paths may have zero payoff
- Essentially free — doubles the number of paths with negligible extra computation
- Works best when the payoff is a monotone function of the terminal stock price

## Control Variates

### Principle
Use a related random variable (the control) whose expected value is known analytically. The control variate technique adjusts the Monte Carlo estimate by the deviation of the simulated control from its known expectation.

### Formula
V_controlled = V_MC - β × (C_simulated - C_exact)

Where:
- V_MC is the raw Monte Carlo estimate
- C_simulated is the simulated value of the control variate
- C_exact is the known analytical value of the control
- β is the optimal control coefficient (estimated from simulation data)

### Optimal β
The optimal coefficient β* = -Cov(V, C) / Var(C), which minimizes the variance of the controlled estimate. In practice, β is estimated from the simulation runs themselves.

### Common Control Variates in Option Pricing
1. **Black-Scholes price** as control for exotic option pricing under stochastic volatility
2. **Geometric Asian option** (closed-form) as control for arithmetic Asian options
3. **Forward price** (known expectation = S₀e^(rT)) as control for any option
4. **European option price** as control for American or barrier options

### Effectiveness
- Variance reduction of 90-99% when the control is highly correlated with the target
- Requires an analytical solution for the control variate
- The closer the control is to the target, the better the reduction
- Can be combined with antithetic variates for even greater improvement

## Importance Sampling

### Principle
Shift the probability distribution to sample more from regions that contribute most to the option value. Instead of sampling from the original distribution f(x), sample from a proposal distribution g(x) and correct with likelihood ratios.

### Formula
V = E_g[payoff(x) × f(x)/g(x)]

The ratio f(x)/g(x) is the likelihood ratio (or Radon-Nikodym derivative) that corrects for sampling from the wrong distribution.

### Application to Options
For deep out-of-the-money options, most standard Monte Carlo paths produce zero payoff. By shifting the mean of the random draws toward the exercise region, more paths produce non-zero payoffs, vastly reducing variance.

### Optimal Drift
The optimal drift shift for a European call with strike K:
μ_shift = [ln(K/S₀) - (r - σ²/2)T] / (σ√T)

This centers the simulation around the exercise price, maximizing the information gained per path.

### Effectiveness
- Variance reduction of 100x-10000x for deep OTM options
- Requires careful choice of the proposal distribution
- Poor choice can increase variance (unlike antithetic/control variates which always help)
- Most beneficial when the event of interest is rare

## Stratified Sampling

### Principle
Divide the probability space [0,1] into N equal strata and draw exactly one uniform random number from each stratum. This ensures systematic coverage of the entire probability distribution, eliminating clustering.

### Implementation
For N paths, instead of drawing U₁,...,Uₙ ~ Uniform(0,1):
- Uᵢ = (i - 1 + Vᵢ)/N, where Vᵢ ~ Uniform(0,1) and i = 1,...,N
- Transform to normal: Zᵢ = Φ⁻¹(Uᵢ)

### Effectiveness
- Typical variance reduction of 2-5x for European options
- Guaranteed to reduce variance (the stratified estimate is always at least as good as standard MC)
- Easy to implement with minimal overhead
- Can be combined with other techniques

## Quasi-Monte Carlo (QMC)

### Principle
Replace pseudorandom numbers with deterministic low-discrepancy sequences (also called quasi-random sequences) that fill the space more uniformly. These sequences minimize gaps and clustering in the sampled points.

### Common Sequences
- **Sobol sequences**: Most popular in finance, handle high dimensions well
- **Halton sequences**: Simple to generate, suitable for moderate dimensions
- **Niederreiter sequences**: Theoretically optimal for certain dimensions

### Convergence
- Standard MC: Error ∝ O(1/√N)
- QMC: Error ∝ O((log N)^d / N), where d is the dimension
- For moderate dimensions (d < 30), QMC convergence is significantly faster

### Randomized QMC
Add a random shift or scramble to the low-discrepancy sequence to enable error estimation via independent replications. This combines the fast convergence of QMC with the statistical framework of MC.

### Effectiveness
- 10-1000x speedup over standard MC for pricing multi-asset options
- Performance degrades in very high dimensions (d > 50)
- Sobol sequences with Brownian bridge construction are the industry standard
- Essential technique for basket options, CDOs, and other multi-factor instruments

## Combining Techniques

### Optimal Combinations
- **Antithetic + Control**: Always beneficial, each addresses different variance sources
- **Antithetic + Stratified**: Complementary, antithetic handles symmetry while stratified handles coverage
- **Importance Sampling + Control Variates**: Powerful for OTM options, control corrects residual variance after drift shift
- **QMC + Antithetic**: QMC provides uniform coverage, antithetic exploits symmetry

### Practical Guidelines
1. Always use antithetic variates — they are free
2. Add control variates when an analytical approximation exists
3. Use importance sampling for deep OTM/ITM options or rare event simulation
4. Use QMC for multi-asset options where dimension is manageable
5. Benchmark variance reduction against standard MC to validate improvement
