# American Options and Early Exercise

## Overview
American options can be exercised at any time before or at expiration, unlike European options which can only be exercised at maturity. This early exercise feature makes American options more valuable (or equally valuable) compared to their European counterparts, but also significantly more challenging to price.

## Early Exercise Premium
The value of an American option equals the European option value plus the early exercise premium:

V_American = V_European + EEP

The early exercise premium (EEP) is always non-negative. For calls on non-dividend-paying stocks, EEP = 0 (early exercise is never optimal), so American and European call prices are identical.

## When is Early Exercise Optimal?

### American Puts
Early exercise of puts can be optimal when:
- The option is deep in-the-money (intrinsic value exceeds time value)
- The interest rate is high (opportunity cost of waiting is significant)
- The stock price is very low (the put is nearly at its maximum value of K)

The optimal exercise boundary S*(t) divides the stock price space: exercise if S < S*(t), continue holding if S > S*(t).

### American Calls on Dividend-Paying Stocks
Early exercise may be optimal just before an ex-dividend date:
- The stock drops by the dividend amount on the ex-date
- If the dividend is large enough, exercising just before captures the dividend
- Decision rule: Exercise if dividend > K × [1 - e^(-r(T-t_ex))]

### American Calls on Non-Dividend Stocks
Never exercise early because:
- The call always has positive time value
- Holding the call provides downside protection (limited loss to premium)
- The present value of the strike is less than the strike itself

## Pricing Methods

### Binomial Tree (Cox-Ross-Rubinstein)
The binomial model is the standard method for American option pricing:
1. Build a recombining tree of stock prices with up factor u = e^(σ√Δt) and down factor d = 1/u
2. At each node, the risk-neutral probability is p = (e^(rΔt) - d) / (u - d)
3. Work backward from expiration, at each node computing:
   - Continuation value: e^(-rΔt) × [p × V_up + (1-p) × V_down]
   - Exercise value: max(K - S, 0) for puts, max(S - K, 0) for calls
   - Node value: max(continuation, exercise)

The tree converges as the number of steps increases, typically requiring 200-1000 steps for accurate pricing.

### Trinomial Tree
Adds a middle node (no price change) to the binomial tree, providing faster convergence and more flexibility in matching the volatility term structure.

### Finite Difference Methods
Solve the Black-Scholes PDE numerically on a grid:
- **Explicit FD**: Fast but requires small time steps for stability (CFL condition)
- **Implicit FD**: Unconditionally stable, solves a system of equations at each step
- **Crank-Nicolson**: Second-order accurate, averages explicit and implicit schemes

The American constraint is imposed as: V(S,t) ≥ max(K-S, 0) at every grid point (for puts).

### Longstaff-Schwartz Monte Carlo (Least Squares Monte Carlo)
The standard Monte Carlo method for American options:

1. **Forward pass**: Simulate N asset price paths forward in time
2. **Backward pass**: At each exercise date (working backward):
   a. Identify in-the-money paths
   b. Regress discounted future payoffs on polynomial basis functions of the current stock price
   c. The regression provides the expected continuation value
   d. Compare intrinsic value to continuation value to determine optimal exercise

### Basis Functions
Common choices for regression basis functions:
- Powers of stock price: 1, S, S², S³
- Laguerre polynomials (optimal for certain payoff structures)
- Hermite polynomials
- Chebyshev polynomials

Using 3-5 basis functions is typically sufficient for single-asset options.

### Bermudan Options
A special case exercisable only on specific dates (e.g., monthly). Priced like American options but with the exercise check only at the specified dates. More common in interest rate and exotic option markets.

## American Option Greeks

### Delta
American option delta differs from European due to the exercise boundary:
- Deep ITM American put: Delta approaches -1 (approaches intrinsic value behavior)
- Near the exercise boundary: Delta can change rapidly
- Computed via finite differences on the pricing grid

### Gamma
Higher gamma near the exercise boundary where the option transitions from holding to exercising. This creates hedging challenges.

### Early Exercise Boundary
The exercise boundary S*(t) separates the exercise region from the continuation region:
- For American puts: S*(t) increases toward K as t → T (the boundary rises to the strike at expiration)
- For American calls (with dividends): Step-down at ex-dividend dates
- The boundary can be computed iteratively using the binomial tree or PDE methods

## Comparison with European Options

| Feature | European | American |
|---------|----------|----------|
| Exercise | At expiry only | Any time |
| Pricing | Analytical (B-S) | Numerical required |
| Call on non-div stock | Same price | Same price |
| Put premium | Lower | Higher (EEP > 0) |
| Hedging | Simpler | Complex (exercise boundary) |
| Greeks | Smooth | Can be discontinuous |
