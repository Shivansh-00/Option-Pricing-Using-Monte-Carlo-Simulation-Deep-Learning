# Deep Learning for Option Pricing

## Overview
Deep learning has emerged as a powerful complement to traditional option pricing methods. Neural networks can learn complex, non-linear pricing functions directly from data, capturing market dynamics that analytical models miss. The hybrid approach combining classical models with deep learning residual corrections offers the best of both worlds.

## Why Deep Learning for Options?

### Limitations of Classical Models
- Black-Scholes assumes constant volatility and normal returns
- Stochastic volatility models require complex calibration
- Monte Carlo is computationally expensive for real-time pricing
- Model risk: choosing the wrong parametric model leads to systematic errors

### Advantages of Deep Learning
- **Universal Approximation**: Neural networks can approximate any continuous function to arbitrary accuracy
- **Data-Driven**: Learns pricing patterns directly from market data without parametric assumptions
- **Speed**: Once trained, inference is nearly instantaneous (microseconds vs seconds for Monte Carlo)
- **Adaptability**: Can incorporate any input features without re-deriving formulas

## Hybrid Residual Learning Approach

### Architecture
The hybrid model combines a classical pricing model with a neural network correction:

Price_hybrid = Price_BS + NN(features)

Where:
- Price_BS is the Black-Scholes price (or any analytical model)
- NN(features) is a neural network that learns the residual error (the difference between the analytical price and the true market price)

### Rationale
1. The classical model provides a strong baseline that captures the bulk of option pricing
2. The neural network only needs to learn the residual — a much easier task
3. This approach inherits the interpretability and theoretical grounding of the classical model
4. The residual captures market-specific effects: volatility smile, term structure, supply-demand imbalances

### Training Process
1. Collect market option prices alongside Black-Scholes prices computed from implied volatility
2. Compute residuals: residual = market_price - BS_price
3. Train the neural network to predict residuals from input features
4. Combined prediction: hybrid_price = BS_price + predicted_residual

## Neural Network Architectures

### Feedforward Networks (MLP)
- Simplest architecture for option pricing
- Input: spot, strike, maturity, rate, volatility, moneyness
- Hidden layers: 3-5 layers with 64-256 neurons each
- Activation: ReLU or GELU for hidden layers, linear for output
- Best for European option pricing where inputs are tabular

### LSTM (Long Short-Term Memory)
- Recurrent architecture that processes sequential data
- Captures temporal dependencies in price series and volatility dynamics
- Input: time series of prices, volumes, and features
- Useful for path-dependent options and volatility forecasting
- Can model regime changes through learned hidden states

### Transformer Models
- Attention-based architecture for capturing long-range dependencies
- Self-attention allows each time step to attend to all other steps
- Superior for capturing complex temporal patterns
- Multi-head attention captures different types of relationships simultaneously
- Position encoding preserves temporal ordering information

## Feature Engineering

### Raw Features
- Spot price, strike price, maturity, risk-free rate
- Historical volatility (multiple windows: 5d, 21d, 63d, 252d)
- Implied volatility from Black-Scholes inversion

### Derived Features
- **Moneyness**: S/K or log(S/K) — normalizes across different price levels
- **Time-scaled Volatility**: σ√T — captures the interaction between volatility and time
- **Volatility Ratio**: IV/RV — indicates over/underpricing of options
- **Skew Proxy**: Difference in IV between OTM puts and ATM options
- **Term Structure Slope**: Difference in IV between short and long maturities

### Normalization
All features should be normalized (z-score or min-max) to ensure stable training. Financial data often has different scales (prices in hundreds, rates in decimals).

## Model Evaluation

### Pricing Accuracy Metrics
- **MAE** (Mean Absolute Error): Average absolute pricing error in dollar terms
- **RMSE** (Root Mean Squared Error): Penalizes large errors more heavily
- **MAPE** (Mean Absolute Percentage Error): Scale-independent accuracy measure
- **R²** (Coefficient of Determination): Proportion of variance explained

### Financial Metrics
- **Hedging P&L**: Test the model's prices for delta hedging performance
- **Implied Volatility Error**: Compare model-implied IV with market IV
- **Greeks Accuracy**: Compare model-derived Greeks with market benchmarks

### Robustness Checks
- Out-of-sample performance across different market regimes
- Stress testing under extreme market conditions
- Stability of predictions over time (concept drift monitoring)

## Production Deployment

### Model Serving
- Export trained models (PyTorch, TensorFlow) for inference
- Use ONNX format for cross-framework compatibility
- Batch inference for portfolio-level pricing
- Real-time inference for single-option pricing

### Model Monitoring
- Track prediction drift over time
- Monitor feature distributions for data drift
- Set alerts for unusually large residuals
- Regular retraining schedule (weekly or monthly)

### Risk Considerations
- Model uncertainty: Ensemble methods or Bayesian approaches
- Extrapolation risk: Neural networks may behave unpredictably outside training distribution
- Adversarial inputs: Extreme market conditions may fool the model
- Regulatory compliance: Need to explain model decisions (XAI)
