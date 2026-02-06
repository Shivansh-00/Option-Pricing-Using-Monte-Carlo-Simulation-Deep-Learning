# System Architecture

The platform is structured as a full-stack, modular system with four core planes:

1. **Data Plane**: ingestion → feature engineering → normalization → sliding windows.
2. **Quant Plane**: Black-Scholes, Monte Carlo (GBM/Heston), Greeks, variance reduction.
3. **AI Plane**: ML models for volatility/regimes, DL residual models and neural MC.
4. **Experience Plane**: FastAPI backend, dashboard UI, RAG explainability.

The architecture is designed for easy experimentation and production deployment.
