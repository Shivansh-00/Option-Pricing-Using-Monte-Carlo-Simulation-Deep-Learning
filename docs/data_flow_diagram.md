# Data Flow Diagram

1. Raw market data lands in `backend/data/raw/`.
2. `data_loader.py` ingests CSV files.
3. `feature_engineering.py` computes returns, vol, skew proxies.
4. `preprocessing.py` normalizes and creates sliding windows.
5. Outputs feed ML/DL training pipelines and pricing requests.
