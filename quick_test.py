"""Quick test of market intelligence + AI endpoints."""
import requests, random, sys

BASE = "http://localhost:8000"
r = requests.post(f"{BASE}/api/v1/auth/login", json={"username": "testuser", "password": "Test1234!"})
if r.status_code != 200:
    r = requests.post(f"{BASE}/api/v1/auth/signup", json={"username": "testuser", "email": "t@test.com", "password": "Test1234!"})
token = r.json()["access_token"]
h = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

tests = [
    ("Market Quote", "GET", "/api/v1/market/quote/AAPL", None),
    ("Market Chain", "GET", "/api/v1/market/chain/AAPL", None),
    ("Market Snapshot", "GET", "/api/v1/market/snapshot/AAPL", None),
    ("Market Health", "GET", "/api/v1/market/health", None),
    ("Mispricing Detect", "POST", "/api/v1/market/mispricing/detect",
     {"spot": 150, "strike": 150, "maturity": 0.5, "rate": 0.05, "volatility": 0.25,
      "option_type": "call", "market_price": 12.0, "bid": 11.5, "ask": 12.5}),
    ("Regime Detect", "POST", "/api/v1/market/regime/detect",
     {"returns": [round(random.gauss(0.0005, 0.015), 6) for _ in range(50)], "vix": 20.0}),
    ("Confidence", "POST", "/api/v1/market/risk/confidence",
     {"spot": 100, "strike": 100, "maturity": 1, "rate": 0.05, "volatility": 0.2}),
    ("VaR", "POST", "/api/v1/market/risk/var",
     {"spot": 100, "strike": 100, "maturity": 1, "rate": 0.05, "volatility": 0.2}),
    ("Reliability", "POST", "/api/v1/market/risk/reliability",
     {"spot": 100, "strike": 100, "maturity": 1, "rate": 0.05, "volatility": 0.2}),
    ("SHAP Explain", "POST", "/api/v1/market/explain/shap",
     {"spot": 100, "strike": 100, "maturity": 1, "rate": 0.05, "volatility": 0.2}),
    ("Benchmark", "POST", "/api/v1/market/benchmark",
     {"spot": 100, "strike": 100, "maturity": 1, "rate": 0.05, "volatility": 0.2}),
    ("AI Explain", "POST", "/api/v1/ai/explain",
     {"question": "What is the Black-Scholes model?"}),
    ("RAG Health", "GET", "/api/v1/ai/rag/health", None),
]

fail = 0
for name, method, path, body in tests:
    try:
        if method == "GET":
            r = requests.get(f"{BASE}{path}", headers=h, timeout=60)
        else:
            r = requests.post(f"{BASE}{path}", headers=h, json=body, timeout=60)
        ok = r.status_code == 200
        detail = "" if ok else r.text[:200]
        mark = "OK" if ok else "FAIL"
        if not ok:
            fail += 1
        print(f"  [{mark}] {name} -> {r.status_code} {detail}", flush=True)
    except Exception as e:
        fail += 1
        print(f"  [ERR] {name} -> {e}", flush=True)

print(f"\n{len(tests)-fail}/{len(tests)} passed", flush=True)
