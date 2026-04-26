"""Quick API test covering all key endpoints with short timeouts."""
import requests
import time
import random

BASE = "http://127.0.0.1:8000"

# Authenticate
r = requests.post(f"{BASE}/api/v1/auth/login", json={"username": "testuser", "password": "Test1234!"})
if r.status_code == 200:
    token = r.json()["access_token"]
else:
    r = requests.post(f"{BASE}/api/v1/auth/signup", json={"username": "testuser", "email": "t@test.com", "password": "Test1234!"})
    token = r.json()["access_token"]

headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
passed = failed = timed_out = 0

def test(name, method, path, body=None, timeout=30):
    global passed, failed, timed_out
    try:
        t0 = time.time()
        if method == "GET":
            r = requests.get(f"{BASE}{path}", headers=headers, timeout=timeout)
        else:
            r = requests.post(f"{BASE}{path}", headers=headers, json=body, timeout=timeout)
        elapsed = round((time.time() - t0) * 1000)
        ok = r.status_code == 200
        if ok:
            passed += 1
        else:
            failed += 1
        detail = "" if ok else r.text[:150]
        mark = "OK" if ok else "FAIL"
        print(f"  [{mark}] {name} -> {r.status_code} ({elapsed}ms)" + (f"\n         {detail}" if detail else ""))
    except requests.exceptions.Timeout:
        timed_out += 1
        print(f"  [TIMEOUT] {name} -> timed out after {timeout}s")
    except Exception as e:
        failed += 1
        print(f"  [ERR] {name} -> {str(e)[:120]}")

print("=" * 65)
print("OPTIQUANT API TEST SUITE")
print("=" * 65)

print("\n--- Health ---")
test("Health",    "GET", "/health")
test("Readiness", "GET", "/ready")
test("Metrics",   "GET", "/metrics")
test("Docs",      "GET", "/docs")

print("\n--- Auth ---")
test("Auth Me", "GET", "/api/v1/auth/me")

print("\n--- Option Pricing ---")
BS = {"spot": 100, "strike": 100, "maturity": 1, "rate": 0.05, "volatility": 0.2}
test("Black-Scholes",  "POST", "/api/v1/pricing/bs", BS)
test("Monte Carlo",    "POST", "/api/v1/pricing/mc",       {**BS, "paths": 5000, "steps": 100})
test("MC Detailed",    "POST", "/api/v1/pricing/mc/detailed", {**BS, "paths": 1000, "steps": 50, "method": "antithetic", "return_paths": False})
test("MC Compare",     "POST", "/api/v1/pricing/mc/compare", BS)
test("Greeks",         "POST", "/api/v1/pricing/greeks",  BS)

print("\n--- ML Volatility ---")
test("IV Predict", "POST", "/api/v1/ml/iv-predict", {"spot": 100, "rate": 0.05, "maturity": 1, "realized_vol": 0.2, "vix": 20.0, "skew": -0.5})
test("Vol Status", "GET",  "/api/v1/ml/vol/status")

print("\n--- Deep Learning ---")
test("DL Forecast",       "POST", "/api/v1/dl/forecast", BS)
test("Market Sentiment",  "POST", "/api/v1/dl/market-sentiment", {"text": "Tech stocks surge on strong earnings", "mode": "transformer"})

print("\n--- Quant Intelligence ---")
test("Jump Diffusion Price",    "POST", "/api/v1/quant/jump-diffusion/price",   {**BS, "option_type": "call"})
test("Arbitrage Scan",          "POST", "/api/v1/quant/arbitrage/scan",          {"spot": 100, "n_options": 10, "regime": 0})
test("Put-Call Parity",         "POST", "/api/v1/quant/arbitrage/put-call-parity", {**BS, "call_price": 12.0, "put_price": 7.0})
test("Uncertainty Quantify",    "POST", "/api/v1/quant/uncertainty/quantify",   {**BS, "n_samples": 100})
test("GPU MC Price",            "POST", "/api/v1/quant/gpu-mc/price",           {**BS, "option_type": "call", "model": "gbm", "variance_reduction": "antithetic", "n_paths": 10000, "n_steps": 100})
returns = [round(random.gauss(0.0005, 0.015), 6) for _ in range(100)]
test("Regime Calibrate",        "POST", "/api/v1/quant/jump-diffusion/calibrate", {"returns": returns})
test("Scenario Analysis",       "POST", "/api/v1/quant/jump-diffusion/scenario",  BS)

print("\n--- Market Data ---")
test("Market Quote AAPL",       "GET",  "/api/v1/market/quote/AAPL")
test("Market Chain AAPL",       "GET",  "/api/v1/market/chain/AAPL")
test("Market Snapshot AAPL",    "GET",  "/api/v1/market/snapshot/AAPL")

print("\n--- Explainability ---")
test("SHAP Explain",            "POST", "/api/v1/market/explain/shap", BS)

print("\n--- AI / RAG ---")
test("AI Explain",              "POST", "/api/v1/ai/explain",       {"question": "What is delta hedging?"}, timeout=30)
test("RAG Health",              "GET",  "/api/v1/ai/rag/health")

print("\n--- PINNs (quick) ---")
test("PINNs Predict", "POST", "/api/v1/quant/pinns/predict", {**BS, "option_type": "call"})
test("PINNs Greeks",  "POST", "/api/v1/quant/pinns/greeks",  BS)

print("\n--- Vol Surface ---")
test("Vol Surface Predict", "POST", "/api/v1/quant/vol-surface/predict", {"spot": 100, "base_vol": 0.2, "regime": 0})

print("\n--- Portfolio Risk ---")
test("Portfolio Risk Report", "POST", "/api/v1/quant/portfolio/risk-report", {"positions": [{"symbol": "AAPL", "quantity": 10, "option_type": "call", **BS}]})

print()
print("=" * 65)
total = passed + failed + timed_out
print(f"RESULTS: {passed}/{total} passed | {failed} failed | {timed_out} timed out")
print("=" * 65)
