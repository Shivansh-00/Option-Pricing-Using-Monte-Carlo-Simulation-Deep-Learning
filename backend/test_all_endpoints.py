"""Comprehensive endpoint tester for all backend API routes."""
import requests
import json
import sys
import time
import random

BASE = "http://127.0.0.1:8000"

def get_token():
    r = requests.post(f"{BASE}/api/v1/auth/login", json={"username": "testuser", "password": "Test1234!"})
    if r.status_code == 200:
        return r.json()["access_token"]
    # signup returns 201
    r = requests.post(f"{BASE}/api/v1/auth/signup", json={"username": "testuser", "email": "t@test.com", "password": "Test1234!"})
    if r.status_code in (200, 201):
        return r.json()["access_token"]
    print(f"AUTH FAILED: {r.status_code} {r.text}", flush=True)
    sys.exit(1)

token = get_token()
headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
results = []

def test(name, method, path, body=None, timeout=120):
    try:
        t0 = time.time()
        if method == "GET":
            r = requests.get(f"{BASE}{path}", headers=headers, timeout=timeout)
        else:
            r = requests.post(f"{BASE}{path}", headers=headers, json=body, timeout=timeout)
        elapsed = round((time.time() - t0) * 1000)
        ok = r.status_code == 200
        detail = ""
        if not ok:
            try:
                detail = r.text[:300]
            except:
                detail = str(r.status_code)
        results.append({"name": name, "ok": ok, "status": r.status_code, "ms": elapsed, "detail": detail})
        mark = "OK" if ok else "FAIL"
        line = f"  [{mark}] {name} -> {r.status_code} ({elapsed}ms)"
        if detail:
            line += f" | {detail[:120]}"
        print(line, flush=True)
    except requests.exceptions.Timeout:
        results.append({"name": name, "ok": False, "status": 0, "ms": 0, "detail": "TIMEOUT"})
        print(f"  [TIMEOUT] {name} -> timeout after {timeout}s", flush=True)
    except Exception as e:
        results.append({"name": name, "ok": False, "status": 0, "ms": 0, "detail": str(e)[:200]})
        print(f"  [ERR] {name} -> {str(e)[:100]}", flush=True)

print("=" * 70, flush=True)
print("COMPREHENSIVE API ENDPOINT TEST", flush=True)
print("=" * 70, flush=True)

# 1. Health
print("\n--- Health & Readiness ---", flush=True)
test("Health", "GET", "/health")
test("Readiness", "GET", "/ready")

# 2. Auth
print("\n--- Authentication ---", flush=True)
test("Auth Me", "GET", "/api/v1/auth/me")

# 3. Pricing
print("\n--- Option Pricing ---", flush=True)
test("Black-Scholes", "POST", "/api/v1/pricing/bs", {"spot": 100, "strike": 100, "maturity": 1, "rate": 0.05, "volatility": 0.2})
test("Monte Carlo", "POST", "/api/v1/pricing/mc", {"spot": 100, "strike": 100, "maturity": 1, "rate": 0.05, "volatility": 0.2, "paths": 10000, "steps": 100})
test("MC Detailed", "POST", "/api/v1/pricing/mc/detailed", {"spot": 100, "strike": 100, "maturity": 1, "rate": 0.05, "volatility": 0.2, "paths": 5000, "steps": 100, "method": "standard", "return_paths": True})
test("MC Compare", "POST", "/api/v1/pricing/mc/compare", {"spot": 100, "strike": 100, "maturity": 1, "rate": 0.05, "volatility": 0.2})
test("Greeks", "POST", "/api/v1/pricing/greeks", {"spot": 100, "strike": 100, "maturity": 1, "rate": 0.05, "volatility": 0.2})

# 4. ML Volatility
print("\n--- ML Volatility ---", flush=True)
test("IV Predict", "POST", "/api/v1/ml/iv-predict", {"spot": 100, "rate": 0.05, "maturity": 1, "realized_vol": 0.2, "vix": 20.0, "skew": -0.5})
test("Vol Train", "POST", "/api/v1/ml/vol/train", {"n_days": 500})
test("Vol Status", "GET", "/api/v1/ml/vol/status")

# 5. Deep Learning
print("\n--- Deep Learning ---", flush=True)
test("DL Forecast", "POST", "/api/v1/dl/forecast", {"spot": 100, "strike": 100, "maturity": 1, "rate": 0.05, "volatility": 0.2})
test("DL Train", "POST", "/api/v1/dl/train", {})
test("Market Sentiment", "POST", "/api/v1/dl/market-sentiment", {"text": "Markets rally as tech stocks surge", "mode": "transformer"})

# 6. Quant Intelligence
print("\n--- PINNs ---", flush=True)
test("PINNs Train", "POST", "/api/v1/quant/pinns/train", {"n_samples": 500, "epochs": 50}, timeout=300)
test("PINNs Predict", "POST", "/api/v1/quant/pinns/predict", {"spot": 100, "strike": 100, "maturity": 1, "rate": 0.05, "volatility": 0.2, "option_type": "call"})
test("PINNs Greeks", "POST", "/api/v1/quant/pinns/greeks", {"spot": 100, "strike": 100, "maturity": 1, "rate": 0.05, "volatility": 0.2})

print("\n--- RL Hedging ---", flush=True)
test("Hedging Train", "POST", "/api/v1/quant/hedging/train", {"spot": 100, "strike": 100, "maturity": 0.25, "rate": 0.05, "volatility": 0.2, "agent_type": "dqn", "episodes": 100}, timeout=300)
test("Hedging Backtest", "POST", "/api/v1/quant/hedging/backtest", {"agent_type": "dqn", "n_scenarios": 50}, timeout=300)
test("Hedging Suggest", "POST", "/api/v1/quant/hedging/suggest", {"spot": 100, "strike": 100, "maturity": 0.25, "rate": 0.05, "volatility": 0.2, "regime": 0, "current_hedge_ratio": 0.5, "current_pnl": 0})

print("\n--- Vol Surface ---", flush=True)
test("Vol Surface Train", "POST", "/api/v1/quant/vol-surface/train", {"n_samples": 200, "epochs": 30}, timeout=300)
test("Vol Surface Predict", "POST", "/api/v1/quant/vol-surface/predict", {"spot": 100, "base_vol": 0.2, "regime": 0})

print("\n--- Jump Diffusion ---", flush=True)
test("Jump Diffusion Price", "POST", "/api/v1/quant/jump-diffusion/price", {"spot": 100, "strike": 100, "maturity": 1, "rate": 0.05, "volatility": 0.2, "option_type": "call"})
# Need 50+ returns for calibrate
returns_data = [round(random.gauss(0.0005, 0.015), 6) for _ in range(100)]
test("Regime Calibrate", "POST", "/api/v1/quant/jump-diffusion/calibrate", {"returns": returns_data})
test("Scenario Analysis", "POST", "/api/v1/quant/jump-diffusion/scenario", {"spot": 100, "strike": 100, "maturity": 1, "rate": 0.05})

print("\n--- Arbitrage ---", flush=True)
test("Arbitrage Scan", "POST", "/api/v1/quant/arbitrage/scan", {"spot": 100, "n_options": 20, "regime": 0})
test("Put-Call Parity", "POST", "/api/v1/quant/arbitrage/put-call-parity", {"spot": 100, "strike": 100, "maturity": 1, "rate": 0.05, "call_price": 12.0, "put_price": 7.0})

print("\n--- Uncertainty ---", flush=True)
test("Uncertainty Quantify", "POST", "/api/v1/quant/uncertainty/quantify", {"spot": 100, "strike": 100, "maturity": 1, "rate": 0.05, "volatility": 0.2, "n_samples": 200})
test("Uncertainty Train", "POST", "/api/v1/quant/uncertainty/train", {"n_samples": 500, "epochs": 20, "method": "bayesian"}, timeout=300)

print("\n--- GPU Monte Carlo ---", flush=True)
test("GPU MC Price", "POST", "/api/v1/quant/gpu-mc/price", {"spot": 100, "strike": 100, "maturity": 1, "rate": 0.05, "volatility": 0.2, "option_type": "call", "model": "gbm", "variance_reduction": "antithetic", "n_paths": 50000, "n_steps": 100})
test("GPU MC Benchmark", "POST", "/api/v1/quant/gpu-mc/benchmark", {"spot": 100, "strike": 100, "maturity": 1, "rate": 0.05, "volatility": 0.2}, timeout=600)

print("\n--- Portfolio Risk ---", flush=True)
test("Portfolio Risk Report", "POST", "/api/v1/quant/portfolio/risk-report", {
    "positions": [
        {"spot": 100, "strike": 100, "maturity": 1, "rate": 0.05, "volatility": 0.2, "option_type": "call", "quantity": 10, "premium_paid": 10.45},
        {"spot": 100, "strike": 110, "maturity": 0.5, "rate": 0.05, "volatility": 0.25, "option_type": "put", "quantity": 5, "premium_paid": 12.0}
    ]
})
test("Portfolio Stress Test", "POST", "/api/v1/quant/portfolio/stress-test", {
    "positions": [
        {"spot": 100, "strike": 100, "maturity": 1, "rate": 0.05, "volatility": 0.2, "option_type": "call", "quantity": 10, "premium_paid": 10.45}
    ]
})

print("\n--- Explainer ---", flush=True)
test("Explain Decision", "POST", "/api/v1/quant/explain/decision", {
    "decision_type": "price",
    "context": {"spot": 100, "strike": 100, "maturity": 1, "volatility": 0.2, "rate": 0.05, "option_type": "call"}
})

print("\n--- Ecosystem Status ---", flush=True)
test("Quant Status", "GET", "/api/v1/quant/status")

# 7. Market Intelligence
print("\n--- Market Intelligence ---", flush=True)
test("Market Quote", "GET", "/api/v1/market/quote/AAPL")
test("Market Chain", "GET", "/api/v1/market/chain/AAPL")
test("Market Snapshot", "GET", "/api/v1/market/snapshot/AAPL")
test("Market Health", "GET", "/api/v1/market/health")
test("Mispricing Detect", "POST", "/api/v1/market/mispricing/detect", {"spot": 150, "strike": 150, "maturity": 0.5, "rate": 0.05, "volatility": 0.25, "option_type": "call", "market_price": 12.0, "bid": 11.5, "ask": 12.5})
test("Regime Detect", "POST", "/api/v1/market/regime/detect", {"returns": [round(random.gauss(0.0005, 0.015), 6) for _ in range(50)], "vix": 20.0})
test("Confidence", "POST", "/api/v1/market/risk/confidence", {"spot": 100, "strike": 100, "maturity": 1, "rate": 0.05, "volatility": 0.2})
test("VaR", "POST", "/api/v1/market/risk/var", {"spot": 100, "strike": 100, "maturity": 1, "rate": 0.05, "volatility": 0.2})
test("Reliability", "POST", "/api/v1/market/risk/reliability", {"spot": 100, "strike": 100, "maturity": 1, "rate": 0.05, "volatility": 0.2})
test("SHAP Explain", "POST", "/api/v1/market/explain/shap", {"spot": 100, "strike": 100, "maturity": 1, "rate": 0.05, "volatility": 0.2})
test("Benchmark", "POST", "/api/v1/market/benchmark", {"spot": 100, "strike": 100, "maturity": 1, "rate": 0.05, "volatility": 0.2})

# 8. RAG / AI Explain
print("\n--- AI Explainer ---", flush=True)
test("AI Explain", "POST", "/api/v1/ai/explain", {"question": "What is the Black-Scholes model?"})
test("RAG Health", "GET", "/api/v1/ai/rag/health")

# Summary
print("\n" + "=" * 70, flush=True)
passed = sum(1 for r in results if r["ok"])
failed = sum(1 for r in results if not r["ok"])
print(f"RESULTS: {passed} passed, {failed} failed, {len(results)} total", flush=True)
print("=" * 70, flush=True)
if failed:
    print("\nFAILED ENDPOINTS:", flush=True)
    for r in results:
        if not r["ok"]:
            print(f"  [{r['status']}] {r['name']}: {r['detail'][:200]}", flush=True)
