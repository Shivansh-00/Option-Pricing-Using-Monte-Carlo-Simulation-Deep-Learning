"""Comprehensive API audit for current OptiQuant routes."""

import random
import time
import requests

BASE = "http://127.0.0.1:8000"


def authenticate(session: requests.Session) -> str:
    login = session.post(
        f"{BASE}/api/v1/auth/login",
        json={"username": "testuser", "password": "Test1234!"},
        timeout=20,
    )
    if login.status_code == 200:
        return login.json()["access_token"]

    signup = session.post(
        f"{BASE}/api/v1/auth/signup",
        json={"username": "testuser", "email": "t@test.com", "password": "Test1234!"},
        timeout=20,
    )
    if signup.status_code in (200, 201):
        return signup.json()["access_token"]

    raise RuntimeError(
        f"Auth failed: login={login.status_code}, signup={signup.status_code}"
    )


def run_test(
    session: requests.Session,
    name: str,
    method: str,
    path: str,
    payload=None,
    timeout: int = 30,
):
    t0 = time.time()
    try:
        if method == "GET":
            response = session.get(f"{BASE}{path}", timeout=timeout)
        else:
            response = session.post(f"{BASE}{path}", json=payload, timeout=timeout)
        elapsed = round((time.time() - t0) * 1000)
        ok = response.status_code == 200
        print(f"[{'OK' if ok else 'FAIL'}] {name}: {response.status_code} {elapsed}ms")
        return ok, response.status_code, elapsed, response.text[:180] if not ok else ""
    except requests.exceptions.Timeout:
        print(f"[TIMEOUT] {name}: timeout after {timeout}s")
        return False, 0, timeout * 1000, "timeout"
    except Exception as exc:
        print(f"[ERROR] {name}: {exc}")
        return False, 0, 0, str(exc)


if __name__ == "__main__":
    session = requests.Session()
    token = authenticate(session)
    session.headers.update(
        {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    )

    bs = {"spot": 100, "strike": 100, "maturity": 1, "rate": 0.05, "volatility": 0.2}
    returns = [round(random.gauss(0.0005, 0.015), 6) for _ in range(100)]

    tests = [
        ("Health", "GET", "/health", None, 15),
        ("Readiness", "GET", "/ready", None, 15),
        ("Metrics", "GET", "/metrics", None, 15),
        ("Auth Me", "GET", "/api/v1/auth/me", None, 20),
        ("Pricing BS", "POST", "/api/v1/pricing/bs", bs, 20),
        ("Pricing MC", "POST", "/api/v1/pricing/mc", {**bs, "paths": 5000, "steps": 100}, 30),
        ("Pricing Greeks", "POST", "/api/v1/pricing/greeks", bs, 20),
        (
            "ML IV",
            "POST",
            "/api/v1/ml/iv-predict",
            {"spot": 100, "rate": 0.05, "maturity": 1, "realized_vol": 0.2, "vix": 20.0, "skew": -0.5},
            30,
        ),
        ("DL Forecast", "POST", "/api/v1/dl/forecast", bs, 40),
        ("Market Quote", "GET", "/api/v1/market/quote/AAPL", None, 20),
        ("AI Explain", "POST", "/api/v1/ai/explain", {"question": "How does RL hedging work?"}, 45),
        ("RAG Health", "GET", "/api/v1/ai/rag/health", None, 20),
        ("Jump Diffusion", "POST", "/api/v1/quant/jump-diffusion/price", {**bs, "option_type": "call"}, 40),
        ("Regime Calibrate", "POST", "/api/v1/quant/jump-diffusion/calibrate", {"returns": returns}, 40),
        (
            "GPU MC",
            "POST",
            "/api/v1/quant/gpu-mc/price",
            {**bs, "option_type": "call", "model": "gbm", "variance_reduction": "antithetic", "n_paths": 10000, "n_steps": 100},
            40,
        ),
        ("PINNs Predict", "POST", "/api/v1/quant/pinns/predict", {**bs, "option_type": "call"}, 30),
        (
            "Portfolio Risk",
            "POST",
            "/api/v1/quant/portfolio/risk-report",
            {"positions": [{"symbol": "AAPL", "quantity": 10, "option_type": "call", **bs}]},
            30,
        ),
    ]

    print("=" * 68)
    print("FULL API AUDIT (CURRENT ROUTES)")
    print("=" * 68)

    failures = []
    for name, method, path, payload, timeout in tests:
        ok, status, _elapsed, detail = run_test(session, name, method, path, payload, timeout)
        if not ok:
            failures.append((name, status, detail))

    print("\n" + "=" * 68)
    print(f"TOTAL: {len(tests)} | OK: {len(tests) - len(failures)} | FAILED/TIMEOUT: {len(failures)}")
    for name, status, detail in failures:
        print(f"  [FAIL] {name}: status={status} | {detail}")
