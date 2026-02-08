from __future__ import annotations

import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

BASE_URL = "http://localhost:8000"

PAYLOAD = {
    "spot": 100,
    "strike": 100,
    "maturity": 1,
    "rate": 0.02,
    "volatility": 0.2,
    "option_type": "call",
    "steps": 252,
    "paths": 20000,
}


def post(path: str, payload: dict) -> int:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{BASE_URL}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=20) as resp:
        resp.read()
        return resp.status


def run_batch(path: str, payload: dict, count: int = 20, workers: int = 5) -> None:
    start = time.time()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(lambda _: post(path, payload), range(count)))
    duration = time.time() - start
    ok = sum(1 for r in results if r == 200)
    print(f"{path} -> {ok}/{count} ok in {duration:.2f}s")


def main() -> None:
    run_batch("/api/v1/pricing/bs", PAYLOAD)
    run_batch("/api/v1/pricing/mc", PAYLOAD)
    run_batch("/api/v1/pricing/greeks", PAYLOAD)
    run_batch("/api/v1/ml/iv-predict", {
        "spot": 100,
        "rate": 0.02,
        "maturity": 1,
        "realized_vol": 0.2,
        "vix": 0.25,
        "skew": 0.1,
    })
    run_batch("/api/v1/ai/explain", {"question": "Explain Black-Scholes", "context": {}})


if __name__ == "__main__":
    main()
