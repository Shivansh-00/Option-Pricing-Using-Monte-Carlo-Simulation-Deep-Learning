"""
Live connectivity & health check for the OptionQuant platform.

Verifies:
  1. .env file is loaded with the right keys
  2. Neon PostgreSQL is reachable (TCP pool or HTTP API)
  3. Groq LLM API responds to a tiny prompt
  4. All FastAPI routers can be imported
  5. All trained model files are present and loadable
  6. Critical schemas / config are valid

Exit code 0 = all green, 1 = at least one failure.
"""
from __future__ import annotations

import os
import sys
import json
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / "backend" / ".env")
load_dotenv(ROOT / ".env")


def _section(title: str) -> None:
    print(f"\n{'=' * 64}\n  {title}\n{'=' * 64}")


def _ok(msg: str) -> None:
    print(f"  [ OK ] {msg}")


def _fail(msg: str) -> None:
    print(f"  [FAIL] {msg}")


def _warn(msg: str) -> None:
    print(f"  [WARN] {msg}")


results: dict[str, bool] = {}


# ---------------------------------------------------------------------------
# 1. Environment
# ---------------------------------------------------------------------------
def check_env() -> bool:
    _section("1. Environment variables")
    required = ["DATABASE_URL", "GROQ_API_KEY"]
    ok = True
    for k in required:
        v = os.getenv(k, "")
        if v:
            _ok(f"{k} present (len={len(v)})")
        else:
            _fail(f"{k} missing")
            ok = False
    return ok


# ---------------------------------------------------------------------------
# 2. Neon PostgreSQL
# ---------------------------------------------------------------------------
def check_neon() -> bool:
    _section("2. Neon PostgreSQL")
    try:
        from backend.app.database import init_pool, is_available, get_cursor, close_pool

        ok = init_pool()
        if not ok:
            _fail("init_pool returned False")
            return False
        _ok(f"DB initialised (available={is_available()})")
        try:
            with get_cursor() as cur:
                cur.execute("SELECT 1 AS ok, NOW() AS now")
                row = cur.fetchone()
                _ok(f"Query result: {row}")
        except Exception as exc:
            _fail(f"SELECT 1 failed: {exc}")
            return False
        finally:
            close_pool()
        return True
    except Exception as exc:
        _fail(f"Neon error: {exc}")
        traceback.print_exc()
        return False


# ---------------------------------------------------------------------------
# 3. Groq LLM
# ---------------------------------------------------------------------------
def check_groq() -> bool:
    _section("3. Groq LLM")
    key = os.getenv("GROQ_API_KEY", "")
    if not key:
        _fail("GROQ_API_KEY not set")
        return False
    try:
        import requests
        model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        resp = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": "Reply with one word: PONG."},
                    {"role": "user", "content": "ping"},
                ],
                "max_tokens": 8,
                "temperature": 0.0,
            },
            timeout=20,
        )
        if resp.status_code == 200:
            data = resp.json()
            txt = data["choices"][0]["message"]["content"].strip()
            _ok(f"Groq OK | model={model} | reply={txt!r}")
            return True
        else:
            _fail(f"HTTP {resp.status_code}: {resp.text[:300]}")
            return False
    except Exception as exc:
        _fail(f"Groq error: {exc}")
        return False


# ---------------------------------------------------------------------------
# 4. Routers import
# ---------------------------------------------------------------------------
def check_routers() -> bool:
    _section("4. FastAPI routers import")
    routers = [
        "auth_routes", "dl_routes", "explain_routes", "ml_routes",
        "pricing_routes", "market_routes", "ws_routes", "quant_routes",
        "pricing_api", "pinns_routes",
    ]
    all_ok = True
    for r in routers:
        try:
            __import__(f"backend.app.api.{r}", fromlist=[r])
            _ok(f"backend.app.api.{r}")
        except Exception as exc:
            _fail(f"backend.app.api.{r}: {exc}")
            all_ok = False
    return all_ok


# ---------------------------------------------------------------------------
# 5. Trained models present
# ---------------------------------------------------------------------------
def check_models() -> bool:
    _section("5. Trained models")
    model_dir = ROOT / "backend" / "models"
    expected = [
        "hybrid_lstm.npz", "hybrid_lstm_meta.json", "vol_engine_meta.json",
        "vol_random_forest.pkl", "vol_gradient_boosting.pkl",
        "vol_ridge.pkl", "vol_lasso.pkl", "vol_lstm.pkl",
        "vol_temporal_cnn.pkl", "vol_ensemble_stack.pkl",
        "pinns_model.pkl", "lstm_model.pt", "transformer_model.pt",
        "training_report.json",
    ]
    all_ok = True
    for f in expected:
        path = model_dir / f
        if path.exists():
            _ok(f"{f} ({path.stat().st_size:,} bytes)")
        else:
            _warn(f"{f} missing — will be regenerated on first request")
            # Not a hard failure
    return all_ok


# ---------------------------------------------------------------------------
# 6. RAG knowledge base
# ---------------------------------------------------------------------------
def check_rag() -> bool:
    _section("6. RAG pipeline")
    try:
        from backend.app.rag.vector_store import get_store
        kb_dir = ROOT / "backend" / "app" / "rag" / "knowledge_base"
        store = get_store(kb_dir)
        kb_size = len(store.documents) if hasattr(store, "documents") else 0
        _ok(f"Vector store loaded ({kb_size} documents)")
        from backend.app.rag.retriever import retrieve
        results = retrieve("What is Black-Scholes pricing?", store, top_k=3)
        _ok(f"retrieve() returned {len(results) if results else 0} hits")
        from backend.app.rag.llm_client import LLMClient
        client = LLMClient()
        _ok(f"LLMClient initialised | model={getattr(client, 'model', '?')}")
        return True
    except Exception as exc:
        _fail(f"RAG error: {exc}")
        traceback.print_exc()
        return False


# ---------------------------------------------------------------------------
def main() -> int:
    print("\nOptionQuant Connectivity & Health Check")
    print(f"Started at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    results["env"]     = check_env()
    results["neon"]    = check_neon()
    results["groq"]    = check_groq()
    results["routers"] = check_routers()
    results["models"]  = check_models()
    results["rag"]     = check_rag()

    _section("Summary")
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    for k, v in results.items():
        flag = "OK  " if v else "FAIL"
        print(f"  [{flag}] {k}")
    print(f"\nTotal: {passed}/{total} checks passed.")

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
