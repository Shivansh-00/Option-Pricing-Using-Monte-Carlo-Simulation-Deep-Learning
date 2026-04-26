"""End-to-end smoke test of the running OptionQuant API."""
import json
import sys
import urllib.error
import urllib.request

BASE = "http://127.0.0.1:8000"

# Login first
login_data = json.dumps({"username": "smoke", "password": "SmokeTest123!"}).encode()
req = urllib.request.Request(
    f"{BASE}/api/v1/auth/login",
    data=login_data,
    headers={"Content-Type": "application/json"},
    method="POST",
)
with urllib.request.urlopen(req, timeout=10) as r:
    tok = json.loads(r.read())["access_token"]

H = {"Authorization": f"Bearer {tok}", "Content-Type": "application/json"}

tests = [
    ("GET",  "/health",                              None),
    ("GET",  "/ready",                               None),
    ("POST", "/api/v1/pricing/bs",                   {"spot":100,"strike":100,"maturity":1,"rate":0.05,"volatility":0.2,"option_type":"call"}),
    ("POST", "/api/v1/pricing/mc",                   {"spot":100,"strike":100,"maturity":1,"rate":0.05,"volatility":0.2,"option_type":"call","paths":5000,"steps":100}),
    ("POST", "/api/v1/pricing/greeks",               {"spot":100,"strike":100,"maturity":1,"rate":0.05,"volatility":0.2,"option_type":"call"}),
    ("POST", "/api/v1/ai/explain",                   {"question":"What is delta hedging?"}),
    ("GET",  "/api/v1/ai/rag/health",                None),
    ("GET",  "/api/v1/ai/rag/stats",                 None),
    ("GET",  "/api/v1/ml/vol/status",                None),
    ("GET",  "/api/v1/dl/status",                    None),
    ("GET",  "/api/v1/pricing/pinns/status",         None),
    ("GET",  "/api/v1/market/health",                None),
]

ok = 0
fail = 0
print(f"\n{'METHOD':6}  {'PATH':45}  STATUS  PREVIEW")
print("-" * 100)
for method, path, body in tests:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(f"{BASE}{path}", data=data, headers=H, method=method)
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            code = r.getcode()
            payload = r.read(180).decode("utf-8", "replace").replace("\n", " ")
            print(f"{method:6}  {path:45}  {code:3}     {payload[:80]}")
            if code < 400:
                ok += 1
            else:
                fail += 1
    except urllib.error.HTTPError as e:
        body_txt = e.read(160).decode("utf-8", "replace")
        print(f"{method:6}  {path:45}  {e.code:3}     {body_txt[:80]}")
        fail += 1
    except Exception as e:
        print(f"{method:6}  {path:45}  ERR     {e}")
        fail += 1

print(f"\nTotal: {ok} OK, {fail} FAIL out of {len(tests)}")
sys.exit(0 if fail == 0 else 1)
