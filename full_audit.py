"""Comprehensive audit of all API endpoints to find errors, timeouts, and issues."""
import requests
import time

BASE = 'http://127.0.0.1:8000'
results = []

def test(name, method, path, payload=None, timeout=30, expected=200):
    url = f'{BASE}{path}'
    try:
        t0 = time.time()
        if method == 'GET':
            r = requests.get(url, timeout=timeout)
        else:
            r = requests.post(url, json=payload, timeout=timeout)
        elapsed = round((time.time() - t0) * 1000)
        ok = r.status_code == expected
        status = 'OK' if ok else 'FAIL'
        snippet = r.text[:120].replace('\n', ' ')
        results.append((status, name, r.status_code, elapsed, snippet))
        print(f"[{status}] {name}: {r.status_code} {elapsed}ms | {snippet}")
    except requests.exceptions.Timeout:
        results.append(('TIMEOUT', name, 0, timeout*1000, 'Request timed out'))
        print(f"[TIMEOUT] {name}: timed out after {timeout}s")
    except Exception as e:
        results.append(('ERROR', name, 0, 0, str(e)[:80]))
        print(f"[ERROR] {name}: {e}")

print("=" * 70)
print("FULL API AUDIT")
print("=" * 70)

# === HEALTH ===
test('Health', 'GET', '/health')
test('Docs', 'GET', '/docs', expected=200)
test('Metrics', 'GET', '/metrics')

# === PRICING ===
pricing_payload = {'S': 150, 'K': 155, 'T': 0.25, 'r': 0.05, 'sigma': 0.2, 'option_type': 'call', 'model': 'black_scholes'}
test('Pricing/BS', 'POST', '/api/v1/pricing/price', pricing_payload, 30)

mc_payload = dict(pricing_payload)
mc_payload['model'] = 'monte_carlo'
test('Pricing/MC', 'POST', '/api/v1/pricing/price', mc_payload, 60)

heston_payload = dict(pricing_payload)
heston_payload['model'] = 'heston'
test('Pricing/Heston', 'POST', '/api/v1/pricing/price', heston_payload, 60)

greeks_payload = {'S': 150, 'K': 155, 'T': 0.25, 'r': 0.05, 'sigma': 0.2, 'option_type': 'call'}
test('Greeks', 'POST', '/api/v1/pricing/greeks', greeks_payload, 30)

# === DL / TRAINING ===
dl_payload = {'S': 150.0, 'K': 155.0, 'T': 0.25, 'r': 0.05, 'sigma': 0.2, 'option_type': 'call'}
test('DL/Predict', 'POST', '/api/v1/dl/predict', dl_payload, 30)
test('DL/Status', 'GET', '/api/v1/dl/status')

# === ML ===
ml_payload = {'S': 150.0, 'K': 155.0, 'T': 0.25, 'r': 0.05, 'sigma': 0.2, 'option_type': 'call'}
test('ML/Predict', 'POST', '/api/v1/ml/predict', ml_payload, 30)
test('ML/Status', 'GET', '/api/v1/ml/status')

# === RISK ===
var_payload = {'S': 150, 'K': 155, 'T': 0.25, 'r': 0.05, 'sigma': 0.2, 'option_type': 'call', 'confidence': 0.95, 'horizon_days': 1, 'portfolio_value': 100000}
test('Risk/VaR', 'POST', '/api/v1/risk/var', var_payload, 30)

# === VOL ENGINE ===
test('Vol/Models', 'GET', '/api/v1/vol/models')
vol_payload = {'ticker': 'AAPL', 'models': ['sabr', 'heston'], 'epochs': 10}
test('Vol/Train', 'POST', '/api/v1/vol/train', vol_payload, 120)

# === PINNS ===
pinns_payload = {'S': 150.0, 'K': 155.0, 'T': 0.25, 'r': 0.05, 'sigma': 0.2, 'option_type': 'call'}
test('PINNs/Price', 'POST', '/api/v1/pinns/price', pinns_payload, 60)
test('PINNs/Status', 'GET', '/api/v1/pinns/status')

# === MARKET DATA ===
test('Market/Quote', 'GET', '/api/v1/market/quote/AAPL', timeout=20)
test('Market/Chain', 'GET', '/api/v1/market/chain/AAPL', timeout=30)

# === AI/RAG ===
rag_payload = {'question': 'What is Black-Scholes model?'}
test('AI/Ask', 'POST', '/api/v1/ai/ask', rag_payload, 30)

# === GRAFANA PROXY ===
test('Grafana/Health', 'GET', '/grafana/api/health', timeout=10)

# === BENCHMARKS / PERFORMANCE ===
test('Benchmark/Pricing', 'GET', '/api/v1/pricing/benchmark', timeout=30)
test('Performance/Summary', 'GET', '/api/v1/performance/summary', timeout=30)

# Summary
print("\n" + "=" * 70)
fails = [r for r in results if r[0] != 'OK']
print(f"TOTAL: {len(results)} | OK: {len(results)-len(fails)} | FAILED/TIMEOUT: {len(fails)}")
for r in fails:
    print(f"  [{r[0]}] {r[1]}: status={r[2]} | {r[4]}")
