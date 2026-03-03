#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════
#  OptiQuant — Production Health Check Script
#  Usage: ./health-check.sh [base-url]
# ══════════════════════════════════════════════════════════════
set -euo pipefail

BASE_URL="${1:-http://localhost:8000}"
PASS=0
FAIL=0

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

check() {
    local name="$1"
    local url="$2"
    local expected="${3:-200}"
    
    STATUS=$(curl -sf -o /dev/null -w '%{http_code}' "$url" 2>/dev/null || echo "000")
    LATENCY=$(curl -sf -o /dev/null -w '%{time_total}' "$url" 2>/dev/null || echo "0")
    
    if [ "$STATUS" = "$expected" ]; then
        echo -e "${GREEN}[PASS]${NC} ${name} — ${STATUS} (${LATENCY}s)"
        PASS=$((PASS + 1))
    else
        echo -e "${RED}[FAIL]${NC} ${name} — Expected ${expected}, got ${STATUS}"
        FAIL=$((FAIL + 1))
    fi
}

echo "═══════════════════════════════════════════════════════════"
echo " OptiQuant Health Check — $(date -u +'%Y-%m-%dT%H:%M:%SZ')"
echo " Target: ${BASE_URL}"
echo "═══════════════════════════════════════════════════════════"
echo ""

# ── Core Endpoints ───────────────────────────────────────────
echo "── Core ──────────────────────────────────────────────────"
check "Health"        "${BASE_URL}/health"
check "Readiness"     "${BASE_URL}/ready"
check "OpenAPI Docs"  "${BASE_URL}/docs"

# ── Auth ─────────────────────────────────────────────────────
echo ""
echo "── Authentication ──────────────────────────────────────"
check "Auth Register (OPTIONS)" "${BASE_URL}/api/v1/auth/register" "405"

# ── Pricing Endpoints ────────────────────────────────────────
echo ""
echo "── Pricing Engine ──────────────────────────────────────"
PRICE_RESP=$(curl -sf -X POST "${BASE_URL}/api/v1/pricing/black-scholes" \
    -H "Content-Type: application/json" \
    -d '{"spot_price":100,"strike_price":105,"risk_free_rate":0.05,"volatility":0.2,"time_to_expiry":1.0,"option_type":"call"}' \
    2>/dev/null || echo "FAIL")

if echo "$PRICE_RESP" | grep -q "price"; then
    echo -e "${GREEN}[PASS]${NC} Black-Scholes pricing"
    PASS=$((PASS + 1))
else
    echo -e "${RED}[FAIL]${NC} Black-Scholes pricing — no price in response"
    FAIL=$((FAIL + 1))
fi

# ── Metrics ──────────────────────────────────────────────────
echo ""
echo "── Observability ───────────────────────────────────────"
check "Prometheus Metrics" "${BASE_URL}/metrics"

# ── Summary ──────────────────────────────────────────────────
TOTAL=$((PASS + FAIL))
echo ""
echo "═══════════════════════════════════════════════════════════"
if [ "$FAIL" -eq 0 ]; then
    echo -e "${GREEN} All ${TOTAL} checks passed${NC}"
else
    echo -e "${RED} ${FAIL}/${TOTAL} checks failed${NC}"
fi
echo "═══════════════════════════════════════════════════════════"

exit "$FAIL"
