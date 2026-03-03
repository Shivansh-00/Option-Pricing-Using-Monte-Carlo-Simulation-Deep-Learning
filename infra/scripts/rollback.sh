#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════
#  OptiQuant — Rollback Script
#  Usage: ./rollback.sh [revision-number]
# ══════════════════════════════════════════════════════════════
set -euo pipefail

REVISION="${1:-0}"  # 0 = previous revision
NAMESPACE="optiquant"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log() { echo -e "${BLUE}[$(date +'%H:%M:%S')]${NC} $1"; }
success() { echo -e "${GREEN}[✓]${NC} $1"; }
error() { echo -e "${RED}[✗]${NC} $1"; exit 1; }

# ── Current State ────────────────────────────────────────────
log "Current deployment state:"
kubectl get deployments -n "$NAMESPACE" -o wide
echo ""

log "Rollout history:"
kubectl rollout history deployment/optiquant-backend -n "$NAMESPACE"
echo ""

# ── Rollback ─────────────────────────────────────────────────
if [ "$REVISION" -eq 0 ]; then
    log "Rolling back to previous revision..."
    kubectl rollout undo deployment/optiquant-backend -n "$NAMESPACE"
    kubectl rollout undo deployment/optiquant-frontend -n "$NAMESPACE"
else
    log "Rolling back to revision ${REVISION}..."
    kubectl rollout undo deployment/optiquant-backend \
        -n "$NAMESPACE" --to-revision="$REVISION"
    kubectl rollout undo deployment/optiquant-frontend \
        -n "$NAMESPACE" --to-revision="$REVISION"
fi

# ── Wait ─────────────────────────────────────────────────────
log "Waiting for rollback to complete..."
kubectl rollout status deployment/optiquant-backend \
    -n "$NAMESPACE" --timeout=300s
kubectl rollout status deployment/optiquant-frontend \
    -n "$NAMESPACE" --timeout=120s

# ── Verify ───────────────────────────────────────────────────
sleep 5
log "Post-rollback status:"
kubectl get pods -n "$NAMESPACE" -o wide
echo ""

HEALTH=$(kubectl exec -n "$NAMESPACE" \
    $(kubectl get pod -n "$NAMESPACE" -l app=optiquant-backend -o jsonpath='{.items[0].metadata.name}') \
    -- curl -sf http://localhost:8000/health 2>/dev/null || echo '{"status":"failed"}')

if echo "$HEALTH" | grep -q "healthy"; then
    success "Rollback successful — system healthy"
else
    error "Rollback completed but health check failed!"
fi
