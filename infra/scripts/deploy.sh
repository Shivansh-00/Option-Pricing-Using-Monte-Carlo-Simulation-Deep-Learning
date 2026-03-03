#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════
#  OptiQuant — Production Deployment Script
#  Usage: ./deploy.sh [staging|production] [image-tag]
# ══════════════════════════════════════════════════════════════
set -euo pipefail

# ── Configuration ────────────────────────────────────────────
ENVIRONMENT="${1:-staging}"
IMAGE_TAG="${2:-latest}"
AWS_REGION="${AWS_REGION:-us-east-1}"
AWS_ACCOUNT_ID="${AWS_ACCOUNT_ID:-}"
ECR_REGISTRY="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
NAMESPACE="optiquant"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log() { echo -e "${BLUE}[$(date +'%H:%M:%S')]${NC} $1"; }
success() { echo -e "${GREEN}[✓]${NC} $1"; }
warn() { echo -e "${YELLOW}[!]${NC} $1"; }
error() { echo -e "${RED}[✗]${NC} $1"; exit 1; }

# ── Pre-flight Checks ───────────────────────────────────────
log "Pre-flight checks..."

command -v aws >/dev/null 2>&1 || error "AWS CLI not found"
command -v kubectl >/dev/null 2>&1 || error "kubectl not found"
command -v docker >/dev/null 2>&1 || error "Docker not found"

if [ -z "$AWS_ACCOUNT_ID" ]; then
    AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
    ECR_REGISTRY="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
fi

success "Pre-flight checks passed"
log "Environment: ${ENVIRONMENT}"
log "Image Tag: ${IMAGE_TAG}"
log "ECR Registry: ${ECR_REGISTRY}"

# ── Build & Push Images ─────────────────────────────────────
log "Building Docker images..."

aws ecr get-login-password --region "$AWS_REGION" | \
    docker login --username AWS --password-stdin "$ECR_REGISTRY"

# Backend
log "Building backend image..."
docker build -t "${ECR_REGISTRY}/optiquant/backend:${IMAGE_TAG}" \
    -f infra/docker/backend.prod.Dockerfile .
docker push "${ECR_REGISTRY}/optiquant/backend:${IMAGE_TAG}"
success "Backend image pushed"

# Frontend
log "Building frontend image..."
docker build -t "${ECR_REGISTRY}/optiquant/frontend:${IMAGE_TAG}" \
    -f infra/docker/frontend.prod.Dockerfile .
docker push "${ECR_REGISTRY}/optiquant/frontend:${IMAGE_TAG}"
success "Frontend image pushed"

# ── Update Kubeconfig ────────────────────────────────────────
EKS_CLUSTER="optiquant-${ENVIRONMENT}-eks"
log "Updating kubeconfig for ${EKS_CLUSTER}..."
aws eks update-kubeconfig --name "$EKS_CLUSTER" --region "$AWS_REGION"
success "Kubeconfig updated"

# ── Deploy to Kubernetes ─────────────────────────────────────
log "Deploying to Kubernetes..."

# Apply namespace and configs
kubectl apply -f infra/k8s/namespace.yaml
kubectl apply -f infra/k8s/backend/configmap.yaml
kubectl apply -f infra/k8s/backend/network-policy.yaml

# Update images
kubectl set image deployment/optiquant-backend \
    backend="${ECR_REGISTRY}/optiquant/backend:${IMAGE_TAG}" \
    -n "$NAMESPACE" --record

kubectl set image deployment/optiquant-frontend \
    frontend="${ECR_REGISTRY}/optiquant/frontend:${IMAGE_TAG}" \
    -n "$NAMESPACE" --record

# Apply autoscaling and ingress
kubectl apply -Rf infra/k8s/autoscaling/
kubectl apply -Rf infra/k8s/ingress/

# ── Wait for Rollout ─────────────────────────────────────────
log "Waiting for backend rollout..."
kubectl rollout status deployment/optiquant-backend \
    -n "$NAMESPACE" --timeout=600s
success "Backend rollout complete"

log "Waiting for frontend rollout..."
kubectl rollout status deployment/optiquant-frontend \
    -n "$NAMESPACE" --timeout=300s
success "Frontend rollout complete"

# ── Post-Deploy Verification ────────────────────────────────
log "Running health checks..."
sleep 10

HEALTH=$(kubectl exec -n "$NAMESPACE" \
    $(kubectl get pod -n "$NAMESPACE" -l app=optiquant-backend -o jsonpath='{.items[0].metadata.name}') \
    -- curl -sf http://localhost:8000/health 2>/dev/null || echo '{"status":"failed"}')

echo "$HEALTH" | python3 -m json.tool 2>/dev/null || echo "$HEALTH"

if echo "$HEALTH" | grep -q "healthy"; then
    success "Health check passed"
else
    warn "Health check returned unexpected status"
fi

# ── Summary ──────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════════"
echo -e "${GREEN} Deployment Complete${NC}"
echo "═══════════════════════════════════════════════════════════"
echo "  Environment: ${ENVIRONMENT}"
echo "  Image Tag:   ${IMAGE_TAG}"
echo "  Namespace:   ${NAMESPACE}"
echo "  Timestamp:   $(date -u +'%Y-%m-%dT%H:%M:%SZ')"
echo ""

# Show pod status
kubectl get pods -n "$NAMESPACE" -o wide
echo ""
kubectl get hpa -n "$NAMESPACE"
