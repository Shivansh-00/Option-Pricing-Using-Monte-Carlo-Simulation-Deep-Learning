#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════
#  OptiQuant — Bootstrap Script
#  One-time setup for AWS infrastructure
#  Usage: ./bootstrap.sh
# ══════════════════════════════════════════════════════════════
set -euo pipefail

AWS_REGION="${AWS_REGION:-us-east-1}"
PROJECT="optiquant"

RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

log() { echo -e "${BLUE}[$(date +'%H:%M:%S')]${NC} $1"; }
success() { echo -e "${GREEN}[✓]${NC} $1"; }

echo "═══════════════════════════════════════════════════════════"
echo " OptiQuant Infrastructure Bootstrap"
echo "═══════════════════════════════════════════════════════════"
echo ""

# ── 1. Create Terraform State Bucket ────────────────────────
log "Creating Terraform state S3 bucket..."
aws s3api create-bucket \
    --bucket "${PROJECT}-terraform-state" \
    --region "$AWS_REGION" \
    2>/dev/null || true

aws s3api put-bucket-versioning \
    --bucket "${PROJECT}-terraform-state" \
    --versioning-configuration Status=Enabled

aws s3api put-bucket-encryption \
    --bucket "${PROJECT}-terraform-state" \
    --server-side-encryption-configuration \
    '{"Rules":[{"ApplyServerSideEncryptionByDefault":{"SSEAlgorithm":"aws:kms"}}]}'

aws s3api put-public-access-block \
    --bucket "${PROJECT}-terraform-state" \
    --public-access-block-configuration \
    "BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true"

success "Terraform state bucket created"

# ── 2. Create DynamoDB Lock Table ────────────────────────────
log "Creating Terraform lock table..."
aws dynamodb create-table \
    --table-name "${PROJECT}-terraform-lock" \
    --attribute-definitions AttributeName=LockID,AttributeType=S \
    --key-schema AttributeName=LockID,KeyType=HASH \
    --billing-mode PAY_PER_REQUEST \
    --region "$AWS_REGION" \
    2>/dev/null || true
success "DynamoDB lock table created"

# ── 3. Create ECR Repositories ──────────────────────────────
log "Creating ECR repositories..."
for repo in backend frontend; do
    aws ecr create-repository \
        --repository-name "${PROJECT}/${repo}" \
        --image-scanning-configuration scanOnPush=true \
        --image-tag-mutability IMMUTABLE \
        --region "$AWS_REGION" \
        2>/dev/null || true
    success "ECR repo: ${PROJECT}/${repo}"
done

# ── 4. Create KMS Key ───────────────────────────────────────
log "Creating KMS key for encryption..."
KMS_KEY=$(aws kms create-key \
    --description "OptiQuant master encryption key" \
    --key-usage ENCRYPT_DECRYPT \
    --origin AWS_KMS \
    --query 'KeyMetadata.KeyId' \
    --output text \
    --region "$AWS_REGION" \
    2>/dev/null || echo "exists")

if [ "$KMS_KEY" != "exists" ]; then
    aws kms create-alias \
        --alias-name "alias/${PROJECT}-master" \
        --target-key-id "$KMS_KEY" \
        --region "$AWS_REGION"
    success "KMS key created: ${KMS_KEY}"
else
    success "KMS key already exists"
fi

# ── 5. Initialize Terraform ─────────────────────────────────
log "Initializing Terraform..."
cd infra/terraform
terraform init
success "Terraform initialized"

echo ""
echo "═══════════════════════════════════════════════════════════"
echo -e "${GREEN} Bootstrap complete!${NC}"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "Next steps:"
echo "  1. Copy terraform.tfvars.example → terraform.tfvars"
echo "  2. Fill in your values (passwords, certs, etc.)"
echo "  3. Run: terraform plan -var-file=terraform.tfvars"
echo "  4. Run: terraform apply -var-file=terraform.tfvars"
echo ""
