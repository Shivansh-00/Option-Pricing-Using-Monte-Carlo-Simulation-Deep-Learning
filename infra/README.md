# OptiQuant — Production Infrastructure

## Directory Structure

```
infra/
├── terraform/          # AWS Infrastructure as Code
│   ├── main.tf         # Root module — provider, backend, modules
│   ├── variables.tf    # Input variables
│   ├── outputs.tf      # Exported values
│   ├── vpc.tf          # VPC, subnets, NAT, IGW
│   ├── eks.tf          # EKS cluster & node groups
│   ├── rds.tf          # RDS PostgreSQL
│   ├── elasticache.tf  # Redis cluster
│   ├── s3.tf           # Model storage + CloudFront CDN
│   ├── alb.tf          # Application Load Balancer
│   ├── iam.tf          # IAM roles & policies
│   ├── security.tf     # Security groups
│   └── terraform.tfvars.example
├── k8s/                # Kubernetes manifests
│   ├── namespace.yaml
│   ├── backend/
│   ├── frontend/
│   ├── redis/
│   ├── monitoring/
│   ├── ingress/
│   └── autoscaling/
├── docker/             # Production-optimized Dockerfiles
│   ├── backend.prod.Dockerfile
│   └── frontend.prod.Dockerfile
├── monitoring/         # Prometheus & Grafana configs
│   ├── prometheus.yml
│   ├── alerting-rules.yml
│   └── grafana-dashboard.json
├── ci-cd/              # GitHub Actions workflows
│   └── .github/workflows/
├── scripts/            # Deployment automation
│   ├── deploy.sh
│   ├── rollback.sh
│   └── health-check.sh
└── docker-compose.prod.yml
```

## Quick Start

```bash
# 1. Local production stack
docker-compose -f infra/docker-compose.prod.yml up --build

# 2. Deploy to AWS
cd infra/terraform
terraform init
terraform plan -var-file=terraform.tfvars
terraform apply -var-file=terraform.tfvars

# 3. Deploy to Kubernetes
kubectl apply -f infra/k8s/namespace.yaml
kubectl apply -Rf infra/k8s/

# 4. CI/CD (automatic on push)
# Copy infra/ci-cd/.github/ to repo root .github/
```
