#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════
#  OptiQuant — Monitoring Stack Launcher
#  Start/stop Grafana + Prometheus + Node Exporter
# ══════════════════════════════════════════════════════════════
set -euo pipefail
cd "$(dirname "$0")/../../.."

COMPOSE_FILE="docker-compose.monitoring.yml"

usage() {
  cat <<EOF
Usage: $(basename "$0") [command]

Commands:
  up        Start the full monitoring stack (default)
  down      Stop the monitoring stack
  restart   Restart all services
  status    Show service status
  logs      Tail logs from all services
  urls      Show access URLs
EOF
}

show_urls() {
  echo ""
  echo "╔══════════════════════════════════════════════╗"
  echo "║        OptiQuant Monitoring Stack            ║"
  echo "╠══════════════════════════════════════════════╣"
  echo "║  Grafana:    http://localhost:3000           ║"
  echo "║             (admin / optiquant2024)          ║"
  echo "║  Prometheus: http://localhost:9090           ║"
  echo "║  Backend:    http://localhost:8000           ║"
  echo "║  Metrics:    http://localhost:8000/metrics   ║"
  echo "╚══════════════════════════════════════════════╝"
  echo ""
}

case "${1:-up}" in
  up)
    echo "Starting OptiQuant monitoring stack..."
    docker compose -f "$COMPOSE_FILE" up -d
    show_urls
    echo "Waiting for services to become healthy..."
    sleep 5
    docker compose -f "$COMPOSE_FILE" ps
    ;;
  down)
    echo "Stopping monitoring stack..."
    docker compose -f "$COMPOSE_FILE" down
    ;;
  restart)
    echo "Restarting monitoring stack..."
    docker compose -f "$COMPOSE_FILE" restart
    show_urls
    ;;
  status)
    docker compose -f "$COMPOSE_FILE" ps
    show_urls
    ;;
  logs)
    docker compose -f "$COMPOSE_FILE" logs -f --tail=100
    ;;
  urls)
    show_urls
    ;;
  *)
    usage
    exit 1
    ;;
esac
