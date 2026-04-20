# OptiQuant Monitoring

This stack provisions Grafana, Prometheus, and Windows host metrics for local development on this workspace.

## Services

- Grafana through the main app: http://localhost:8000/grafana/
- Grafana direct container port: http://localhost:3000
- Prometheus: http://localhost:9090
- Backend metrics: http://localhost:8000/metrics
- Windows exporter: scraped by Prometheus on port 9182 inside Docker networking

## Start

PowerShell:

```powershell
.\infra\monitoring\scripts\start-monitoring.ps1 up
```

Bash:

```bash
./infra/monitoring/scripts/start-monitoring.sh up
```

Direct Docker Compose:

```powershell
docker compose -f docker-compose.monitoring.yml up -d
```

## Dashboards

Provisioned dashboards are loaded from `infra/monitoring/dashboards`.

- `optiquant-overview.json`
- `optiquant-pricing.json`
- `optiquant-models.json`
- `optiquant-rag.json`
- `optiquant-websocket.json`

## Credentials

- Grafana user: `admin`
- Grafana password: `optiquant2024`

## Notes

- This compose file is tuned for a Windows host by using `windows-exporter` instead of Linux `node-exporter`.
- Prometheus scrapes the locally running backend through `host.docker.internal:8000` so Grafana can be embedded in the main app at `/grafana/`.
- Protected API metrics appear after authenticated endpoints are called.
- Prometheus datasource is provisioned with explicit UID `prometheus` so dashboard imports remain stable.
