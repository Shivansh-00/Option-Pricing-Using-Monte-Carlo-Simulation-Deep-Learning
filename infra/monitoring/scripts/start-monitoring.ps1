param(
    [ValidateSet('up','down','restart','status','logs','urls')]
    [string]$Command = 'up'
)

$ErrorActionPreference = 'Stop'
Set-Location (Resolve-Path "$PSScriptRoot\..\..\..")

function Show-Urls {
    Write-Host ''
    Write-Host 'OptiQuant Monitoring Stack'
    Write-Host '  Grafana:    http://localhost:3000   (admin / optiquant2024)'
    Write-Host '  Prometheus: http://localhost:9090'
    Write-Host '  Backend:    http://localhost:8000'
    Write-Host '  Metrics:    http://localhost:8000/metrics'
    Write-Host ''
}

switch ($Command) {
    'up' {
        docker compose -f docker-compose.monitoring.yml up -d
        Show-Urls
        docker compose -f docker-compose.monitoring.yml ps
    }
    'down' {
        docker compose -f docker-compose.monitoring.yml down
    }
    'restart' {
        docker compose -f docker-compose.monitoring.yml restart
        Show-Urls
    }
    'status' {
        docker compose -f docker-compose.monitoring.yml ps
        Show-Urls
    }
    'logs' {
        docker compose -f docker-compose.monitoring.yml logs -f --tail=100
    }
    'urls' {
        Show-Urls
    }
}
