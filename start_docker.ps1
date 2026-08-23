# start_docker.ps1
# ═════════════════════════════════════════════════════════════════════════
# Lance le projet (docker compose up) en mettant d'abord à jour PUBLIC_HOST
# dans .env avec l'IP LAN réelle de la machine.
#
# Pourquoi : PUBLIC_HOST sert à construire les liens dans les emails d'alerte
# (voir alert_manager.py) -- "localhost" ne veut rien dire pour quelqu'un qui
# lit l'email depuis son téléphone. Cette IP change à chaque fois que la
# machine change de réseau (autre site, autre box) ; ce script évite d'avoir
# à la mettre à jour à la main dans .env avant chaque démarrage.
#
# Détection : se fait ICI, sur l'hôte Windows (Get-NetIPConfiguration voit
# les vraies interfaces réseau -- Wi-Fi/Ethernet). Impossible de faire cette
# détection depuis l'intérieur du conteneur Docker : il est isolé sur un
# réseau bridge interne (172.18.x.x) et ne voit jamais l'IP LAN réelle de
# l'hôte, quelle que soit la machine.
#
# Usage : .\start_docker.ps1
# ═════════════════════════════════════════════════════════════════════════

$envPath = Join-Path $PSScriptRoot ".env"

if (-not (Test-Path $envPath)) {
    Write-Host "ERREUR : .env introuvable ($envPath) -- copiez .env.example en .env d'abord." -ForegroundColor Red
    exit 1
}

# Adaptateur avec passerelle par défaut active = interface réellement connectée
# au réseau local (exclut les adaptateurs virtuels Docker/WSL/VPN sans gateway)
$adapter = Get-NetIPConfiguration | Where-Object {
    $_.IPv4DefaultGateway -and $_.NetAdapter.Status -eq 'Up'
} | Select-Object -First 1

if (-not $adapter) {
    Write-Host "ATTENTION : aucune interface reseau active detectee -- PUBLIC_HOST inchange." -ForegroundColor Yellow
} else {
    $ip = $adapter.IPv4Address.IPAddress
    $content = Get-Content $envPath -Raw

    if ($content -match '(?m)^PUBLIC_HOST=') {
        $newContent = $content -replace '(?m)^PUBLIC_HOST=.*$', "PUBLIC_HOST=$ip"
    } else {
        $newContent = $content.TrimEnd() + "`nPUBLIC_HOST=$ip`n"
    }

    if ($newContent -ne $content) {
        Set-Content -Path $envPath -Value $newContent -Encoding utf8 -NoNewline
        Write-Host "PUBLIC_HOST mis a jour -> $ip ($($adapter.InterfaceAlias))" -ForegroundColor Green
    } else {
        Write-Host "PUBLIC_HOST deja a jour -> $ip" -ForegroundColor Green
    }
}

Write-Host "`nLancement de docker compose..." -ForegroundColor Cyan
docker compose up -d
