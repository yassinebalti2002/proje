# ══════════════════════════════════════════════════════════════════════════════
#  Dockerfile — Maintenance Prédictive API v3.1.0
#  PFE ISG Bizerte / Novation City — 2025-2026
#  Compatible : Docker local · Render · Raspberry Pi 4 (ARM64)
# ══════════════════════════════════════════════════════════════════════════════
#
#  Build local :
#    docker build -t maintenance-predictive .
#
#  Run local :
#    docker run -p 8000:8000 maintenance-predictive
#
#  Run avec MariaDB local :
#    docker run -p 8000:8000 \
#               -e MARIADB_HOST=192.168.120.58 \
#               -e MARIADB_PASSWORD=xxx \
#               maintenance-predictive
#
#  Build ARM64 (Raspberry Pi 4) :
#    docker buildx build --platform linux/arm64 -t maintenance-predictive:arm64 .
# ══════════════════════════════════════════════════════════════════════════════

FROM python:3.11-slim

LABEL maintainer="Mohamed Yassine Balti — ISG Bizerte / Novation City"
LABEL version="3.1.0"
LABEL description="API FastAPI — Détection anomalies roulements IFM (6 modèles ML)"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    PORT=8000 \
    API_HOST=0.0.0.0 \
    DEMO_MODE=false \
    MARIADB_HOST=192.168.120.58 \
    MARIADB_PORT=3306 \
    MARIADB_USER=root \
    MARIADB_DATABASE=ai_cp

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir scipy pyod

# Code source
COPY api_unified_pythagore.py .
COPY signal_processing.py .
COPY alert_manager.py .
COPY reporting_module.py .
COPY train_rul_model.py .
COPY realtime_mariadb.py .
COPY realtime_ifm_direct.py .
COPY gateway_ifm_simulator.py .

# Modèles ML pré-entraînés (inclus dans l'image pour déploiement cloud)
COPY models/ ./models/

RUN mkdir -p /app/data /app/logs /app/reports

# Port exposé (Render injecte automatiquement $PORT)
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8000}/health || exit 1

CMD ["sh", "-c", "python -u api_unified_pythagore.py"]
