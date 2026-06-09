# ══════════════════════════════════════════════════════════════════════════════
#  Dockerfile — Maintenance Prédictive API
#  PFE ISG Bizerte / Novation City — 2025-2026
# ══════════════════════════════════════════════════════════════════════════════
#
#  Construction :
#    docker build -t maintenance-predictive .
#
#  Lancement (PC standard) :
#    docker run -p 8000:8000 -v $(pwd)/models:/app/models maintenance-predictive
#
#  Lancement avec gateway IFM :
#    docker run -p 8000:8000 \
#               -v $(pwd)/models:/app/models \
#               -v $(pwd)/data:/app/data \
#               -e IFM_GATEWAY_HOST=192.168.1.50 \
#               maintenance-predictive
#
#  ── Vers le déploiement Edge (Raspberry Pi 4) ─────────────────────────────
#  Build multi-architecture (cross-compile depuis un PC x86) :
#    docker buildx build --platform linux/arm64 -t maintenance-predictive:arm64 .
#    docker save maintenance-predictive:arm64 | ssh pi@192.168.1.X docker load
#
#  Prérequis Edge : Raspberry Pi 4 (4 Go RAM minimum), Docker installé,
#  même réseau local que la gateway IFM (192.168.1.x).
#
#  ⚠️  LIMITE ACTUELLE : Les modèles .pkl (pyod ECOD) ne sont pas exportables
#  en ONNX nativement. Cette image utilise les modèles sklearn/pyod originaux
#  via Python — fonctionnel sur ARM64 mais non optimisé pour embarqué.
#  Une vraie optimisation Edge nécessiterait d'exporter IF et OCSVM via
#  sklearn-onnx et de remplacer ECOD par un modèle ONNX-compatible.
# ══════════════════════════════════════════════════════════════════════════════

# Image de base — slim pour réduire la taille (compatible ARM64 via buildx)
FROM python:3.11-slim

# Métadonnées
LABEL maintainer="ISG Bizerte — PFE Maintenance Prédictive"
LABEL version="3.1.0"
LABEL description="API FastAPI — Détection anomalies roulements industriels"

# Variables d'environnement
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    # Adresse gateway IFM (override via -e IFM_GATEWAY_HOST=...)
    IFM_GATEWAY_HOST=192.168.1.50 \
    IFM_GATEWAY_PORT=80 \
    IFM_PORTS="1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19" \
    # API
    API_PORT=8000 \
    API_HOST=0.0.0.0

# Répertoire de travail
WORKDIR /app

# Dépendances système (minimales)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copie des requirements en premier (cache Docker optimisé)
COPY requirements.txt .

# Installation des dépendances Python
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir scipy pyod

# Copie du code source (sans venv, sans sql, sans données volumineuses)
COPY api_unified_pythagore.py .
COPY alert_manager.py .
COPY gateway_ifm_simulator.py .
COPY realtime_ifm_direct.py .

# Copie des modèles pré-entraînés
# Le dossier models/ DOIT exister avec les .pkl avant le build
COPY models/ ./models/

# Création des dossiers de données persistants
RUN mkdir -p /app/data /app/logs

# Port exposé
EXPOSE 8000

# Health check Docker (vérifie que l'API répond)
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Point d'entrée — API FastAPI via Uvicorn
CMD ["python", "-u", "api_unified_pythagore.py"]
