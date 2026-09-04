# ══════════════════════════════════════════════════════════════════════════════
#  Dockerfile — Maintenance Prédictive API v3.1.0
#  PFE ISG Bizerte / Novation City — 2025-2026
#  Compatible : Docker local · Render · Raspberry Pi 4 (ARM64)
# ══════════════════════════════════════════════════════════════════════════════
#
#  Build local :
#    docker build -t maintenance-predictive .
#
#  Run local (copier .env.example → .env et remplir les valeurs) :
#    docker run -p 8000:8000 --env-file .env maintenance-predictive
#
#  Run avec variables d'environnement explicites :
#    docker run -p 8000:8000 \
#               -e API_KEYS=<votre_cle_secrete> \
#               -e MARIADB_HOST=192.168.120.58 \
#               -e MARIADB_USER=app_user \
#               -e MARIADB_PASSWORD=<votre_mot_de_passe> \
#               maintenance-predictive
#
#  Build ARM64 (Raspberry Pi 4) :
#    docker buildx build --platform linux/arm64 -t maintenance-predictive:arm64 .
# ══════════════════════════════════════════════════════════════════════════════

FROM python:3.11-slim-bookworm

LABEL maintainer="Mohamed Yassine Balti — ISG Bizerte / Novation City"
LABEL version="3.1.0"
LABEL description="API FastAPI — Détection anomalies roulements IFM (6 modèles ML)"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    PORT=8000 \
    API_HOST=0.0.0.0 \
    DEMO_MODE=false \
    # Connexion MariaDB — valeurs par défaut NON root, à surcharger via --env-file .env
    MARIADB_HOST=192.168.120.58 \
    MARIADB_PORT=3306 \
    MARIADB_USER=app_user \
    MARIADB_DATABASE=ai_cp
    # MARIADB_PASSWORD et API_KEYS ne sont PAS définis ici — injectés uniquement via .env

WORKDIR /app

# Créer un utilisateur non-root pour exécuter l'application
RUN groupadd --gid 1001 appgroup && \
    useradd --uid 1001 --gid appgroup --shell /bin/bash --create-home appuser

RUN apt-get update && apt-get upgrade -y && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir scipy pyod

# Code source
COPY api_unified_pythagore.py .
COPY core.py .
COPY routers/ ./routers/
COPY auth.py .
COPY user_auth.py .
COPY rate_limiter.py .
COPY signal_processing.py .
COPY alert_manager.py .
COPY reporting_module.py .
COPY train_rul_model.py .
COPY gateway_ifm_simulator.py .
COPY realtime_mariadb.py .
COPY config.py .
COPY generate_dataset_from_sql.py .
COPY train_model_v3_unsupervised.py .
# Servie par GET /pipeline (api_unified_pythagore.py) -- oubliée ici jusqu'à
# présent, ce qui rendait le bouton "Pipeline Upload" du dashboard mort
# (404) une fois déployé en Docker (fonctionnait seulement en exécution
# locale hors conteneur, où le fichier est présent à côté du script).
COPY pipeline_upload.html .
COPY login.html .
COPY register.html .
COPY forgot-password.html .
COPY reset-password.html .
COPY admin-users.html .
COPY kpi_history.html .
COPY tasks_history.html .
COPY theme-toggle.js .
COPY theme-toggle.css .

# Modèles ML pré-entraînés (inclus dans l'image pour déploiement cloud)
COPY models/ ./models/

RUN mkdir -p /app/data /app/logs /app/reports && \
    chown -R appuser:appgroup /app

# Basculer vers l'utilisateur non-root
USER appuser

# Port exposé (Render injecte automatiquement $PORT)
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8000}/health || exit 1

CMD ["sh", "-c", "python -u api_unified_pythagore.py"]
