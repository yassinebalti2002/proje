"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  API Unifiée — Maintenance Prédictive Roulements                            ║
║  PFE — Surveillance de 20 capteurs IFM — Novation City                      ║
║                                                                              ║
║  Port : 8000                                                                 ║
║  Docs : http://localhost:8000/docs                                           ║
║                                                                              ║
║  Endpoints principaux :                                                      ║
║    POST /v1/predict              → Détection anomalie (IF+LOF+OCSVM+ECOD)   ║
║    POST /v1/predict-rul          → Estimation RUL (Remaining Useful Life)   ║
║    GET  /v1/health-score/{id}    → Score santé moteur                        ║
║    GET  /v1/history/{id}         → Historique prédictions par capteur  [NEW] ║
║    GET  /v1/alert-level/{id}     → Niveau d'alerte actuel capteur      [NEW] ║
║    GET  /health                  → Health check                              ║
║    GET  /metrics                 → Métriques modèle V3 (AUC=0.9475 CV)          ║
║    GET  /sensors                 → Liste capteurs depuis full_data           ║
║    GET  /anomalies               → Anomalies filtrées                        ║
╚══════════════════════════════════════════════════════════════════════════════╝

Point d'entrée : composition de l'app FastAPI à partir des routers (voir
routers/) et de l'état partagé (voir core.py) — ce fichier ne contient
lui-même aucune logique métier.
"""

import os
import logging
from pathlib import Path

import core

try:
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse
    from fastapi.middleware.cors import CORSMiddleware
    from contextlib import asynccontextmanager
    import uvicorn
    FASTAPI_OK = True
except ImportError:
    FASTAPI_OK = False
    print("FastAPI non installé. Lance : pip install fastapi uvicorn pydantic")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [API] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

API_VERSION = core.API_VERSION

if FASTAPI_OK:

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Remplace @app.on_event('startup') — recommandé depuis FastAPI 0.93."""
        log.info(f"Démarrage API Unifiée V{API_VERSION}")
        core.load_all_models()
        core.load_history_from_disk()
        if core.USER_AUTH_OK:
            try:
                core.init_users_table()
            except Exception as e:
                log.warning(f"init_users_table() a échoué : {e}")
        yield
        # Shutdown : sauvegarder une dernière fois avant arrêt
        core._last_persist = 0.0
        core.save_history_to_disk()
        log.info("API arrêtée proprement — historique sauvegardé")

    class UTF8JSONResponse(JSONResponse):
        """Force le charset=utf-8 dans Content-Type — sans ça, PowerShell 5.1
        (Invoke-RestMethod) décode le JSON en ISO-8859-1 par défaut et corrompt
        les caractères accentués (ex: "ÉLEVÉ" -> "ÃLEVÃ")."""
        media_type = "application/json; charset=utf-8"

    app = FastAPI(
        title="Maintenance Prédictive — API Unifiée",
        description=(
            "Système complet de surveillance de 20 capteurs IFM — Novation City.\n\n"
            "**Modèle IA** : Ensemble non supervisé IF+ECOD+HBOS+COPOD (SoftVote), LOF/OCSVM calculés à titre diagnostic uniquement\n\n"
            "**Données** : Capteurs IFM VVB001 → MySQL ai_cp (1 648 886 mesures, nov 2025 – mar 2026)\n\n"
            "**PFE ISG Bizerte** — Détection d'anomalies + Estimation RUL roulements\n\n"
            "**Auth** : En-tête `X-API-Key` obligatoire sur tous les endpoints /v1/*"
        ),
        version=API_VERSION,
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
        default_response_class=UTF8JSONResponse,
    )

    # CORS restreint aux origines définies via CORS_ORIGINS (prod) ou * (dev)
    _cors_origins_env = os.getenv("CORS_ORIGINS", "")
    _cors_origins = (
        [o.strip() for o in _cors_origins_env.split(",") if o.strip()]
        if _cors_origins_env
        else ["*"]
    )
    if "*" in _cors_origins:
        log.warning("CORS ouvert à toutes les origines — acceptable en dev, à restreindre en prod via CORS_ORIGINS")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    # Comptes utilisateurs (register/login humain) — additif, n'affecte pas
    # l'auth par clé API (require_api_key/require_admin_key) utilisée ailleurs
    if core.USER_AUTH_OK:
        app.include_router(core.user_auth_router, prefix="/v1/auth", tags=["Authentification utilisateurs"])

    from routers import system, predict, spectral, data, alerts, reporting, pipeline, pages
    app.include_router(system.router)
    app.include_router(predict.router)
    app.include_router(spectral.router)
    app.include_router(data.router)
    app.include_router(alerts.router)
    app.include_router(reporting.router)
    app.include_router(pipeline.router)
    app.include_router(pages.router)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys, io
    # Force UTF-8 pour eviter UnicodeEncodeError sur Windows (cp1252)
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

    print("\n" + "=" * 80)
    print("  MAINTENANCE PREDICTIVE -- API UNIFIEE V3.1")
    print("=" * 80)
    print(f"  URL     : http://localhost:8000")
    print(f"  Docs    : http://localhost:8000/docs")
    print(f"  Redoc   : http://localhost:8000/redoc")
    print(f"\n  Endpoints IA :")
    print(f"    POST /v1/predict              -> Anomalie (IF+LOF+OCSVM+ECOD) | vote 2/4")
    print(f"    POST /v1/predict-rul          -> RUL (Remaining Useful Life)")
    print(f"    POST /v1/iot-predict          -> Predict+RUL direct IoT sans BDD [NEW]")
    print(f"    GET  /v1/health-score/{{sensor_id}}  -> Score sante 0-100")
    print(f"    GET  /v1/history/{{sensor_id}}       -> Historique predictions")
    print(f"    GET  /v1/alert-level/{{sensor_id}}   -> Niveau alerte dashboard")
    print(f"\n  Endpoints systeme :")
    print(f"    GET  /health    -> Health check + modeles charges")
    print(f"    GET  /metrics   -> F1=0.298 | AUC=0.9475 | CV 3-fold par capteur")
    print(f"    GET  /sensors   -> 20 capteurs IFM")
    print(f"    GET  /anomalies -> Anomalies filtrees")
    print("=" * 80 + "\n")

    if not FASTAPI_OK:
        print("Installe les dépendances : pip install fastapi uvicorn pydantic scipy")
    else:
        port = int(os.environ.get("PORT", 8000))
        # TLS optionnel -- si TLS_CERT_FILE/TLS_KEY_FILE sont definis (voir
        # generate_selfsigned_cert.py pour un certificat de test), l'API sert
        # directement en HTTPS. Sinon comportement inchange (HTTP simple),
        # retro-compatible avec tous les deploiements existants.
        tls_cert = os.environ.get("TLS_CERT_FILE", "").strip()
        tls_key  = os.environ.get("TLS_KEY_FILE", "").strip()
        ssl_kwargs = {}
        if tls_cert and tls_key:
            if Path(tls_cert).exists() and Path(tls_key).exists():
                ssl_kwargs = {"ssl_certfile": tls_cert, "ssl_keyfile": tls_key}
                print(f"  TLS actif — HTTPS sur le port {port} (cert: {tls_cert})")
            else:
                print(f"  ATTENTION : TLS_CERT_FILE/TLS_KEY_FILE definis mais introuvables — HTTP sans chiffrement")
        uvicorn.run(app, host="0.0.0.0", port=port, reload=False, **ssl_kwargs)
