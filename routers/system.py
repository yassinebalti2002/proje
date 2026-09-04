"""
routers/system.py
==================
Endpoints système : accueil, health check, métriques modèle, model card,
limites connues du système.
"""

import json
import logging
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, Request, Depends

import core
from auth import require_api_key
from rate_limiter import make_rate_limiter

log = logging.getLogger(__name__)
router = APIRouter(tags=["Système"])


@router.get("/")
def root():
    return {
        "api":     "Maintenance Prédictive — API Unifiée",
        "version": core.API_VERSION,
        "port":    8000,
        "docs":    "http://localhost:8000/docs",
        "endpoints": {
            "POST /v1/predict":               "Détection anomalie temps réel (6 modèles, stacking)",
            "POST /v1/predict-rul":           "Estimation RUL (Remaining Useful Life)",
            "POST /v1/iot-predict":           "Prédiction directe IoT sans base de données [NEW]",
            "GET  /v1/health-score/{id}":     "Score santé moteur (0-100)",
            "GET  /v1/history/{id}":          "Historique prédictions par capteur",
            "GET  /v1/alert-level/{id}":      "Niveau alerte actuel — dashboard",
            "GET  /v1/kpi-history":           "Instantanés périodiques des KPIs du parc",
            "GET  /v1/tasks-history":         "Journal unifié des tâches système",
            "POST /v1/auth/register":         "Créer un compte utilisateur (dashboard)",
            "POST /v1/auth/login":            "Connexion — JWT",
            "GET  /health":                   "Health check API",
            "GET  /metrics":                  "Métriques modèle (voir models/metrics_v3.csv — valeurs live)",
            "GET  /sensors":                  "Liste 20 capteurs IFM",
            "GET  /anomalies":                "Anomalies filtrées par score",
            "GET  /docs":                     "Documentation Swagger — liste complète des 30 endpoints",
        }
    }


@router.get("/health")
def health():
    from datetime import datetime
    return {
        "status":         "ok",
        "models_loaded":  len(core.models) >= 4,
        "models":         list(core.models.keys()),
        "features_count": len(core.features_list),
        "version":        core.API_VERSION,
        "n_sensors_in_memory": len(core.anomaly_history),
        "timestamp":      datetime.now().isoformat(),
    }


@router.get("/metrics", summary="Métriques du modèle V3 (F1, AUC, Accuracy)")
def get_metrics(request: Request, _rl=Depends(make_rate_limiter(30))):
    """
    Retourne les métriques de performance du modèle non supervisé (lues en direct
    depuis models/metrics_v3.csv — modèle V8 : F1=0,8052, AUC=0,9868, holdout par capteur).
    """
    path = Path(core.METRICS_PATH)
    if not path.exists():
        # Chercher dans tous les emplacements possibles
        for candidate in [
            core.MODEL_DIR / "metrics_v2.csv",
            core.MODEL_DIR / "metrics_v3.csv",
            core.PROJECT_DIR / "metrics_v2.csv",
        ]:
            if candidate.exists():
                path = candidate
                break
        else:
            return {
                "message": "Métriques non disponibles — lance step3_model.py",
                "hint":    "Le fichier metrics_v2.csv doit être dans le dossier models/",
            }
    try:
        # latin-1 accepte tous les octets 0-255 — toujours safe sur Windows
        df_m = pd.read_csv(path, encoding="latin-1")
        # Format long (metric,value) → dict {metric_name: value}
        if "metric" in df_m.columns and "value" in df_m.columns:
            m = df_m.set_index("metric")["value"].to_dict()
        else:
            m = df_m.iloc[0].to_dict()
        # RAPPORT : stacking LogisticRegression sur tous les modèles chargés
        softvote_models = [k for k in ["if","lof","ocsvm","ecod","hbos","copod"] if k in core.models]
        n_models = len(softvote_models)
        ens_names = m.get("ensemble", " + ".join(k.upper() for k in softvote_models))
        # Coefficients réels du stacking (weights_if, weights_lof, ... dans metrics_v3.csv)
        # — fallback à un poids égal si le modèle n'a pas encore été ré-entraîné avec le stacking.
        learned_weights = {
            k.upper(): round(float(m[f"weights_{k}"]), 4)
            for k in softvote_models if f"weights_{k}" in m
        }
        weights_out = learned_weights or {k.upper(): round(1/max(n_models,1), 2) for k in softvote_models}
        return {
            "model_version": m.get("model_version", "V7"),
            "ensemble":      ens_names,
            "voting":        m.get("voting", "SoftVote seuil optimal"),
            "dataset":       m.get("dataset", "ai_cp full_data — 20 capteurs IFM"),
            "window_size":   int(float(m.get("window_size", 20))),
            "n_features":    int(float(m.get("n_features",  31))),
            "augmentation":  m.get("augmentation", "x3"),
            "pca_variance":  float(m.get("pca_variance", 0.95)),
            "f1_score":      round(float(m.get("f1_score",  0)), 4),
            "accuracy":      round(float(m.get("accuracy",  0)), 4),
            "precision":     round(float(m.get("precision", 0)), 4),
            "recall":        round(float(m.get("recall",    0)), 4),
            "auc_roc":       round(float(m.get("auc_roc",   0)), 4),
            "n_anomalies":   int(float(m.get("n_anomalies", 0))),
            "n_total":       int(float(m.get("n_total",     0))),
            "contamination": round(float(m.get("contamination", 0)), 4),
            "weights": weights_out,
            "source_file": str(path.name),
        }
    except Exception as e:
        return {"message": f"Erreur lecture métriques : {e}"}


@router.get("/v1/model-card",
            summary="Fiche technique : provenance des données, méthodologie, limites connues")
def get_model_card(request: Request, _rl=Depends(make_rate_limiter(30))):
    """
    Expose de façon structurée et programmatique la provenance des données
    d'entraînement et les limites de chaque modèle -- pratique standard
    (« model card » / « data sheet ») pour tout système ML utilisé en
    production. Précise aussi qu'un modèle RUL entraîné sur données
    synthétiques a été expérimenté puis écarté du pipeline de production
    (voir rul_estimation.data_provenance.note ci-dessous).

    Objectif : qu'un intégrateur tiers puisse vérifier PROGRAMMATIQUEMENT
    (pas seulement dans une doc PDF qu'on peut oublier de lire) sur quoi
    repose une prédiction avant de l'utiliser pour une décision critique.
    """
    from datetime import datetime
    card = {
        "generated_at": datetime.now().isoformat(),
        "api_version": core.API_VERSION,
        "anomaly_detection": {
            "models": ["IsolationForest", "LOF", "OCSVM", "ECOD", "HBOS", "COPOD"],
            "fusion": "Stacking LogisticRegression",
            "data_provenance": {
                "source": "ai_cp.full_data — mesures réelles capteurs IFM (20 capteurs, Novation City)",
                "real_measurements": True,
                "label_type": "heuristique (percentiles composites, PAS de panne confirmée étiquetée)",
                "label_caveat": (
                    "Les 'anomalies' d'entraînement sont définies par des seuils statistiques "
                    "(percentile de vibration/température/kurtosis), pas par des pannes réelles "
                    "confirmées. Le modèle apprend à reproduire cette règle, pas à prédire des "
                    "défaillances observées."
                ),
            },
            "evaluation": {},
        },
        "rul_estimation": {
            "components": ["heuristique (seuils industriels fixes + tendance réelle + historique réel du capteur)"],
            "data_provenance": {
                "source": "Calcul déterministe sur les mesures réelles reçues via /v1/predict-rul — aucun modèle entraîné, aucune donnée fabriquée",
                "real_measurements": True,
                "note": (
                    "Un modèle ML (GradientBoostingRegressor) avait été expérimenté sur des courbes "
                    "de dégradation SYNTHÉTIQUES (loi de Weibull), faute de panne réelle confirmée "
                    "sur la période de collecte (nov. 2025 - mai 2026, voir train_rul_model.py). "
                    "Il n'est plus chargé ni appelé dans le pipeline de production depuis cette "
                    "version : le RUL retourné par cette API provient uniquement de la formule "
                    "heuristique ci-dessus, calculée à 100% à partir des mesures réelles."
                ),
            },
            "evaluation": {},
        },
        "known_limitations": [
            "current_mean systématiquement à 0 — le courant électrique réel existe (table motor_mesure) mais n'est pas encore intégré au pipeline de features",
            "Détection binaire uniquement (anomalie/normal) — pas de classification du type de défaut (bille/piste intérieure/extérieure)",
            "Rate limiting par IP peu fiable derrière un reverse proxy sans configuration proxy_headers",
        ],
    }

    # Metriques anomalies -- reutilise le meme fichier que /metrics
    try:
        path = core.METRICS_PATH if Path(core.METRICS_PATH).exists() else None
        if path:
            df_m = pd.read_csv(path, encoding="latin-1")
            m = df_m.set_index("metric")["value"].to_dict() if "metric" in df_m.columns else df_m.iloc[0].to_dict()
            card["anomaly_detection"]["evaluation"] = {
                "auc_roc":    round(float(m.get("auc_roc", 0)), 4),
                "f1_score":   round(float(m.get("f1_score", 0)), 4),
                "precision":  round(float(m.get("precision", 0)), 4),
                "recall":     round(float(m.get("recall", 0)), 4),
                "n_total":    int(float(m.get("n_total", 0))),
                "n_anomalies": int(float(m.get("n_anomalies", 0))),
                "trained_at": m.get("trained_at", "inconnu"),
                "evaluation_method": m.get("evaluation", "inconnu"),
            }
    except Exception as e:
        card["anomaly_detection"]["evaluation"] = {"error": str(e)}

    # Pas de métriques RUL ici : depuis que le modèle ML Weibull est retiré du
    # pipeline de production, il n'y a plus de modèle "entraîné" à évaluer pour
    # le RUL -- c'est une formule déterministe (compute_rul), pas un modèle.
    # metrics_rul_v1.json reste sur disque pour archive/expérimentation mais
    # ne décrit plus ce que l'API retourne réellement.

    return card


@router.get("/v1/system-limits",
            summary="Limites connues et lacunes techniques du système")
def get_system_limits(request: Request, _key: str = Depends(require_api_key), _rl=Depends(make_rate_limiter(30))):
    """
    Documente honnêtement les limitations techniques identifiées.
    Utile pour la transparence et la soutenance PFE.
    """
    return {
        "version": core.API_VERSION,
        "limits": [
            {
                "id": "L1",
                "titre": "Alertes externes — email actif, webhook/SMS non configurés",
                "description": (
                    "Canal email SMTP configuré et vérifié (envoi réel testé avec succès, "
                    "cooldown 300s/capteur, seuil health_score<=85, niveaux URGENT/CRITIQUE "
                    "uniquement — cohérent avec RUL <= 7 jours). Webhook (Slack/Teams/Discord) "
                    "et SMS (Twilio) restent désactivés faute de compte externe fourni — le "
                    "module alert_manager.py les supporte déjà si besoin plus tard."
                ),
                "statut": "EMAIL_ACTIF",
                "fichier": "alert_manager.py",
                "activation_restante": "Webhook/SMS : renseigner url/credentials dans alert_config.json",
                "note_deploiement": "Avant remise au client : remplacer les destinataires email de test par les vraies adresses du client dans alert_config.json"
            },
            {
                "id": "L2",
                "titre": "4 features d'accélération toujours nulles",
                "description": (
                    "acc_p2p, acc_z2p, acc_crest, acc_rms = 0.0 dans toutes les prédictions. "
                    "La gateway IFM AL1352 transmet ces valeurs sur un seul axe Y dans une "
                    "sous-clé séparée du JSON, et la consolidation multi-lignes ne les aligne "
                    "pas avec la mesure principale. Ces 4 features ont variance nulle et "
                    "n'apportent aucune information discriminante au modèle."
                ),
                "statut": "LIMITATION_MATERIELLE",
                "impact": "4/25 features neutralisées — PCA les absorbe sans effet"
            },
            {
                "id": "L3",
                "titre": "Courant électrique absent",
                "description": (
                    "current_mean = 0.0 dans 100%% des prédictions. Les capteurs IFM VVB001/VSE002 "
                    "mesurent uniquement vibrations et température. Aucun capteur de courant "
                    "(pince ampèremétrique, transducteur de courant) n'est intégré au banc d'essai. "
                    "Le poids courant (15%%) du health_score est systématiquement neutralisé."
                ),
                "statut": "LIMITATION_MATERIELLE",
                "impact": "health_score calculé sur 85%% de son potentiel — poids courant inactif"
            },
            {
                "id": "L4",
                "titre": "Pas de déploiement Edge computing",
                "description": (
                    "L'API tourne sur un PC Windows standard (localhost:8000). "
                    "Il n'existe pas de Dockerfile, de configuration ARM (Raspberry Pi / Jetson), "
                    "ni de modèles ONNX/TFLite optimisés pour embarqué. "
                    "L'export ONNX des modèles pyod (ECOD) n'est pas nativement supporté "
                    "par sklearn-onnx et nécessiterait une refonte du pipeline d'inférence."
                ),
                "statut": "NON_IMPLEMENTE",
                "solution_envisagee": (
                    "Exporter IF et OCSVM via sklearn-onnx, "
                    "déployer sur Raspberry Pi 4 avec onnxruntime, "
                    "conteneuriser avec Docker."
                )
            },
            {
                "id": "L5",
                "titre": "Faux niveaux URGENT sur capteurs sains",
                "description": (
                    "La formule RUL était trop sensible : des capteurs avec health_score > 90 "
                    "recevaient un niveau URGENT. Correction appliquée dans cette version : "
                    "tout capteur avec health_score >= 85 est forcé en niveau OK, "
                    "avec un RUL plancher de 500 heures."
                ),
                "statut": "CORRIGE_PARTIELLEMENT",
                "correctif": "Filtre health_score >= 85 → forcer alert_level = OK (compute_rul)"
            },
            {
                "id": "L6",
                "titre": "RUL heuristique — pas de modèle entraîné",
                "description": (
                    "Le RUL est calculé par formule empirique (deg_instant * 0.5 + deg_rate * 0.3 "
                    "+ hist_factor * 0.2), pas par un modèle de régression entraîné. "
                    "Un modèle supervisé (Weibull, LSTM de dégradation, Cox) nécessite "
                    "des données de défaillances réelles confirmées. Aucun des 20 capteurs "
                    "n'a atteint la défaillance complète pendant la période de collecte "
                    "(nov. 2025 → mai 2026), rendant l'entraînement supervisé impossible."
                ),
                "statut": "LIMITATION_DONNEES",
                "impact": "RUL estimatif uniquement — précision non validée sur défaillances réelles"
            },
            {
                "id": "L7",
                "titre": "BPFO/BPFI/BSF non mesurables avec le fs par défaut de l'analyse spectrale",
                "description": (
                    "GET /v1/spectral/{id} et POST /v1/spectral-analysis utilisent fs=100Hz par "
                    "défaut (Nyquist = 50Hz). Pour un moteur à 1450 tr/min sur roulement SKF "
                    "6205-2RS, BPFO≈86Hz, BPFI≈131Hz et BSF≈56Hz dépassent tous cette limite : "
                    "leur SNR est structurellement 0 (aucun bin de fréquence n'existe à ces "
                    "valeurs), pas une preuve d'absence de défaut. Seule FTF≈9.6Hz (défaut de "
                    "cage) est mesurable. Cause plus profonde : le signal analysé est la suite "
                    "des vibration_z reçues à chaque appel /v1/predict (espacées de plusieurs "
                    "secondes), pas une forme d'onde d'accéléromètre haute fréquence -- "
                    "augmenter fs ne suffirait pas sans un vrai capteur haute fréquence en amont."
                ),
                "statut": "LIMITATION_MATERIELLE",
                "correctif": "Champ 'measurable' ajouté par fréquence (signal_processing.py) pour distinguer 'non mesurable' de 'absent'",
                "impact": "Diagnostic BPFO/BPFI/BSF non fiable en l'état — seul FTF est théoriquement exploitable"
            }
        ]
    }
