"""
routers/data.py
================
Endpoints de consultation des données : résultats consolidés, liste des
capteurs, historique et niveau d'alerte par capteur, anomalies filtrées.
"""

import logging
from datetime import datetime

import numpy as np
from fastapi import APIRouter, Request, Depends, HTTPException
from fastapi.encoders import jsonable_encoder

import core
from core import safe_trend
from rate_limiter import make_rate_limiter

log = logging.getLogger(__name__)
router = APIRouter()


# ── Résultats JSON temps réel (pour encadrant / export) ───────────────
@router.get(
    "/v1/results",
    tags=["IA / Prédiction"],
    summary="Derniers résultats de tous les capteurs (GET — navigateur)",
    description=(
        "Retourne en JSON les dernières prédictions (anomalie + RUL) "
        "pour chaque capteur actif. Endpoint GET accessible directement "
        "depuis un navigateur ou curl. Mis à jour à chaque appel /v1/predict."
    )
)
def get_results(request: Request, sensor_id: str = None, _rl=Depends(make_rate_limiter(60))):
    """
    GET /v1/results          → tous les capteurs actifs
    GET /v1/results?sensor_id=8f7f2f7e  → un capteur précis
    """
    if not core._latest_results:
        return {
            "status":  "en_attente",
            "message": "Aucune prédiction reçue pour l'instant. Démarrez le moteur temps réel.",
            "results": []
        }
    if sensor_id:
        entry = core._latest_results.get(sensor_id)
        if not entry:
            raise HTTPException(status_code=404, detail=f"Capteur '{sensor_id}' non trouvé dans les résultats actifs.")
        return jsonable_encoder({
            "status":    "ok",
            "n_sensors": 1,
            "results":   [entry]
        })
    return jsonable_encoder({
        "status":    "ok",
        "timestamp": datetime.now().isoformat(),
        "api":       "Maintenance Prédictive — ISG Bizerte",
        "version":   core.API_VERSION,
        "n_sensors": len(core._latest_results),
        "results":   list(core._latest_results.values())
    })


# ── Liste capteurs ────────────────────────────────────────────────────
@router.get("/sensors", tags=["Données"])
def get_sensors(request: Request, _rl=Depends(make_rate_limiter(60))):
    # ── Priorité 1 : anomaly_history temps réel (rempli par /v1/predict) ──
    if core.anomaly_history:
        try:
            sensors_list = []
            for sid, dq in core.anomaly_history.items():
                hist = list(dq)
                if not hist:
                    continue
                scores = [e.get("score", 0) for e in hist]
                n_anom = sum(1 for s in scores if s >= 0.5)
                avg_s  = round(sum(scores) / len(scores), 3)
                sensors_list.append({
                    "sensor_id":    sid,
                    "n_measures":   len(hist),
                    "n_anomalies":  n_anom,
                    "anomaly_rate": round(n_anom / len(hist), 3),
                    "avg_score":    avg_s,
                    "avg_health":   round(max(0, 100 - avg_s * 100), 1),
                })
            if sensors_list:
                return {"sensors": sensors_list, "source": "realtime"}
        except Exception as e:
            log.warning(f"/sensors fallback erreur : {e}")

    # ── Priorité 2 : df_results historique (fichier CSV pré-calculé) ──
    if core.df_results is not None:
        try:
            summary = (
                core.df_results.groupby("sensor_id")
                .agg(
                    n_measures  =("is_anomaly", "count"),
                    n_anomalies =("is_anomaly", "sum"),
                    avg_score   =("anomaly_score", "mean"),
                    avg_health  =("health_score", "mean"),
                )
                .reset_index()
            )
            summary["anomaly_rate"] = (
                summary["n_anomalies"] / summary["n_measures"]
            ).round(3)
            return {"sensors": summary.to_dict(orient="records"), "source": "historical"}
        except Exception as e:
            return {"sensors": [], "error": str(e)}

    return {"sensors": [], "message": "Aucune donnée — lance realtime_mariadb.py"}


# ══════════════════════════════════════════════════════════════════════
#  GET /v1/history/{sensor_id} — Historique prédictions [NEW]
# ══════════════════════════════════════════════════════════════════════
@router.get(
    "/v1/history/{sensor_id}",
    tags=["IA / Prédiction"],
    summary="Historique des prédictions d'un capteur",
    description=(
        "Retourne les N dernières prédictions enregistrées en mémoire "
        "pour un capteur donné. Utile pour visualiser la tendance de dégradation "
        "dans un dashboard ou pour debug.\n\n"
        "**Note** : L'historique est en RAM — réinitialisé au redémarrage de l'API."
    )
)
def get_history(request: Request, sensor_id: str, limit: int = 20, _rl=Depends(make_rate_limiter(60))):
    """
    Retourne l'historique glissant des scores d'anomalie pour un capteur.
    Maximum HISTORY_WINDOW (50) entrées gardées en mémoire.
    """
    if sensor_id not in core.anomaly_history or not core.anomaly_history[sensor_id]:
        raise HTTPException(
            status_code=404,
            detail=f"Aucun historique pour le capteur '{sensor_id}'. "
                   f"Lance d'abord POST /v1/predict avec ce sensor_id."
        )

    hist = list(core.anomaly_history[sensor_id])
    hist_limited = hist[-limit:]  # Garder les N plus récents

    scores = [e["score"] for e in hist if not np.isnan(e.get("score", np.nan))]
    if not scores:
        scores = [0.0]
    recent = scores[-10:]

    # Tendance : en hausse = dégradation, en baisse = amélioration
    trend_val = safe_trend(scores[-5:]) if len(scores) >= 5 else 0.0
    if trend_val > 0.02:
        trend_label = "DÉGRADATION"
    elif trend_val < -0.02:
        trend_label = "AMÉLIORATION"
    else:
        trend_label = "STABLE"

    return {
        "sensor_id":      sensor_id,
        "n_total":        len(hist),
        "n_returned":     len(hist_limited),
        "avg_score":      round(float(np.mean(recent)), 4),
        "max_score":      round(float(np.max(scores)), 4),
        "anomaly_rate":   round(sum(1 for s in recent if s >= 0.5) / len(recent), 3),
        "trend":          trend_label,
        "trend_value":    round(trend_val, 6),
        "history":        hist_limited,
        "timestamp":      datetime.now().isoformat(),
    }


# ══════════════════════════════════════════════════════════════════════
#  GET /v1/alert-level/{sensor_id} — Niveau d'alerte actuel [NEW]
# ══════════════════════════════════════════════════════════════════════
@router.get(
    "/v1/alert-level/{sensor_id}",
    tags=["IA / Prédiction"],
    summary="Niveau d'alerte actuel d'un capteur",
    description=(
        "Retourne le niveau d'alerte consolidé d'un capteur basé sur "
        "ses 10 dernières prédictions en mémoire.\n\n"
        "**Niveaux** : OK → ATTENTION → URGENT → CRITIQUE\n\n"
        "Conçu pour alimenter des feux tricolores dans un dashboard."
    )
)
def get_alert_level(request: Request, sensor_id: str, _rl=Depends(make_rate_limiter(60))):
    """
    Calcule le niveau d'alerte consolidé à partir de l'historique en mémoire.
    Retourne une réponse simple pour dashboard / feux tricolores.
    """
    if sensor_id not in core.anomaly_history or not core.anomaly_history[sensor_id]:
        return {
            "sensor_id":   sensor_id,
            "alert_level": "INCONNU",
            "color":       "gray",
            "message":     "Aucune prédiction reçue pour ce capteur.",
            "timestamp":   datetime.now().isoformat(),
        }

    hist   = list(core.anomaly_history[sensor_id])
    scores = [e["score"] for e in hist[-10:] if not np.isnan(e.get("score", np.nan))]
    if not scores:
        return {"sensor_id": sensor_id, "alert_level": "INCONNU", "color": "gray",
                "message": "Scores invalides.", "timestamp": datetime.now().isoformat()}
    avg    = float(np.nanmean(scores))
    if np.isnan(avg): avg = 0.0
    trend  = safe_trend(scores) if len(scores) >= 3 else 0.0
    anomaly_rate = sum(1 for s in scores if s >= 0.5) / len(scores)

    # Calcul niveau d'alerte consolidé
    if avg >= 0.75 or anomaly_rate >= 0.80:
        alert, color, icon = "CRITIQUE",  "red",    "🔴"
    elif avg >= 0.50 or anomaly_rate >= 0.50:
        alert, color, icon = "URGENT",    "orange", "🟠"
    elif avg >= 0.25 or anomaly_rate >= 0.20:
        alert, color, icon = "ATTENTION", "yellow", "🟡"
    else:
        alert, color, icon = "OK",        "green",  "🟢"

    messages = {
        "OK":        "Fonctionnement nominal.",
        "ATTENTION": "Surveillance renforcée recommandée.",
        "URGENT":    "Intervention recommandée sous 72h.",
        "CRITIQUE":  "ARRÊT IMMÉDIAT recommandé.",
    }

    return {
        "sensor_id":    sensor_id,
        "alert_level":  alert,
        "color":        color,
        "icon":         icon,
        "avg_score":    round(avg, 4),
        "anomaly_rate": round(anomaly_rate, 3),
        "trend":        "↑ HAUSSE" if trend > 0.02 else ("↓ BAISSE" if trend < -0.02 else "→ STABLE"),
        "n_measures":   len(scores),
        "message":      messages[alert],
        "timestamp":    datetime.now().isoformat(),
    }


# ── Anomalies filtrées ────────────────────────────────────────────────
@router.get("/anomalies", tags=["Données"])
def get_anomalies(request: Request, min_score: float = 0.5, limit: int = 100, _rl=Depends(make_rate_limiter(60))):
    if core.df_results is None:
        return {"anomalies": []}
    df_a = core.df_results[core.df_results["anomaly_score"] >= min_score]
    cols = [c for c in ["sensor_id", "motor_id", "motor_name",
                         "anomaly_score", "risk_level",
                         "temp_cur", "vib_z_cur"] if c in df_a.columns]
    return {
        "n_anomalies": len(df_a),
        "anomalies":   df_a[cols].dropna(how="all").head(limit).to_dict(orient="records"),
    }
