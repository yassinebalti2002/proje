"""
routers/predict.py
===================
Endpoints de prédiction ML : détection d'anomalie, RUL, et le combiné
IoT (sans base de données).
"""

import logging
from collections import deque
from datetime import datetime

import numpy as np
from fastapi import APIRouter, Request, Depends, HTTPException

import core
from core import (
    MeasurePoint, PredictRequest, PredictResponse,
    RULRequest, RULResponse,
    IoTMeasurementRequest, IoTPredictResponse,
    extract_features, run_ensemble, compute_rul, update_history,
)
from auth import require_api_key
from rate_limiter import make_rate_limiter

log = logging.getLogger(__name__)
router = APIRouter(tags=["IA / Prédiction"])


# ══════════════════════════════════════════════════════════════════════════
#  POST /v1/predict — Détection anomalie
# ══════════════════════════════════════════════════════════════════════════
@router.post(
    "/v1/predict",
    response_model=PredictResponse,
    summary="Détection d'anomalie temps réel",
    description=(
        "Reçoit l'historique de mesures d'un capteur et retourne le diagnostic "
        "d'anomalie basé sur le stacking (LogisticRegression) des 6 modèles IF/LOF/OCSVM/ECOD/HBOS/COPOD.\n\n"
        "**Format données** : compatible avec le champ `data` de la table `full_data` "
        "(SensorNodeId, Temperature, Vibration RMS X/Y/Z)."
    )
)
def predict_anomaly(request: Request, req: PredictRequest, _key: str = Depends(require_api_key), _rl=Depends(make_rate_limiter(60))):
    if not req.history:
        raise HTTPException(status_code=400, detail="history ne peut pas être vide")

    # 1. Extraction features
    feat = extract_features(req.history)

    # 1b. Alimente le buffer brut vibration_z (pour GET /v1/spectral/{id},
    # lecture publique — voir core._raw_vib_buffers)
    _vib_pts = [h.vibration_z for h in req.history if h.vibration_z is not None]
    if _vib_pts:
        _buf = core._raw_vib_buffers.setdefault(req.sensor_id, deque(maxlen=core.RAW_VIB_BUFFER_SIZE))
        _buf.extend(_vib_pts)

    # 2. Construction vecteur pour le modèle
    if core.features_list:
        X = np.array([[feat.get(c, np.nan) for c in core.features_list]], dtype="float32")
        X = np.nan_to_num(X, nan=0.0)
    else:
        # Fallback : vecteur par défaut si features non chargées
        X = np.array([[
            feat.get("temp_mean", 35),
            feat.get("vib_z_rms_w", 300),
            feat.get("vib_z_kurt", 0),
            feat.get("vib_x_rms_w", 200),
            feat.get("vib_y_rms_w", 200),
        ]], dtype="float32")

    # 3. Inférence ensemble (avec filtre persistance par capteur)
    sensor_id_pred = req.sensor_id if hasattr(req, "sensor_id") and req.sensor_id else "default"
    result = run_ensemble(X, sensor_id=sensor_id_pred)

    # 4. Calcul anomaly_score normalisé (0–1)
    anomaly_score = round(result["confidence"], 4)
    # V6 : boost contextuel base sur les valeurs physiques reelles
    if result["is_anomaly"]:
        vib_rms = feat.get("vib_z_rms_w", 0) or 0
        if vib_rms > 1000:
            anomaly_score = min(1.0, anomaly_score + 0.15)
        elif vib_rms > 600:
            anomaly_score = min(1.0, anomaly_score + 0.05)

    # 5. Niveau de risque
    if anomaly_score >= 0.75:   risk = "CRITIQUE"
    elif anomaly_score >= 0.50: risk = "ÉLEVÉ"
    elif anomaly_score >= 0.25: risk = "MODÉRÉ"
    else:                       risk = "FAIBLE"

    # Cohérence prediction/risk : tant que le filtre de persistance (V9) n'a
    # pas confirmé l'anomalie (is_anomaly=False), ne pas afficher CRITIQUE/
    # ÉLEVÉ. Le score brut peut déjà être élevé dès la 1re mesure suspecte,
    # avant confirmation sur k fenêtres consécutives -- sans ce plafond, la
    # réponse pouvait annoncer prediction="NORMAL" avec risk_level="CRITIQUE"
    # en même temps (observé en test réel), contradictoire pour un dashboard.
    if not result["is_anomaly"] and risk in ("CRITIQUE", "ÉLEVÉ"):
        risk = "MODÉRÉ"

    # Cohérence health/risk : capteur sain (health ≥ 85) non confirmé anomalie → FAIBLE
    # Évite MODÉRÉ+Health=99 qui est contradictoire pour l'utilisateur
    _health_val = feat.get("health_score", 0) or 0
    if _health_val >= 85 and not result["is_anomaly"]:
        risk = "FAIBLE"

    # 6. Mise à jour historique moteur (pour RUL)
    update_history(req.sensor_id, anomaly_score, result["confidence"])

    # 6b. Pas d'alerte externe ici : le risque d'anomalie (risk_level) ne dit
    # rien du temps restant avant défaillance. Sur demande, les alertes
    # email/webhook/SMS sont désormais déclenchées uniquement depuis
    # /v1/predict-rul, dont alert_level (URGENT/CRITIQUE) correspond
    # précisément à un RUL <= 7 jours -- voir ce endpoint plus bas.

    # 7. Features utiles à retourner (pour debug / dashboard)
    # NaN -> None (jamais la valeur NaN brute) : Starlette sert allow_nan=False,
    # un float NaN dans la réponse fait planter la sérialisation JSON en 500
    # (observé sous charge concurrente sur un même capteur -- ValueError:
    # "Out of range float values are not JSON compliant").
    feat_summary = {
        k: (round(v, 4) if not np.isnan(v) else None) if isinstance(v, float) else v
        for k, v in feat.items()
    }

    _ts = datetime.now().isoformat()
    core._latest_results.setdefault(req.sensor_id, {}).update({
        "sensor_id":    req.sensor_id,
        "motor_id":     req.motor_id,
        "timestamp":    _ts,
        "predict": {
            "prediction":        result["label"],
            "is_anomaly":        result["is_anomaly"],
            "confidence":        result["confidence"],
            "anomaly_score":     anomaly_score,
            "risk_level":        risk,
            "votes":             result["votes"],
            "individual_models": result["individual"],
            "individual_scores": result["individual_scores"],
            "features":          feat_summary,
        }
    })

    return PredictResponse(
        sensor_id         = req.sensor_id,
        motor_id          = req.motor_id,
        timestamp         = _ts,
        prediction        = result["label"],
        is_anomaly        = result["is_anomaly"],
        confidence        = result["confidence"],
        votes             = result["votes"],
        risk_level        = risk,
        anomaly_score     = anomaly_score,
        individual_models = result["individual"],
        individual_scores = result["individual_scores"],
        features          = feat_summary,
    )


# ══════════════════════════════════════════════════════════════════════════
#  POST /v1/predict-rul — Remaining Useful Life
# ══════════════════════════════════════════════════════════════════════════
@router.post(
    "/v1/predict-rul",
    response_model=RULResponse,
    summary="Estimation du Remaining Useful Life (RUL)",
    description=(
        "Estime le temps restant avant défaillance du roulement en heures.\n\n"
        "**Méthode** :\n"
        "- Score de dégradation instantané (position par rapport aux seuils industriels)\n"
        "- Tendance temporelle des vibrations et de la température (régression linéaire)\n"
        "- Historique des anomalies précédentes du moteur (fenêtre glissante)\n\n"
        "**Minimum requis** : 3 mesures dans `history` pour calculer la tendance.\n\n"
        "**Seuils utilisés** (roulements industriels) :\n"
        "- Température critique : > 60°C\n"
        "- Vibration Z RMS critique : > 1000 mg\n"
        "- Kurtosis critique : > 7"
    )
)
def predict_rul(request: Request, req: RULRequest, _key: str = Depends(require_api_key), _rl=Depends(make_rate_limiter(60))):
    if len(req.history) < 3:
        raise HTTPException(
            status_code=400,
            detail="Minimum 3 mesures requises pour estimer le RUL (calcul de tendance)"
        )

    # 1. Extraction features temporelles
    feat = extract_features(req.history)

    # 2. Enrichissement features spectrales (si signal_processing disponible)
    if core.SIGNAL_PROCESSING_OK:
        try:
            from signal_processing import extract_spectral_features
            vib_series = [h.vibration_z for h in req.history if h.vibration_z is not None]
            if len(vib_series) >= 8:
                spec_feat = extract_spectral_features(vib_series, fs=100.0, rpm=1450.0)
                feat.update(spec_feat)
        except Exception as _e:
            log.debug(f"Spectral features RUL ignorées : {_e}")

    # 3. Essai modèle RUL ML dédié (GradientBoosting)
    ml_rul_result = None
    if core.RUL_ML_ENABLED and core._rul_predictor is not None:
        try:
            ml_rul_result = core._rul_predictor.predict(feat)
            log.debug(f"RUL ML : {ml_rul_result['rul_hours']}h ({ml_rul_result['model_type']})")
        except Exception as _e:
            log.warning(f"RUL ML échoué, fallback heuristique : {_e}")

    # 4. Calcul RUL heuristique (toujours calculé pour les tendances)
    predict_result_data = {
        "prediction":    req.prediction    or "NORMAL",
        "votes":         req.votes         or 0,
        "confidence":    req.confidence    or 0.0,
        "risk_level":    req.risk_level     or "OK",
        "anomaly_score": req.anomaly_score or 0.0,
    }
    rul_heuristic = compute_rul(req.history, feat, req.sensor_id, predict_result_data)

    # 5. Sélection du résultat final
    # Stratégie : heuristique pour les heures RUL (calibrée sur les plages CDC),
    # ML pour détecter une dégradation plus précoce → peut élever le niveau d'alerte.
    # Le ML seul ne fixe plus les heures car son dataset synthétique n'a pas de
    # défaillances réelles → ses valeurs absolues restent surestimées.
    _LEVEL_ORDER = ["OK", "ATTENTION", "URGENT", "CRITIQUE"]
    _predict_risk = (req.risk_level or "OK").upper()
    _ml_alert     = (ml_rul_result or {}).get("alert_level", "OK")
    _heur_alert   = rul_heuristic["alert_level"]

    # Choisir le niveau d'alerte le plus sévère parmi heuristique et ML
    _ml_idx   = _LEVEL_ORDER.index(_ml_alert)   if _ml_alert   in _LEVEL_ORDER else 0
    _heur_idx = _LEVEL_ORDER.index(_heur_alert) if _heur_alert in _LEVEL_ORDER else 0
    alert_level = _LEVEL_ORDER[max(_ml_idx, _heur_idx)]

    # Heures RUL : toujours issues de l'heuristique (plages CDC garanties)
    # On réinterpolele dans la plage du niveau d'alerte final si celui-ci a été
    # aggravé par le ML (le RUL doit alors être dans la plage du nouveau niveau)
    if alert_level != _heur_alert and alert_level in core.RUL_TABLE:
        _rmin, _rmax = core.RUL_TABLE[alert_level]
        _deg = rul_heuristic["degradation_rate"] / 100.0
        _ratio = min(1.0, max(0.0, _deg))
        rul_hours = round(max(0.0, _rmax - _ratio * (_rmax - _rmin)), 1)
    else:
        rul_hours = rul_heuristic["rul_hours"]
    rul_days      = round(rul_hours / 24.0, 2)

    _ml_model_used = (ml_rul_result or {}).get("model_type", "none")
    confidence     = "HAUTE" if len(req.history) >= 10 else ("MOYENNE" if len(req.history) >= 5 else "FAIBLE")
    recommendation = rul_heuristic["recommendation"]
    trend_detail   = rul_heuristic["trend"]
    trend_detail["rul_model"] = f"heuristic_CDC + ML_{_ml_model_used}"

    # 6. Mise à jour historique
    deg_score = rul_heuristic["degradation_rate"] / 100.0
    update_history(req.sensor_id, deg_score, 1.0)

    # 7. Alerte externe si RUL sous seuil CDC (URGENT < 7j, CRITIQUE < 3j)
    if core.ALERTS_ENABLED and core._alert_manager and alert_level in ("URGENT", "CRITIQUE"):
        core._alert_manager.send_alert(
            sensor_id   = req.sensor_id,
            risk_level  = alert_level,
            health_score= rul_heuristic["health_score"],
            rul_hours   = rul_hours,
            vib_total   = None,
            temperature = None,
            votes       = 0
        )

    _ts_rul = datetime.now().isoformat()
    core._latest_results.setdefault(req.sensor_id, {}).update({
        "sensor_id": req.sensor_id,
        "timestamp": _ts_rul,
        "rul": {
            "rul_hours":        rul_hours,
            "rul_days":         rul_days,
            "health_score":     rul_heuristic["health_score"],
            "degradation_rate": rul_heuristic["degradation_rate"],
            "alert_level":      alert_level,
            "recommendation":   recommendation,
            "confidence":       confidence,
        }
    })

    return RULResponse(
        sensor_id        = req.sensor_id,
        motor_id          = req.motor_id,
        timestamp        = _ts_rul,
        rul_hours        = rul_hours,
        rul_days         = rul_days,
        degradation_rate = rul_heuristic["degradation_rate"],
        health_score     = rul_heuristic["health_score"],
        confidence       = confidence,
        alert_level      = alert_level,
        recommendation   = recommendation,
        trend            = trend_detail,
    )


# ══════════════════════════════════════════════════════════════════════════
#  POST /v1/iot-predict — Predict sans base de données (IoT direct)
# ══════════════════════════════════════════════════════════════════════════
@router.post(
    "/v1/iot-predict",
    response_model=IoTPredictResponse,
    summary="Prédiction directe depuis données IoT (sans base de données)",
    description=(
        "Endpoint destiné à la **production sans accès MariaDB**.\n\n"
        "Le collègue qui reçoit les données IoT envoie **une mesure à la fois** "
        "(température + vibration X/Y/Z consolidées). "
        "Le serveur maintient une **fenêtre glissante de 10 mesures** par capteur "
        "et retourne la prédiction d'anomalie **ET** le RUL en un seul appel.\n\n"
        "**Format d'entrée** : mesure consolidée issue des 3 lignes `full_data` "
        "(`gph='temperature'` + `gph='vibration_x'` + `gph='vibration_y'`).\n\n"
        "**Exemple d'utilisation** :\n"
        "```\n"
        "POST /v1/iot-predict\n"
        "{\n"
        '  "sensor_id": "8f7f2f7e",\n'
        '  "temperature": 32.5,\n'
        '  "vibration_x": 266.0,\n'
        '  "vibration_y": 273.0,\n'
        '  "vibration_z": 280.0\n'
        "}\n"
        "```"
    )
)
def iot_predict(request: Request, req: IoTMeasurementRequest, _key: str = Depends(require_api_key), _rl=Depends(make_rate_limiter(60))):
    # 1. Construire un MeasurePoint depuis la mesure brute
    point = MeasurePoint(
        timestamp   = req.timestamp or datetime.now().isoformat(),
        temperature = req.temperature,
        vibration_x = req.vibration_x,
        vibration_y = req.vibration_y,
        vibration_z = req.vibration_z,
        current     = req.current or 0.0,
        acc_p2p     = req.acc_p2p,
        acc_rms     = req.acc_rms,
        acc_crest   = req.acc_crest,
        acc_z2p     = req.acc_z2p,
    )

    # 2. Ajouter à la fenêtre glissante serveur
    if req.sensor_id not in core.iot_windows:
        core.iot_windows[req.sensor_id] = deque(maxlen=core.IOT_WINDOW_SIZE)
    core.iot_windows[req.sensor_id].append(point)
    history = list(core.iot_windows[req.sensor_id])
    window_size = len(history)

    # 3. Extraction features
    feat = extract_features(history)

    # 4. Construction vecteur pour les modèles
    if core.features_list:
        X = np.array([[feat.get(c, np.nan) for c in core.features_list]], dtype="float32")
        X = np.nan_to_num(X, nan=0.0)
    else:
        X = np.array([[
            feat.get("temp_mean", 35),
            feat.get("vib_z_rms_w", 300),
            feat.get("vib_z_kurt", 0),
            feat.get("vib_x_rms_w", 200),
            feat.get("vib_y_rms_w", 200),
        ]], dtype="float32")

    # 5. Inférence ensemble (avec filtre persistance par capteur)
    result = run_ensemble(X, sensor_id=req.sensor_id)

    # 6. Calcul anomaly_score
    anomaly_score = round(result["confidence"], 4)
    if result["is_anomaly"]:
        vib_rms = feat.get("vib_z_rms_w", 0) or 0
        if vib_rms > 1000:
            anomaly_score = min(1.0, anomaly_score + 0.15)
        elif vib_rms > 600:
            anomaly_score = min(1.0, anomaly_score + 0.05)

    # 7. Niveau de risque
    if anomaly_score >= 0.75:   risk = "CRITIQUE"
    elif anomaly_score >= 0.50: risk = "ÉLEVÉ"
    elif anomaly_score >= 0.25: risk = "MODÉRÉ"
    else:                       risk = "FAIBLE"

    # Cohérence prediction/risk — voir /v1/predict pour l'explication complète.
    if not result["is_anomaly"] and risk in ("CRITIQUE", "ÉLEVÉ"):
        risk = "MODÉRÉ"

    _health_val = feat.get("health_score", 0) or 0
    if _health_val >= 85 and not result["is_anomaly"]:
        risk = "FAIBLE"

    # 8. Mise à jour historique anomalies
    update_history(req.sensor_id, anomaly_score, result["confidence"])

    # 9. RUL — calculé uniquement si >= 3 mesures disponibles
    rul_hours = rul_days = health_score = alert_level = recommendation = None
    if window_size >= 3:
        predict_result_data = {
            "prediction":    result["label"],
            "votes":         result["votes"],
            "confidence":    result["confidence"],
            "risk_level":    risk,
            "anomaly_score": anomaly_score,
        }
        try:
            rul_result = compute_rul(history, feat, req.sensor_id, predict_result_data)
            rul_hours      = rul_result["rul_hours"]
            rul_days       = rul_result["rul_days"]
            health_score   = rul_result["health_score"]
            alert_level    = rul_result["alert_level"]
            recommendation = rul_result["recommendation"]
        except Exception as _rul_err:
            log.warning(f"RUL IoT échoué pour {req.sensor_id} : {_rul_err}")

    # 10. Alerte externe — uniquement si RUL <= 7 jours (alert_level
    # URGENT ou CRITIQUE, voir RUL_TABLE). Le risque d'anomalie seul
    # (risk_level) ne dit rien du temps restant avant défaillance, donc
    # ne déclenche plus d'alerte ici (aligné avec /v1/predict-rul).
    if core.ALERTS_ENABLED and core._alert_manager and alert_level in ("URGENT", "CRITIQUE"):
        core._alert_manager.send_alert(
            sensor_id    = req.sensor_id,
            risk_level   = alert_level,
            health_score = health_score if health_score is not None else feat.get("health_score", 0),
            rul_hours    = rul_hours,
            vib_total    = feat.get("vib_total"),
            temperature  = feat.get("temp_cur"),
            votes        = result["votes"]
        )

    # 11. Features résumées
    # NaN -> None, jamais la valeur brute (voir le meme correctif sur /v1/predict).
    feat_summary = {
        k: (round(v, 4) if not np.isnan(v) else None) if isinstance(v, float) else v
        for k, v in feat.items()
    }

    return IoTPredictResponse(
        sensor_id         = req.sensor_id,
        motor_id          = req.motor_id,
        timestamp         = datetime.now().isoformat(),
        window_size       = window_size,
        prediction        = result["label"],
        is_anomaly        = result["is_anomaly"],
        confidence        = result["confidence"],
        votes             = result["votes"],
        risk_level        = risk,
        anomaly_score     = anomaly_score,
        individual_models = result["individual"],
        individual_scores = result["individual_scores"],
        rul_hours         = rul_hours,
        rul_days          = rul_days,
        health_score      = health_score,
        alert_level       = alert_level,
        recommendation    = recommendation,
        features          = feat_summary,
    )


# ══════════════════════════════════════════════════════════════════════════
#  GET /v1/health-score/{sensor_id}
# ══════════════════════════════════════════════════════════════════════════
@router.get(
    "/v1/health-score/{sensor_id}",
    summary="Score de santé d'un moteur",
)
def get_health_score(request: Request, sensor_id: str, _rl=Depends(make_rate_limiter(60))):
    """
    Retourne le score de santé (0–100) normalisé par capteur.
    Utilise la baseline propre au capteur pour éviter le biais global.
    """
    from core import safe_trend
    if sensor_id not in core.anomaly_history or not core.anomaly_history[sensor_id]:
        return {
            "sensor_id":    sensor_id,
            "health_score": 100.0,
            "status":       "Aucun historique disponible pour ce capteur",
            "n_records":    0,
        }

    hist   = list(core.anomaly_history[sensor_id])
    scores = [e["score"] for e in hist if not np.isnan(e.get("score", np.nan))]
    if not scores:
        return {"sensor_id": sensor_id, "health_score": 100.0, "status": "Scores invalides", "n_records": 0}
    recent = scores[-10:]

    # Score brut
    raw_health = 100 * (1 - float(np.nanmean(recent)))

    # Normalisation par baseline capteur — corrige le biais global 43-48
    # Si baseline connue : on recentre le score autour de 100 (baseline = 0% dégradation)
    baseline = core.sensor_baseline.get(sensor_id)
    if baseline is not None and baseline > 0 and not np.isnan(baseline):
        # Score relatif : combien on a dégradé par rapport à la baseline
        degradation_relative = max(0.0, float(np.nanmean(recent)) - baseline)
        health = round(100 * (1 - degradation_relative / max(baseline, 0.01)), 1)
        health = max(0.0, min(100.0, health))
        score_method = "relatif_baseline"
    else:
        health = round(max(0.0, min(100.0, raw_health)), 1)
        score_method = "brut_en_attente_baseline"

    anomaly_rate = round(sum(1 for s in recent if s >= 0.5) / len(recent), 3)
    trend_val    = safe_trend(scores[-5:]) if len(scores) >= 5 else 0.0

    return {
        "sensor_id":     sensor_id,
        "health_score":  health,
        "health_raw":    round(raw_health, 1),
        "baseline":      round(baseline, 4) if baseline else None,
        "score_method":  score_method,
        "anomaly_rate":  anomaly_rate,
        "n_records":     len(hist),
        "last_score":    round(scores[-1], 4),
        "trend":         "DÉGRADATION" if trend_val > 0.01 else (
                         "AMÉLIORATION" if trend_val < -0.01 else "STABLE"),
        "timestamp":     hist[-1]["timestamp"],
    }
