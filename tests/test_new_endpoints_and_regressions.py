"""
test_new_endpoints_and_regressions.py
======================================
Tests complémentaires écrits lors de l'audit du 2026-07-25 :

  1. Tests unitaires (sans serveur) pour reporting_module.py :
     - régression XSS stocké (sensor_id / motor_id échappés dans le HTML)
     - régression crash sur timestamp manquant/None dans l'historique
  2. Tests d'intégration (nécessitent l'API démarrée) pour des endpoints
     qui n'étaient pas encore couverts par la suite existante :
     - POST /v1/iot-predict
     - GET  /v1/alert-level/{sensor_id}
     - GET  /v1/history/{sensor_id} (contenu, pas seulement l'auth)

Usage :
    pytest tests/test_new_endpoints_and_regressions.py -q
    pytest tests/test_new_endpoints_and_regressions.py -q --api-key <cle>
"""

import requests

from reporting_module import _build_sensor_table, _build_anomaly_history_section


# ══════════════════════════════════════════════════════════════════════════
#  Unitaires — reporting_module.py (aucun serveur requis)
# ══════════════════════════════════════════════════════════════════════════

def _sensor_summary(**overrides):
    base = {
        "sensor_id": "8f7f2f7e",
        "motor_id": "Motor_1604",
        "risk_level": "OK",
        "health_score": 90,
        "anomaly_count": 0,
        "anomaly_rate": 0,
        "rul_hours": 500,
        "rul_days": 20,
        "total_measures": 10,
        "last_seen": "2026-07-25T00:00:00",
    }
    base.update(overrides)
    return base


def test_sensor_table_escapes_sensor_id():
    """Un sensor_id malveillant ne doit jamais produire de HTML/JS actif."""
    payload = "<script>alert(document.cookie)</script>"
    html_out = _build_sensor_table([_sensor_summary(sensor_id=payload)])
    assert "<script>" not in html_out
    assert "&lt;script&gt;" in html_out


def test_sensor_table_escapes_motor_id():
    payload = "<img src=x onerror=alert(1)>"
    html_out = _build_sensor_table([_sensor_summary(motor_id=payload)])
    assert "<img src=x onerror" not in html_out
    assert "&lt;img" in html_out


def test_anomaly_history_section_handles_missing_timestamp():
    """entrée sans 'timestamp' -> ne doit pas lever d'exception (avant : .get() sans garde contre None)."""
    history = {"8f7f2f7e": [{"score": 0.9}]}  # pas de clé "timestamp"
    html_out = _build_anomaly_history_section(history)
    assert "8f7f2f7e" in html_out


def test_anomaly_history_section_handles_none_timestamp():
    """entrée avec 'timestamp': None -> ne doit pas lever de TypeError sur None[:19]."""
    history = {"8f7f2f7e": [{"score": 0.9, "timestamp": None}]}
    html_out = _build_anomaly_history_section(history)
    assert "8f7f2f7e" in html_out


def test_anomaly_history_section_escapes_sensor_id():
    payload = "<b>xss</b>"
    html_out = _build_anomaly_history_section({payload: [{"score": 0.1, "timestamp": "2026-07-25T00:00:00"}]})
    assert "<b>xss</b>" not in html_out
    assert "&lt;b&gt;" in html_out


# ══════════════════════════════════════════════════════════════════════════
#  Intégration — endpoints non couverts par test_api_final.py / test_security.py
# ══════════════════════════════════════════════════════════════════════════

SENSOR_ID = "8f7f2f7e"

IOT_MEASURE = {
    "sensor_id": SENSOR_ID,
    "motor_id": "Motor_1604",
    "temperature": 32.7,
    "vibration_x": 266,
    "vibration_y": 273,
    "vibration_z": 280,
}


def test_iot_predict_basic_flow(base, headers):
    """POST /v1/iot-predict doit accepter une mesure unique et répondre 200
    avec un historique glissant géré côté serveur (fenêtre < 3 -> RUL=None)."""
    r = requests.post(f"{base}/v1/iot-predict", json=IOT_MEASURE, headers=headers, timeout=15)
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["sensor_id"] == SENSOR_ID
    assert d["prediction"] in ("ANOMALY", "NORMAL")
    assert 0.0 <= d["confidence"] <= 1.0
    assert "window_size" in d


def test_iot_predict_requires_auth(base):
    r = requests.post(f"{base}/v1/iot-predict", json=IOT_MEASURE, timeout=15)
    assert r.status_code == 401


def test_alert_level_unknown_sensor_returns_inconnu(base, headers):
    r = requests.get(f"{base}/v1/alert-level/capteur-jamais-vu-xyz", headers=headers, timeout=10)
    assert r.status_code == 200
    d = r.json()
    assert d["alert_level"] == "INCONNU"
    assert d["color"] == "gray"


def test_alert_level_known_sensor_after_predict(base, headers):
    # Assure qu'il existe un historique pour SENSOR_ID (test_iot_predict_basic_flow
    # ou la suite test_api_final.py l'a déjà peuplé dans la plupart des cas ;
    # on repousse une mesure ici pour ne pas dépendre de l'ordre d'exécution).
    requests.post(f"{base}/v1/iot-predict", json=IOT_MEASURE, headers=headers, timeout=15)
    r = requests.get(f"{base}/v1/alert-level/{SENSOR_ID}", headers=headers, timeout=10)
    assert r.status_code == 200
    d = r.json()
    assert d["alert_level"] in ("OK", "ATTENTION", "URGENT", "CRITIQUE", "INCONNU")


def test_history_known_sensor_structure(base, headers):
    requests.post(f"{base}/v1/iot-predict", json=IOT_MEASURE, headers=headers, timeout=15)
    r = requests.get(f"{base}/v1/history/{SENSOR_ID}", headers=headers, timeout=10)
    assert r.status_code == 200
    d = r.json()
    assert d["sensor_id"] == SENSOR_ID
    assert isinstance(d["avg_score"], float)
    assert isinstance(d["max_score"], float)
    assert d["trend"] in ("DÉGRADATION", "AMÉLIORATION", "STABLE")


def test_history_unknown_sensor_404(base, headers):
    r = requests.get(f"{base}/v1/history/capteur-jamais-vu-xyz", headers=headers, timeout=10)
    assert r.status_code == 404
