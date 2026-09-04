"""
test_rul_heuristic_and_history.py
==================================
Tests écrits lors de l'audit du 2026-08-27 : couvrent les changements de cette
session qui n'étaient pas encore testés automatiquement.

  1. RUL 100% heuristique : le modèle ML entraîné sur des courbes de
     dégradation synthétiques (Weibull, train_rul_model.py) a été retiré du
     pipeline de production -- ces tests verrouillent qu'aucune trace du
     modèle ML ne réapparaît dans la réponse de /v1/predict-rul.
  2. measurement_timestamp : le vrai horodatage de la mesure (distinct de
     l'heure de calcul) doit être renvoyé tel quel par /v1/predict et
     /v1/predict-rul -- c'est ce qui permet au dashboard de distinguer une
     mesure fraîche (temps réel) d'une mesure rejouée (mode --replay).
  3. Nouveaux endpoints d'historique : /v1/kpi-history, /v1/tasks-history,
     /v1/pipeline/jobs/history.

Usage :
    pytest tests/test_rul_heuristic_and_history.py -q
"""

import requests

SENSOR_ID = "8f7f2f7e"

HISTORY_5PTS = [
    {"timestamp": "2026-05-18T13:00:00", "temperature": 35 + i, "vibration_x": 300,
     "vibration_y": 200, "vibration_z": 400 + i * 20}
    for i in range(5)
]


# ══════════════════════════════════════════════════════════════════════════
#  RUL 100% heuristique (plus de modèle ML Weibull)
# ══════════════════════════════════════════════════════════════════════════

def test_predict_rul_uses_heuristic_only(base, headers):
    """trend.rul_model doit valoir exactement 'heuristic_CDC' -- toute autre
    valeur (ex: 'heuristic_CDC + ML_...') indiquerait que le modèle Weibull
    influence encore le résultat."""
    payload = {"sensor_id": SENSOR_ID, "history": HISTORY_5PTS}
    r = requests.post(f"{base}/v1/predict-rul", json=payload, headers=headers, timeout=15)
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["trend"]["rul_model"] == "heuristic_CDC"


def test_predict_rul_response_has_no_ml_fields(base, headers):
    """Régression : d'anciens champs liés au modèle ML (ex: model_type) ne
    doivent plus apparaître nulle part dans la réponse."""
    payload = {"sensor_id": SENSOR_ID, "history": HISTORY_5PTS}
    r = requests.post(f"{base}/v1/predict-rul", json=payload, headers=headers, timeout=15)
    assert r.status_code == 200, r.text
    assert "model_type" not in r.text
    assert "GradientBoosting" not in r.text


# ══════════════════════════════════════════════════════════════════════════
#  measurement_timestamp -- distinction LIVE vs REJEU
# ══════════════════════════════════════════════════════════════════════════

def test_predict_measurement_timestamp_matches_input(base, headers):
    """measurement_timestamp doit être le vrai timestamp de la dernière
    mesure envoyée (2026-05-18T13:00:00 ici), jamais l'heure de calcul."""
    payload = {"sensor_id": SENSOR_ID, "history": HISTORY_5PTS}
    r = requests.post(f"{base}/v1/predict", json=payload, headers=headers, timeout=15)
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["measurement_timestamp"] == "2026-05-18T13:00:00"
    # L'heure de calcul, elle, doit rester proche de "maintenant" -- donc
    # différente du timestamp de mesure fourni ci-dessus.
    assert d["timestamp"] != d["measurement_timestamp"]


def test_predict_rul_measurement_timestamp_matches_input(base, headers):
    payload = {"sensor_id": SENSOR_ID, "history": HISTORY_5PTS}
    r = requests.post(f"{base}/v1/predict-rul", json=payload, headers=headers, timeout=15)
    assert r.status_code == 200, r.text
    assert r.json()["measurement_timestamp"] == "2026-05-18T13:00:00"


def test_predict_measurement_timestamp_none_when_absent(base, headers):
    """Si le client n'envoie aucun timestamp, measurement_timestamp doit être
    None plutôt qu'une valeur inventée."""
    history_no_ts = [
        {"temperature": 35 + i, "vibration_x": 300, "vibration_y": 200, "vibration_z": 400 + i * 20}
        for i in range(5)
    ]
    payload = {"sensor_id": SENSOR_ID, "history": history_no_ts}
    r = requests.post(f"{base}/v1/predict", json=payload, headers=headers, timeout=15)
    assert r.status_code == 200, r.text
    assert r.json()["measurement_timestamp"] is None


# ══════════════════════════════════════════════════════════════════════════
#  Nouveaux endpoints d'historique
# ══════════════════════════════════════════════════════════════════════════

def test_kpi_history_public_no_key_required(base):
    """GET /v1/kpi-history est public (rate-limité seulement, comme /sensors)."""
    r = requests.get(f"{base}/v1/kpi-history?limit=5", timeout=10)
    assert r.status_code == 200, r.text
    d = r.json()
    assert "snapshots" in d
    assert isinstance(d["snapshots"], list)


def test_kpi_history_snapshot_shape(base):
    r = requests.get(f"{base}/v1/kpi-history?limit=1", timeout=10)
    assert r.status_code == 200
    d = r.json()
    if d["snapshots"]:
        snap = d["snapshots"][-1]
        for key in ("timestamp", "n_sensors", "avg_score", "avg_health",
                    "n_ok", "n_attention", "n_urgent", "n_critical"):
            assert key in snap, f"clé manquante dans un instantané KPI : {key}"


def test_tasks_history_requires_auth(base):
    r = requests.get(f"{base}/v1/tasks-history", timeout=10)
    assert r.status_code == 401


def test_tasks_history_authenticated_shape(base, headers):
    r = requests.get(f"{base}/v1/tasks-history?limit=10", headers=headers, timeout=10)
    assert r.status_code == 200, r.text
    d = r.json()
    assert "total" in d and "counts" in d and "events" in d
    for e in d["events"]:
        assert e["type"] in ("pipeline", "alert", "report")


def test_pipeline_jobs_history_requires_auth(base):
    r = requests.get(f"{base}/v1/pipeline/jobs/history", timeout=10)
    assert r.status_code == 401


def test_pipeline_jobs_history_authenticated_shape(base, headers):
    r = requests.get(f"{base}/v1/pipeline/jobs/history?limit=5", headers=headers, timeout=10)
    assert r.status_code == 200, r.text
    d = r.json()
    assert "total" in d and "jobs" in d
