"""
test_security.py
================
Tests d'intégration ciblés sur la sécurité de l'API :
  1. Auth — 401 sans clé, 200 avec clé valide, 401 avec clé invalide
  2. Rate limiting — 429 après dépassement de quota
  3. CORS — en-têtes présents
  4. Endpoints publics — /health et /docs accessibles sans clé

Prérequis : API démarrée avec API_KEYS définie.

Usage :
    pytest tests/test_security.py -v --api-key <votre_cle>
    pytest tests/test_security.py -v  # utilise API_KEYS depuis l'env
"""

import time
import pytest
import requests


# ── Payload minimal valide pour /v1/predict ────────────────────────────────────
_PAYLOAD_PREDICT = {
    "sensor_id": "8f7f2f7e",
    "motor_id": "Motor_test",
    "history": [
        {"temperature": 32.0, "vibration_x": 265.0, "vibration_y": 270.0, "vibration_z": 275.0}
        for _ in range(5)
    ],
}

_PAYLOAD_RUL = {
    "sensor_id": "8f7f2f7e",
    "history": [
        {"temperature": 32.0, "vibration_x": 265.0, "vibration_y": 270.0, "vibration_z": 275.0}
        for _ in range(5)
    ],
}


# ══════════════════════════════════════════════════════════════════════════════
#  1. Endpoints PUBLICS — accessibles sans clé
# ══════════════════════════════════════════════════════════════════════════════

class TestEndpointsPublics:
    def test_health_sans_cle(self, base):
        """GET /health doit retourner 200 sans X-API-Key (utilisé par Docker healthcheck)."""
        r = requests.get(f"{base}/health", timeout=5)
        assert r.status_code == 200
        assert r.json().get("status") == "ok"

    def test_root_sans_cle(self, base):
        """GET / doit retourner 200 sans X-API-Key."""
        r = requests.get(f"{base}/", timeout=5)
        assert r.status_code == 200

    def test_docs_sans_cle(self, base):
        """GET /docs (Swagger UI) doit être accessible sans clé."""
        r = requests.get(f"{base}/docs", timeout=5)
        assert r.status_code == 200

    def test_redoc_sans_cle(self, base):
        """GET /redoc doit être accessible sans clé."""
        r = requests.get(f"{base}/redoc", timeout=5)
        assert r.status_code == 200


# ══════════════════════════════════════════════════════════════════════════════
#  2. Authentification — X-API-Key
# ══════════════════════════════════════════════════════════════════════════════

class TestAuthentification:
    def test_predict_sans_cle_retourne_401(self, base):
        """POST /v1/predict sans X-API-Key → 401 Unauthorized."""
        r = requests.post(f"{base}/v1/predict", json=_PAYLOAD_PREDICT, timeout=10)
        assert r.status_code == 401, f"Attendu 401, reçu {r.status_code}"

    def test_predict_cle_invalide_refusee(self, base):
        """
        POST /v1/predict avec X-API-Key incorrect → accès refusé.

        Codes attendus :
          401 — clé invalide (serveur configuré avec API_KEYS)
          503 — aucune clé configurée côté serveur (serveur démarré sans API_KEYS)
        Dans les deux cas l'accès est bloqué : c'est le comportement correct.
        """
        r = requests.post(
            f"{base}/v1/predict",
            json=_PAYLOAD_PREDICT,
            headers={"X-API-Key": "cle-invalide-xxxxxxxxxxx"},
            timeout=10,
        )
        assert r.status_code in (401, 503), (
            f"Attendu 401 ou 503 (accès refusé), reçu {r.status_code}. "
            "Un 200 signifierait que l'auth ne fonctionne pas !"
        )

    def test_predict_cle_valide_retourne_200(self, base, headers, api_key):
        """POST /v1/predict avec X-API-Key valide → 200."""
        if not api_key:
            pytest.skip("Pas de clé API disponible — passe --api-key <cle>")
        r = requests.post(
            f"{base}/v1/predict",
            json=_PAYLOAD_PREDICT,
            headers=headers,
            timeout=60,  # modèles ML lourds — première inférence peut prendre du temps
        )
        assert r.status_code == 200, f"Attendu 200, recu {r.status_code} : {r.text[:200]}"

    def test_metrics_public_sans_cle(self, base):
        """GET /metrics sans X-API-Key → 200 (endpoint public — lecture seule dashboard)."""
        r = requests.get(f"{base}/metrics", timeout=5)
        assert r.status_code == 200, f"Attendu 200, reçu {r.status_code}"

    def test_sensors_public_sans_cle(self, base):
        """GET /sensors sans X-API-Key → 200 (endpoint public — lecture seule dashboard)."""
        r = requests.get(f"{base}/sensors", timeout=5)
        assert r.status_code == 200, f"Attendu 200, reçu {r.status_code}"

    def test_anomalies_public_sans_cle(self, base):
        """GET /anomalies sans X-API-Key → 200 (endpoint public — lecture seule dashboard)."""
        r = requests.get(f"{base}/anomalies", timeout=5)
        assert r.status_code == 200, f"Attendu 200, reçu {r.status_code}"

    def test_health_score_public_sans_cle(self, base):
        """GET /v1/health-score/{id} sans X-API-Key → 200 (endpoint public)."""
        r = requests.get(f"{base}/v1/health-score/8f7f2f7e", timeout=5)
        assert r.status_code in (200, 404), f"Attendu 200/404, reçu {r.status_code}"

    def test_history_public_sans_cle(self, base):
        """GET /v1/history/{id} sans X-API-Key → 200 (endpoint public)."""
        r = requests.get(f"{base}/v1/history/8f7f2f7e", timeout=5)
        assert r.status_code in (200, 404), f"Attendu 200/404, reçu {r.status_code}"

    def test_predict_rul_sans_cle_retourne_401(self, base):
        """POST /v1/predict-rul sans X-API-Key → 401."""
        r = requests.post(f"{base}/v1/predict-rul", json=_PAYLOAD_RUL, timeout=10)
        assert r.status_code == 401

    def test_reponse_401_contient_detail(self, base):
        """Le corps de la réponse 401 doit contenir un message clair."""
        r = requests.post(f"{base}/v1/predict", json=_PAYLOAD_PREDICT, timeout=10)
        body = r.json()
        assert "detail" in body, "La réponse 401 doit avoir un champ 'detail'"
        assert len(body["detail"]) > 5


# ══════════════════════════════════════════════════════════════════════════════
#  3. Rate Limiting
# ══════════════════════════════════════════════════════════════════════════════

class TestRateLimiting:
    def test_predict_rate_limit_429(self, base):
        """
        65 requêtes SIMULTANÉES sur /sensors (limite 60/minute) → au moins une doit
        retourner 429.

        /sensors est public (pas besoin de clé). On utilise ThreadPoolExecutor pour
        envoyer toutes les requêtes en parallèle.
        """
        import concurrent.futures

        def _send(_):
            try:
                return requests.get(f"{base}/sensors", timeout=10).status_code
            except Exception:
                return -1

        with concurrent.futures.ThreadPoolExecutor(max_workers=65) as pool:
            statuses = list(pool.map(_send, range(65)))

        assert 429 in statuses, (
            f"Aucun 429 reçu parmi {len(statuses)} requêtes parallèles sur /sensors — "
            "le rate limiter n'est pas actif (limite attendue : 60/minute)"
        )

    def test_health_non_rate_limite(self, base):
        """GET /health est public et ne doit pas être bloqué même après beaucoup d'appels."""
        statuses = [
            requests.get(f"{base}/health", timeout=5).status_code
            for _ in range(10)
        ]
        assert all(s == 200 for s in statuses), f"Statuts : {statuses}"


# ══════════════════════════════════════════════════════════════════════════════
#  4. CORS
# ══════════════════════════════════════════════════════════════════════════════

class TestCORS:
    def test_cors_header_present_sur_health(self, base):
        """
        La réponse GET /health avec un header Origin doit inclure
        Access-Control-Allow-Origin (ajouté par le CORSMiddleware FastAPI).
        On utilise GET, pas OPTIONS : FastAPI ne route pas les preflight sur /health.
        """
        r = requests.get(
            f"{base}/health",
            headers={"Origin": "http://localhost:3000"},
            timeout=5,
        )
        assert r.status_code == 200
        response_headers_lower = {k.lower(): v for k, v in r.headers.items()}
        assert "access-control-allow-origin" in response_headers_lower, (
            "Le header Access-Control-Allow-Origin est absent de la réponse /health. "
            "Vérifier que CORSMiddleware est bien ajouté à l'app FastAPI."
        )

    def test_cors_header_present_sur_predict(self, base, headers, api_key):
        """La réponse /v1/predict doit inclure le header CORS."""
        if not api_key:
            pytest.skip("Pas de clé API disponible")
        r = requests.post(
            f"{base}/v1/predict",
            json=_PAYLOAD_PREDICT,
            headers={**headers, "Origin": "http://localhost:3000"},
            timeout=15,
        )
        assert "access-control-allow-origin" in {k.lower(): v for k, v in r.headers.items()}
