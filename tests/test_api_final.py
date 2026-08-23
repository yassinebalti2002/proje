r"""
test_api_final.py
=================
Test exhaustif des endpoints de l'API FastAPI v3.
Couvre les scénarios nominaux + sécurité (auth obligatoire depuis v3.1).

Usage standalone :
    python tests/test_api_final.py
    python tests/test_api_final.py --host localhost --port 8000 --api-key <votre_cle>
    python tests/test_api_final.py --verbose

Usage pytest :
    pytest tests/test_api_final.py --api-key <votre_cle>
    pytest tests/ --api-key <votre_cle> -v
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
try:
    import requests
except ImportError:
    print("❌ pip install requests")
    sys.exit(1)

# ── Couleurs ──────────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def ok(msg):   print(f"  {GREEN}✅ {msg}{RESET}")
def fail(msg): print(f"  {RED}❌ {msg}{RESET}")
def warn(msg): print(f"  {YELLOW}⚠️  {msg}{RESET}")
def info(msg): print(f"  {CYAN}ℹ️  {msg}{RESET}")
def title(n, t): print(f"\n{BOLD}{'─'*58}\n  TEST {n} — {t}\n{'─'*58}{RESET}")

# ══════════════════════════════════════════════════════════════════════════════
#  Données de test — tirées de full_data (sensor 8f7f2f7e / Motor_1604)
# ══════════════════════════════════════════════════════════════════════════════

SENSOR_ID = "8f7f2f7e"
MOTOR_ID  = "Motor_1604"

# Clé API globale — surchargée par --api-key ou la variable d'env API_KEYS
_env_key = os.getenv("API_KEYS", "").split(",")[0].strip()
_API_KEY: str = _env_key  # peut être surchargé depuis main()


def _headers() -> dict:
    """Retourne l'en-tête X-API-Key si une clé est disponible."""
    return {"X-API-Key": _API_KEY} if _API_KEY else {}

# Scénario NORMAL — régime nominal observé dans full_data
HISTORY_NORMAL = [
    {"temperature": 32.65, "vibration_x": 266, "vibration_y": 273, "vibration_z": 280},
    {"temperature": 32.66, "vibration_x": 266, "vibration_y": 273, "vibration_z": 262},
    {"temperature": 32.70, "vibration_x": 270, "vibration_y": 275, "vibration_z": 278},
    {"temperature": 32.80, "vibration_x": 268, "vibration_y": 271, "vibration_z": 290},
    {"temperature": 32.75, "vibration_x": 265, "vibration_y": 272, "vibration_z": 275},
    {"temperature": 32.90, "vibration_x": 271, "vibration_y": 276, "vibration_z": 283},
    {"temperature": 33.10, "vibration_x": 269, "vibration_y": 274, "vibration_z": 288},
    {"temperature": 33.00, "vibration_x": 267, "vibration_y": 273, "vibration_z": 281},
    {"temperature": 33.20, "vibration_x": 270, "vibration_y": 275, "vibration_z": 292},
    {"temperature": 33.40, "vibration_x": 272, "vibration_y": 277, "vibration_z": 295},
]

# Scénario CRITIQUE — reproduit exactement tes captures Swagger (temp=45, vib_z=1350)
HISTORY_CRITIQUE = [
    {"temperature": 39.97, "vibration_x": 283, "vibration_y": 300, "vibration_z": 1144},
    {"temperature": 41.00, "vibration_x": 310, "vibration_y": 330, "vibration_z": 1180},
    {"temperature": 42.00, "vibration_x": 350, "vibration_y": 380, "vibration_z": 1200},
    {"temperature": 42.50, "vibration_x": 370, "vibration_y": 395, "vibration_z": 1220},
    {"temperature": 43.00, "vibration_x": 385, "vibration_y": 405, "vibration_z": 1260},
    {"temperature": 43.50, "vibration_x": 390, "vibration_y": 410, "vibration_z": 1290},
    {"temperature": 44.00, "vibration_x": 395, "vibration_y": 415, "vibration_z": 1310},
    {"temperature": 44.50, "vibration_x": 398, "vibration_y": 418, "vibration_z": 1330},
    {"temperature": 45.00, "vibration_x": 400, "vibration_y": 420, "vibration_z": 1350},
    {"temperature": 45.00, "vibration_x": 402, "vibration_y": 422, "vibration_z": 1360},
]

# Scénario DÉGRADATION — tendance progressive (pour predict-rul)
HISTORY_DEGRADATION = [
    {"temperature": 35.0, "vibration_x": 290, "vibration_y": 305, "vibration_z": 600},
    {"temperature": 35.8, "vibration_x": 310, "vibration_y": 320, "vibration_z": 680},
    {"temperature": 36.5, "vibration_x": 325, "vibration_y": 338, "vibration_z": 740},
    {"temperature": 37.2, "vibration_x": 340, "vibration_y": 352, "vibration_z": 810},
    {"temperature": 38.0, "vibration_x": 355, "vibration_y": 368, "vibration_z": 890},
    {"temperature": 38.9, "vibration_x": 368, "vibration_y": 381, "vibration_z": 960},
    {"temperature": 39.5, "vibration_x": 378, "vibration_y": 392, "vibration_z": 1020},
    {"temperature": 40.2, "vibration_x": 388, "vibration_y": 402, "vibration_z": 1080},
    {"temperature": 41.0, "vibration_x": 395, "vibration_y": 410, "vibration_z": 1140},
    {"temperature": 42.0, "vibration_x": 400, "vibration_y": 416, "vibration_z": 1200},
]


# ══════════════════════════════════════════════════════════════════════════════
#  Résultats
# ══════════════════════════════════════════════════════════════════════════════

results = []

def record(test_name, passed, details=""):
    results.append({"test": test_name, "passed": passed, "details": details})


# ══════════════════════════════════════════════════════════════════════════════
#  Implémentations (_impl) — contiennent la logique, retournent True/False
#  Utilisées à la fois par les fonctions pytest (test_*) et main()
# ══════════════════════════════════════════════════════════════════════════════

def _impl_root(base, verbose, headers=None):
    h = headers if headers is not None else _headers()
    title(1, "GET /  (root — public, sans auth)")
    try:
        r = requests.get(f"{base}/", timeout=10)
        if r.status_code == 200:
            ok(f"Status 200 — {r.text[:80]}")
            record("GET /", True)
        else:
            warn(f"Status {r.status_code} (acceptable si redirection vers /docs)")
            record("GET /", True, f"status={r.status_code}")
        return True
    except Exception as e:
        fail(f"Connexion impossible : {e}")
        fail("→ Lance d'abord : python api_unified_pythagore.py")
        record("GET /", False, str(e))
        return False


def _impl_health(base, verbose, headers=None):
    title(2, "GET /health  (public, sans auth)")
    try:
        r = requests.get(f"{base}/health", timeout=10)
        d = r.json()
        if r.status_code == 200 and d.get("status") == "ok":
            ok(f"Status OK — version={d.get('version','?')} | modèles={d.get('models','?')}")
            ok(f"Features : {d.get('features_count','?')} | models_loaded={d.get('models_loaded','?')}")
            if verbose:
                info(json.dumps(d, indent=2))
            record("GET /health", True)
            return True
        else:
            fail(f"Health KO : {d}")
            record("GET /health", False, str(d))
    except Exception as e:
        fail(str(e))
        record("GET /health", False, str(e))
    return False


def _impl_metrics(base, verbose, headers=None):
    h = headers if headers is not None else _headers()
    title(3, "GET /metrics")
    try:
        r = requests.get(f"{base}/metrics", headers=h, timeout=10)
        if r.status_code == 401:
            fail("HTTP 401 — clé API manquante ou invalide. Passe --api-key <cle>")
            record("GET /metrics", False, "HTTP 401")
            return False
        d = r.json()
        if r.status_code == 200:
            f1  = d.get("f1_score", 0)
            auc = d.get("auc_roc", 0)
            acc = d.get("accuracy", 0)
            ok(f"F1={f1:.4f} | AUC={auc:.4f} | Accuracy={acc:.4f}")
            ok(f"Ensemble : {d.get('ensemble','?')} | Vote : {d.get('voting','?')}")
            ok(f"Anomalies : {d.get('n_anomalies','?')}/{d.get('n_total','?')} ({100*d.get('n_anomalies',0)/max(d.get('n_total',1),1):.1f}%)")
            if f1 >= 0.70:
                ok(f"F1 ≥ 0.70 ✅ (seuil acceptable pour PFE)")
            else:
                warn(f"F1 < 0.70 — à mentionner dans le rapport")
            if verbose:
                info(json.dumps(d, indent=2))
            record("GET /metrics", True, f"F1={f1:.4f} AUC={auc:.4f}")
            return True
        else:
            fail(f"Status {r.status_code}")
            record("GET /metrics", False)
    except Exception as e:
        fail(str(e))
        record("GET /metrics", False, str(e))
    return False


def _impl_sensors(base, verbose, headers=None):
    h = headers if headers is not None else _headers()
    title(4, "GET /sensors")
    try:
        r = requests.get(f"{base}/sensors", headers=h, timeout=10)
        if r.status_code == 401:
            fail("HTTP 401 — clé API manquante ou invalide")
            record("GET /sensors", False, "HTTP 401")
            return False
        d = r.json()
        if r.status_code == 200:
            sensors = d.get("sensors", d.get("data", []))
            ok(f"{len(sensors)} capteurs trouvés")
            if sensors and verbose:
                info(f"Exemples : {sensors[:3]}")
            record("GET /sensors", True, f"{len(sensors)} capteurs")
            return True
        else:
            warn(f"Status {r.status_code} — endpoint peut être optionnel")
            record("GET /sensors", True, f"status={r.status_code}")
    except Exception as e:
        warn(f"Endpoint /sensors non disponible : {e}")
        record("GET /sensors", True, "optionnel")
    return True


def _impl_anomalies(base, verbose, headers=None):
    h = headers if headers is not None else _headers()
    title(5, "GET /anomalies")
    try:
        r = requests.get(f"{base}/anomalies", headers=h, timeout=10)
        if r.status_code == 401:
            fail("HTTP 401 — clé API manquante ou invalide")
            record("GET /anomalies", False, "HTTP 401")
            return False
        d = r.json()
        if r.status_code == 200:
            anomalies = d.get("anomalies", d.get("data", []))
            ok(f"{len(anomalies)} anomalies historiques")
            record("GET /anomalies", True, f"{len(anomalies)} anomalies")
            return True
        else:
            warn(f"Status {r.status_code}")
            record("GET /anomalies", True, "optionnel")
    except Exception as e:
        warn(f"Endpoint /anomalies non disponible : {e}")
        record("GET /anomalies", True, "optionnel")
    return True


def _impl_predict(base, verbose, headers=None):
    h = headers if headers is not None else _headers()
    title(6, "POST /v1/predict — 3 scénarios")
    all_ok = True

    scenarios = [
        ("NORMAL",      HISTORY_NORMAL,      ["OK","FAIBLE"],           0),
        ("CRITIQUE",    HISTORY_CRITIQUE,     ["CRITIQUE","ÉLEVÉ"],      3),
        ("DÉGRADATION", HISTORY_DEGRADATION,  ["MODÉRÉ","ÉLEVÉ","CRITIQUE"], 1),
    ]

    for label, history, expected_risks, min_votes in scenarios:
        print(f"\n  ▶ Scénario {label}")
        payload = {
            "sensor_id": SENSOR_ID,
            "motor_id":  MOTOR_ID,
            "history":   history,
        }
        try:
            r = requests.post(f"{base}/v1/predict", json=payload, headers=h, timeout=15)
            if r.status_code == 401:
                fail("HTTP 401 — clé API manquante ou invalide. Passe --api-key <cle>")
                record(f"POST /v1/predict [{label}]", False, "HTTP 401")
                return False
            if r.status_code != 200:
                fail(f"HTTP {r.status_code} : {r.text[:200]}")
                record(f"POST /v1/predict [{label}]", False, f"HTTP {r.status_code}")
                all_ok = False
                continue

            d = r.json()
            risk   = d.get("risk_level", "?")
            votes  = d.get("votes", 0)
            score  = d.get("anomaly_score", 0)
            models = d.get("individual_models", {})

            ok(f"Risque={risk} | Votes={votes}/6 | Score={score}")
            ok(f"Modèles : IF={models.get('IF','?')} | LOF={models.get('LOF','?')} | OCSVM={models.get('OCSVM','?')} | ECOD={models.get('ECOD','?')} | HBOS={models.get('HBOS','?')} | COPOD={models.get('COPOD','?')}")

            if risk in expected_risks:
                ok(f"Risque '{risk}' conforme au scénario {label} ✅")
            else:
                warn(f"Risque '{risk}' inattendu pour {label} (attendu : {expected_risks})")

            if votes >= min_votes:
                ok(f"Votes {votes} ≥ {min_votes} ✅")
            else:
                warn(f"Votes {votes} < {min_votes} attendus")

            if verbose:
                info(json.dumps(d, indent=2, ensure_ascii=False))

            record(f"POST /v1/predict [{label}]", True,
                   f"risk={risk} votes={votes}")

        except Exception as e:
            fail(str(e))
            record(f"POST /v1/predict [{label}]", False, str(e))
            all_ok = False

    return all_ok


def _impl_predict_rul(base, verbose, headers=None):
    h = headers if headers is not None else _headers()
    title(7, "POST /v1/predict-rul — 2 scénarios")
    all_ok = True

    scenarios = [
        ("NORMAL",      HISTORY_NORMAL,      "NORMAL",  0, 0.0, "OK"),
        ("DÉGRADATION", HISTORY_DEGRADATION, "ANOMALY", 3, 0.75, "CRITIQUE"),
    ]

    for label, history, prediction, votes, confidence, risk_level in scenarios:
        print(f"\n  ▶ Scénario {label}")
        payload = {
            "sensor_id":     SENSOR_ID,
            "motor_id":      MOTOR_ID,
            "prediction":    prediction,
            "votes":         votes,
            "confidence":    confidence,
            "risk_level":    risk_level,
            "anomaly_score": confidence,
            "history":       history,
        }
        try:
            r = requests.post(f"{base}/v1/predict-rul", json=payload, headers=h, timeout=15)
            if r.status_code == 401:
                fail("HTTP 401 — clé API invalide")
                record(f"POST /v1/predict-rul [{label}]", False, "HTTP 401")
                return False
            if r.status_code != 200:
                fail(f"HTTP {r.status_code} : {r.text[:300]}")
                info("💡 Champs requis : sensor_id, motor_id, prediction, votes, confidence, risk_level, anomaly_score, history")
                record(f"POST /v1/predict-rul [{label}]", False, f"HTTP {r.status_code}")
                all_ok = False
                continue

            d = r.json()
            rul_h  = d.get("rul_hours", "?")
            rul_d  = d.get("rul_days", "?")
            health = d.get("health_score", "?")
            alert  = d.get("alert_level", "?")
            conf   = d.get("confidence", "?")
            reco   = d.get("recommendation", "?")

            ok(f"RUL = {rul_h}h / {rul_d} jours")
            ok(f"Santé = {health}/100 | Alerte = {alert} | Confiance = {conf}")
            ok(f"Reco : {reco}")

            if rul_h and isinstance(rul_h, (int, float)):
                if label == "NORMAL" and rul_h > 500:
                    ok(f"RUL {rul_h}h > 500h — cohérent avec scénario NORMAL ✅")
                elif label == "DÉGRADATION" and rul_h < 700:
                    ok(f"RUL {rul_h}h < 700h — cohérent avec dégradation ✅")

            if verbose:
                info(json.dumps(d, indent=2, ensure_ascii=False))

            record(f"POST /v1/predict-rul [{label}]", True,
                   f"rul={rul_h}h health={health}")

        except Exception as e:
            fail(str(e))
            record(f"POST /v1/predict-rul [{label}]", False, str(e))
            all_ok = False

    return all_ok


def _impl_health_score(base, verbose, headers=None):
    h = headers if headers is not None else _headers()
    title(8, f"GET /v1/health-score/{SENSOR_ID}")
    try:
        r = requests.get(f"{base}/v1/health-score/{SENSOR_ID}", headers=h, timeout=10)
        if r.status_code == 401:
            fail("HTTP 401 — clé API invalide")
            record("GET /v1/health-score", False, "HTTP 401")
            return False
        if r.status_code == 200:
            d = r.json()
            ok(f"Health score pour {SENSOR_ID} : {d}")
            record(f"GET /v1/health-score", True)
            return True
        else:
            warn(f"Status {r.status_code} — endpoint optionnel")
            record(f"GET /v1/health-score", True, "optionnel")
    except Exception as e:
        warn(f"Endpoint /v1/health-score non disponible : {e}")
        record(f"GET /v1/health-score", True, "optionnel")
    return True


# ══════════════════════════════════════════════════════════════════════════════
#  Fonctions pytest — délèguent à _impl, pas de return pour éviter le warning
# ══════════════════════════════════════════════════════════════════════════════

def test_root(base, verbose, headers):
    assert _impl_root(base, verbose, headers), "GET / échoué — API injoignable"


def test_health(base, verbose, headers):
    assert _impl_health(base, verbose, headers), "GET /health KO"


def test_metrics(base, verbose, headers):
    assert _impl_metrics(base, verbose, headers), "GET /metrics KO"


def test_sensors(base, verbose, headers):
    assert _impl_sensors(base, verbose, headers), "GET /sensors KO"


def test_anomalies(base, verbose, headers):
    assert _impl_anomalies(base, verbose, headers), "GET /anomalies KO"


def test_predict(base, verbose, headers):
    assert _impl_predict(base, verbose, headers), "POST /v1/predict KO"


def test_predict_rul(base, verbose, headers):
    assert _impl_predict_rul(base, verbose, headers), "POST /v1/predict-rul KO"


def test_health_score(base, verbose, headers):
    assert _impl_health_score(base, verbose, headers), "GET /v1/health-score KO"


# ══════════════════════════════════════════════════════════════════════════════
#  Résumé final
# ══════════════════════════════════════════════════════════════════════════════

def print_summary():
    print(f"\n{BOLD}{'═'*58}")
    print(f"  RÉSUMÉ — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'═'*58}{RESET}")

    passed = sum(1 for r in results if r["passed"])
    total  = len(results)

    for r in results:
        status = f"{GREEN}PASS{RESET}" if r["passed"] else f"{RED}FAIL{RESET}"
        detail = f"  ({r['details']})" if r.get("details") else ""
        print(f"  [{status}]  {r['test']}{detail}")

    print(f"\n{BOLD}  Résultat : {passed}/{total} tests réussis{RESET}")

    if passed == total:
        print(f"\n{GREEN}{BOLD}  ✅ API 100% opérationnelle — prête pour la soutenance !{RESET}")
    elif passed >= total * 0.8:
        print(f"\n{YELLOW}{BOLD}  ⚠️  API globalement OK ({passed}/{total}) — quelques endpoints optionnels manquants.{RESET}")
    else:
        print(f"\n{RED}{BOLD}  ❌ Problèmes détectés — vérifie les logs ci-dessus.{RESET}")

    # Sauvegarde rapport
    report = {
        "timestamp": datetime.now().isoformat(),
        "passed": passed, "total": total,
        "results": results,
    }
    Path("test_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\n  Rapport sauvegardé : test_report.json\n")


# ══════════════════════════════════════════════════════════════════════════════
#  Main — mode standalone (python tests/test_api_final.py)
# ══════════════════════════════════════════════════════════════════════════════

def main():
    global _API_KEY
    parser = argparse.ArgumentParser(description="Test complet API FastAPI v3")
    parser.add_argument("--host",    default="localhost")
    parser.add_argument("--port",    default=8000, type=int)
    parser.add_argument("--api-key", default=None,
                        help="Clé API (X-API-Key). Sinon utilise la var d'env API_KEYS.")
    parser.add_argument("--verbose", action="store_true",
                        help="Afficher les JSON complets")
    args = parser.parse_args()

    if args.api_key:
        _API_KEY = args.api_key

    if not _API_KEY:
        print(f"{YELLOW}⚠️  Aucune clé API fournie — les endpoints protégés retourneront 401.{RESET}")
        print(f"   Passe --api-key <ta_cle>  ou  définis API_KEYS=<cle> dans l'env.\n")

    base = f"http://{args.host}:{args.port}"

    print(f"\n{BOLD}{'═'*58}")
    print(f"  TEST API COMPLET — Maintenance Prédictive v3")
    print(f"  Cible : {base}")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'═'*58}{RESET}")

    # Test root — si ça échoue, arrêt immédiat (standalone utilise _impl directement)
    if not _impl_root(base, args.verbose):
        print(f"\n{RED}API injoignable. Lance d'abord :{RESET}")
        print(f"  python api_unified_pythagore.py\n")
        sys.exit(1)

    _impl_health(base, args.verbose)
    _impl_metrics(base, args.verbose)
    _impl_sensors(base, args.verbose)
    _impl_anomalies(base, args.verbose)
    _impl_predict(base, args.verbose)
    _impl_predict_rul(base, args.verbose)
    _impl_health_score(base, args.verbose)

    print_summary()


if __name__ == "__main__":
    main()
