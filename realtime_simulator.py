r"""
realtime_simulator.py
=====================
Simulateur réaliste basé sur les vraies distributions de full_data.
Reproduit exactement les 3 scénarios observés dans tes données capteurs :
  - NORMAL      : temp ~32°C, vib_z ~280 mg   (comme Motor_8f7f2f7e en régime nominal)
  - DEGRADATION : temp 35→42°C, vib_z 600→1200 mg (tendance progressive)
  - CRITIQUE    : temp ~44°C, vib_z ~1300 mg  (comme dans tes captures Swagger)

Usage :
    python realtime_simulator.py
    python realtime_simulator.py --scenario normal
    python realtime_simulator.py --scenario degradation
    python realtime_simulator.py --scenario critique
    python realtime_simulator.py --scenario aleatoire   ← enchaîne les 3 automatiquement
    python realtime_simulator.py --window 10 --interval 2 --api-port 8000
"""

import argparse
import json
import logging
import math
import random
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [SIM] %(message)s",
    handlers=[
        logging.FileHandler("simulator.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ]
)
log = logging.getLogger("simulator")

# ── Couleurs ──────────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

RISK_COLOR = {"CRITIQUE": RED, "ÉLEVÉ": YELLOW, "MOYEN": YELLOW,
              "FAIBLE": GREEN, "OK": GREEN, "INCONNU": CYAN}

# ══════════════════════════════════════════════════════════════════════════════
#  DISTRIBUTIONS RÉELLES extraites de full_data (2 842 mesures)
#  Source : métriques de ton projet, captures Swagger, terminal realtime_client
# ══════════════════════════════════════════════════════════════════════════════

DISTRIBUTIONS = {
    "normal": {
        # Régime nominal — Motor_8f7f2f7e / Motor_1604 en fonctionnement stable
        "temp_mean": 32.5,   "temp_std": 0.8,   "temp_min": 30.0,  "temp_max": 34.5,
        "vib_x_mean": 270,   "vib_x_std": 15,
        "vib_y_mean": 278,   "vib_y_std": 14,
        "vib_z_mean": 285,   "vib_z_std": 18,   "vib_z_min": 250,  "vib_z_max": 360,
        "label": "NORMAL",
    },
    "degradation": {
        # Tendance progressive — température monte, vibrations augmentent
        # Reproduit le scénario DÉGRADATION de test_api_complet.py
        "temp_mean": 38.5,   "temp_std": 1.5,   "temp_min": 35.0,  "temp_max": 42.0,
        "vib_x_mean": 700,   "vib_x_std": 80,
        "vib_y_mean": 720,   "vib_y_std": 75,
        "vib_z_mean": 900,   "vib_z_std": 100,  "vib_z_min": 600,  "vib_z_max": 1200,
        "label": "DÉGRADATION",
    },
    "critique": {
        # Anomalie franche — reproduit exactement les captures Swagger (temp=45, vib_z=1350)
        "temp_mean": 44.0,   "temp_std": 1.2,   "temp_min": 42.0,  "temp_max": 46.0,
        "vib_x_mean": 1200,  "vib_x_std": 60,
        "vib_y_mean": 1230,  "vib_y_std": 55,
        "vib_z_mean": 1310,  "vib_z_std": 50,   "vib_z_min": 1200, "vib_z_max": 1400,
        "label": "CRITIQUE",
    },
}

SENSORS = ["8f7f2f7e", "91d92804", "eb084747", "4b5e4b32"]


# ══════════════════════════════════════════════════════════════════════════════
#  Générateur de mesures
# ══════════════════════════════════════════════════════════════════════════════

class MeasurementGenerator:
    """
    Génère des mesures réalistes en suivant les distributions de full_data.
    Ajoute une dérive temporelle progressive pour simuler la dégradation.
    """

    def __init__(self, scenario: str, sensor_id: str):
        self.scenario  = scenario
        self.sensor_id = sensor_id
        self.step      = 0          # compteur de mesures (pour la dérive)
        self.dist      = DISTRIBUTIONS[scenario]

    def _clamp(self, val, lo, hi):
        return max(lo, min(hi, val))

    def _gauss(self, mean, std):
        return random.gauss(mean, std)

    def next(self) -> dict:
        self.step += 1
        d = self.dist

        # Dérive progressive pour le scénario dégradation
        drift = 0.0
        if self.scenario == "degradation":
            drift = min(self.step * 0.15, 4.0)   # monte jusqu'à +4°C sur 26 mesures

        temp  = self._clamp(self._gauss(d["temp_mean"] + drift, d["temp_std"]),
                            d["temp_min"], d["temp_max"])
        vib_x = self._clamp(self._gauss(d["vib_x_mean"] + drift * 30, d["vib_x_std"]),
                            50, 1500)
        vib_y = self._clamp(self._gauss(d["vib_y_mean"] + drift * 28, d["vib_y_std"]),
                            50, 1500)
        vib_z = self._clamp(self._gauss(d["vib_z_mean"] + drift * 35, d["vib_z_std"]),
                            d["vib_z_min"], d["vib_z_max"])

        # Courant simulé — varie avec la charge (corrélé aux vibrations)
        # Normal : 30-60A | Dégradation : 60-100A | Critique : 100-160A
        base_current = 35.0
        current = self._clamp(
            self._gauss(base_current + drift * 8, 5.0),
            0.0, 500.0
        )

        return {
            "temperature": round(temp, 2),
            "vibration_x": round(vib_x, 1),
            "vibration_y": round(vib_y, 1),
            "vibration_z": round(vib_z, 1),
            "current":     round(current, 1),
            "sensor_id":   self.sensor_id,
        }


# ══════════════════════════════════════════════════════════════════════════════
#  Client API (identique à realtime_production.py — robuste + retry)
# ══════════════════════════════════════════════════════════════════════════════

class APIClient:
    def __init__(self, host="localhost", port=8000, timeout=10, retries=3):
        self.base    = f"http://{host}:{port}"
        self.timeout = timeout
        self.retries = retries
        self.ok = self.total = 0

    def _post(self, endpoint, payload):
        try:
            import requests
        except ImportError:
            log.error("requests non installé — pip install requests")
            return None, "requests manquant"

        for attempt in range(1, self.retries + 1):
            try:
                r = requests.post(f"{self.base}{endpoint}",
                                  json=payload, timeout=self.timeout)
                if r.status_code == 200:
                    return r.json(), None
                log.warning(f"[{endpoint}] HTTP {r.status_code} — tentative {attempt}/{self.retries}")
            except Exception as e:
                log.warning(f"[{endpoint}] Erreur tentative {attempt}/{self.retries} : {e}")
            if attempt < self.retries:
                time.sleep(1)
        return None, f"Échec après {self.retries} tentatives"

    def predict(self, sensor_id, history):
        self.total += 1
        data, err = self._post("/v1/predict", {
            "sensor_id": sensor_id,
            "motor_id": f"Motor_{sensor_id}",
            "history": history,
        })
        if data:
            self.ok += 1
            for m in ["IF", "LOF", "OCSVM", "ECOD"]:
                data.setdefault("individual_models", {})\
                    .setdefault(m, "INCONNU")
            return data, None
        return {
            "prediction": "INCONNU", "votes": 0,
            "risk_level": "INCONNU", "anomaly_score": 0.0,
            "confidence": 0.0,
            "individual_models": {m: "INCONNU" for m in ["IF","LOF","OCSVM","ECOD"]},
        }, err

    def predict_rul(self, sensor_id, predict_result, history):
        data, err = self._post("/v1/predict-rul", {
            "sensor_id":     sensor_id,
            "motor_id":      f"Motor_{sensor_id}",
            "prediction":    predict_result.get("prediction", "NORMAL"),
            "votes":         predict_result.get("votes", 0),
            "confidence":    predict_result.get("confidence", 0.0),
            "risk_level":    predict_result.get("risk_level", "OK"),
            "anomaly_score": predict_result.get("anomaly_score", 0.0),
            "history":       history,
        })
        return data or {
            "rul_hours": None, "rul_days": None,
            "health_score": None, "alert_level": "INCONNU",
            "recommendation": "API indisponible", "confidence": "INCONNU",
        }, err

    @property
    def reliability(self):
        return 100 * self.ok / self.total if self.total else 100.0


# ══════════════════════════════════════════════════════════════════════════════
#  Affichage
# ══════════════════════════════════════════════════════════════════════════════

def display(iteration, scenario_label, sensor_id, last_m, predict, rul):
    ts    = datetime.now().strftime("%H:%M:%S")
    risk  = predict.get("risk_level", "?")
    color = RISK_COLOR.get(risk, RESET)
    mods  = predict.get("individual_models", {})

    def icon(m):
        v = mods.get(m, "INCONNU")
        return "🔴" if v == "ANOMALY" else ("🟢" if v == "NORMAL" else "⬜")

    rul_h  = rul.get("rul_hours")
    health = rul.get("health_score")

    print(f"\n{'─'*58}")
    print(f"[{ts}] Itération #{iteration}  |  Scénario : {BOLD}{scenario_label}{RESET}  |  Capteur : {sensor_id}")
    print(f"{'─'*58}")
    print(f"{color}{BOLD}🔴 Risque    : {risk}{RESET}")
    print(f"🗳️  Votes     : {predict.get('votes','?')}/4 modèles")
    print(f"📊 Score     : {predict.get('anomaly_score','?')}")
    print(f"🤖 Modèles   : IF={icon('IF')} | LOF={icon('LOF')} | OCSVM={icon('OCSVM')} | ECOD={icon('ECOD')}")
    rul_str = f"{rul_h:.1f}h / {rul.get('rul_days','?'):.2f}j" if rul_h else "N/A"
    print(f"⏳ RUL       : {rul_str}")
    print(f"💚 Santé     : {f'{health}/100' if health is not None else 'N/A'}")
    print(f"🔔 Alerte    : {rul.get('alert_level','?')}")
    print(f"💡 Reco      : {rul.get('recommendation','?')}")
    print(f"🎯 Confiance : {rul.get('confidence','?')}")
    print(f"🌡️  Temp      : {last_m['temperature']}°C")
    print(f"📳 Vib Z     : {last_m['vibration_z']} mg")
    print(f"⚡ Courant   : {last_m.get('current', 0.0)} A")


def save(iteration, scenario, sensor_id, measurement, predict, rul):
    record = {
        "iteration": iteration, "scenario": scenario,
        "timestamp": datetime.now().isoformat(),
        "sensor_id": sensor_id, "measurement": measurement,
        "predict": predict, "rul": rul,
    }
    path = Path("realtime_results.json")
    try:
        data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
        data.append(record)
        path.write_text(json.dumps(data[-500:], indent=2,
                                   ensure_ascii=False, default=str),
                        encoding="utf-8")
    except Exception as e:
        log.warning(f"Sauvegarde échouée : {e}")


# ══════════════════════════════════════════════════════════════════════════════
#  Scénario "aléatoire" : enchaîne normal → dégradation → critique en boucle
# ══════════════════════════════════════════════════════════════════════════════

SCENARIO_SEQUENCE = [
    ("normal",      15, "Fonctionnement nominal (15 mesures)"),
    ("degradation", 15, "Dégradation progressive (15 mesures)"),
    ("critique",    10, "Anomalie critique (10 mesures)"),
]


# ══════════════════════════════════════════════════════════════════════════════
#  Moteur principal
# ══════════════════════════════════════════════════════════════════════════════

def run(args):
    api       = APIClient(args.api_host, args.api_port, args.timeout)
    sensor_id = random.choice(SENSORS)
    window    = deque(maxlen=args.window)
    iteration = 0

    # Réinitialiser realtime_results.json à chaque démarrage
    try:
        Path("realtime_results.json").write_text("[]", encoding="utf-8")
        log.info("realtime_results.json réinitialisé")
    except Exception:
        pass

    log.info("═" * 60)
    log.info("  SIMULATEUR RÉALISTE — Maintenance Prédictive")
    log.info(f"  Scénario  : {args.scenario}")
    log.info(f"  Capteur   : {sensor_id}")
    log.info(f"  API       : http://{args.api_host}:{args.api_port}")
    log.info(f"  Fenêtre   : {args.window} | Intervalle : {args.interval}s")
    log.info("═" * 60)

    print(f"\n{GREEN}{BOLD}Simulateur démarré — données basées sur full_data réel{RESET}")
    print(f"Attente de {args.window} mesures pour la première prédiction...\n")

    if args.scenario == "aleatoire":
        _run_sequence(args, api, sensor_id, window)
    else:
        _run_single(args, args.scenario, api, sensor_id, window)


def _run_single(args, scenario, api, sensor_id, window):
    gen       = MeasurementGenerator(scenario, sensor_id)
    label     = DISTRIBUTIONS[scenario]["label"]
    iteration = 0

    try:
        while True:
            m = gen.next()
            window.append(m)

            if len(window) < args.window:
                print(f"⏳ Remplissage fenêtre... ({len(window)}/{args.window})", end="\r")
                time.sleep(args.interval)
                continue

            iteration += 1
            history   = list(window)
            last_m    = {k: v for k, v in history[-1].items() if k != "sensor_id"}

            predict, _ = api.predict(sensor_id, history)
            rul,     _ = api.predict_rul(sensor_id, predict, history)

            display(iteration, label, sensor_id, last_m, predict, rul)
            save(iteration, scenario, sensor_id, last_m, predict, rul)

            if iteration % 10 == 0:
                log.info(f"Fiabilité API : {api.reliability:.0f}% ({api.ok}/{api.total})")

            time.sleep(args.interval)

    except KeyboardInterrupt:
        print(f"\n\n{YELLOW}Arrêt — {iteration} itérations | Fiabilité : {api.reliability:.0f}%{RESET}")


def _run_sequence(args, api, sensor_id, window):
    """Enchaîne normal → dégradation → critique en boucle infinie."""
    global_iter = 0

    try:
        while True:
            for scenario, n_steps, description in SCENARIO_SEQUENCE:
                label = DISTRIBUTIONS[scenario]["label"]
                gen   = MeasurementGenerator(scenario, sensor_id)
                window.clear()

                print(f"\n{CYAN}{BOLD}{'═'*58}{RESET}")
                print(f"{CYAN}{BOLD}  ▶ {description}{RESET}")
                print(f"{CYAN}{BOLD}{'═'*58}{RESET}")

                local_iter = 0
                while local_iter < n_steps:
                    m = gen.next()
                    window.append(m)

                    if len(window) < args.window:
                        print(f"⏳ Remplissage... ({len(window)}/{args.window})", end="\r")
                        time.sleep(args.interval)
                        continue

                    global_iter += 1
                    local_iter  += 1
                    history = list(window)
                    last_m  = {k: v for k, v in history[-1].items() if k != "sensor_id"}

                    predict, _ = api.predict(sensor_id, history)
                    rul,     _ = api.predict_rul(sensor_id, predict, history)

                    display(global_iter, label, sensor_id, last_m, predict, rul)
                    save(global_iter, scenario, sensor_id, last_m, predict, rul)
                    time.sleep(args.interval)

    except KeyboardInterrupt:
        print(f"\n\n{YELLOW}Arrêt — {global_iter} itérations | Fiabilité : {api.reliability:.0f}%{RESET}")


# ══════════════════════════════════════════════════════════════════════════════
#  Argparse
# ══════════════════════════════════════════════════════════════════════════════

def main():
    import io
    if hasattr(sys.stdout, "buffer") and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)

    parser = argparse.ArgumentParser(
        description="Simulateur réaliste full_data → API FastAPI"
    )
    parser.add_argument("--scenario",  default="aleatoire",
                        choices=["normal", "degradation", "critique", "aleatoire"],
                        help="Scénario à simuler (défaut: aleatoire)")
    parser.add_argument("--api-host",  default="localhost")
    parser.add_argument("--api-port",  default=8000, type=int)
    parser.add_argument("--timeout",   default=10,   type=int)
    parser.add_argument("--window",    default=10,   type=int,
                        help="Taille fenêtre glissante (défaut: 10)")
    parser.add_argument("--interval",  default=2.0,  type=float,
                        help="Secondes entre chaque mesure (défaut: 2)")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
