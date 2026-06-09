r"""
realtime_mariadb.py
===================
Moteur de production temps réel : MariaDB IoT → API FastAPI
Chaîne complète : Capteurs IFM → Gateway → MariaDB → prédictions ML

Architecture :
    Capteurs IFM (20x)
        ↓ IO-Link
    Gateway IFM (HTTP REST)
        ↓ INSERT
    MariaDB — table full_data (serveur IoT, réseau local)
        ↓ SELECT WHERE id > last_id  (polling toutes les 2s)
    realtime_mariadb.py  (ta machine)
        ↓ POST /v1/predict + /v1/predict-rul
    API FastAPI v3.1  (localhost:8000)
        ↓ résultats
    Affichage terminal + realtime_results.json

Usage :
    python realtime_mariadb.py
    python realtime_mariadb.py --host 192.168.1.50 --user root --password monpass
    python realtime_mariadb.py --host 192.168.1.50 --database ai_cp --table full_data
"""

import argparse
import io as _io
import json
import logging
import math
import os
import sys
import time
from collections import deque, defaultdict
from datetime import datetime
from pathlib import Path

# V6 : forcer UTF-8 au niveau module pour eviter UnicodeEncodeError sur Windows CP1252
if hasattr(sys.stdout, "buffer") and getattr(sys.stdout, "encoding", "utf-8").lower() != "utf-8":
    sys.stdout = _io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = _io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [MARIADB] %(message)s",
    handlers=[
        logging.FileHandler("realtime_mariadb.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ]
)
log = logging.getLogger("mariadb")

# ── Couleurs terminal ──────────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"
RISK_COLOR = {
    "CRITIQUE":  RED,
    "ÉLEVÉ":     YELLOW,
    "MODÉRÉ":    YELLOW,
    "FAIBLE":    GREEN,
    "OK":        GREEN,
    "INCONNU":   CYAN,
}

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION — modifie ces valeurs ou utilise les arguments CLI
# ══════════════════════════════════════════════════════════════════════════════
DEFAULT_CONFIG = {
    "host":     os.environ.get("MARIADB_HOST",     "192.168.1.50"),
    "port":     int(os.environ.get("MARIADB_PORT", "3306")),
    "user":     os.environ.get("MARIADB_USER",     "root"),
    "password": os.environ.get("MARIADB_PASSWORD", "ton_mot_de_passe"),
    "database": os.environ.get("MARIADB_DATABASE", "ai_cp"),
    "table":    os.environ.get("MARIADB_TABLE",    "full_data"),
}


# Limite de capteurs surveillés simultanément
MAX_SENSORS = 25

# ══════════════════════════════════════════════════════════════════════════════
#  LECTEUR MariaDB — polling incrémental sur full_data
# ══════════════════════════════════════════════════════════════════════════════

class MariaDBReader:
    """
    Lit les nouvelles lignes de full_data par polling incrémental.

    Structure de full_data :
        id            INT          → curseur de polling
        SensorNodeId  VARCHAR      → identifiant capteur (ex: 8f7f2f7e)
        timestamp     DATETIME     → horodatage
        gph           VARCHAR      → type mesure : temperature | vibration_x | vibration_y
        data          LONGTEXT     → JSON avec Temperature, Vibration.RMS.X/Y/Z
        type          VARCHAR      → 'res'

    Stratégie de consolidation :
        3 lignes (gph=temperature + vibration_x + vibration_y) avec le même
        MeasDetails.Id forment une session complète.
        Ce reader regroupe les lignes par MeasDetails.Id avant d'envoyer à l'API.
    """

    def __init__(self, host, port, user, password, database, table):
        self.host     = host
        self.port     = port
        self.user     = user
        self.password = password
        self.database = database
        self.table    = table
        self.conn     = None
        self.cursor   = None
        self.last_id  = 0

        # Buffer de consolidation : MeasDetails.Id → {temp, vib_x, vib_y, vib_z, sensor_id, ts}
        self._pending = defaultdict(dict)
        self._PENDING_TTL = 60  # secondes — purge les sessions incomplètes

    def connect(self):
        """Connexion à MariaDB via mysql-connector-python."""
        try:
            import mysql.connector
        except ImportError:
            log.error("mysql-connector-python non installé")
            log.error("→ pip install mysql-connector-python")
            sys.exit(1)

        try:
            self.conn = mysql.connector.connect(
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password,
                database=self.database,
                connection_timeout=10,
                autocommit=True,
            )
            self.cursor = self.conn.cursor(dictionary=True)
            log.info(f"✅ Connecté à MariaDB : {self.host}:{self.port}/{self.database}/{self.table}")

            # Démarrer à partir de la dernière ligne existante
            self.cursor.execute(f"SELECT MAX(id) as max_id FROM `{self.table}`")
            row = self.cursor.fetchone()
            max_id = row["max_id"] if row and row["max_id"] else 0

            # Mode replay — rejoue les N dernières lignes réelles
            replay = getattr(self, "replay", 0)
            if replay > 0:
                self.last_id = max(0, max_id - replay)
                log.info(f"Mode REPLAY — curseur reculé à id={self.last_id} ({replay} lignes à rejouer)")
                print(f"\n  ▶️  REPLAY MODE — {replay} dernières lignes réelles seront traitées\n")
            else:
                self.last_id = max_id
                log.info(f"Curseur initialisé à id={self.last_id} (seules les nouvelles lignes seront traitées)")

        except Exception as e:
            log.error(f"Connexion MariaDB échouée : {e}")
            log.error(f"  Host     : {self.host}:{self.port}")
            log.error(f"  Database : {self.database}")
            log.error(f"  User     : {self.user}")
            log.error("Causes fréquentes :")
            log.error("  • IP incorrecte → ping 192.168.x.x pour tester")
            log.error("  • Port 3306 bloqué par le pare-feu du serveur IoT")
            log.error("  • Utilisateur sans accès distant → GRANT ALL ON *.* TO 'user'@'%'")
            log.error("  • Mot de passe incorrect")
            sys.exit(1)

    def _reconnect(self):
        """Reconnexion automatique si la connexion est perdue."""
        log.warning("Connexion MariaDB perdue — reconnexion...")
        try:
            self.connect()
            log.info("Reconnexion réussie")
        except Exception as e:
            log.error(f"Reconnexion échouée : {e}")

    def _is_alive(self):
        """Vérifie que la connexion est vivante."""
        try:
            self.conn.ping(reconnect=False)
            return True
        except Exception:
            return False

    def fetch_current(self, sensor_id: str) -> float:
        """
        Lit le courant le plus récent depuis la table motor_mesure.
        Retourne 0.0 si la table n'existe pas ou si le capteur n'a pas de courant.
        Appelé une fois par session complète.
        """
        try:
            self.cursor.execute(
                """
                SELECT courant FROM motor_mesure
                WHERE LOWER(id_cp) = LOWER(%s)
                  AND courant > 0
                ORDER BY date DESC
                LIMIT 1
                """,
                (sensor_id,)
            )
            row = self.cursor.fetchone()
            if row and row.get("courant") is not None:
                return float(row["courant"])
        except Exception:
            pass  # motor_mesure absente ou erreur → courant = 0
        return 0.0

    def poll(self, batch_size=100):
        """
        Lit les nouvelles lignes depuis last_id.
        Consolide les 3 lignes gph en sessions complètes.
        Retourne une liste de sessions prêtes pour l'API.
        """
        if not self._is_alive():
            self._reconnect()
            return []

        try:
            query = f"""
                SELECT id, SensorNodeId, gph, data
                FROM `{self.table}`
                WHERE id > %s
                ORDER BY id ASC
                LIMIT %s
            """
            self.cursor.execute(query, (self.last_id, batch_size))
            rows = self.cursor.fetchall()
        except Exception as e:
            log.error(f"Erreur SELECT : {e}")
            return []

        if not rows:
            return []

        # Purge TTL — évite la fuite mémoire si un capteur n'envoie pas les 3 gph
        now = time.time()
        expired = [k for k, v in self._pending.items()
                   if now - v.get("ts", now) > self._PENDING_TTL]
        for k in expired:
            log.warning(f"Buffer pending expiré (TTL {self._PENDING_TTL}s) → suppression clé {k}")
            del self._pending[k]
        if len(self._pending) > 0:
            log.debug(f"Buffer pending : {len(self._pending)} sessions en attente")

        # Avancer le curseur
        self.last_id = rows[-1]["id"]
        log.debug(f"{len(rows)} nouvelles lignes reçues, curseur → {self.last_id}")

        # Consolider les lignes par MeasDetails.Id
        completed_sessions = []
        for row in rows:
            session = self._process_row(row)
            if session:
                completed_sessions.append(session)

        return completed_sessions

    def _process_row(self, row):
        """
        Parse une ligne full_data et l'intègre dans le buffer de consolidation.
        Retourne une session complète quand les 3 gph sont disponibles, sinon None.
        """
        sensor_id = row.get("SensorNodeId", "unknown")
        gph       = row.get("gph", "")

        # Parser le JSON du champ data
        # Fix : le connecteur MySQL peut retourner data comme str OU comme dict
        try:
            raw = row.get("data") or ""
            if isinstance(raw, dict):
                data = raw
            elif isinstance(raw, (bytes, bytearray)):
                data = json.loads(raw.decode("utf-8"))
            elif isinstance(raw, str) and raw.strip():
                data = json.loads(raw)
            else:
                data = {}
        except (json.JSONDecodeError, TypeError, UnicodeDecodeError):
            # Parfois le JSON est mal formé (double quotes échappées)
            try:
                data = json.loads(str(row["data"]).replace('\\"', '"'))
            except Exception:
                log.warning(f"JSON invalide pour id={row['id']} gph={gph}")
                return None

        # Extraire le MeasDetails.Id pour regrouper les 3 lignes
        # Fix : data peut être str ou dict selon le connecteur MySQL
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except Exception:
                data = {}
        meas_id = data.get("MeasDetails", {}).get("Id")
        if not meas_id:
            meas_id = f"{sensor_id}_{row['id']}"

        key = f"{sensor_id}_{meas_id}"

        # Remplir le buffer selon le type de mesure
        if gph == "temperature":
            self._pending[key]["sensor_id"] = sensor_id
            self._pending[key]["temperature"] = data.get("Temperature")
            self._pending[key].setdefault("ts", time.time())  # horodatage TTL
            # La ligne temperature contient aussi la vibration Z RMS
            vib_rms = data.get("Vibration", {}).get("RMS", {})
            if "Z" in vib_rms:
                self._pending[key]["vibration_z"] = vib_rms["Z"]

        elif gph == "vibration_x":
            vib_rms = data.get("Vibration", {}).get("RMS", {})
            if "X" in vib_rms:
                self._pending[key]["vibration_x"] = vib_rms["X"]

        elif gph == "vibration_y":
            vib_rms = data.get("Vibration", {}).get("RMS", {})
            if "Y" in vib_rms:
                self._pending[key]["vibration_y"] = vib_rms["Y"]

        elif gph == "acceleration":
            vib = data.get("Vibration", {})
            mes = data.get("mesure", {})
            self._pending[key]["sensor_id"] = sensor_id
            # Ancienne structure (mesure.x/y/z)
            if mes:
                self._pending[key]["vibration_x"] = mes.get("x") or self._pending[key].get("vibration_x")
                self._pending[key]["vibration_y"] = mes.get("y") or self._pending[key].get("vibration_y")
                self._pending[key]["vibration_z"] = mes.get("z") or self._pending[key].get("vibration_z")
            # Nouvelles features accélération IFM (A-P2P, A-Z2P, Crest, A-RMS)
            if vib.get("A-P2P", {}).get("Y") is not None:
                self._pending[key]["acc_p2p"]   = float(vib["A-P2P"]["Y"])
            if vib.get("A-Z2P", {}).get("Y") is not None:
                self._pending[key]["acc_z2p"]   = float(vib["A-Z2P"]["Y"])
            if vib.get("Crest", {}).get("Y") is not None:
                self._pending[key]["acc_crest"] = float(vib["Crest"]["Y"])
            if vib.get("A-RMS", {}).get("Y") is not None:
                self._pending[key]["acc_rms"]   = float(vib["A-RMS"]["Y"])

        # Session complète si temp + vib_z + vib_x + vib_y disponibles (3 axes requis)
        s = self._pending[key]
        has_temp = s.get("temperature") is not None
        has_vibz = s.get("vibration_z") is not None
        has_vibx = s.get("vibration_x") is not None
        has_viby = s.get("vibration_y") is not None

        if has_temp and has_vibz and has_vibx and has_viby:
            session = {
                "sensor_id":   s.get("sensor_id", sensor_id),
                "temperature": float(s["temperature"]),
                "vibration_x": float(s.get("vibration_x", 0) or 0),
                "vibration_y": float(s.get("vibration_y", 0) or 0),
                "vibration_z": float(s["vibration_z"]),
            }
            # Accélération depuis gph='acceleration' si disponible (V4+)
            if s.get("acc_p2p")   is not None: session["acc_p2p"]   = float(s["acc_p2p"])
            if s.get("acc_z2p")   is not None: session["acc_z2p"]   = float(s["acc_z2p"])
            if s.get("acc_crest") is not None: session["acc_crest"] = float(s["acc_crest"])
            if s.get("acc_rms")   is not None: session["acc_rms"]   = float(s["acc_rms"])
            # Dérivation acc_p2p depuis vibration RMS (mg → GE) si non fourni par capteur
            # Les capteurs IFM VVB001 transmettent vibration en mg = accélération en milli-g
            # 1 GE = 1000 mg ; P2P ≈ 2√2 × RMS (approximation signal vibratoire)
            if "acc_p2p" not in session:
                import math
                vx = float(s.get("vibration_x", 0) or 0) / 1000.0  # mg → GE
                vy = float(s.get("vibration_y", 0) or 0) / 1000.0
                vz = float(s["vibration_z"]) / 1000.0
                acc_rms_ge = math.sqrt(vx**2 + vy**2 + vz**2)
                session["acc_p2p"]   = round(2 * math.sqrt(2) * acc_rms_ge, 4)
                session["acc_rms"]   = round(acc_rms_ge, 4)
                session["acc_crest"] = round(session["acc_p2p"] / acc_rms_ge, 4) if acc_rms_ge > 0 else 0.0
            # Ajouter courant depuis motor_mesure (V5)
            session["current"] = self.fetch_current(sensor_id)
            # Nettoyer le buffer
            del self._pending[key]
            return session

        return None

    def close(self):
        try:
            if self.conn:
                self.conn.close()
                log.info("Connexion MariaDB fermée proprement")
        except Exception:
            pass


# ══════════════════════════════════════════════════════════════════════════════
#  CLIENT API FastAPI — identique à realtime_production.py
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
            log.error("requests non installé → pip install requests")
            return None, "requests manquant"

        for attempt in range(1, self.retries + 1):
            try:
                r = requests.post(
                    f"{self.base}{endpoint}",
                    json=payload, timeout=self.timeout
                )
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
            "motor_id":  f"Motor_{sensor_id}",
            "history":   history,
        })
        if data:
            self.ok += 1
            for m in ["IF", "LOF", "OCSVM", "ECOD"]:
                data.setdefault("individual_models", {}).setdefault(m, "INCONNU")
            return data, None
        return {
            "prediction": "INCONNU", "votes": 0, "risk_level": "INCONNU",
            "anomaly_score": 0.0, "confidence": 0.0,
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
            "rul_hours": None, "rul_days": None, "health_score": None,
            "alert_level": "INCONNU", "recommendation": "API indisponible",
        }, err

    @property
    def reliability(self):
        return 100 * self.ok / self.total if self.total else 100.0


# ══════════════════════════════════════════════════════════════════════════════
#  AFFICHAGE TERMINAL
# ══════════════════════════════════════════════════════════════════════════════

def display(iteration, sensor_id, last_m, predict, rul):
    ts    = datetime.now().strftime("%H:%M:%S")
    risk  = predict.get("risk_level", "?")
    color = RISK_COLOR.get(risk, RESET)
    mods  = predict.get("individual_models", {})

    def icon(m):
        v = mods.get(m, "INCONNU")
        return "🔴" if v == "ANOMALY" else ("🟢" if v == "NORMAL" else "⬜")

    rul_h  = rul.get("rul_hours")
    health = rul.get("health_score")
    rul_str = f"{rul_h:.1f}h / {rul.get('rul_days', 0):.2f}j" if rul_h else "N/A"

    print(f"\n{'─'*60}")
    print(f"[{ts}]  #{iteration}  |  Capteur : {sensor_id}  |  Source : MariaDB RÉEL")
    print(f"{'─'*60}")
    print(f"{color}{BOLD}🔴 Risque    : {risk}{RESET}")
    print(f"🗳️  Votes     : {predict.get('votes','?')}/4 modèles")
    print(f"📊 Score     : {predict.get('anomaly_score','?')}")
    print(f"🤖 Modèles   : IF={icon('IF')} | LOF={icon('LOF')} | OCSVM={icon('OCSVM')} | ECOD={icon('ECOD')}")
    print(f"⏳ RUL       : {rul_str}")
    print(f"💚 Santé     : {f'{health}/100' if health is not None else 'N/A'}")
    print(f"🔔 Alerte    : {rul.get('alert_level','?')}")
    print(f"💡 Reco      : {rul.get('recommendation','?')}")
    print(f"🌡️  Temp      : {last_m.get('temperature','?')}°C")
    print(f"📳 Vib Z     : {last_m.get('vibration_z','?')} mg")
    print(f"📳 Vib total : {last_m.get('vibration_total','?')} mg")


def save_result(iteration, sensor_id, measurement, predict, rul):
    record = {
        "iteration": iteration,
        "timestamp": datetime.now().isoformat(),
        "sensor_id": sensor_id,
        "source":    "mariadb_realtime",
        "measurement": measurement,
        "predict":   predict,
        "rul":       rul,
    }
    path = Path("realtime_results.json")
    try:
        data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
        data.append(record)
        # Garder les 50 dernières entrées par capteur (évite qu'un seul capteur monopolise le fichier)
        per_sensor = defaultdict(list)
        for e in data:
            per_sensor[e.get("sensor_id", "unknown")].append(e)
        data = [e for entries in per_sensor.values() for e in entries[-50:]]
        data.sort(key=lambda e: e.get("timestamp", ""))
        path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8"
        )
    except Exception as e:
        log.warning(f"Sauvegarde JSON échouée : {e}")


# ══════════════════════════════════════════════════════════════════════════════
#  MOTEUR PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

def run(args):
    log.info("═" * 65)
    log.info("  MOTEUR TEMPS RÉEL — MariaDB IoT → API FastAPI")
    log.info(f"  MariaDB : {args.host}:{args.port}/{args.database}/{args.table}")
    log.info(f"  API     : http://{args.api_host}:{args.api_port}")
    log.info(f"  Fenêtre : {args.window} mesures | Poll : {args.poll}s")
    log.info("═" * 65)

    # Réinitialiser realtime_results.json à chaque démarrage (évite accumulation multi-runs)
    try:
        Path("realtime_results.json").write_text("[]", encoding="utf-8")
        log.info("realtime_results.json réinitialisé")
    except Exception:
        pass

    # Composants
    reader = MariaDBReader(
        host=args.host, port=args.port,
        user=args.user, password=args.password,
        database=args.database, table=args.table,
    )
    reader.replay = getattr(args, 'replay', 0)  # mode replay

    api = APIClient(
        host=args.api_host, port=args.api_port,
        timeout=args.timeout, retries=args.retries,
    )

    # Connexion MariaDB
    reader.connect()

    # Fenêtres glissantes par capteur (chaque capteur a sa propre fenêtre)
    windows = defaultdict(lambda: deque(maxlen=args.window))

    iteration = 0
    waiting_msg_shown = {}

    log.info("✅ Prêt — en attente de données réelles depuis les capteurs IFM...")
    print(f"\n{GREEN}{BOLD}Moteur démarré — données réelles depuis MariaDB IoT{RESET}")
    print(f"Réseau local : {args.host}:{args.port}/{args.database}\n")

    try:
        while True:
            sessions = reader.poll(batch_size=args.batch)

            if sessions:
                for session in sessions:
                    sensor_id = session["sensor_id"]

                    # Limite de capteurs simultanés
                    if sensor_id not in windows and len(windows) >= MAX_SENSORS:
                        log.warning(
                            f"MAX_SENSORS={MAX_SENSORS} atteint — capteur {sensor_id} ignoré"
                        )
                        continue

                    windows[sensor_id].append(session)
                    waiting_msg_shown[sensor_id] = False

                    # Log nouvelle mesure
                    log.info(
                        f"Mesure reçue — capteur={sensor_id} "
                        f"temp={session['temperature']}°C "
                        f"vib_z={session['vibration_z']}mg "
                        f"fenêtre={len(windows[sensor_id])}/{args.window}"
                    )
            else:
                # Afficher un message d'attente une seule fois par capteur
                for sid in windows:
                    if len(windows[sid]) < args.window and not waiting_msg_shown.get(sid):
                        print(
                            f"⏳ Capteur {sid} : {len(windows[sid])}/{args.window} mesures...",
                            end="\r"
                        )

            # Prédictions pour chaque capteur ayant une fenêtre pleine
            for sensor_id, window in windows.items():
                if len(window) < args.window:
                    continue

                iteration += 1
                history = list(window)

                # Calcul vib_total pour affichage (math importé en haut du fichier)
                last = history[-1]
                vt = round(math.sqrt(
                    last.get("vibration_x",0)**2 +
                    last.get("vibration_y",0)**2 +
                    last.get("vibration_z",0)**2
                ), 1)
                last_m = {
                    "temperature":    last.get("temperature"),
                    "vibration_z":    last.get("vibration_z"),
                    "vibration_total": vt,
                }

                # Appels API (history sans sensor_id pour la compatibilité)
                hist_clean = [
                    {k: v for k, v in m.items() if k != "sensor_id"}
                    for m in history
                ]

                predict, _ = api.predict(sensor_id, hist_clean)
                rul,     _ = api.predict_rul(sensor_id, predict, hist_clean)

                # Affichage
                display(iteration, sensor_id, last_m, predict, rul)

                # Log alerte critique
                if predict.get("risk_level") in ("CRITIQUE", "ÉLEVÉ"):
                    log.warning(
                        f"🔴 {predict.get('risk_level')} — "
                        f"capteur={sensor_id} | votes={predict.get('votes')}/4 | "
                        f"temp={last.get('temperature')}°C | vib_z={last.get('vibration_z')}mg"
                    )

                # Fiabilité API toutes les 10 itérations
                if iteration % 10 == 0:
                    log.info(
                        f"Fiabilité API : {api.reliability:.0f}% "
                        f"({api.ok}/{api.total}) | "
                        f"Capteurs actifs : {len(windows)}"
                    )

                # Sauvegarde
                save_result(iteration, sensor_id, last_m, predict, rul)

            time.sleep(args.poll)

    except KeyboardInterrupt:
        print(f"\n\n{YELLOW}Arrêt demandé — {iteration} prédictions effectuées{RESET}")
        log.info(
            f"Arrêt — {iteration} itérations | "
            f"Fiabilité API : {api.reliability:.0f}% | "
            f"Capteurs : {list(windows.keys())}"
        )
    finally:
        reader.close()
        print(f"\n{GREEN}Résultats sauvegardés dans realtime_results.json{RESET}\n")


# ══════════════════════════════════════════════════════════════════════════════
#  DIAGNOSTIC RAPIDE
# ══════════════════════════════════════════════════════════════════════════════

def run_diagnostic(args):
    """Test rapide de connexion + lecture + API avant de lancer le moteur."""
    print(f"\n{BOLD}{'='*60}")
    print(f"  DIAGNOSTIC — MariaDB IoT → API")
    print(f"{'='*60}{RESET}\n")

    # 1. MySQL connector
    print("1. Dépendance mysql-connector-python...")
    try:
        import mysql.connector
        print(f"   {GREEN}✅ mysql-connector-python installé{RESET}")
    except ImportError:
        print(f"   {RED}❌ Non installé → pip install mysql-connector-python{RESET}")
        return False

    # 2. Connexion MariaDB
    print(f"\n2. Connexion MariaDB {args.host}:{args.port}/{args.database}...")
    try:
        conn = mysql.connector.connect(
            host=args.host, port=args.port,
            user=args.user, password=args.password,
            database=args.database, connection_timeout=10
        )
        print(f"   {GREEN}✅ Connexion établie{RESET}")
    except Exception as e:
        print(f"   {RED}❌ Connexion échouée : {e}{RESET}")
        return False

    # 3. Table
    print(f"\n3. Vérification table '{args.table}'...")
    cur = conn.cursor(dictionary=True)
    try:
        cur.execute(f"SELECT COUNT(*) as n FROM `{args.table}`")
        n = cur.fetchone()["n"]
        print(f"   {GREEN}✅ Table trouvée — {n} lignes{RESET}")

        cur.execute(
            f"SELECT id, SensorNodeId, gph FROM `{args.table}` "
            f"ORDER BY id DESC LIMIT 5"
        )
        rows = cur.fetchall()
        print(f"   Dernières lignes reçues :")
        for r in rows:
            print(f"     id={r['id']} | capteur={r['SensorNodeId']} | gph={r['gph']}")
    except Exception as e:
        print(f"   {RED}❌ Erreur table : {e}{RESET}")
        conn.close()
        return False

    # 4. API
    print(f"\n4. API FastAPI http://{args.api_host}:{args.api_port}...")
    try:
        import requests
        r = requests.get(f"http://{args.api_host}:{args.api_port}/health", timeout=5)
        d = r.json()
        if d.get("status") == "ok":
            print(f"   {GREEN}✅ API OK — version={d.get('version')} | modèles={d.get('models')}{RESET}")
        else:
            print(f"   {YELLOW}⚠️  API répond mais status={d.get('status')}{RESET}")
    except Exception as e:
        print(f"   {RED}❌ API injoignable : {e}{RESET}")
        print(f"   → Lance d'abord : python api_unified_pythagore.py")

    conn.close()
    print(f"\n{GREEN}{BOLD}✅ Diagnostic OK — Lance maintenant : python realtime_mariadb.py{RESET}\n")
    return True


# ══════════════════════════════════════════════════════════════════════════════
#  ARGPARSE
# ══════════════════════════════════════════════════════════════════════════════

def main():
    import io as _io
    # V6 : correction encodage Unicode sur Windows (CP1252 -> UTF-8)
    if hasattr(sys.stdout, "buffer") and getattr(sys.stdout, "encoding", "utf-8").lower() != "utf-8":
        sys.stdout = _io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = _io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)

    parser = argparse.ArgumentParser(
        description="Moteur temps reel MariaDB IoT -> API FastAPI (maintenance predictive)"
    )
    # MariaDB IoT
    parser.add_argument("--host",     default=DEFAULT_CONFIG["host"],
                        help=f"IP serveur MariaDB (défaut: {DEFAULT_CONFIG['host']})")
    parser.add_argument("--port",     default=DEFAULT_CONFIG["port"], type=int,
                        help="Port MariaDB (défaut: 3306)")
    parser.add_argument("--user",     default=DEFAULT_CONFIG["user"],
                        help=f"Utilisateur MariaDB (défaut: {DEFAULT_CONFIG['user']})")
    parser.add_argument("--password", default=DEFAULT_CONFIG["password"],
                        help="Mot de passe MariaDB")
    parser.add_argument("--database", default=DEFAULT_CONFIG["database"],
                        help=f"Nom base de données (défaut: {DEFAULT_CONFIG['database']})")
    parser.add_argument("--table",    default=DEFAULT_CONFIG["table"],
                        help=f"Table capteurs (défaut: {DEFAULT_CONFIG['table']})")
    # API
    parser.add_argument("--api-host", default="localhost")
    parser.add_argument("--api-port", default=8000, type=int)
    parser.add_argument("--timeout",  default=10,   type=int)
    parser.add_argument("--retries",  default=3,    type=int)
    # Moteur
    parser.add_argument("--window",   default=20,   type=int,
                        help="Taille fenetre glissante par capteur (defaut: 20 — V6 ameliore)")
    parser.add_argument("--poll",     default=2.0,  type=float,
                        help="Intervalle polling MariaDB en secondes (défaut: 2)")
    parser.add_argument("--batch",    default=100,  type=int,
                        help="Nombre max de lignes lues par poll (défaut: 100)")
    # Mode
    parser.add_argument("--diagnostic", action="store_true",
                        help="Lance uniquement le diagnostic sans démarrer le moteur")
    parser.add_argument("--replay", type=int, default=0, metavar="N",
                        help="Rejoue les N dernières lignes existantes (données réelles sans capteurs connectés)")

    args = parser.parse_args()

    if args.diagnostic:
        run_diagnostic(args)
    else:
        print(f"\nTest de connexion avant demarrage...")
        if run_diagnostic(args):
            print(f"Demarrage du moteur dans 3s...\n")
            time.sleep(3)
            run(args)
        else:
            print(f"\nCorrige les erreurs puis relance.\n")
            sys.exit(1)


if __name__ == "__main__":
    main()
