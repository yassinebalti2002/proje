r"""
realtime_ifm_direct.py
======================
Moteur temps réel SANS MariaDB : Capteurs IFM → Gateway HTTP → API FastAPI
Chaîne complète : Capteurs IFM (IO-Link) → Gateway IFM (HTTP REST) → prédictions ML

Architecture :
    Capteurs IFM (19x)
        ↓ IO-Link
    Gateway IFM AL1350 / AL1352 (HTTP REST)
        ↓ GET /iolinkmaster/port[N]/iolinkdevice/pdin  (polling toutes les 2s)
    realtime_ifm_direct.py  (ta machine)
        ↓ POST /v1/predict + /v1/predict-rul
    API FastAPI v3.1  (localhost:8000)
        ↓ résultats
    Affichage terminal + realtime_results.json

Différences vs realtime_mariadb.py :
    ✅ Plus besoin de MariaDB ni de mysql-connector-python
    ✅ Lecture directe de la gateway IFM via HTTP REST (même réseau local)
    ✅ Conservation de toute la logique de consolidation des 3 gph
    ✅ Fenêtre glissante, retry, affichage terminal : identiques
    ✅ Un seul fichier SQL toujours dans le projet (ai_cp.sql) mais non utilisé ici

Usage :
    python realtime_ifm_direct.py
    python realtime_ifm_direct.py --gateway 192.168.1.50 --ports 1 2 3 4 5
    python realtime_ifm_direct.py --gateway 192.168.1.50 --poll 2 --window 10
    python realtime_ifm_direct.py --diagnostic
"""

import argparse
import json
import logging
import math
import os
import sys
import time
from collections import deque, defaultdict
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [IFM_DIRECT] %(message)s",
    handlers=[
        logging.FileHandler("realtime_ifm_direct.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ]
)
log = logging.getLogger("ifm_direct")

# ── Couleurs terminal ──────────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"
RISK_COLOR = {
    "CRITIQUE": RED,
    "ÉLEVÉ":    YELLOW,
    "MODÉRÉ":   YELLOW,
    "FAIBLE":   GREEN,
    "OK":       GREEN,
    "INCONNU":  CYAN,
}

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION — modifie ces valeurs ou utilise les arguments CLI
# ══════════════════════════════════════════════════════════════════════════════
DEFAULT_CONFIG = {
    "gateway_host": os.environ.get("IFM_GATEWAY_HOST", "192.168.1.50"),
    "gateway_port": int(os.environ.get("IFM_GATEWAY_PORT", "80")),
    # Liste des ports IO-Link à surveiller (1 à 8 selon le modèle de gateway)
    "ports":        [int(p) for p in os.environ.get("IFM_PORTS", "1 2 3 4").split()],
    # Timeout HTTP vers la gateway
    "gateway_timeout": int(os.environ.get("IFM_TIMEOUT", "5")),
}

MAX_SENSORS = 25  # même limite que realtime_mariadb.py


# ══════════════════════════════════════════════════════════════════════════════
#  LECTEUR GATEWAY IFM — polling HTTP direct
# ══════════════════════════════════════════════════════════════════════════════

class IFMGatewayReader:
    """
    Lit les données de la gateway IFM AL1350/AL1352 directement via HTTP REST.

    Endpoints REST IFM typiques :
      GET http://<gateway>/iolinkmaster/port[N]/iolinkdevice/pdin
          → retourne le process data input brut (JSON ou hex)

      GET http://<gateway>/iolinkmaster/port[N]/iolinkdevice/getdata
          → retourne les valeurs décodées : Temperature, Vibration.RMS.X/Y/Z

      GET http://<gateway>/iolinkmaster/port[N]/iolinkdevice/productname
          → identifiant du capteur sur ce port

    La gateway expose aussi une API "datastorage" :
      GET http://<gateway>/dataStorage
          → dernier snapshot de tous les ports

    Ce reader supporte deux modes auto-détectés :
      Mode A — /getdata  (AL1352 firmware ≥ 2.x, JSON structuré)
      Mode B — /pdin     (AL1350 firmware 1.x, payload hex → décodé manuellement)

    Structure session complète (compatible API FastAPI) :
        {
          "sensor_id":   "port1_VVB001",
          "temperature": 32.5,
          "vibration_x": 270.0,
          "vibration_y": 278.0,
          "vibration_z": 285.0,
          # optionnel V5 :
          "acc_p2p": ..., "acc_z2p": ..., "acc_crest": ..., "acc_rms": ...
        }
    """

    def __init__(self, gateway_host, gateway_port=80, ports=None, timeout=5):
        self.host    = gateway_host
        self.port    = gateway_port
        self.ports   = ports or [1, 2, 3, 4]
        self.timeout = timeout
        self.base    = f"http://{gateway_host}:{gateway_port}"

        # Buffer de consolidation : sensor_id → {temp, vib_x, vib_y, vib_z, ...}
        # Utilisé quand temp et vibrations arrivent sur des endpoints séparés
        self._pending   = defaultdict(dict)
        self._PENDING_TTL = 60  # secondes

        # Cache des identifiants capteurs par port (évite un GET à chaque cycle)
        self._sensor_ids = {}   # port → sensor_id string
        self._mode       = {}   # port → "getdata" | "pdin" | "datastorage"

        # Compteurs
        self.total_polls   = 0
        self.total_sessions = 0

        try:
            import requests as _r
            self._requests = _r
        except ImportError:
            log.error("requests non installé → pip install requests")
            sys.exit(1)

    # ──────────────────────────────────────────────────────────────────────────
    #  Connexion / détection du mode
    # ──────────────────────────────────────────────────────────────────────────

    def connect(self):
        """
        Vérifie la disponibilité de la gateway et détecte le mode d'accès
        pour chaque port configuré.
        """
        log.info(f"Connexion à la gateway IFM : {self.base}")
        log.info(f"Ports configurés : {self.ports}")

        # Test racine
        try:
            r = self._requests.get(self.base, timeout=self.timeout)
            log.info(f"✅ Gateway joignable — HTTP {r.status_code}")
        except Exception as e:
            log.error(f"❌ Gateway inaccessible : {e}")
            log.error(f"  → Vérifie l'IP : ping {self.host}")
            log.error(f"  → Vérifie que la gateway est sous tension et sur le même réseau")
            sys.exit(1)

        # Détection mode et sensor_id pour chaque port
        for p in self.ports:
            self._detect_port_mode(p)

        active = [p for p, m in self._mode.items() if m != "absent"]
        if not active:
            log.error("Aucun port actif trouvé sur la gateway")
            log.error("→ Vérifie que les capteurs IFM sont branchés sur les ports IO-Link")
            sys.exit(1)

        log.info(f"✅ Ports actifs : {active}")

    def _detect_port_mode(self, port_num: int):
        """
        Détecte si le port répond en mode 'getdata' (structuré) ou 'pdin' (hex).
        Stocke aussi le sensor_id (productname ou fallback).
        """
        # 1. Essai productname → label lisible (pour affichage seulement)
        # L'ID stable reste toujours "port{N}" pour éviter les changements de nom
        stable_id = f"port{port_num}"
        try:
            url = f"{self.base}/iolinkmaster/port[{port_num}]/iolinkdevice/productname"
            r = self._requests.get(url, timeout=self.timeout)
            if r.status_code == 200:
                data = r.json()
                name = data.get("data", {}).get("value", "")
                name = "".join(c for c in str(name) if c.isalnum() or c in "-_")
                # sensor_id = port stable + nom lisible
                self._sensor_ids[port_num] = f"{stable_id}_{name}" if name else stable_id
            else:
                self._sensor_ids[port_num] = stable_id
        except Exception:
            self._sensor_ids[port_num] = stable_id

        # 2. Essai mode getdata (firmware ≥ 2.x)
        try:
            url = f"{self.base}/iolinkmaster/port[{port_num}]/iolinkdevice/getdata"
            r = self._requests.get(url, timeout=self.timeout)
            if r.status_code == 200:
                self._mode[port_num] = "getdata"
                log.info(
                    f"  Port {port_num} → mode getdata | sensor_id={self._sensor_ids[port_num]}"
                )
                return
        except Exception:
            pass

        # 3. Essai mode pdin (firmware 1.x)
        try:
            url = f"{self.base}/iolinkmaster/port[{port_num}]/iolinkdevice/pdin"
            r = self._requests.get(url, timeout=self.timeout)
            if r.status_code == 200:
                self._mode[port_num] = "pdin"
                log.info(
                    f"  Port {port_num} → mode pdin | sensor_id={self._sensor_ids[port_num]}"
                )
                return
        except Exception:
            pass

        # 4. Essai datastorage global (certains firmwares)
        try:
            url = f"{self.base}/dataStorage"
            r = self._requests.get(url, timeout=self.timeout)
            if r.status_code == 200:
                self._mode[port_num] = "datastorage"
                log.info(
                    f"  Port {port_num} → mode datastorage | sensor_id={self._sensor_ids[port_num]}"
                )
                return
        except Exception:
            pass

        self._mode[port_num] = "absent"
        log.warning(f"  Port {port_num} → aucun capteur détecté")

    # ──────────────────────────────────────────────────────────────────────────
    #  Polling principal
    # ──────────────────────────────────────────────────────────────────────────

    def poll(self) -> list:
        """
        Interroge tous les ports actifs et retourne les sessions complètes
        prêtes pour l'API (même format que MariaDBReader.poll).
        """
        self.total_polls += 1
        sessions = []

        # Purge TTL du buffer de consolidation
        now = time.time()
        expired = [k for k, v in self._pending.items()
                   if now - v.get("_ts", 0) > self._PENDING_TTL]
        for k in expired:
            log.warning(f"Buffer pending expiré (TTL {self._PENDING_TTL}s) → suppression {k}")
            del self._pending[k]

        for port_num in self.ports:
            mode = self._mode.get(port_num, "absent")
            if mode == "absent":
                continue

            try:
                if mode == "getdata":
                    session = self._read_getdata(port_num)
                elif mode == "pdin":
                    session = self._read_pdin(port_num)
                elif mode == "datastorage":
                    session = self._read_datastorage(port_num)
                else:
                    session = None

                if session:
                    sessions.append(session)
                    self.total_sessions += 1

            except Exception as e:
                log.warning(f"Erreur lecture port {port_num} ({mode}) : {e}")

        return sessions

    # ──────────────────────────────────────────────────────────────────────────
    #  Mode A — /getdata  (JSON structuré, firmware ≥ 2.x)
    # ──────────────────────────────────────────────────────────────────────────

    def _read_getdata(self, port_num: int) -> dict | None:
        """
        Lit l'endpoint /getdata de la gateway IFM.

        Réponse typique AL1352 firmware 2.x (capteur VSE002/VVB001) :
        {
          "data": {
            "Temperature": 32.5,
            "Vibration": {
              "RMS": {"X": 270.0, "Y": 278.0, "Z": 285.0},
              "A-P2P": {"Y": 1.2},
              "A-Z2P": {"Y": 0.6},
              "Crest": {"Y": 3.1},
              "A-RMS": {"Y": 0.4}
            },
            "MeasDetails": {"Id": "abc123"}
          }
        }
        """
        url = f"{self.base}/iolinkmaster/port[{port_num}]/iolinkdevice/getdata"
        r = self._requests.get(url, timeout=self.timeout)

        if r.status_code != 200:
            return None

        try:
            payload = r.json()
        except ValueError:
            return None

        # Extraire les valeurs — deux structures possibles selon firmware
        data = payload.get("data", payload)

        # Température
        temp = data.get("Temperature") or data.get("temperature")
        if temp is None:
            # Certains firmwares imbriquent dans "values"
            temp = data.get("values", {}).get("Temperature")
        if temp is None:
            log.debug(f"Port {port_num} getdata : température absente dans {list(data.keys())}")
            return None

        # Vibrations RMS
        vib   = data.get("Vibration", data.get("vibration", {}))
        rms   = vib.get("RMS", vib.get("rms", {}))
        vib_x = float(rms.get("X", rms.get("x", 0)) or 0)
        vib_y = float(rms.get("Y", rms.get("y", 0)) or 0)
        vib_z = float(rms.get("Z", rms.get("z", 0)) or 0)

        # Session minimale valide : temp + vib_z
        if vib_z == 0:
            log.warning(f"Port {port_num} getdata : vib_z=0 → capteur peut-être arrêté")
            return None

        sensor_id = self._sensor_ids.get(port_num, f"port{port_num}")
        session = {
            "sensor_id":   sensor_id,
            "temperature": float(temp),
            "vibration_x": vib_x,
            "vibration_y": vib_y,
            "vibration_z": vib_z,
        }

        # Accélérations V5 (optionnel)
        for key, path in [
            ("acc_p2p",   ("A-P2P", "Y")),
            ("acc_z2p",   ("A-Z2P", "Y")),
            ("acc_crest", ("Crest",  "Y")),
            ("acc_rms",   ("A-RMS",  "Y")),
        ]:
            val = vib.get(path[0], {}).get(path[1])
            if val is not None:
                session[key] = float(val)

        return session

    # ──────────────────────────────────────────────────────────────────────────
    #  Mode B — /pdin  (payload hex brut, firmware 1.x)
    # ──────────────────────────────────────────────────────────────────────────

    def _read_pdin(self, port_num: int) -> dict | None:
        """
        Lit l'endpoint /pdin de la gateway IFM (firmware 1.x).

        Réponse typique :
        {
          "data": {
            "value": "0x02050AE0011E00B400C800D2"
          }
        }

        Le payload hex IFM (VSE002 / VVB001 ISDU) se décode ainsi :
          Bytes 0-1  : température (int16, /10 → °C)
          Bytes 2-3  : vibration X RMS (uint16, mg)
          Bytes 4-5  : vibration Y RMS (uint16, mg)
          Bytes 6-7  : vibration Z RMS (uint16, mg)
          Bytes 8-9  : accélération A-P2P (uint16, /100 → g)
          Bytes 10-11: accélération A-Z2P
          Bytes 12-13: Crest factor
          Bytes 14-15: A-RMS

        ⚠️  Le layout exact dépend du modèle de capteur IFM (VSE002 ≠ VVB001 ≠ VVB021).
           Ajuste les offsets si nécessaire d'après la datasheet de ton capteur.
        """
        url = f"{self.base}/iolinkmaster/port[{port_num}]/iolinkdevice/pdin"
        r = self._requests.get(url, timeout=self.timeout)

        if r.status_code != 200:
            return None

        try:
            payload = r.json()
        except ValueError:
            return None

        hex_str = payload.get("data", {}).get("value", "")
        if not hex_str:
            return None

        # Nettoyer le préfixe 0x et les espaces
        hex_str = hex_str.replace("0x", "").replace("0X", "").replace(" ", "")

        try:
            raw = bytes.fromhex(hex_str)
        except ValueError:
            log.debug(f"Port {port_num} pdin : hex invalide → '{hex_str}'")
            return None

        if len(raw) < 8:
            log.debug(f"Port {port_num} pdin : payload trop court ({len(raw)} bytes)")
            return None

        # Décodage IFM VVB001 / VSE002 (big-endian)
        # Température : bytes 0-1 — int16 big-endian, unité = 0.1 °C
        temp_raw = int.from_bytes(raw[0:2], "big", signed=True)
        temp     = temp_raw / 10.0

        # Vibrations RMS (mg) : bytes 2-7 — uint16 big-endian
        vib_x = int.from_bytes(raw[2:4], "big", signed=False)
        vib_y = int.from_bytes(raw[4:6], "big", signed=False)
        vib_z = int.from_bytes(raw[6:8], "big", signed=False)

        if vib_z == 0:
            return None

        sensor_id = self._sensor_ids.get(port_num, f"port{port_num}")
        session = {
            "sensor_id":   sensor_id,
            "temperature": float(temp),
            "vibration_x": float(vib_x),
            "vibration_y": float(vib_y),
            "vibration_z": float(vib_z),
        }

        # Accélérations V5 si payload assez long (≥ 16 bytes)
        if len(raw) >= 16:
            session["acc_p2p"]   = int.from_bytes(raw[8:10],  "big") / 100.0
            session["acc_z2p"]   = int.from_bytes(raw[10:12], "big") / 100.0
            session["acc_crest"] = int.from_bytes(raw[12:14], "big") / 10.0
            session["acc_rms"]   = int.from_bytes(raw[14:16], "big") / 100.0

        return session

    # ──────────────────────────────────────────────────────────────────────────
    #  Mode C — /dataStorage  (snapshot global de tous les ports)
    # ──────────────────────────────────────────────────────────────────────────

    def _read_datastorage(self, port_num: int) -> dict | None:
        """
        Lit l'endpoint /dataStorage de la gateway (un seul appel couvre tous les ports).
        Utilisé uniquement si les modes getdata/pdin ne sont pas disponibles.

        Réponse typique :
        {
          "data": [
            {
              "port": 1,
              "productname": "VVB001",
              "processdata": {
                "Temperature": 32.5,
                "Vibration": { "RMS": {"X": 270, "Y": 278, "Z": 285} }
              }
            },
            ...
          ]
        }
        """
        url = f"{self.base}/dataStorage"
        r = self._requests.get(url, timeout=self.timeout)

        if r.status_code != 200:
            return None

        try:
            payload = r.json()
        except ValueError:
            return None

        entries = payload.get("data", [])
        if isinstance(entries, dict):
            entries = [entries]

        for entry in entries:
            try:
                if int(entry.get("port", -1)) != port_num:
                    continue
            except (ValueError, TypeError):
                continue

            pd_data = entry.get("processdata", {})
            temp    = pd_data.get("Temperature")
            vib     = pd_data.get("Vibration", {}).get("RMS", {})
            vib_z   = float(vib.get("Z", 0) or 0)

            if temp is None or vib_z == 0:
                continue   # ← continue, pas return None

            sensor_id = self._sensor_ids.get(port_num, f"port{port_num}")
            return {
                "sensor_id":   sensor_id,
                "temperature": float(temp),
                "vibration_x": float(vib.get("X", 0) or 0),
                "vibration_y": float(vib.get("Y", 0) or 0),
                "vibration_z": vib_z,
            }

        return None

    # ──────────────────────────────────────────────────────────────────────────
    #  Utilitaires
    # ──────────────────────────────────────────────────────────────────────────

    def close(self):
        log.info(
            f"IFMGatewayReader fermé — "
            f"{self.total_sessions} sessions produites en {self.total_polls} polls"
        )


# ══════════════════════════════════════════════════════════════════════════════
#  CLIENT API FastAPI — identique à realtime_mariadb.py
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
            "individual_models": {m: "INCONNU" for m in ["IF", "LOF", "OCSVM", "ECOD"]},
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
#  AFFICHAGE TERMINAL — identique à realtime_mariadb.py
# ══════════════════════════════════════════════════════════════════════════════

def display(iteration, sensor_id, last_m, predict, rul, source="IFM_DIRECT"):
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
    print(f"[{ts}]  #{iteration}  |  Capteur : {sensor_id}  |  Source : {source}")
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
        "iteration":   iteration,
        "timestamp":   datetime.now().isoformat(),
        "sensor_id":   sensor_id,
        "source":      "ifm_direct_realtime",
        "measurement": measurement,
        "predict":     predict,
        "rul":         rul,
    }
    path = Path("realtime_results.json")
    try:
        data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
        data.append(record)
        path.write_text(
            json.dumps(data[-500:], indent=2, ensure_ascii=False, default=str),
            encoding="utf-8"
        )
    except Exception as e:
        log.warning(f"Sauvegarde JSON échouée : {e}")


# ══════════════════════════════════════════════════════════════════════════════
#  MOTEUR PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

def run(args):
    log.info("═" * 65)
    log.info("  MOTEUR TEMPS RÉEL — IFM Gateway HTTP direct → API FastAPI")
    log.info(f"  Gateway : http://{args.gateway}:{args.gateway_port}  ports={args.ports}")
    log.info(f"  API     : http://{args.api_host}:{args.api_port}")
    log.info(f"  Fenêtre : {args.window} mesures | Poll : {args.poll}s")
    log.info("═" * 65)

    # Composants
    reader = IFMGatewayReader(
        gateway_host=args.gateway,
        gateway_port=args.gateway_port,
        ports=args.ports,
        timeout=args.gateway_timeout,
    )
    api = APIClient(
        host=args.api_host, port=args.api_port,
        timeout=args.timeout, retries=args.retries,
    )

    # Connexion gateway (avec détection de mode par port)
    reader.connect()

    # Fenêtres glissantes par capteur
    windows = defaultdict(lambda: deque(maxlen=args.window))

    iteration = 0
    waiting_shown = {}

    log.info("✅ Prêt — polling gateway IFM en cours...")
    print(f"\n{GREEN}{BOLD}Moteur démarré — données réelles depuis la gateway IFM HTTP{RESET}")
    print(f"Gateway : http://{args.gateway}:{args.gateway_port}  |  Ports : {args.ports}\n")

    try:
        while True:
            sessions = reader.poll()

            if sessions:
                for session in sessions:
                    sensor_id = session["sensor_id"]

                    # Limite de capteurs simultanés
                    if sensor_id not in windows and len(windows) >= MAX_SENSORS:
                        log.warning(f"MAX_SENSORS={MAX_SENSORS} atteint — capteur {sensor_id} ignoré")
                        continue

                    windows[sensor_id].append(session)
                    waiting_shown[sensor_id] = False

                    log.info(
                        f"Mesure reçue — capteur={sensor_id} "
                        f"temp={session['temperature']}°C "
                        f"vib_z={session['vibration_z']}mg "
                        f"fenêtre={len(windows[sensor_id])}/{args.window}"
                    )
            else:
                # Message d'attente (une seule fois par capteur)
                for sid in windows:
                    if len(windows[sid]) < args.window and not waiting_shown.get(sid):
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

                # vib_total pour l'affichage
                last = history[-1]
                vx = last.get("vibration_x", 0) or 0
                vy = last.get("vibration_y", 0) or 0
                vz = last.get("vibration_z", 0) or 0
                vt = round(math.sqrt(vx**2 + vy**2 + vz**2), 1)
                last_m = {
                    "temperature":     last.get("temperature"),
                    "vibration_x":     vx,
                    "vibration_y":     vy,
                    "vibration_z":     vz,
                    "vibration_total": vt,
                    "current":         last.get("current", 0),
                }

                # Appels API (history sans sensor_id)
                hist_clean = [
                    {k: v for k, v in m.items() if k != "sensor_id"}
                    for m in history
                ]

                predict, err1 = api.predict(sensor_id, hist_clean)
                rul,     err2 = api.predict_rul(sensor_id, predict, hist_clean)

                display(iteration, sensor_id, last_m, predict, rul)

                if predict.get("risk_level") in ("CRITIQUE", "ÉLEVÉ"):
                    log.warning(
                        f"🔴 {predict.get('risk_level')} — "
                        f"capteur={sensor_id} | votes={predict.get('votes')}/4 | "
                        f"temp={last.get('temperature')}°C | vib_z={last.get('vibration_z')}mg"
                    )

                if iteration % 10 == 0:
                    log.info(
                        f"Fiabilité API : {api.reliability:.0f}% "
                        f"({api.ok}/{api.total}) | "
                        f"Capteurs actifs : {len(windows)} | "
                        f"Sessions IFM reçues : {reader.total_sessions}"
                    )

                save_result(iteration, sensor_id, last_m, predict, rul)

            time.sleep(args.poll)

    except KeyboardInterrupt:
        print(f"\n\n{YELLOW}Arrêt demandé — {iteration} prédictions effectuées{RESET}")
        log.info(
            f"Arrêt — {iteration} itérations | "
            f"Fiabilité API : {api.reliability:.0f}% | "
            f"Capteurs : {list(windows.keys())} | "
            f"Sessions IFM : {reader.total_sessions}"
        )
    finally:
        reader.close()
        print(f"\n{GREEN}Résultats sauvegardés dans realtime_results.json{RESET}\n")


# ══════════════════════════════════════════════════════════════════════════════
#  DIAGNOSTIC
# ══════════════════════════════════════════════════════════════════════════════

def run_diagnostic(args):
    print(f"\n{BOLD}{'='*60}")
    print(f"  DIAGNOSTIC — IFM Gateway HTTP → API")
    print(f"{'='*60}{RESET}\n")

    import requests as req

    # 1. requests
    print("1. Dépendance requests...")
    print(f"   {GREEN}✅ requests installé{RESET}")

    # 2. Gateway
    print(f"\n2. Gateway IFM http://{args.gateway}:{args.gateway_port} ...")
    try:
        r = req.get(f"http://{args.gateway}:{args.gateway_port}", timeout=args.gateway_timeout)
        print(f"   {GREEN}✅ Gateway joignable — HTTP {r.status_code}{RESET}")
    except Exception as e:
        print(f"   {RED}❌ Gateway inaccessible : {e}{RESET}")
        print(f"   → ping {args.gateway}")
        return False

    # 3. Ports
    print(f"\n3. Lecture ports {args.ports} ...")
    for p in args.ports:
        ok = False
        for ep in ["getdata", "pdin"]:
            try:
                url = f"http://{args.gateway}:{args.gateway_port}/iolinkmaster/port[{p}]/iolinkdevice/{ep}"
                r = req.get(url, timeout=args.gateway_timeout)
                if r.status_code == 200:
                    print(f"   Port {p} → {GREEN}✅ {ep} OK{RESET}")
                    ok = True
                    break
            except Exception:
                pass
        if not ok:
            print(f"   Port {p} → {YELLOW}⚠️  aucune réponse (capteur absent ou mode inconnu){RESET}")

    # 4. API
    print(f"\n4. API FastAPI http://{args.api_host}:{args.api_port} ...")
    try:
        r = req.get(f"http://{args.api_host}:{args.api_port}/health", timeout=5)
        d = r.json()
        if d.get("status") == "ok":
            print(f"   {GREEN}✅ API OK — version={d.get('version')} | modèles={d.get('models')}{RESET}")
        else:
            print(f"   {YELLOW}⚠️  API répond mais status={d.get('status')}{RESET}")
    except Exception as e:
        print(f"   {RED}❌ API injoignable : {e}{RESET}")
        print(f"   → Lance d'abord : python api_unified_pythagore.py")

    print(f"\n{GREEN}{BOLD}✅ Diagnostic terminé — Lance : python realtime_ifm_direct.py{RESET}\n")
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
        description="Moteur temps reel IFM Gateway HTTP -> API FastAPI (sans MariaDB)"
    )
    # Gateway IFM
    parser.add_argument("--gateway",         default=DEFAULT_CONFIG["gateway_host"],
                        help=f"IP gateway IFM (defaut: {DEFAULT_CONFIG['gateway_host']})")
    parser.add_argument("--gateway-port",    default=DEFAULT_CONFIG["gateway_port"], type=int,
                        help="Port HTTP gateway (defaut: 80)")
    parser.add_argument("--ports",           default=DEFAULT_CONFIG["ports"],
                        nargs="+", type=int,
                        help=f"Ports IO-Link a surveiller (defaut: {DEFAULT_CONFIG['ports']})")
    parser.add_argument("--gateway-timeout", default=DEFAULT_CONFIG["gateway_timeout"], type=int,
                        help="Timeout HTTP gateway en secondes (defaut: 5)")
    # API
    parser.add_argument("--api-host",  default="localhost")
    parser.add_argument("--api-port",  default=8000, type=int)
    parser.add_argument("--timeout",   default=10,   type=int)
    parser.add_argument("--retries",   default=3,    type=int)
    # Moteur
    parser.add_argument("--window",    default=20,   type=int,
                        help="Taille fenetre glissante par capteur (defaut: 20 — V6 ameliore)")
    parser.add_argument("--poll",      default=2.0,  type=float,
                        help="Intervalle polling gateway en secondes (defaut: 2)")
    # Mode
    parser.add_argument("--diagnostic", action="store_true",
                        help="Lance uniquement le diagnostic")

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
