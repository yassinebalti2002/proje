"""
gateway_ifm_simulator.py
========================
Simulateur de Gateway IFM AL1350/AL1352
Émule exactement les endpoints HTTP REST de la vraie gateway IFM
avec les valeurs RÉELLES de vos 19 capteurs.

Usage :
    python gateway_ifm_simulator.py
    python gateway_ifm_simulator.py --host 0.0.0.0 --port 80
    python gateway_ifm_simulator.py --mode realiste
    python gateway_ifm_simulator.py --mode critique   (simule pannes)

Endpoints simulés (identiques à la vraie gateway IFM) :
    GET /                                                    → info gateway
    GET /iolinkmaster/port[N]/iolinkdevice/productname       → nom du capteur
    GET /iolinkmaster/port[N]/iolinkdevice/getdata           → données JSON
    GET /iolinkmaster/port[N]/iolinkdevice/pdin              → données hex
    GET /dataStorage                                         → snapshot tous ports

Puis lancer realtime_ifm_direct.py :
    python realtime_ifm_direct.py --gateway 127.0.0.1 --ports 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19
"""

import argparse
import json
import math
import random
import time
import struct
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn

class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True
from urllib.parse import urlparse

# ══════════════════════════════════════════════════════════════════════════════
#  PROFILS RÉELS DES 19 CAPTEURS IFM
#  Valeurs extraites directement de dataset_2026_with_acc.csv
# ══════════════════════════════════════════════════════════════════════════════

CAPTEURS = {
    # port → profil réel du capteur
    1:  {"id": "07da47b8", "model": "VVB001", "temp_moy": 37.7, "temp_std": 1.4, "vib_x": 370, "vib_y": 24,  "vib_z": 214, "vib_std": 114, "statut": "normal"},
    2:  {"id": "0ff416d2", "model": "VVB001", "temp_moy": 35.5, "temp_std": 1.7, "vib_x": 518, "vib_y": 435, "vib_z": 358, "vib_std": 101, "statut": "normal"},
    3:  {"id": "2c6254af", "model": "VVB001", "temp_moy": 33.1, "temp_std": 1.3, "vib_x": 579, "vib_y": 37,  "vib_z": 328, "vib_std": 62,  "statut": "normal"},
    4:  {"id": "3a782f1b", "model": "VSE002", "temp_moy": 32.6, "temp_std": 4.5, "vib_x": 64,  "vib_y": 55,  "vib_z": 60,  "vib_std": 65,  "statut": "normal"},
    5:  {"id": "4b5e4b32", "model": "VVB001", "temp_moy": 45.6, "temp_std": 2.9, "vib_x": 606, "vib_y": 384, "vib_z": 450, "vib_std": 171, "statut": "attention"},
    6:  {"id": "53cb61b2", "model": "VVB001", "temp_moy": 33.7, "temp_std": 1.4, "vib_x": 496, "vib_y": 684, "vib_z": 536, "vib_std": 163, "statut": "attention"},
    7:  {"id": "68c11f06", "model": "VVB001", "temp_moy": 43.7, "temp_std": 2.9, "vib_x": 331, "vib_y": 558, "vib_z": 307, "vib_std": 134, "statut": "normal"},
    8:  {"id": "6e0c1740", "model": "VVB001", "temp_moy": 30.7, "temp_std": 2.4, "vib_x": 447, "vib_y": 375, "vib_z": 286, "vib_std": 299, "statut": "normal"},
    9:  {"id": "718fd2af", "model": "VVB001", "temp_moy": 46.5, "temp_std": 0.7, "vib_x": 232, "vib_y": 256, "vib_z": 217, "vib_std": 18,  "statut": "normal"},
    10: {"id": "8f7f2f7e", "model": "VVB001", "temp_moy": 38.7, "temp_std": 2.1, "vib_x": 365, "vib_y": 283, "vib_z": 293, "vib_std": 62,  "statut": "normal"},
    11: {"id": "91d92804", "model": "VVB001", "temp_moy": 52.1, "temp_std": 5.0, "vib_x": 376, "vib_y": 663, "vib_z": 870, "vib_std": 332, "statut": "critique"},
    12: {"id": "99695e98", "model": "VSE002", "temp_moy": 31.2, "temp_std": 3.3, "vib_x": 54,  "vib_y": 58,  "vib_z": 56,  "vib_std": 62,  "statut": "normal"},
    13: {"id": "a6a46be1", "model": "VVB001", "temp_moy": 37.4, "temp_std": 2.1, "vib_x": 433, "vib_y": 324, "vib_z": 337, "vib_std": 106, "statut": "normal"},
    14: {"id": "aa7b02a1", "model": "VVB001", "temp_moy": 49.9, "temp_std": 0.9, "vib_x": 330, "vib_y": 155, "vib_z": 400, "vib_std": 23,  "statut": "attention"},
    15: {"id": "b2acdf45", "model": "VSE002", "temp_moy": 28.6, "temp_std": 1.3, "vib_x": 121, "vib_y": 156, "vib_z": 138, "vib_std": 140, "statut": "normal"},
    16: {"id": "bc59bf5f", "model": "VVB001", "temp_moy": 38.0, "temp_std": 2.7, "vib_x": 606, "vib_y": 606, "vib_z": 606, "vib_std": 192, "statut": "attention"},
    17: {"id": "d9508e77", "model": "VVB001", "temp_moy": 35.7, "temp_std": 1.8, "vib_x": 487, "vib_y": 370, "vib_z": 307, "vib_std": 85,  "statut": "normal"},
    18: {"id": "eb084747", "model": "VVB001", "temp_moy": 40.2, "temp_std": 6.2, "vib_x": 201, "vib_y": 203, "vib_z": 675, "vib_std": 607, "statut": "critique"},
    19: {"id": "f48c25f9", "model": "VSE002", "temp_moy": 34.0, "temp_std": 4.7, "vib_x": 79,  "vib_y": 73,  "vib_z": 60,  "vib_std": 67,  "statut": "normal"},
}

GATEWAY_ID = "6f85d70d"
MODE = "realiste"  # "realiste" | "critique"

# ══════════════════════════════════════════════════════════════════════════════
#  GÉNÉRATEUR DE VALEURS RÉALISTES
# ══════════════════════════════════════════════════════════════════════════════

def generer_mesure(port_num: int) -> dict:
    """
    Génère une mesure réaliste pour le capteur sur ce port.
    Simule les variations naturelles observées dans les données réelles.
    """
    capteur = CAPTEURS.get(port_num)
    if not capteur:
        return None

    t = time.time()
    # Variation sinusoïdale lente (cycle de ~5 minutes) + bruit gaussien
    cycle = math.sin(t / 300.0 + port_num)

    # Température avec dérive lente et bruit
    temp = capteur["temp_moy"] + cycle * capteur["temp_std"] * 0.5
    temp += random.gauss(0, capteur["temp_std"] * 0.3)
    temp = round(max(20.0, min(65.0, temp)), 2)

    # Vibrations avec variation selon statut
    factor = 1.0
    if MODE == "critique" and capteur["statut"] == "critique":
        # Simuler une dégradation progressive
        factor = 1.0 + 0.5 * abs(math.sin(t / 60.0))

    std = capteur["vib_std"] * 0.4

    vib_x = max(1, int(capteur["vib_x"] * factor + random.gauss(0, std)))
    vib_y = max(1, int(capteur["vib_y"] * factor + random.gauss(0, std)))
    vib_z = max(1, int(capteur["vib_z"] * factor + random.gauss(0, std)))

    # Accélérations (calculées depuis vib_z)
    acc_p2p   = round(vib_z * 0.0028 + random.gauss(0, 0.05), 3)
    acc_z2p   = round(vib_z * 0.0014 + random.gauss(0, 0.02), 3)
    acc_crest = round(1.4 + random.gauss(0, 0.15), 2)
    acc_rms   = round(vib_z * 0.0009 + random.gauss(0, 0.01), 3)

    return {
        "temp":       temp,
        "vib_x":      vib_x,
        "vib_y":      vib_y,
        "vib_z":      vib_z,
        "acc_p2p":    max(0, acc_p2p),
        "acc_z2p":    max(0, acc_z2p),
        "acc_crest":  max(1.0, acc_crest),
        "acc_rms":    max(0, acc_rms),
        "sensor_id":  capteur["id"],
        "meas_id":    int(t) % 100000,
    }


def build_getdata_response(port_num: int) -> dict:
    """Format JSON AL1352 firmware ≥ 2.x — identique à la vraie gateway"""
    m = generer_mesure(port_num)
    if not m:
        return None

    return {
        "data": {
            "Temperature": m["temp"],
            "Vibration": {
                "RMS": {
                    "X": m["vib_x"],
                    "Y": m["vib_y"],
                    "Z": m["vib_z"]
                },
                "A-P2P":  {"Y": m["acc_p2p"]},
                "A-Z2P":  {"Y": m["acc_z2p"]},
                "Crest":  {"Y": m["acc_crest"]},
                "A-RMS":  {"Y": m["acc_rms"]}
            },
            "Timestamp":    int(time.time()),
            "SensorNodeId": m["sensor_id"],
            "SourceAddress": f"{port_num * 100000}",
            "GatewayId":    GATEWAY_ID,
            "MeasDetails": {
                "FftSize":     4096,
                "FftWindow":   2,
                "G-range":     4,
                "Precision":   1,
                "BinSize":     1000,
                "ValueOffset": 0,
                "Trigger":     4,
                "Id":          m["meas_id"]
            },
            "Type": "scalar"
        }
    }


def build_pdin_response(port_num: int) -> dict:
    """Format hex AL1350 firmware 1.x — payload binaire encodé"""
    m = generer_mesure(port_num)
    if not m:
        return None

    # Encoder en big-endian selon layout IFM VVB001/VSE002
    temp_raw = int(m["temp"] * 10)  # int16, unité 0.1°C
    raw = struct.pack(">hHHHHHHH",
        temp_raw,
        int(m["vib_x"]),
        int(m["vib_y"]),
        int(m["vib_z"]),
        int(m["acc_p2p"] * 100),
        int(m["acc_z2p"] * 100),
        int(m["acc_crest"] * 10),
        int(m["acc_rms"] * 100),
    )
    hex_str = "0x" + raw.hex().upper()
    return {"data": {"value": hex_str}}


def build_datastorage_response() -> dict:
    """Snapshot global de tous les ports actifs"""
    entries = []
    for port_num, capteur in CAPTEURS.items():
        m = generer_mesure(port_num)
        if not m:
            continue
        entries.append({
            "port":        port_num,
            "productname": capteur["model"],
            "sensorid":    capteur["id"],
            "SensorNodeId": capteur["id"],
            "processdata": {
                "Temperature": m["temp"],
                "SensorNodeId": capteur["id"],
                "Vibration": {
                    "RMS": {
                        "X": m["vib_x"],
                        "Y": m["vib_y"],
                        "Z": m["vib_z"]
                    },
                    "A-P2P":  {"Y": m["acc_p2p"]},
                    "A-Z2P":  {"Y": m["acc_z2p"]},
                    "Crest":  {"Y": m["acc_crest"]},
                    "A-RMS":  {"Y": m["acc_rms"]}
                }
            }
        })
    return {"data": entries}


# ══════════════════════════════════════════════════════════════════════════════
#  SERVEUR HTTP — simule la gateway IFM
# ══════════════════════════════════════════════════════════════════════════════

class IFMGatewayHandler(BaseHTTPRequestHandler):

    def log_message(self, fmt, *args):
        # Afficher uniquement les requêtes de données (pas les health checks)
        path = self.path
        if "getdata" in path or "pdin" in path or "dataStorage" in path:
            ts = datetime.now().strftime("%H:%M:%S")
            port = self._extract_port(path)
            capteur = CAPTEURS.get(port, {})
            sid = capteur.get("id", "?")
            print(f"[{ts}] GET {path} → capteur {sid} (port {port})")

    def _extract_port(self, path: str) -> int:
        """Extrait le numéro de port depuis /iolinkmaster/port[N]/...
        Supporte les crochets encodés (%5B%5D) que requests envoie automatiquement.
        """
        import re
        # Supporter port[N] et port%5BN%5D (encodage URL des crochets)
        m = re.search(r"port(?:\[|%5B)(\d+)(?:\]|%5D)", path, re.IGNORECASE)
        return int(m.group(1)) if m else 0

    def _send_json(self, data: dict, status: int = 200):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _send_404(self):
        self.send_response(404)
        self.end_headers()

    def do_GET(self):
        path = urlparse(self.path).path

        # ── Racine — info gateway ────────────────────────────────────────────
        if path in ("/", ""):
            self._send_json({
                "gateway": "IFM AL1352 Simulator",
                "firmware": "2.3.1",
                "gatewayId": GATEWAY_ID,
                "ports": len(CAPTEURS),
                "mode": MODE,
                "timestamp": int(time.time()),
                "simulator": True,
                "version": "ISG Bizerte / Novation City"
            })
            return

        # ── productname ──────────────────────────────────────────────────────
        if "productname" in path:
            port = self._extract_port(path)
            capteur = CAPTEURS.get(port)
            if capteur:
                self._send_json({"data": {"value": capteur["model"]}})
            else:
                self._send_404()
            return

        # ── getdata (mode principal AL1352) ──────────────────────────────────
        if "getdata" in path:
            port = self._extract_port(path)
            if port not in CAPTEURS:
                self._send_404()
                return
            resp = build_getdata_response(port)
            self._send_json(resp)
            return

        # ── pdin (mode hex AL1350 firmware 1.x) ─────────────────────────────
        if "pdin" in path:
            port = self._extract_port(path)
            if port not in CAPTEURS:
                self._send_404()
                return
            resp = build_pdin_response(port)
            self._send_json(resp)
            return

        # ── dataStorage (snapshot global) ────────────────────────────────────
        if "dataStorage" in path or "datastorage" in path.lower():
            self._send_json(build_datastorage_response())
            return

        self._send_404()


# ══════════════════════════════════════════════════════════════════════════════
#  POINT D'ENTRÉE
# ══════════════════════════════════════════════════════════════════════════════

def main():
    global MODE

    parser = argparse.ArgumentParser(
        description="Simulateur Gateway IFM AL1350/AL1352 — ISG Bizerte / Novation City"
    )
    parser.add_argument("--host",  default="127.0.0.1",
                        help="Adresse d'écoute (défaut: 127.0.0.1)")
    parser.add_argument("--port",  default=80, type=int,
                        help="Port HTTP (défaut: 80)")
    parser.add_argument("--mode",  default="realiste",
                        choices=["realiste", "critique"],
                        help="realiste=valeurs normales | critique=simule pannes (défaut: realiste)")
    args = parser.parse_args()
    MODE = args.mode

    print("═" * 65)
    print("  SIMULATEUR GATEWAY IFM AL1350/AL1352")
    print("  ISG Bizerte / Novation City — Maintenance Prédictive")
    print("═" * 65)
    print(f"  Adresse   : http://{args.host}:{args.port}")
    print(f"  Capteurs  : {len(CAPTEURS)} capteurs IFM simulés (valeurs réelles)")
    print(f"  Mode      : {MODE.upper()}")
    print(f"  Gateway ID: {GATEWAY_ID}")
    print("═" * 65)
    print()
    print("  Capteurs simulés :")
    for port, c in CAPTEURS.items():
        statut_icon = "🔴" if c["statut"] == "critique" else ("🟠" if c["statut"] == "attention" else "🟢")
        print(f"    Port {port:2d} → {c['id']}  {c['model']}  "
              f"temp≈{c['temp_moy']:.0f}°C  vib_z≈{c['vib_z']}mg  {statut_icon}")
    print()
    print("  Puis dans un autre terminal :")
    print(f"  python realtime_ifm_direct.py --gateway {args.host} --gateway-port {args.port} "
          f"--ports {' '.join(str(p) for p in CAPTEURS.keys())}")
    print()
    print("  Ctrl+C pour arrêter")
    print()

    # Port 80 nécessite les droits admin sur Windows
    # Si erreur, utiliser --port 8080
    try:
        server = ThreadingHTTPServer((args.host, args.port), IFMGatewayHandler)
        print(f"✅ Simulateur démarré sur http://{args.host}:{args.port}")
        print(f"   En attente de requêtes de realtime_ifm_direct.py ...\n")
        server.serve_forever()
    except PermissionError:
        print(f"❌ Port {args.port} refusé (droits admin requis pour port < 1024)")
        print(f"   Relancez avec : python gateway_ifm_simulator.py --port 8080")
        print(f"   Et adaptez realtime_ifm_direct.py : --gateway-port 8080")
    except OSError as e:
        print(f"❌ Erreur réseau : {e}")
        print(f"   Port {args.port} peut-être déjà utilisé → essayez --port 8081")
    except KeyboardInterrupt:
        print("\n✅ Simulateur arrêté proprement.")


if __name__ == "__main__":
    main()
