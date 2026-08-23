"""
health_watchdog.py
===================
Surveillance externe de la disponibilité du système (API + MariaDB), avec
alerte automatique en cas de panne prolongée.

Pourquoi un processus séparé : si l'API elle-même plante, elle ne peut pas
s'auto-signaler. Un watchdog externe, indépendant, est le seul moyen fiable
de détecter et notifier une panne complète du service.

Fonctionnement :
    - Interroge GET /health toutes les --interval secondes (défaut 30s)
    - Vérifie aussi la connectivité TCP vers MariaDB (port 3306 par défaut)
    - Après --fail-threshold échecs consécutifs (défaut 3, soit ~90s), déclenche
      une alerte via AlertManager (email/webhook/SMS, mêmes canaux que les
      alertes capteurs) et l'écrit dans watchdog.log
    - Revient à la normale -> log un message de rétablissement (pas d'alerte
      de plus, évite le double bruit)

Usage :
    python health_watchdog.py
    python health_watchdog.py --interval 15 --fail-threshold 2
    python health_watchdog.py --api-url http://localhost:8000 --no-db-check
"""

import argparse
import json
import logging
import socket
import time
import urllib.request
import urllib.error
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [WATCHDOG] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler("watchdog.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("watchdog")

try:
    from alert_manager import AlertManager
    _alert_mgr = AlertManager()
except Exception as e:
    _alert_mgr = None
    log.warning(f"AlertManager indisponible ({e}) — les alertes systeme seront seulement journalisees")


def check_api(url: str, timeout: float = 5.0) -> tuple[bool, str]:
    try:
        with urllib.request.urlopen(f"{url}/health", timeout=timeout) as r:
            data = json.loads(r.read().decode("utf-8"))
            if data.get("status") == "ok":
                return True, f"OK (modeles={data.get('models_loaded')})"
            return False, f"reponse inattendue : {data}"
    except urllib.error.URLError as e:
        return False, f"injoignable : {e.reason}"
    except Exception as e:
        return False, f"erreur : {e}"


def check_db(host: str, port: int, timeout: float = 3.0) -> tuple[bool, str]:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True, "OK"
    except Exception as e:
        return False, f"injoignable : {e}"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--api-url", default="http://localhost:8000")
    parser.add_argument("--db-host", default="localhost")
    parser.add_argument("--db-port", type=int, default=3306)
    parser.add_argument("--interval", type=float, default=30.0, help="Secondes entre deux verifications")
    parser.add_argument("--fail-threshold", type=int, default=3, help="Echecs consecutifs avant alerte")
    parser.add_argument("--no-db-check", action="store_true", help="Ne pas verifier MariaDB")
    parser.add_argument("--once", action="store_true", help="Un seul cycle puis quitte (pour tests)")
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("  WATCHDOG — Surveillance API + MariaDB")
    log.info(f"  API : {args.api_url} | Intervalle : {args.interval}s | Seuil : {args.fail_threshold}")
    log.info("=" * 60)

    api_fail_count = 0
    db_fail_count = 0
    api_was_down = False
    db_was_down = False

    while True:
        api_ok, api_msg = check_api(args.api_url)
        if api_ok:
            if api_was_down:
                log.info(f"✅ API rétablie — {api_msg}")
                api_was_down = False
            else:
                log.info(f"✅ API OK — {api_msg}")
            api_fail_count = 0
        else:
            api_fail_count += 1
            log.warning(f"❌ API KO ({api_fail_count}/{args.fail_threshold}) — {api_msg}")
            if api_fail_count >= args.fail_threshold and not api_was_down:
                api_was_down = True
                msg = f"API injoignable depuis {api_fail_count * args.interval:.0f}s ({api_msg})"
                log.error(f"🔴 ALERTE — {msg}")
                if _alert_mgr:
                    _alert_mgr.send_system_alert(msg)

        if not args.no_db_check:
            db_ok, db_msg = check_db(args.db_host, args.db_port)
            if db_ok:
                if db_was_down:
                    log.info(f"✅ MariaDB rétablie — {db_msg}")
                    db_was_down = False
                else:
                    log.info(f"✅ MariaDB OK — {db_msg}")
                db_fail_count = 0
            else:
                db_fail_count += 1
                log.warning(f"❌ MariaDB KO ({db_fail_count}/{args.fail_threshold}) — {db_msg}")
                if db_fail_count >= args.fail_threshold and not db_was_down:
                    db_was_down = True
                    msg = f"MariaDB injoignable depuis {db_fail_count * args.interval:.0f}s ({db_msg})"
                    log.error(f"🔴 ALERTE — {msg}")
                    if _alert_mgr:
                        _alert_mgr.send_system_alert(msg)

        if args.once:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
