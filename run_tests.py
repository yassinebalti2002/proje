"""
run_tests.py
============
Script unique pour lancer TOUS les tests du projet dans l'ordre.

Étapes :
  1. Tests unitaires ML   (pas besoin de serveur)
  2. Tests de sécurité    (serveur requis)
  3. Tests d'intégration  (serveur requis)

Usage :
    python run_tests.py                          # tests unitaires seulement
    python run_tests.py --with-server            # tous les tests (serveur auto-démarré)
    python run_tests.py --api-key <cle>          # avec clé API explicite
    python run_tests.py --api-key <cle> --with-server

Prérequis :
    pip install pytest slowapi python-dotenv
    Définir API_KEYS dans .env ou via --api-key
"""

import argparse
import io
import os
import subprocess
import sys
import time
import requests
from dotenv import load_dotenv

load_dotenv()

# Force UTF-8 sur Windows (PowerShell cp1252 → UnicodeEncodeError sinon)
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf_8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

BASE_URL = "https://localhost:8000"

# Certificat auto-signé (voir generate_selfsigned_cert.py) -- ce script ne
# vérifie pas le certificat contre une CA publique, cohérent avec le
# verify=False déjà utilisé côté client interne (realtime_mariadb.py).
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def banner(title: str):
    sep = "=" * 60
    print(f"\n{BOLD}{sep}")
    print(f"  {title}")
    print(f"{sep}{RESET}")


def wait_for_api(timeout: int = 90) -> bool:
    """
    Attend que l'API soit prête ET que les modèles ML soient chargés.
    /health retourne 200 immédiatement, mais models_loaded peut être False
    pendant le chargement. On attend models_loaded == True avant de continuer.
    """
    print(f"  Attente demarrage API + chargement modeles ML", end="", flush=True)
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(f"{BASE_URL}/health", timeout=3, verify=False)
            if r.status_code == 200:
                data = r.json()
                if data.get("models_loaded", False):
                    print(f"  {GREEN}OK (modeles charges){RESET}")
                    return True
                else:
                    print("m", end="", flush=True)  # modèles en cours
        except Exception:
            pass
        print(".", end="", flush=True)
        time.sleep(2)
    print(f"  {RED}TIMEOUT{RESET}")
    return False


def run_pytest(args_extra: list) -> int:
    """Lance pytest avec les arguments donnés, retourne le code de sortie."""
    cmd = [sys.executable, "-m", "pytest"] + args_extra
    result = subprocess.run(cmd)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="Lanceur de tests complet")
    parser.add_argument("--api-key",     default=None, help="Clé API (X-API-Key)")
    parser.add_argument("--with-server", action="store_true",
                        help="Démarre l'API automatiquement avant les tests d'intégration")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", default=8000, type=int)
    args = parser.parse_args()

    # Résolution de la clé API
    api_key = args.api_key or os.getenv("API_KEYS", "").split(",")[0].strip()
    if not api_key:
        print(f"{YELLOW}⚠️  Aucune clé API — les tests d'intégration seront partiels.{RESET}")
        print(f"   Passe --api-key <ta_cle>  ou  définis API_KEYS dans .env\n")

    common_flags = ["-v", "--tb=short"]
    key_flag = ["--api-key", api_key] if api_key else []
    host_flags = ["--host", args.host, "--port", str(args.port)]

    # ──────────────────────────────────────────────────────────────────────────
    banner("ÉTAPE 1 / 3 — Tests unitaires ML (sans serveur)")
    # ──────────────────────────────────────────────────────────────────────────
    rc1 = run_pytest(["tests/test_ml_functions.py"] + common_flags)
    if rc1 == 0:
        print(f"\n{GREEN}✅ Tests unitaires : TOUS PASSÉS{RESET}")
    else:
        print(f"\n{RED}❌ Tests unitaires : ÉCHEC (code {rc1}){RESET}")

    # ──────────────────────────────────────────────────────────────────────────
    # Démarrage du serveur si demandé
    # ──────────────────────────────────────────────────────────────────────────
    server_proc = None
    if args.with_server:
        banner("Démarrage du serveur API")
        env = os.environ.copy()
        if api_key:
            env["API_KEYS"] = api_key
        server_proc = subprocess.Popen(
            [sys.executable, "api_unified_pythagore.py"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=env,
        )
        if not wait_for_api(timeout=90):
            print(f"{RED}L'API n'a pas démarré. Vérifiez les logs.{RESET}")
            server_proc.terminate()
            sys.exit(1)

    # Vérifier si le serveur est déjà en cours (ou vient d'être démarré)
    server_available = False
    try:
        r = requests.get(f"{BASE_URL}/health", timeout=3, verify=False)
        server_available = r.status_code == 200
    except Exception:
        pass

    # None = "jamais lancé" (distinct de 0 = "lancé et réussi") -- avec 0 comme
    # valeur initiale, si server_available restait False, le résumé final
    # affichait à tort "PASS" pour des tests sécurité/intégration qui n'avaient
    # jamais tourné (rc==0 est aussi le code de succès pytest).
    rc2, rc3 = None, None

    if server_available:
        # ──────────────────────────────────────────────────────────────────────
        banner("ÉTAPE 2 / 3 — Tests de sécurité (auth + rate limiting)")
        # ──────────────────────────────────────────────────────────────────────
        rc2 = run_pytest(
            ["tests/test_security.py"] + common_flags + key_flag + host_flags
        )
        if rc2 == 0:
            print(f"\n{GREEN}✅ Tests sécurité : TOUS PASSÉS{RESET}")
        else:
            print(f"\n{RED}❌ Tests sécurité : ÉCHEC (code {rc2}){RESET}")

        # ──────────────────────────────────────────────────────────────────────
        banner("ÉTAPE 3 / 3 — Tests d'intégration API (endpoints complets)")
        # ──────────────────────────────────────────────────────────────────────
        rc3 = run_pytest(
            ["tests/test_api_final.py"] + common_flags + key_flag + host_flags
        )
        if rc3 == 0:
            print(f"\n{GREEN}✅ Tests d'intégration : TOUS PASSÉS{RESET}")
        else:
            print(f"\n{RED}❌ Tests d'intégration : ÉCHEC (code {rc3}){RESET}")
    else:
        print(f"\n{YELLOW}⏭️  Étapes 2 et 3 ignorées — serveur non disponible.{RESET}")
        print(f"   Lance l'API puis relance avec --with-server ou manuellement.\n")

    # ──────────────────────────────────────────────────────────────────────────
    banner("RÉSUMÉ FINAL")
    # ──────────────────────────────────────────────────────────────────────────
    steps = [
        ("Tests unitaires ML",    rc1),
        ("Tests sécurité",        rc2),
        ("Tests d'intégration",   rc3),
    ]
    all_passed = True
    for name, rc in steps:
        if rc is None:
            status = f"{YELLOW}SKIP (jamais lancé — serveur indisponible){RESET}"
            all_passed = False
        elif rc == 0:
            status = f"{GREEN}PASS{RESET}"
        elif rc == 5:
            status = f"{YELLOW}SKIP (aucun test collecté){RESET}"
        else:
            status = f"{RED}FAIL (code {rc}){RESET}"
            all_passed = False
        print(f"  {status}  {name}")

    if server_proc:
        server_proc.terminate()

    if all_passed:
        print(f"\n{GREEN}{BOLD}✅ Tous les tests passent — projet prêt pour la production !{RESET}\n")
    else:
        print(f"\n{RED}{BOLD}❌ Des tests ont échoué — voir les détails ci-dessus.{RESET}\n")

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
