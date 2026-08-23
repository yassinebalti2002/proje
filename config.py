# ==============================================================================
#  CONFIG.PY — Fichier de configuration central
#  Modifiez uniquement ce fichier quand l'adresse IP ou les paramètres changent.
#  Tous les autres fichiers Python lisent leurs valeurs depuis ici.
#
#  Les valeurs par défaut ci-dessous sont surchargeables via variables
#  d'environnement (.env) — évite d'avoir un mot de passe en clair comme
#  seule source de vérité sur le disque.
# ==============================================================================

import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ── Serveur IoT / MariaDB ──────────────────────────────────────────────────────
MARIADB_HOST     = os.getenv("MARIADB_HOST", "192.168.120.58")   # ← IP du serveur IoT sur le réseau local
MARIADB_PORT     = int(os.getenv("MARIADB_PORT", "3306"))
MARIADB_USER     = os.getenv("MARIADB_USER", "root")
MARIADB_PASSWORD = os.getenv("MARIADB_PASSWORD", "")
MARIADB_DATABASE = os.getenv("MARIADB_DATABASE", "ai_cp")
MARIADB_TABLE    = "full_data"


# ── Gateway IFM (accès direct HTTP) ───────────────────────────────────────────
IFM_GATEWAY_HOST = os.getenv("IFM_GATEWAY_HOST", "192.168.120.58")   # ← même serveur IoT, port HTTP 80
IFM_GATEWAY_PORT = int(os.getenv("IFM_GATEWAY_PORT", "80"))

# ── API FastAPI (locale) ───────────────────────────────────────────────────────
API_HOST         = "localhost"
API_PORT         = 8000

# ── Serveur dashboard HTTP ─────────────────────────────────────────────────────
DASHBOARD_PORT   = 3000

# ── Liste des capteurs IFM surveillés ─────────────────────────────────────────
SENSORS = [
    "8f7f2f7e", "0ff416d2", "b2acdf45", "6e0c1740", "07da47b8",
    "aa7b02a1", "eb084747", "2c6254af", "53cb61b2", "a6a46be1",
    "bc59bf5f", "d9508e77", "f48c25f9", "68c11f06",
    "4b5e4b32", "3a782f1b", "718fd2af", "91d92804", "99695e98",
    "ed6fa322",
]
