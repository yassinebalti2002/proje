"""
auth.py
=======
Authentification par API Key pour l'API Maintenance Prédictive, avec 2 niveaux
de privilège (RBAC minimal) et journal d'audit.

Fonctionnement :
  - Le client envoie l'en-tête  X-API-Key: <clé>
  - La clé est comparée (en temps constant) à deux listes : ADMIN et OPERATOR
  - Les endpoints "opérateur" (prédiction, consultation) acceptent les deux rôles
  - Les endpoints "admin" (déclenchement de ré-entraînement) exigent le rôle admin
  - Chaque tentative d'authentification (succès ou échec) est journalisée dans
    audit.log — sans jamais écrire la clé en clair (seulement une empreinte
    courte, suffisante pour corréler "quelle clé" sans pouvoir la reconstituer)

Configuration (variables d'environnement) :
  API_KEYS=cle1,cle2              # clés ADMIN — accès complet (retro-compatible)
  API_KEYS_OPERATOR=cle3,cle4     # clés OPERATEUR — accès lecture/prédiction,
                                   # PAS le déclenchement de ré-entraînement
  API_KEY_HEADER=X-API-Key        # nom de l'en-tête (défaut conservé ci-dessus)

Génération d'une clé sûre (Python) :
  python -c "import secrets; print(secrets.token_hex(32))"
"""

import hashlib
import hmac
import json
import os
import logging
from datetime import datetime, timezone
from pathlib import Path
from fastapi import HTTPException, Request, Security, status
from fastapi.security.api_key import APIKeyHeader

log = logging.getLogger("auth")

_HEADER_NAME = os.getenv("API_KEY_HEADER", "X-API-Key")
_api_key_header = APIKeyHeader(name=_HEADER_NAME, auto_error=False)

AUDIT_LOG_PATH = Path(os.getenv("AUDIT_LOG_PATH", "audit.log"))

ROLE_ADMIN = "admin"
ROLE_OPERATOR = "operator"


# ══════════════════════════════════════════════════════════════════════════
#  Chargement des clés — 2 rôles
# ══════════════════════════════════════════════════════════════════════════

def _parse_keys(env_var: str) -> set[str]:
    raw = os.getenv(env_var, "")
    return {k.strip() for k in raw.split(",") if k.strip()}


def _load_keys() -> tuple[set[str], set[str]]:
    """
    Charge les clés ADMIN (API_KEYS, historique) et OPERATOR (API_KEYS_OPERATOR,
    nouveau, optionnel). Si API_KEYS_OPERATOR est absent, seul le rôle admin
    existe -- comportement identique à avant l'introduction du RBAC.
    """
    admin = _parse_keys("API_KEYS")
    operator = _parse_keys("API_KEYS_OPERATOR")
    if not admin and not operator:
        log.warning(
            "API_KEYS non défini — toutes les requêtes seront refusées (401). "
            "Définissez API_KEYS=<votre_cle> dans votre .env ou variable d'environnement."
        )
    else:
        log.info(
            f"Auth initialisée — {len(admin)} clé(s) admin, {len(operator)} clé(s) opérateur"
        )
    return admin, operator


ADMIN_KEYS: set[str] = set()
OPERATOR_KEYS: set[str] = set()
ADMIN_KEYS, OPERATOR_KEYS = _load_keys()

# Retro-compatibilite : union des deux roles. Certains appelants externes
# (tests, scripts) verifient juste "cette cle est-elle valide, tous roles
# confondus ?" sans se soucier du RBAC -- VALID_API_KEYS repond a ca.
VALID_API_KEYS: set[str] = ADMIN_KEYS | OPERATOR_KEYS


def _constant_time_compare(a: str, b: str) -> bool:
    """Comparaison en temps constant — résiste aux timing attacks."""
    return hmac.compare_digest(a.encode(), b.encode())


def _resolve_role(api_key: str) -> str | None:
    """Retourne le rôle associé à la clé (admin prioritaire), ou None si invalide."""
    if any(_constant_time_compare(api_key, k) for k in ADMIN_KEYS):
        return ROLE_ADMIN
    if any(_constant_time_compare(api_key, k) for k in OPERATOR_KEYS):
        return ROLE_OPERATOR
    return None


def _fingerprint(api_key: str) -> str:
    """Empreinte courte non réversible -- permet de corréler des logs sans
    jamais écrire la clé elle-même sur disque."""
    return hashlib.sha256(api_key.encode()).hexdigest()[:12]


# ══════════════════════════════════════════════════════════════════════════
#  Journal d'audit
# ══════════════════════════════════════════════════════════════════════════

def _audit(result: str, request: Request | None, api_key: str, role: str | None) -> None:
    """Ajoute une ligne JSON au journal d'audit. Ne doit jamais lever
    d'exception -- une panne d'écriture du log ne doit pas casser l'API."""
    try:
        entry = {
            "ts":     datetime.now(timezone.utc).isoformat(),
            "result": result,                                    # GRANTED | DENIED
            "role":   role,
            "key":    _fingerprint(api_key) if api_key else None,
            "ip":     (request.client.host if request and request.client else None),
            "method": (request.method if request else None),
            "path":   (request.url.path if request else None),
        }
        with AUDIT_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        log.warning(f"Écriture audit.log échouée : {e}")


# ══════════════════════════════════════════════════════════════════════════
#  Dépendances FastAPI
# ══════════════════════════════════════════════════════════════════════════

def require_api_key(request: Request, api_key: str = Security(_api_key_header)) -> str:
    """
    Dépendance FastAPI pour les endpoints de niveau OPÉRATEUR (accepte les
    clés admin ET opérateur). Utilisation inchangée par rapport à avant le RBAC :

        @app.post("/v1/predict")
        def predict(req: ..., _: str = Depends(require_api_key)):
            ...
    """
    if not api_key:
        _audit("DENIED", request, "", None)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="En-tête X-API-Key manquant.",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    if not ADMIN_KEYS and not OPERATOR_KEYS:
        _audit("DENIED", request, api_key, None)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Aucune clé API configurée côté serveur. Contactez l'administrateur.",
        )
    role = _resolve_role(api_key)
    if role is None:
        _audit("DENIED", request, api_key, None)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Clé API invalide.",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    _audit("GRANTED", request, api_key, role)
    return api_key


def require_admin_key(request: Request, api_key: str = Security(_api_key_header)) -> str:
    """
    Dépendance FastAPI pour les endpoints de niveau ADMIN uniquement (ex:
    déclenchement d'un ré-entraînement via /v1/pipeline/upload). Une clé
    opérateur valide reçoit 403 (authentifiée mais pas autorisée), distinct
    du 401 (non authentifiée du tout).
    """
    if not api_key:
        _audit("DENIED", request, "", None)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="En-tête X-API-Key manquant.",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    role = _resolve_role(api_key)
    if role is None:
        _audit("DENIED", request, api_key, None)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Clé API invalide.",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    if role != ROLE_ADMIN:
        _audit("FORBIDDEN", request, api_key, role)
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Cette opération nécessite une clé API de niveau admin.",
        )
    _audit("GRANTED", request, api_key, role)
    return api_key


def reload_keys() -> dict:
    """Recharge les clés depuis l'environnement sans redémarrage (hot reload)."""
    global ADMIN_KEYS, OPERATOR_KEYS, VALID_API_KEYS
    ADMIN_KEYS, OPERATOR_KEYS = _load_keys()
    VALID_API_KEYS = ADMIN_KEYS | OPERATOR_KEYS
    return {"admin": len(ADMIN_KEYS), "operator": len(OPERATOR_KEYS)}
