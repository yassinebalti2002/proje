"""
user_auth.py
============
Authentification par compte utilisateur (username + mot de passe) pour les
clients humains (dashboard, future app mobile) — complémentaire à auth.py
(clés API pour l'accès machine-à-machine), qui reste inchangé.

Fonctionnement :
  - POST /v1/auth/register         crée un compte (mot de passe hashé bcrypt),
                                    statut "pending" -- ne peut pas se connecter
                                    tant qu'un admin ne l'a pas validé
  - POST /v1/auth/login            vérifie les identifiants ET le statut du
                                    compte, renvoie un JWT (HS256) si approuvé
  - POST /v1/auth/forgot-password  envoie par email un lien de réinitialisation
  - POST /v1/auth/reset-password   applique un nouveau mot de passe via ce lien
  - GET  /v1/auth/me               renvoie l'identité déduite du JWT (Bearer token)
  - GET  /v1/auth/admin/pending           liste les comptes en attente (clé API admin)
  - POST /v1/auth/admin/{id}/approve      valide un compte en attente (clé API admin)
  - POST /v1/auth/admin/{id}/reject       refuse un compte en attente (clé API admin)

Validation des inscriptions :
  Un compte nouvellement créé a status="pending" et ne peut pas se connecter
  (login renvoie 403) tant qu'un administrateur ne l'a pas approuvé via les
  endpoints /v1/auth/admin/*. Ces endpoints réutilisent l'authentification par
  clé API admin (require_admin_key, voir auth.py) déjà utilisée pour les autres
  actions sensibles de cette API -- pas de rôle web séparé à gérer : quiconque
  détient une clé API_KEYS (admin) peut valider/refuser un compte via la page
  admin-users.html ou directement en HTTP.

Configuration (variables d'environnement) :
  JWT_SECRET                  clé de signature JWT — DOIT différer de API_KEYS
  JWT_EXPIRE_MINUTES          durée de validité d'un token (défaut 60)
  RESET_TOKEN_EXPIRE_MINUTES  durée de validité d'un lien de réinitialisation (défaut 30)

Tables users + password_resets : créées au démarrage par init_users_table()
(voir lifespan() dans api_unified_pythagore.py) via CREATE TABLE IF NOT
EXISTS, sans bloquer le démarrage de l'API si la base est injoignable.

L'envoi de l'email de réinitialisation réutilise le compte SMTP configuré
dans alert_config.json (voir alert_manager.py) — même émetteur que les
alertes capteurs, mais indépendant du flag "enabled" de celles-ci.
"""

import hashlib
import logging
import os
import secrets
import threading
from datetime import datetime, timedelta, timezone

import bcrypt
import jwt
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, EmailStr, Field

from config import MARIADB_HOST, MARIADB_PORT, MARIADB_USER, MARIADB_PASSWORD, MARIADB_DATABASE
from rate_limiter import make_rate_limiter
from auth import require_admin_key

log = logging.getLogger("user_auth")

JWT_SECRET = os.getenv("JWT_SECRET", "")
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_MINUTES = int(os.getenv("JWT_EXPIRE_MINUTES", "60"))
RESET_TOKEN_EXPIRE_MINUTES = int(os.getenv("RESET_TOKEN_EXPIRE_MINUTES", "30"))

if not JWT_SECRET:
    log.warning(
        "JWT_SECRET non défini — l'authentification utilisateur refusera toutes "
        "les requêtes. Définissez JWT_SECRET dans votre .env (voir .env.example)."
    )

# Hash bcrypt précalculé (mot de passe factice) utilisé pour égaliser le temps
# de réponse quand le username n'existe pas -- évite qu'un attaquant déduise
# l'existence d'un compte à partir du délai de réponse du endpoint /login.
_DUMMY_HASH = bcrypt.hashpw(b"dummy-password-for-timing", bcrypt.gensalt())

_bearer_scheme = HTTPAuthorizationCredentials  # type hint helper only
_security = HTTPBearer(auto_error=False)


# ══════════════════════════════════════════════════════════════════════════
#  Accès base de données (connexion courte par requête)
# ══════════════════════════════════════════════════════════════════════════

def _get_db_connection():
    import mysql.connector
    return mysql.connector.connect(
        host=MARIADB_HOST,
        port=MARIADB_PORT,
        user=MARIADB_USER,
        password=MARIADB_PASSWORD,
        database=MARIADB_DATABASE,
        connection_timeout=10,
        autocommit=True,
    )


def init_users_table() -> None:
    """Crée la table users si absente. Ne lève jamais -- une base injoignable
    au démarrage ne doit pas empêcher l'API de démarrer (les endpoints
    /register et /login échoueront alors individuellement en 503)."""
    try:
        conn = _get_db_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                  id            INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
                  username      VARCHAR(50)  NOT NULL,
                  email         VARCHAR(255) NOT NULL,
                  firstname     VARCHAR(100) NOT NULL,
                  lastname      VARCHAR(100) NOT NULL,
                  password_hash VARCHAR(255) NOT NULL,
                  role          VARCHAR(20)  NOT NULL DEFAULT 'user',
                  status        VARCHAR(20)  NOT NULL DEFAULT 'approved',
                  created_at    DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
                  last_login    DATETIME     NULL,
                  UNIQUE KEY uq_users_username (username),
                  UNIQUE KEY uq_users_email (email)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                """
            )
            # Migration pour les bases créées avant l'ajout de la validation
            # admin -- ADD COLUMN IF NOT EXISTS (MariaDB) ; DEFAULT 'approved'
            # pour ne pas verrouiller les comptes déjà existants et actifs.
            # Les nouvelles inscriptions passent explicitement status='pending'
            # dans l'INSERT de register() ci-dessous.
            try:
                cur.execute(
                    "ALTER TABLE users ADD COLUMN IF NOT EXISTS "
                    "status VARCHAR(20) NOT NULL DEFAULT 'approved'"
                )
            except Exception as e:
                log.warning(f"Migration colonne users.status ignorée : {e}")
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS password_resets (
                  id            INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
                  user_id       INT UNSIGNED NOT NULL,
                  token_hash    CHAR(64)     NOT NULL,
                  expires_at    DATETIME     NOT NULL,
                  used_at       DATETIME     NULL,
                  created_at    DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
                  KEY idx_token_hash (token_hash),
                  CONSTRAINT fk_password_resets_user FOREIGN KEY (user_id)
                    REFERENCES users(id) ON DELETE CASCADE
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                """
            )
            cur.close()
            log.info("Tables users + password_resets prêtes")
        finally:
            conn.close()
    except Exception as e:
        log.warning(
            f"Impossible d'initialiser la table users : {e} — /v1/auth/register "
            "et /v1/auth/login répondront 503 tant que la base est injoignable."
        )


# ══════════════════════════════════════════════════════════════════════════
#  Mots de passe (bcrypt) et JWT (PyJWT, HS256)
# ══════════════════════════════════════════════════════════════════════════

def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))


def _hash_reset_token(token: str) -> str:
    """SHA-256 du token de réinitialisation -- seul le hash est stocké en base
    (même logique que _fingerprint() dans auth.py pour les clés API) : si la
    table password_resets fuitait, elle ne contiendrait aucun token exploitable."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _send_reset_email(to_email: str, token: str) -> None:
    """Envoie le lien de réinitialisation par email, en arrière-plan (l'appel
    SMTP peut prendre plusieurs secondes -- ne doit pas bloquer la réponse
    HTTP de /forgot-password)."""
    def _worker():
        import core  # import tardif : core importe ce module -> import en tête créerait un cycle
        if not core.ALERTS_ENABLED or core._alert_manager is None:
            log.warning("Email de réinitialisation non envoyé — AlertManager indisponible")
            return
        from alert_manager import PUBLIC_HOST
        reset_url = f"http://{PUBLIC_HOST}:3000/reset-password.html?token={token}"
        subject = "Réinitialisation de votre mot de passe — Novation City"
        text = (
            "Réinitialisation de mot de passe — Novation City\n"
            f"{'='*40}\n"
            "Une demande de réinitialisation de mot de passe a été effectuée pour ce compte.\n"
            f"Lien (valable {RESET_TOKEN_EXPIRE_MINUTES} minutes) :\n{reset_url}\n\n"
            "Si vous n'êtes pas à l'origine de cette demande, ignorez cet email."
        )
        html = f"""
<!DOCTYPE html>
<html lang="fr">
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width"></head>
<body style="font-family:Arial,sans-serif;background:#f3f4f6;padding:20px;margin:0;">
  <div style="max-width:600px;margin:0 auto;background:#fff;border-radius:12px;
              box-shadow:0 4px 12px rgba(0,0,0,0.1);overflow:hidden;">
    <div style="background:#00857a;color:#fff;padding:24px;text-align:center;">
      <h1 style="margin:0;font-size:22px;">🔑 Réinitialisation de mot de passe</h1>
    </div>
    <div style="padding:24px;">
      <p>Une demande de réinitialisation de mot de passe a été effectuée pour ce compte.</p>
      <p>Ce lien est valable <strong>{RESET_TOKEN_EXPIRE_MINUTES} minutes</strong>.</p>
      <div style="margin:24px 0;text-align:center;">
        <a href="{reset_url}"
           style="background:#00857a;color:#fff;padding:12px 24px;
                  border-radius:8px;text-decoration:none;font-weight:bold;">
          Réinitialiser mon mot de passe
        </a>
      </div>
      <p style="color:#6b7280;font-size:13px;">
        Si vous n'êtes pas à l'origine de cette demande, ignorez cet email —
        votre mot de passe actuel reste inchangé.
      </p>
    </div>
    <div style="background:#f9fafb;padding:16px;text-align:center;
                font-size:12px;color:#6b7280;">
      Système de Maintenance Prédictive — ISG Bizerte / Novation City
    </div>
  </div>
</body>
</html>
"""
        core._alert_manager.send_generic_email(to_email, subject, html, text)

    threading.Thread(target=_worker, daemon=True, name="reset-email").start()


def create_token(user_id: int, username: str, role: str) -> tuple[str, int]:
    expires_in = JWT_EXPIRE_MINUTES * 60
    payload = {
        "sub": str(user_id),
        "username": username,
        "role": role,
        "exp": datetime.now(timezone.utc) + timedelta(minutes=JWT_EXPIRE_MINUTES),
        "iat": datetime.now(timezone.utc),
    }
    token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)
    return token, expires_in


def require_user_token(creds: HTTPAuthorizationCredentials | None = Depends(_security)) -> dict:
    """Dépendance FastAPI pour les endpoints nécessitant un utilisateur
    connecté (JWT). Distincte de require_api_key/require_admin_key
    (auth.py) qui restent utilisées pour l'accès machine-à-machine."""
    if creds is None or not creds.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="En-tête Authorization: Bearer <token> manquant.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if not JWT_SECRET:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentification utilisateur non configurée côté serveur.",
        )
    try:
        claims = jwt.decode(creds.credentials, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token expiré.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token invalide.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return claims


# ══════════════════════════════════════════════════════════════════════════
#  Modèles Pydantic
# ══════════════════════════════════════════════════════════════════════════

class RegisterRequest(BaseModel):
    firstname: str = Field(..., min_length=1, max_length=100)
    lastname: str = Field(..., min_length=1, max_length=100)
    email: EmailStr
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=8, max_length=72)


class RegisterResponse(BaseModel):
    id: int
    username: str
    email: str
    status: str = "pending"


class PendingUserResponse(BaseModel):
    id: int
    username: str
    email: str
    firstname: str
    lastname: str
    created_at: str


class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user: dict


class ForgotPasswordRequest(BaseModel):
    email: EmailStr


class ResetPasswordRequest(BaseModel):
    token: str = Field(..., min_length=1)
    new_password: str = Field(..., min_length=8, max_length=72)


# ══════════════════════════════════════════════════════════════════════════
#  Endpoints
# ══════════════════════════════════════════════════════════════════════════

router = APIRouter()

_INVALID_CREDENTIALS_MSG = "Identifiant ou mot de passe incorrect."


@router.post(
    "/register",
    response_model=RegisterResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Créer un compte utilisateur",
)
def register(req: RegisterRequest, _rl=Depends(make_rate_limiter(10))):
    import mysql.connector

    password_hash = hash_password(req.password)
    try:
        conn = _get_db_connection()
    except Exception as e:
        log.warning(f"DB injoignable pour /register : {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Base de données utilisateurs indisponible.",
        )
    try:
        cur = conn.cursor()
        try:
            cur.execute(
                "INSERT INTO users (username, email, firstname, lastname, password_hash, status) "
                "VALUES (%s, %s, %s, %s, %s, 'pending')",
                (req.username, req.email, req.firstname, req.lastname, password_hash),
            )
            user_id = cur.lastrowid
        except mysql.connector.IntegrityError:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Ce nom d'utilisateur ou cet email est déjà utilisé.",
            )
        finally:
            cur.close()
    finally:
        conn.close()

    log.info(f"Nouvel utilisateur inscrit (en attente de validation admin) : {req.username} (id={user_id})")
    return RegisterResponse(id=user_id, username=req.username, email=req.email, status="pending")


@router.post("/login", response_model=TokenResponse, summary="Se connecter")
def login(req: LoginRequest, _rl=Depends(make_rate_limiter(10))):
    try:
        conn = _get_db_connection()
    except Exception as e:
        log.warning(f"DB injoignable pour /login : {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Base de données utilisateurs indisponible.",
        )
    try:
        cur = conn.cursor(dictionary=True)
        try:
            cur.execute(
                "SELECT id, username, password_hash, role, status FROM users WHERE username = %s",
                (req.username,),
            )
            row = cur.fetchone()
        finally:
            cur.close()

        if row is None:
            verify_password(req.password, _DUMMY_HASH.decode("utf-8"))
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=_INVALID_CREDENTIALS_MSG,
            )
        if not verify_password(req.password, row["password_hash"]):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=_INVALID_CREDENTIALS_MSG,
            )
        # Identifiants corrects -- on peut maintenant renseigner le statut du
        # compte sans risque d'énumération (l'appelant vient de prouver qu'il
        # est bien le propriétaire de ce compte).
        if row["status"] == "pending":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Compte en attente de validation par un administrateur.",
            )
        if row["status"] == "rejected":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Ce compte a été refusé par un administrateur. Contactez le support.",
            )

        cur = conn.cursor()
        try:
            cur.execute("UPDATE users SET last_login = NOW() WHERE id = %s", (row["id"],))
        finally:
            cur.close()
    finally:
        conn.close()

    token, expires_in = create_token(row["id"], row["username"], row["role"])
    return TokenResponse(
        access_token=token,
        expires_in=expires_in,
        user={"id": row["id"], "username": row["username"], "role": row["role"]},
    )


@router.post(
    "/forgot-password",
    summary="Demander un lien de réinitialisation de mot de passe",
)
def forgot_password(req: ForgotPasswordRequest, _rl=Depends(make_rate_limiter(5))):
    # Message générique dans tous les cas (compte existant ou non, DB
    # joignable ou non) -- ne jamais révéler si un email correspond à un
    # compte, sinon l'endpoint devient un outil d'énumération de comptes.
    generic_msg = {"message": "Si un compte existe avec cet email, un lien de réinitialisation a été envoyé."}
    try:
        conn = _get_db_connection()
    except Exception as e:
        log.warning(f"DB injoignable pour /forgot-password : {e}")
        return generic_msg

    try:
        cur = conn.cursor(dictionary=True)
        try:
            cur.execute("SELECT id FROM users WHERE email = %s", (req.email,))
            row = cur.fetchone()
        finally:
            cur.close()

        if row is not None:
            token = secrets.token_urlsafe(32)
            token_hash = _hash_reset_token(token)
            cur = conn.cursor()
            try:
                cur.execute(
                    "INSERT INTO password_resets (user_id, token_hash, expires_at) "
                    "VALUES (%s, %s, DATE_ADD(NOW(), INTERVAL %s MINUTE))",
                    (row["id"], token_hash, RESET_TOKEN_EXPIRE_MINUTES),
                )
            finally:
                cur.close()
            _send_reset_email(req.email, token)
            log.info(f"Lien de réinitialisation généré pour user_id={row['id']}")
    finally:
        conn.close()

    return generic_msg


@router.post(
    "/reset-password",
    summary="Réinitialiser le mot de passe à partir d'un token reçu par email",
)
def reset_password(req: ResetPasswordRequest, _rl=Depends(make_rate_limiter(10))):
    token_hash = _hash_reset_token(req.token)
    _invalid = HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail="Lien de réinitialisation invalide ou expiré.",
    )
    try:
        conn = _get_db_connection()
    except Exception as e:
        log.warning(f"DB injoignable pour /reset-password : {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Base de données utilisateurs indisponible.",
        )
    try:
        cur = conn.cursor(dictionary=True)
        try:
            cur.execute(
                "SELECT id, user_id FROM password_resets "
                "WHERE token_hash = %s AND used_at IS NULL AND expires_at > NOW()",
                (token_hash,),
            )
            row = cur.fetchone()
        finally:
            cur.close()

        if row is None:
            raise _invalid

        new_hash = hash_password(req.new_password)
        cur = conn.cursor()
        try:
            cur.execute("UPDATE users SET password_hash = %s WHERE id = %s", (new_hash, row["user_id"]))
            cur.execute("UPDATE password_resets SET used_at = NOW() WHERE id = %s", (row["id"],))
        finally:
            cur.close()
    finally:
        conn.close()

    log.info(f"Mot de passe réinitialisé pour user_id={row['user_id']}")
    return {"message": "Mot de passe réinitialisé avec succès."}


@router.get("/me", summary="Identité de l'utilisateur connecté")
def me(claims: dict = Depends(require_user_token)):
    return {
        "id": int(claims["sub"]),
        "username": claims["username"],
        "role": claims["role"],
    }


# ══════════════════════════════════════════════════════════════════════════
#  Endpoints admin — validation des inscriptions
#  Protégés par require_admin_key (clé API admin, voir auth.py) plutôt que
#  par un rôle web séparé : cette API a déjà deux niveaux de clé API
#  (admin/opérateur) pour ses actions sensibles (ex: ré-entraînement), on
#  réutilise le même mécanisme ici pour ne pas dupliquer un système d'auth.
# ══════════════════════════════════════════════════════════════════════════

@router.get(
    "/admin/pending",
    response_model=list[PendingUserResponse],
    summary="Lister les comptes en attente de validation (clé API admin requise)",
)
def list_pending_users(_admin_key: str = Depends(require_admin_key)):
    try:
        conn = _get_db_connection()
    except Exception as e:
        log.warning(f"DB injoignable pour /admin/pending : {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Base de données utilisateurs indisponible.",
        )
    try:
        cur = conn.cursor(dictionary=True)
        try:
            cur.execute(
                "SELECT id, username, email, firstname, lastname, created_at "
                "FROM users WHERE status = 'pending' ORDER BY created_at ASC"
            )
            rows = cur.fetchall()
        finally:
            cur.close()
    finally:
        conn.close()

    return [
        PendingUserResponse(
            id=r["id"], username=r["username"], email=r["email"],
            firstname=r["firstname"], lastname=r["lastname"],
            created_at=r["created_at"].isoformat() if r["created_at"] else "",
        )
        for r in rows
    ]


def _set_user_status(user_id: int, new_status: str) -> None:
    """Bascule le statut d'un compte -- factorisé car approve/reject ne
    diffèrent que par la valeur écrite."""
    try:
        conn = _get_db_connection()
    except Exception as e:
        log.warning(f"DB injoignable pour changement de statut utilisateur : {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Base de données utilisateurs indisponible.",
        )
    try:
        cur = conn.cursor()
        try:
            cur.execute("SELECT id FROM users WHERE id = %s", (user_id,))
            if cur.fetchone() is None:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Utilisateur introuvable.",
                )
            cur.execute("UPDATE users SET status = %s WHERE id = %s", (new_status, user_id))
        finally:
            cur.close()
    finally:
        conn.close()


@router.post(
    "/admin/{user_id}/approve",
    summary="Valider un compte en attente (clé API admin requise)",
)
def approve_user(user_id: int, _admin_key: str = Depends(require_admin_key)):
    _set_user_status(user_id, "approved")
    log.info(f"Compte utilisateur id={user_id} approuvé par un administrateur")
    return {"message": "Compte validé."}


@router.post(
    "/admin/{user_id}/reject",
    summary="Refuser un compte en attente (clé API admin requise)",
)
def reject_user(user_id: int, _admin_key: str = Depends(require_admin_key)):
    _set_user_status(user_id, "rejected")
    log.info(f"Compte utilisateur id={user_id} refusé par un administrateur")
    return {"message": "Compte refusé."}
