"""
user_auth.py
============
Authentification par compte utilisateur (username + mot de passe) pour les
clients humains (dashboard, future app mobile) — complémentaire à auth.py
(clés API pour l'accès machine-à-machine), qui reste inchangé.

Fonctionnement :
  - POST /v1/auth/register  crée un compte (mot de passe hashé bcrypt)
  - POST /v1/auth/login     vérifie les identifiants, renvoie un JWT (HS256)
  - GET  /v1/auth/me        renvoie l'identité déduite du JWT (Bearer token)

Configuration (variables d'environnement) :
  JWT_SECRET          clé de signature JWT — DOIT différer de API_KEYS
  JWT_EXPIRE_MINUTES  durée de validité d'un token (défaut 60)

Table users : créée au démarrage par init_users_table() (voir lifespan()
dans api_unified_pythagore.py) via CREATE TABLE IF NOT EXISTS, sans bloquer
le démarrage de l'API si la base est injoignable.
"""

import logging
import os
from datetime import datetime, timedelta, timezone

import bcrypt
import jwt
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, EmailStr, Field

from config import MARIADB_HOST, MARIADB_PORT, MARIADB_USER, MARIADB_PASSWORD, MARIADB_DATABASE
from rate_limiter import make_rate_limiter

log = logging.getLogger("user_auth")

JWT_SECRET = os.getenv("JWT_SECRET", "")
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_MINUTES = int(os.getenv("JWT_EXPIRE_MINUTES", "60"))

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
                  created_at    DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
                  last_login    DATETIME     NULL,
                  UNIQUE KEY uq_users_username (username),
                  UNIQUE KEY uq_users_email (email)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                """
            )
            cur.close()
            log.info("Table users prête")
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


class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user: dict


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
                "INSERT INTO users (username, email, firstname, lastname, password_hash) "
                "VALUES (%s, %s, %s, %s, %s)",
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

    log.info(f"Nouvel utilisateur inscrit : {req.username} (id={user_id})")
    return RegisterResponse(id=user_id, username=req.username, email=req.email)


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
                "SELECT id, username, password_hash, role FROM users WHERE username = %s",
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


@router.get("/me", summary="Identité de l'utilisateur connecté")
def me(claims: dict = Depends(require_user_token)):
    return {
        "id": int(claims["sub"]),
        "username": claims["username"],
        "role": claims["role"],
    }
