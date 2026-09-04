"""
Tests d'intégration live pour /v1/auth/* (register, login, me, admin/*).
Mêmes conventions que test_api_final.py : requêtes via `requests` contre un
serveur déjà lancé (voir conftest.py pour --host/--port), pas de TestClient.
register/login/me sont publics (pas de X-API-Key) -- protégés par rate
limiting. Les endpoints /v1/auth/admin/* exigent une clé API admin (voir
`headers` dans conftest.py) -- un compte nouvellement inscrit reste
status="pending" et ne peut pas se connecter tant qu'il n'est pas approuvé
via ces endpoints (voir user_auth.py).
"""
import time

import requests


def _unique_username() -> str:
    return f"testuser_{int(time.time() * 1000)}"


def _register_payload(username: str) -> dict:
    return {
        "firstname": "Test",
        "lastname": "User",
        "email": f"{username}@example.com",
        "username": username,
        "password": "SuperSecret123",
    }


def _register_and_approve(base, headers, username=None) -> tuple[str, dict]:
    """Inscrit un utilisateur puis l'approuve via /v1/auth/admin/{id}/approve
    (clé API admin) -- un compte fraîchement créé est status="pending" et ne
    peut pas se connecter tant qu'il n'est pas validé. Renvoie
    (username, payload) prêt à être utilisé pour /login."""
    username = username or _unique_username()
    payload = _register_payload(username)
    reg = requests.post(f"{base}/v1/auth/register", json=payload)
    assert reg.status_code == 201, reg.text
    user_id = reg.json()["id"]
    approve = requests.post(f"{base}/v1/auth/admin/{user_id}/approve", headers=headers)
    assert approve.status_code == 200, approve.text
    return username, payload


def test_register_success(base):
    username = _unique_username()
    res = requests.post(f"{base}/v1/auth/register", json=_register_payload(username))
    assert res.status_code == 201, res.text
    body = res.json()
    assert body["username"] == username
    assert body["status"] == "pending"
    assert "password" not in body
    assert "password_hash" not in body


def test_register_duplicate_username(base):
    username = _unique_username()
    payload = _register_payload(username)

    first = requests.post(f"{base}/v1/auth/register", json=payload)
    assert first.status_code == 201, first.text

    dup_payload = dict(payload)
    dup_payload["email"] = f"other_{username}@example.com"
    second = requests.post(f"{base}/v1/auth/register", json=dup_payload)
    assert second.status_code == 409


def test_login_success(base, headers):
    username, payload = _register_and_approve(base, headers)

    res = requests.post(
        f"{base}/v1/auth/login",
        json={"username": username, "password": payload["password"]},
    )
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["access_token"]
    assert body["token_type"] == "bearer"
    assert body["user"]["username"] == username


def test_login_pending_account_rejected(base):
    """Un compte fraîchement inscrit (status="pending") ne doit pas pouvoir
    se connecter, même avec le bon mot de passe, tant qu'un admin ne l'a
    pas validé via /v1/auth/admin/{id}/approve."""
    username = _unique_username()
    payload = _register_payload(username)
    reg = requests.post(f"{base}/v1/auth/register", json=payload)
    assert reg.status_code == 201, reg.text

    res = requests.post(
        f"{base}/v1/auth/login",
        json={"username": username, "password": payload["password"]},
    )
    assert res.status_code == 403
    assert "attente" in res.json()["detail"].lower()


def test_admin_approve_flow(base, headers):
    """Le compte apparaît dans /admin/pending avant validation, disparaît
    après, et le login ne fonctionne qu'après l'approbation."""
    username = _unique_username()
    payload = _register_payload(username)
    reg = requests.post(f"{base}/v1/auth/register", json=payload)
    assert reg.status_code == 201, reg.text
    user_id = reg.json()["id"]

    pending = requests.get(f"{base}/v1/auth/admin/pending", headers=headers)
    assert pending.status_code == 200, pending.text
    assert any(u["id"] == user_id for u in pending.json())

    approve = requests.post(f"{base}/v1/auth/admin/{user_id}/approve", headers=headers)
    assert approve.status_code == 200, approve.text

    pending_after = requests.get(f"{base}/v1/auth/admin/pending", headers=headers)
    assert pending_after.status_code == 200, pending_after.text
    assert all(u["id"] != user_id for u in pending_after.json())

    res = requests.post(
        f"{base}/v1/auth/login",
        json={"username": username, "password": payload["password"]},
    )
    assert res.status_code == 200, res.text


def test_admin_reject_flow(base, headers):
    username = _unique_username()
    payload = _register_payload(username)
    reg = requests.post(f"{base}/v1/auth/register", json=payload)
    assert reg.status_code == 201, reg.text
    user_id = reg.json()["id"]

    reject = requests.post(f"{base}/v1/auth/admin/{user_id}/reject", headers=headers)
    assert reject.status_code == 200, reject.text

    res = requests.post(
        f"{base}/v1/auth/login",
        json={"username": username, "password": payload["password"]},
    )
    assert res.status_code == 403
    assert "refusé" in res.json()["detail"].lower()


def test_admin_endpoints_require_admin_key(base):
    res = requests.get(f"{base}/v1/auth/admin/pending")
    assert res.status_code == 401


def test_login_wrong_password(base):
    username = _unique_username()
    payload = _register_payload(username)
    reg = requests.post(f"{base}/v1/auth/register", json=payload)
    assert reg.status_code == 201, reg.text

    res = requests.post(
        f"{base}/v1/auth/login",
        json={"username": username, "password": "wrong-password"},
    )
    assert res.status_code == 401


def test_login_unknown_username(base):
    res = requests.post(
        f"{base}/v1/auth/login",
        json={"username": _unique_username(), "password": "whatever123"},
    )
    assert res.status_code == 401


def test_me_with_valid_token(base, headers):
    username, payload = _register_and_approve(base, headers)

    login = requests.post(
        f"{base}/v1/auth/login",
        json={"username": username, "password": payload["password"]},
    )
    assert login.status_code == 200, login.text
    token = login.json()["access_token"]

    res = requests.get(
        f"{base}/v1/auth/me",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert res.status_code == 200, res.text
    assert res.json()["username"] == username


def test_me_without_token(base):
    res = requests.get(f"{base}/v1/auth/me")
    assert res.status_code == 401
