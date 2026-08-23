"""
Tests d'intégration live pour /v1/auth/* (register, login, me).
Mêmes conventions que test_api_final.py : requêtes via `requests` contre un
serveur déjà lancé (voir conftest.py pour --host/--port), pas de TestClient.
Ces endpoints sont publics (pas de X-API-Key) -- protégés par rate limiting.
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


def test_register_success(base):
    username = _unique_username()
    res = requests.post(f"{base}/v1/auth/register", json=_register_payload(username))
    assert res.status_code == 201, res.text
    body = res.json()
    assert body["username"] == username
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


def test_login_success(base):
    username = _unique_username()
    payload = _register_payload(username)
    reg = requests.post(f"{base}/v1/auth/register", json=payload)
    assert reg.status_code == 201, reg.text

    res = requests.post(
        f"{base}/v1/auth/login",
        json={"username": username, "password": payload["password"]},
    )
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["access_token"]
    assert body["token_type"] == "bearer"
    assert body["user"]["username"] == username


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


def test_me_with_valid_token(base):
    username = _unique_username()
    payload = _register_payload(username)
    reg = requests.post(f"{base}/v1/auth/register", json=payload)
    assert reg.status_code == 201, reg.text

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
