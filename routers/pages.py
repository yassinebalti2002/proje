"""
routers/pages.py
=================
Pages HTML statiques servies directement par l'API (fonctionnent même sans
le conteneur "dashboard" nginx) : login, register, mot de passe oublié,
et validation des comptes (admin).
"""

from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import HTMLResponse, FileResponse

router = APIRouter(tags=["Authentification utilisateurs"], include_in_schema=False)

_PROJECT_DIR = Path(__file__).parent.parent


# Chemins avec suffixe .html (pas juste /login) pour que les liens relatifs
# login.html <-> register.html restent valides, qu'ils soient servis par
# nginx (dashboard, volumes montés sous ce même nom) ou par l'API ici.
@router.get("/login.html")
def get_login_page():
    html_path = _PROJECT_DIR / "login.html"
    if html_path.exists():
        return FileResponse(str(html_path), media_type="text/html")
    return HTMLResponse("<h1>login.html introuvable</h1>", status_code=404)


@router.get("/register.html")
def get_register_page():
    html_path = _PROJECT_DIR / "register.html"
    if html_path.exists():
        return FileResponse(str(html_path), media_type="text/html")
    return HTMLResponse("<h1>register.html introuvable</h1>", status_code=404)


@router.get("/forgot-password.html")
def get_forgot_password_page():
    html_path = _PROJECT_DIR / "forgot-password.html"
    if html_path.exists():
        return FileResponse(str(html_path), media_type="text/html")
    return HTMLResponse("<h1>forgot-password.html introuvable</h1>", status_code=404)


@router.get("/reset-password.html")
def get_reset_password_page():
    html_path = _PROJECT_DIR / "reset-password.html"
    if html_path.exists():
        return FileResponse(str(html_path), media_type="text/html")
    return HTMLResponse("<h1>reset-password.html introuvable</h1>", status_code=404)


@router.get("/admin-users.html")
def get_admin_users_page():
    html_path = _PROJECT_DIR / "admin-users.html"
    if html_path.exists():
        return FileResponse(str(html_path), media_type="text/html")
    return HTMLResponse("<h1>admin-users.html introuvable</h1>", status_code=404)


# Bouton clair/sombre partagé par toutes les pages HTML (dashboards inclus) --
# servi ici pour que les pages fonctionnent aussi quand l'API (port 8000) est
# accédée directement, sans passer par nginx (port 3000, voir docker-compose.yml).
@router.get("/theme-toggle.js")
def get_theme_toggle_js():
    path = _PROJECT_DIR / "theme-toggle.js"
    if path.exists():
        return FileResponse(str(path), media_type="text/javascript")
    return HTMLResponse("theme-toggle.js introuvable", status_code=404)


@router.get("/theme-toggle.css")
def get_theme_toggle_css():
    path = _PROJECT_DIR / "theme-toggle.css"
    if path.exists():
        return FileResponse(str(path), media_type="text/css")
    return HTMLResponse("theme-toggle.css introuvable", status_code=404)
