"""
routers/pages.py
=================
Pages HTML statiques servies directement par l'API (fonctionnent même sans
le conteneur "dashboard" nginx) : login, register.
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
