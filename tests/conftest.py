"""Fixtures pytest partagées entre tous les fichiers de test."""
import os
import pytest
import requests


def pytest_addoption(parser):
    parser.addoption("--host",    default="localhost", help="Hôte API")
    parser.addoption("--port",    default=8000,        type=int, help="Port API")
    parser.addoption("--api-key", default=None,        help="Clé API (X-API-Key). Utilise API_KEYS env si absent.")


@pytest.fixture(scope="session")
def base(request):
    host = request.config.getoption("--host")
    port = request.config.getoption("--port")
    # HTTPS : l'API est TLS-only depuis l'activation du certificat auto-signé
    # (voir generate_selfsigned_cert.py / docker-compose.yml) -- voir la
    # fixture autouse ci-dessous pour la désactivation de la vérification
    # du certificat côté tests.
    return f"https://{host}:{port}"


@pytest.fixture(scope="session", autouse=True)
def _disable_tls_verify_for_selfsigned():
    """
    Certificat auto-signé -> requests lèverait SSLError sur chaque appel
    sans ceci. Patch requests.Session.request (utilisé en interne par
    requests.get/post/...) pour que verify=False s'applique par défaut à
    tous les appels des fichiers de test, sans devoir éditer ~25 appels
    individuels dans test_security.py / test_api_final.py.
    """
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    orig_request = requests.Session.request

    def patched(self, method, url, **kwargs):
        kwargs.setdefault("verify", False)
        return orig_request(self, method, url, **kwargs)

    requests.Session.request = patched
    yield
    requests.Session.request = orig_request


@pytest.fixture(scope="session")
def api_key(request):
    """Clé API : --api-key > variable d'env API_KEYS (première clé)."""
    key = request.config.getoption("--api-key")
    if not key:
        env = os.getenv("API_KEYS", "")
        key = env.split(",")[0].strip() if env else ""
    return key


@pytest.fixture(scope="session")
def headers(api_key):
    """En-tête HTTP avec la clé API — injecté dans tous les tests d'intégration."""
    return {"X-API-Key": api_key} if api_key else {}


@pytest.fixture
def verbose(request):
    return request.config.getoption("-v", default=False, skip=True)
