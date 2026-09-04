"""Fixtures pytest partagées entre tous les fichiers de test."""
import os
import pytest
import requests

# Charge .env AVANT toute collecte de test -- sans ça, API_KEYS est absent de
# l'environnement et le fallback os.environ.setdefault("API_KEYS", "test-key-unit")
# de test_ml_functions.py (nécessaire à SES tests unitaires isolés) devient la
# valeur effective pour toute la session pytest, y compris pour les fichiers
# d'intégration qui tapent la vraie API -- provoquant des 401 qui n'ont rien à
# voir avec un bug de code (observé : besoin de charger .env manuellement
# avant de lancer pytest).
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


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
    url = f"https://{host}:{port}"

    # Purge le rate limiter une fois en début de session : sans ça, le quota
    # (ex: 10 appels/60s sur /v1/auth/register) peut déjà être partiellement
    # consommé par un run pytest précédent contre le même serveur encore
    # chaud, et fait échouer des tests sans rapport avec le rate limiting.
    # No-op silencieux si le serveur ne tourne pas avec TEST_MODE=1.
    try:
        requests.post(f"{url}/v1/_test/reset-rate-limit", timeout=3, verify=False)
    except Exception:
        pass

    return url


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
