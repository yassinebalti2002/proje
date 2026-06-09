"""Fixtures pytest pour test_api_final.py."""
import pytest


def pytest_addoption(parser):
    parser.addoption("--host", default="localhost", help="Hôte API")
    parser.addoption("--port", default=8000,        type=int, help="Port API")


@pytest.fixture
def base(request):
    host = request.config.getoption("--host")
    port = request.config.getoption("--port")
    return f"http://{host}:{port}"


@pytest.fixture
def verbose(request):
    return request.config.getoption("-v", default=False, skip=True)
