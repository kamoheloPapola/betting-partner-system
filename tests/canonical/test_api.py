import pytest

from fastapi.testclient import TestClient
from src.canonical.api import create_app


@pytest.fixture
def api_client(tmp_path,api_loop):
    db=tmp_path/"missing.sqlite3"
    with TestClient(create_app(db),base_url="http://127.0.0.1",backend_options={"loop_factory":api_loop}) as client:
        yield client


def test_missing_database_is_visible_setup_state_without_file_creation(tmp_path,api_client):
    r=api_client.get("/api/health")
    assert r.status_code==200 and r.json()["state"]=="NEEDS_SETUP"
    assert not (tmp_path/"missing.sqlite3").exists()


def test_no_mutating_http_routes_and_loopback_host_guard(api_client):
    assert api_client.post("/api/forecasts").status_code==405
    assert api_client.post("/api/run").status_code==404
    for host in ("external.example","testserver"):
        assert api_client.get("/api/health",headers={"host":host}).status_code==400


def test_dashboard_assets_and_security_policy(api_client):
    page=api_client.get("/")
    assert page.status_code==200 and "Frozen forecasts" in page.text
    assert "script-src 'self'" in page.headers["Content-Security-Policy"]
    script=api_client.get("/app.js").text
    assert "textContent" in script and "innerHTML" not in script
    assert api_client.get("/style.css").status_code==200


def test_api_bootstrap_does_not_allow_application_network(api_loop):
    import socket
    with pytest.raises(AssertionError,match="NETWORK_FORBIDDEN"):
        socket.create_connection(("example.com",443))

