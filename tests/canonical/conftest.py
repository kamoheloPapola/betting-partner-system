"""Canonical tests are offline and do not load root config/ML fixtures."""
import socket
import asyncio
import pytest


@pytest.fixture(autouse=True)
def unset_heartbeat(monkeypatch):
    from src.canonical.heartbeat import ENV_VAR

    monkeypatch.delenv(ENV_VAR, raising=False)


@pytest.fixture(autouse=True)
def no_network(monkeypatch,request):
    if "api_loop" in request.fixturenames:
        request.getfixturevalue("api_loop")
    def denied(*args, **kwargs):
        raise AssertionError("NETWORK_FORBIDDEN_IN_CANONICAL_TESTS")
    monkeypatch.setattr(socket, "getaddrinfo", denied)
    monkeypatch.setattr(socket, "create_connection", denied)
    monkeypatch.setattr(socket.socket, "connect", denied)
    monkeypatch.setattr(socket.socket, "connect_ex", denied)
    monkeypatch.setattr(socket.socket, "sendto", denied)


@pytest.fixture
def api_loop():
    # Windows' private asyncio socketpair is created before the application guard.
    loop=asyncio.new_event_loop()
    yield lambda:loop
    if not loop.is_closed():loop.close()

