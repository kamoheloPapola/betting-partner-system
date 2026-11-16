"""All test transports are fake. Accidental network calls fail immediately."""
import socket
import pytest

@pytest.fixture(autouse=True)
def forbid_network(monkeypatch):
    def denied(*args,**kwargs):
        raise AssertionError("NETWORK_FORBIDDEN_IN_OFFLINE_TESTS")
    monkeypatch.setattr(socket,"create_connection",denied)
    monkeypatch.setattr(socket.socket,"connect",denied)
