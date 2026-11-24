import json
import os
import socket
from threading import Event
from types import SimpleNamespace

import pytest

from src.canonical import cli, heartbeat
from src.canonical.clock import CanonicalError


@pytest.mark.parametrize("url", [
    pytest.param(None, id="unset"),
    pytest.param("", id="empty"),
    pytest.param(" \t ", id="whitespace-only"),
    pytest.param("http://example.test/check", id="non-https"),
    pytest.param("relative/path", id="malformed-relative"),
    pytest.param("https:///check", id="malformed-missing-host"),
    pytest.param("https://example.test:bad/check", id="malformed-port"),
    pytest.param("https://example.test/check#fragment", id="malformed-fragment"),
    pytest.param("https://name:secret@example.test/check", id="malformed-userinfo"),
    pytest.param("https://example.test/a\nb", id="malformed-newline"),
    pytest.param("https://[::1]extra/check", id="malformed-ipv6"),
    pytest.param("https://example..test/check", id="malformed-hostname"),
])
def test_unconfigured_heartbeat_sends_nothing(monkeypatch, url):
    if url is not None:
        monkeypatch.setenv(heartbeat.ENV_VAR, url)
    receipt = heartbeat.send("run", "SUCCESS", [], transport=lambda *_: pytest.fail("Unexpected ping"))
    assert receipt == {"state": "DISABLED", "kind": "SUCCESS"}
    assert heartbeat.status(receipt) == {"state": "NOT_CONFIGURED", "kind": "SUCCESS"}
    assert heartbeat.status() == {"state": "NOT_CONFIGURED", "kind": None}


def test_offline_guard_blocks_default_heartbeat_connection():
    assert heartbeat.ENV_VAR not in os.environ
    with pytest.raises(AssertionError, match="^NETWORK_FORBIDDEN_IN_CANONICAL_TESTS$"):
        heartbeat.HTTPSConnection("127.0.0.1", port=0, timeout=0.1).connect()
    with pytest.raises(AssertionError, match="^NETWORK_FORBIDDEN_IN_CANONICAL_TESTS$"):
        socket.getaddrinfo("127.0.0.1", 0, flags=socket.AI_NUMERICHOST)


@pytest.mark.parametrize("error", [RuntimeError, CanonicalError])
def test_scheduler_ready_masks_exceptions(tmp_path, monkeypatch, capsys, error):
    canary = "https://readiness-canary.example.test/private-fragment"
    monkeypatch.setenv(heartbeat.ENV_VAR, canary)

    def unavailable(*args, **kwargs):
        raise error(canary)

    monkeypatch.setattr(cli, "Store", unavailable)
    assert cli.main(["--database", str(tmp_path / "unused.sqlite3"), "scheduler-ready"]) != 0
    output = capsys.readouterr()
    assert json.loads(output.out) == {"state": "NOT_READY", "error_type": error.__name__}
    assert output.err == ""


@pytest.mark.parametrize(("error", "count"), [(TypeError, 2), (KeyboardInterrupt, 1)])
def test_send_masks_unexpected_transport_exceptions(monkeypatch, error, count):
    monkeypatch.setenv(heartbeat.ENV_VAR, "https://heartbeat.example.test/canary")
    calls = []

    def transport(url, body, timeout):
        calls.append(timeout)
        raise error(url.geturl())

    assert heartbeat.send(None, "FAILED", [], transport=transport) == {
        "state": "FAILED", "kind": "FAILURE", "error_type": error.__name__,
    }
    assert calls == [10] * count


def test_send_masks_error_before_transport(monkeypatch):
    def invalid_endpoint():
        raise RuntimeError("https://heartbeat.example.test/canary")

    monkeypatch.setattr(heartbeat, "_endpoint", invalid_endpoint)
    assert heartbeat.send("run", "SUCCESS", []) == {
        "state": "FAILED", "kind": "SUCCESS", "error_type": "RuntimeError",
    }


@pytest.mark.parametrize(("responses", "attempts", "state"), [
    ([204], 1, "SENT"), ([503, 204], 2, "SENT"), ([429, 502], 2, "FAILED"),
    ([302], 1, "FAILED"), ([404], 1, "FAILED"),
])
def test_retry_only_transient_delivery_failures(monkeypatch, responses, attempts, state):
    monkeypatch.setenv(heartbeat.ENV_VAR, "https://heartbeat.example.test/check")
    sent = []

    def transport(url, body, timeout):
        sent.append((url.path, body, timeout))
        return responses[len(sent) - 1]

    assert heartbeat.status() == {"state": "NOT_EVALUATED", "kind": None}
    assert heartbeat.send("run", "FAILED", ["PL"], transport=transport) == {"state": state, "kind": "FAILURE"}
    assert len(sent) == attempts and sent == [sent[0]] * attempts


def test_https_transport_sends_json_as_get_body(monkeypatch):
    monkeypatch.setenv(heartbeat.ENV_VAR, "https://heartbeat.example.test:8443/check?key=test-value")
    calls = []

    class Response:
        status = 204

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    class Connection:
        sock = None

        def __init__(self, host, port, *, timeout, context):
            calls.append((host, port, timeout, context.check_hostname, context.keylog_filename))

        def connect(self):
            pass

        def request(self, method, target, *, body, headers):
            calls.append((method, target, json.loads(body), headers["Content-Type"]))

        def getresponse(self):
            return Response()

        def close(self):
            pass

    monkeypatch.setattr(heartbeat, "HTTPSConnection", Connection)
    assert heartbeat.send("run", "SUCCESS", []) == {"state": "SENT", "kind": "SUCCESS"}
    assert calls == [
        ("heartbeat.example.test", 8443, 10, True, None),
        ("GET", "/check?key=test-value", {"run_id": "run", "state": "SUCCESS", "withheld_leagues": []},
         "application/json"),
    ]


def test_slow_connect_times_out_and_cannot_send_late(monkeypatch):
    url = heartbeat.urlsplit("https://heartbeat.example.test/check")
    release = Event()
    closed = Event()
    sent = []

    class Connection:
        sock = None

        def __init__(self, *args, **kwargs):
            pass

        def connect(self):
            release.wait(timeout=3)

        def request(self, *args, **kwargs):
            sent.append(True)

        def close(self):
            closed.set()

    monkeypatch.setattr(heartbeat, "HTTPSConnection", Connection)
    try:
        assert heartbeat._https_get(url, b"{}", timeout=0.05) is None
        assert not closed.is_set()
    finally:
        release.set()
    assert closed.wait(timeout=3)
    assert sent == []


def test_slow_response_is_aborted_on_deadline(monkeypatch):
    monkeypatch.setenv(heartbeat.ENV_VAR, "https://heartbeat.example.test/check")
    finished = Event()
    stopped = Event()

    class Connection:
        sock = SimpleNamespace(shutdown=lambda how: stopped.set())

        def __init__(self, *args, **kwargs):
            pass

        def connect(self):
            pass

        def request(self, *args, **kwargs):
            pass

        def getresponse(self):
            stopped.wait(timeout=3)
            raise OSError("Connection closed")

        def close(self):
            finished.set()

    monkeypatch.setattr(heartbeat, "HTTPSConnection", Connection)
    monkeypatch.setattr(heartbeat.ssl.SSLContext, "load_default_certs", lambda self: None)
    assert heartbeat._https_get(heartbeat._endpoint(), b"{}", timeout=0.1) is None
    assert stopped.is_set() and finished.wait(timeout=3)
