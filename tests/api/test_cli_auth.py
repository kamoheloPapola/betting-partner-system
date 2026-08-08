from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

from src.api.routes import cli as cli_module


ADMIN_TOKEN = "test-admin-token"
VIEWER_TOKEN = "test-viewer-token"


class _FakeRegistry:
    def __init__(self) -> None:
        self.manifest = {"active_models": {}, "shadow_models": {}}

    def reload(self) -> None:
        return None


class _FakeDriftOrchestrator:
    status_file = Path("unused-status.json")
    confidence_state_file = Path("unused-confidence.json")

    def inspect_global_state(self):
        return {"status": "OK"}


class _FakePredictor:
    def predict_for_show_predictions(self, **_kwargs):
        return []


def _cli_route_cases():
    cases = []
    for route in cli_module.router.routes:
        if not isinstance(route, APIRoute):
            continue
        for method in sorted(route.methods - {"HEAD", "OPTIONS"}):
            cases.append(pytest.param(method, route.path, id=f"{method}-{route.path}"))
    return cases


CLI_ROUTE_CASES = _cli_route_cases()


@pytest.fixture
def cli_client(monkeypatch, tmp_path):
    monkeypatch.setenv(
        "CLI_AUTH_TOKENS",
        json.dumps({ADMIN_TOKEN: "admin", VIEWER_TOKEN: "viewer"}),
    )
    monkeypatch.setattr(cli_module, "ModelRegistry", _FakeRegistry)
    monkeypatch.setattr(cli_module, "DriftOrchestrator", _FakeDriftOrchestrator)
    monkeypatch.setattr(cli_module, "Predictor", _FakePredictor)
    monkeypatch.setattr(cli_module, "get_model_state", lambda: "LOCKED")
    monkeypatch.setattr(
        cli_module.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=0, stdout="", stderr=""),
    )
    monkeypatch.setattr(_FakeDriftOrchestrator, "status_file", tmp_path / "status.json")
    monkeypatch.setattr(
        _FakeDriftOrchestrator,
        "confidence_state_file",
        tmp_path / "confidence.json",
    )

    app = FastAPI()
    app.include_router(cli_module.router)
    return TestClient(app)


@pytest.mark.parametrize(("method", "path"), CLI_ROUTE_CASES)
@pytest.mark.parametrize(
    ("auth_case", "headers", "expected_status", "expected_body"),
    [
        ("missing", {}, 401, {"detail": "unauthorized"}),
        ("bad", {"Authorization": "Bearer invalid-token"}, 401, {"detail": "unauthorized"}),
        ("wrong-role", {"Authorization": f"Bearer {VIEWER_TOKEN}"}, 403, {"detail": "forbidden"}),
        ("admin", {"Authorization": f"Bearer {ADMIN_TOKEN}"}, 200, None),
    ],
)
def test_every_cli_route_enforces_authentication_and_authorization(
    cli_client,
    method,
    path,
    auth_case,
    headers,
    expected_status,
    expected_body,
):
    request_kwargs = {"headers": headers}
    if method != "GET":
        request_kwargs["json"] = {}

    response = cli_client.request(method, path, **request_kwargs)

    assert response.status_code == expected_status, auth_case
    if expected_body is not None:
        assert response.json() == expected_body


@pytest.mark.parametrize(
    "authorization",
    ["not-a-bearer-header", "Basic dXNlcjpwYXNz", "Bearer"],
)
def test_malformed_authorization_headers_return_generic_401(cli_client, authorization):
    response = cli_client.get("/cli/help", headers={"Authorization": authorization})

    assert response.status_code == 401
    assert response.json() == {"detail": "unauthorized"}


def test_auth_denial_log_contains_incident_context(cli_client, caplog):
    with caplog.at_level(logging.WARNING, logger="src.api.auth"):
        response = cli_client.get("/cli/help")

    assert response.status_code == 401
    message = next(
        record.getMessage()
        for record in caplog.records
        if "cli_auth_denied" in record.getMessage()
    )
    assert "caller_ip=testclient" in message
    assert "route=/cli/help" in message
    assert "reason=missing_or_malformed_credentials" in message
    assert re.search(r"timestamp=\d{4}-\d{2}-\d{2}T", message)


def test_wrong_role_is_distinct_in_logs_without_exposing_token(cli_client, caplog):
    with caplog.at_level(logging.WARNING, logger="src.api.auth"):
        response = cli_client.get(
            "/cli/help",
            headers={"Authorization": f"Bearer {VIEWER_TOKEN}"},
        )

    assert response.status_code == 403
    messages = "\n".join(record.getMessage() for record in caplog.records)
    assert "cli_auth_forbidden" in messages
    assert "role=viewer" in messages
    assert VIEWER_TOKEN not in messages


@pytest.mark.parametrize("configured_tokens", [None, "not-json", "[]"])
def test_missing_or_invalid_token_configuration_fails_closed(
    cli_client,
    monkeypatch,
    configured_tokens,
):
    if configured_tokens is None:
        monkeypatch.delenv("CLI_AUTH_TOKENS", raising=False)
    else:
        monkeypatch.setenv("CLI_AUTH_TOKENS", configured_tokens)

    response = cli_client.get(
        "/cli/help",
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
    )

    assert response.status_code == 401
    assert response.json() == {"detail": "unauthorized"}
