from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from starlette.requests import Request

from src.api import auth
from src.config import startup
from src.config.env_contract import SCHEDULER_PROCESS
from src.config.model_state import ModelStateError
from src.ml import artifact_signing
from src.monitoring import alert_pipeline, nightly_heartbeat


def _request(path: str = "/cli/train") -> Request:
    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": path,
            "headers": [],
            "query_string": b"",
            "scheme": "http",
            "server": ("testserver", 80),
            "client": ("203.0.113.10", 32100),
        }
    )


def test_cli_denial_reaches_shared_dispatch(monkeypatch):
    events = []
    monkeypatch.setattr(auth, "dispatch_alert", lambda **event: events.append(event))

    auth._log_denied(_request(), reason="invalid_credentials")

    assert events == [
        {
            "source": "cli_auth_denial",
            "severity": "WARNING",
            "message": "CLI authorization denied",
            "context": {
                "caller_ip": "203.0.113.10",
                "route": "/cli/train",
                "reason": "invalid_credentials",
                "status_code": 401,
            },
        }
    ]


def test_artifact_rejection_reaches_shared_dispatch(monkeypatch, tmp_path):
    events = []
    monkeypatch.setattr(
        artifact_signing,
        "dispatch_alert",
        lambda **event: events.append(event),
    )
    artifact_path = tmp_path / "model.pkl"

    with pytest.raises(artifact_signing.ArtifactVerificationError):
        artifact_signing._reject(
            artifact_path,
            version="2.0.0",
            reason="digest_mismatch",
            detail="tampered",
        )

    assert events[0]["source"] == "artifact_verification_failure"
    assert events[0]["severity"] == "CRITICAL"
    assert events[0]["context"] == {
        "path": str(artifact_path),
        "version": "2.0.0",
        "reason": "digest_mismatch",
    }


def test_model_lock_block_reaches_shared_dispatch(monkeypatch):
    events = []

    def blocked(_process):
        raise ModelStateError("deployment is unlocked")

    monkeypatch.setattr(startup, "run_preflight", blocked)
    monkeypatch.setattr(startup, "dispatch_alert", lambda **event: events.append(event))

    assert startup.main([SCHEDULER_PROCESS]) == 3
    assert events[0]["source"] == "nightly_pipeline_blocked"
    assert events[0]["severity"] == "CRITICAL"
    assert events[0]["context"]["exit_code"] == 3


def test_heartbeat_staleness_dispatches_but_fresh_and_first_run_do_not(
    monkeypatch,
    tmp_path,
):
    events = []
    heartbeat_file = tmp_path / "nightly_heartbeat.json"
    now = datetime(2026, 8, 8, 12, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(
        nightly_heartbeat,
        "dispatch_alert",
        lambda **event: events.append(event),
    )

    first = nightly_heartbeat.check_nightly_heartbeat(
        heartbeat_file=heartbeat_file,
        now=now,
        max_age_hours=30,
    )
    assert first.state == "not_initialized"
    assert events == []

    never_completed = nightly_heartbeat.check_nightly_heartbeat(
        heartbeat_file=heartbeat_file,
        now=now + timedelta(hours=31),
        max_age_hours=30,
    )
    assert never_completed.state == "missing"
    assert events[0]["source"] == "scheduler_heartbeat"
    assert events[0]["severity"] == "CRITICAL"
    events.clear()

    nightly_heartbeat.record_nightly_success(
        heartbeat_file=heartbeat_file,
        now=now - timedelta(hours=2),
    )
    fresh = nightly_heartbeat.check_nightly_heartbeat(
        heartbeat_file=heartbeat_file,
        now=now,
        max_age_hours=30,
    )
    assert fresh.state == "fresh"
    assert events == []

    stale = nightly_heartbeat.check_nightly_heartbeat(
        heartbeat_file=heartbeat_file,
        now=now + timedelta(hours=31),
        max_age_hours=30,
    )
    assert stale.state == "stale"
    assert events[0]["source"] == "scheduler_heartbeat"
    assert events[0]["severity"] == "CRITICAL"
    assert events[0]["context"]["max_age_hours"] == 30.0


def test_dispatch_failure_falls_back_to_error_log(monkeypatch):
    class BrokenAlerter:
        def send_alert(self, *args, **kwargs):
            raise OSError("channel unavailable")

    fallback_calls = []
    monkeypatch.setattr(alert_pipeline, "Alerter", BrokenAlerter)
    monkeypatch.setattr(
        alert_pipeline.logger,
        "exception",
        lambda message, *args: fallback_calls.append((message, args)),
    )

    sent = alert_pipeline.dispatch_alert(
        source="cli_auth_denial",
        severity="WARNING",
        message="CLI authorization denied",
    )

    assert sent is False
    assert fallback_calls == [
        (
            "operational_alert_dispatch_failed source=%s severity=%s message=%s",
            ("cli_auth_denial", "WARNING", "CLI authorization denied"),
        )
    ]


def test_dispatch_uses_existing_cross_instance_dedup(monkeypatch, tmp_path):
    from src.monitoring import alerter

    monkeypatch.setattr(alerter, "ALERT_HISTORY_FILE", tmp_path / "alerts.json")
    monkeypatch.setattr(alert_pipeline, "capture_alert", lambda **kwargs: None)

    first = alert_pipeline.dispatch_alert(
        source="cli_auth_denial",
        severity="WARNING",
        message="repeated probe",
        context={"caller_ip": "203.0.113.10", "route": "/cli/train"},
    )
    second = alert_pipeline.dispatch_alert(
        source="cli_auth_denial",
        severity="WARNING",
        message="repeated probe",
        context={"caller_ip": "203.0.113.10", "route": "/cli/train"},
    )

    assert first is True
    assert second is False
