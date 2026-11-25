import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api.cache import MODEL_HEALTH_CACHE_KEY, PredictionCache, prediction_cache, prediction_cache_key
import src.api.main as api_main
import src.config as config
import src.monitoring.drift_orchestrator as drift_orchestrator


client = TestClient(api_main.app)


def _write_manifest(tmp_path, manifest):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


def _model_entry(payload, market):
    return next(model for model in payload["models"] if model["market"] == market)


def test_april_go_snapshot_cannot_claim_live_health(tmp_path, monkeypatch):
    archived = tmp_path / "rolling_90d_status.json"
    archived.write_text(json.dumps({"status": "GO", "evaluated_at": "2026-04-19T22:37:32Z"}))
    monkeypatch.setattr(api_main.DriftGuardrail, "STATUS_FILE", archived)
    monkeypatch.setattr(api_main, "DRIFT_STATE_DIR", tmp_path)
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    (tmp_path / "PL_drift_status.json").write_bytes(archived.read_bytes())

    for decision in (api_main._read_prediction_guard_status(), api_main._read_league_drift_status_from_state_dir("PL")):
        assert (decision.status, decision.reason) == ("STOP", "LEGACY_DRIFT_STALE")
    health = client.get("/health")
    assert health.status_code == 200
    assert set(health.json()) == {"status", "model_version", "last_update", "scope"}
    assert health.json()["scope"] == "process_liveness"

    drift = client.get("/drift-status")
    assert drift.status_code == 200
    assert drift.json() == {"status": "historical", "last_check": None, "alerts": []}

    assert archived.exists()


@pytest.mark.parametrize("method, path, message", [
    ("GET", "/cli/inspect-drift", "Use canonical status for current health."),
    ("POST", "/cli/reset-drift", "Legacy drift files are historical records."),
])
def test_retired_drift_routes_return_gone(method, path, message):
    response = client.request(method, path)
    assert response.status_code == 410 and response.json() == {"detail": message}


NOW = datetime(2026, 11, 19, 12, tzinfo=timezone.utc)


@pytest.mark.parametrize("evaluated_at,expected,reason", [
    pytest.param((NOW - timedelta(hours=1)).isoformat(), "GO", None, id="fresh-go"),
    pytest.param((NOW - timedelta(days=2)).isoformat(), "STOP", "LEGACY_DRIFT_STALE", id="exactly-two-days"),
    pytest.param((NOW - timedelta(days=2) + timedelta(microseconds=1)).isoformat(), "GO", None, id="just-under-two-days"),
    pytest.param(NOW.replace(tzinfo=None).isoformat(), "STOP", "LEGACY_DRIFT_STALE", id="naive"),
    pytest.param((NOW + timedelta(seconds=1)).isoformat(), "STOP", "LEGACY_DRIFT_STALE", id="future"),
    pytest.param(42, "STOP", "LEGACY_DRIFT_STALE", id="non-string"),
    pytest.param(None, "STOP", "LEGACY_DRIFT_UNAVAILABLE", id="missing-file"),
])
def test_legacy_drift_freshness_matches_guard_and_dashboard(tmp_path, monkeypatch, evaluated_at, expected, reason):
    class FixedDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return NOW.astimezone(tz) if tz else NOW.replace(tzinfo=None)

    archived = tmp_path / "rolling_90d_status.json"
    monkeypatch.setattr(api_main, "datetime", FixedDateTime)
    monkeypatch.setattr(api_main.DriftGuardrail, "STATUS_FILE", archived)
    monkeypatch.setattr(api_main, "DRIFT_STATE_DIR", tmp_path)
    if evaluated_at is not None:
        archived.write_text(json.dumps({"status": "GO", "evaluated_at": evaluated_at}), encoding="utf-8")
        (tmp_path / "PL_drift_status.json").write_bytes(archived.read_bytes())
    for decision in (api_main._read_prediction_guard_status(), api_main._read_league_drift_status_from_state_dir("PL")):
        assert (decision.status, decision.reason) == (expected, reason)
    dashboard = client.get("/model-health").json()
    assert dashboard["global_drift_status"] == ("UNKNOWN" if reason else expected)
    assert dashboard["global_drift_reason"] == reason



@pytest.mark.parametrize("cache_state", ("cold", "warm"))
@pytest.mark.parametrize("state,allowed,league", [
    pytest.param("fresh", True, "PL", id="fresh"),
    pytest.param("just-under-two-days", True, "PL", id="just-under-two-days"),
    pytest.param("stale", False, "PL", id="stale"),
    pytest.param("missing", False, "PL", id="missing"),
    pytest.param("missing-league", False, "PL", id="missing-league"),
    pytest.param("unreadable-global", False, "PL", id="unreadable-global"),
    pytest.param("unreadable-league", False, "PL", id="unreadable-league"),
    pytest.param("stop", False, "PL", id="stop"),
    pytest.param("malformed", False, "PL", id="malformed"),
    pytest.param("malformed-time", False, "PL", id="malformed-time"),
    pytest.param("tz-naive", False, "PL", id="tz-naive"),
    pytest.param("future", False, "PL", id="future"),
    pytest.param("exactly-two-days", False, "PL", id="exactly-two-days"),
    pytest.param("stale-league", False, "PL", id="stale-league-PL"),
    pytest.param("stale-league", False, "BL1", id="stale-league-BL1"),
    pytest.param("stale-league", False, "SA", id="stale-league-SA"),
    pytest.param("stale-league", False, "PD", id="stale-league-PD"),
])
def test_prediction_guard_precedes_cache_and_predictor(
    tmp_path, monkeypatch, caplog, cache_state, state, allowed, league,
):
    class FixedDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return NOW.astimezone(tz) if tz else NOW.replace(tzinfo=None)

    fresh = (NOW - timedelta(hours=1)).isoformat()
    timestamp = {
        "just-under-two-days": (NOW - timedelta(days=2) + timedelta(microseconds=1)).isoformat(),
        "stale": (NOW - timedelta(days=3)).isoformat(),
        "malformed-time": "yesterday",
        "tz-naive": NOW.replace(tzinfo=None).isoformat(),
        "future": (NOW + timedelta(seconds=1)).isoformat(),
        "exactly-two-days": (NOW - timedelta(days=2)).isoformat(),
    }.get(state, fresh)
    archive = tmp_path / "rolling_90d_status.json"
    if state != "missing":
        status = "STOP" if state == "stop" else "GO"
        contents = "{" if state == "malformed" else json.dumps({"status": status, "evaluated_at": timestamp})
        archive.write_text(contents, encoding="utf-8")
    league_state = tmp_path / f"{league}_drift_status.json"
    if state != "missing-league":
        league_time = (NOW - timedelta(days=3)).isoformat() if state == "stale-league" else fresh
        league_state.write_text(json.dumps({"status": "GO", "evaluated_at": league_time}), encoding="utf-8")

    if state == "unreadable-global":
        def read_global(path, *args, **kwargs):
            if Path(path) == archive:
                raise PermissionError("fixture access refused")
            return open(path, *args, **kwargs)

        monkeypatch.setattr(drift_orchestrator, "open", read_global, raising=False)
    elif state == "unreadable-league":
        read_text = Path.read_text

        def read_league(path, *args, **kwargs):
            if path == league_state:
                raise PermissionError("fixture access refused")
            return read_text(path, *args, **kwargs)

        monkeypatch.setattr(Path, "read_text", read_league)

    cache = PredictionCache()
    key = prediction_cache_key(league)
    cached = [{"match_id": "cached"}]
    predicted = [{"match_id": "computed"}]
    if cache_state == "warm":
        cache.set(key, cached)
    calls = []

    def predict(self, *, league):
        calls.append(league)
        return predicted

    monkeypatch.setattr(api_main, "datetime", FixedDateTime)
    monkeypatch.setattr(api_main.DriftGuardrail, "STATUS_FILE", archive)
    monkeypatch.setattr(api_main, "DRIFT_STATE_DIR", tmp_path)
    monkeypatch.setattr(api_main, "prediction_cache", cache)
    monkeypatch.setattr(api_main.Predictor, "predict_upcoming", predict)
    monkeypatch.setattr(api_main.Predictor, "__init__", lambda self: None)

    result = api_main._load_or_compute_predictions(league)
    expected = (cached if cache_state == "warm" else predicted) if allowed else []
    assert result == expected
    assert calls == ([league] if allowed and cache_state == "cold" else [])
    assert cache.get(key) == (expected if allowed else None)

    block_messages = [record.getMessage() for record in caplog.records
                      if record.name == api_main.__name__ and record.getMessage().startswith("Prediction blocked ")]
    if allowed:
        assert block_messages == []
    else:
        reason = {
            "missing": api_main.LEGACY_DRIFT_UNAVAILABLE,
            "missing-league": api_main.LEGACY_DRIFT_UNAVAILABLE,
            "unreadable-global": api_main.LEGACY_DRIFT_UNAVAILABLE,
            "unreadable-league": api_main.LEGACY_DRIFT_UNAVAILABLE,
            "malformed": api_main.LEGACY_DRIFT_UNAVAILABLE,
            "stop": "drift_guardrail_stop",
        }.get(state, api_main.LEGACY_DRIFT_STALE)
        assert block_messages == [f"Prediction blocked league={league} reason={reason}"]


def test_stale_stop_reason_is_logged_reported_and_returned(tmp_path, monkeypatch, caplog):
    archived = tmp_path / "rolling_90d_status.json"
    monkeypatch.setattr(api_main.DriftGuardrail, "STATUS_FILE", archived)
    monkeypatch.setattr(api_main, "DRIFT_STATE_DIR", tmp_path)
    monkeypatch.setattr(api_main, "_LAST_DRIFT_DECISIONS", {})
    alerts = []
    monkeypatch.setattr(api_main, "capture_alert", lambda **event: alerts.append(event))
    fresh = datetime.now(timezone.utc).isoformat()
    archived.write_text(json.dumps({"status": "STOP", "evaluated_at": fresh}), encoding="utf-8")
    assert api_main._read_prediction_guard_status().reason is None
    archived.write_text(json.dumps({"status": "GO", "evaluated_at": "2026-01-01T00:00:00Z"}), encoding="utf-8")
    response = client.post("/api/v1/predictions/trigger", json={"league": "PL"})
    assert response.status_code == 200 and response.json()["blocked"]
    assert response.json()["reason"] == "LEGACY_DRIFT_STALE"
    assert "LEGACY_DRIFT_STALE" in caplog.text
    assert alerts[0]["context"]["reason"] == "LEGACY_DRIFT_STALE"


def test_model_health_preserves_individual_drift_with_fresh_global_state(tmp_path, monkeypatch):
    manifest_path = _write_manifest(tmp_path, {
        "active_models": {"poisson_pl": "poisson", "corner_pl": "corner"},
        "poisson": {"name": "poisson", "drift_status": "STOP"},
        "corner": {"name": "corner"},
    })
    monkeypatch.setattr(api_main.ModelRegistry, "MANIFEST_FILE", manifest_path)
    prediction_cache.invalidate(MODEL_HEALTH_CACHE_KEY)
    global_file = api_main.DriftGuardrail.STATUS_FILE
    confidence_file = api_main.DriftOrchestrator.DEFAULT_CONFIDENCE_STATE_FILE
    global_file.parent.mkdir(parents=True)
    confidence_file.write_text(json.dumps({"market_status": {"poisson": "GO", "corner": "WATCH"}}), encoding="utf-8")
    fresh = {"status": "GO", "evaluated_at": datetime.now(timezone.utc).isoformat()}
    stale = {**fresh, "evaluated_at": "2026-04-19T22:37:32Z"}

    # An outdated response must not change the cached model entries used after recovery.
    for state, expected in ((stale, {"poisson": "UNKNOWN", "corner": "UNKNOWN"}),
                            (fresh, {"poisson": "STOP", "corner": "WATCH"})):
        global_file.write_text(json.dumps(state), encoding="utf-8")
        payload = client.get("/model-health").json()
        assert {row["market"]: row["drift_status"] for row in payload["models"]} == expected
        assert {market: rows[0]["drift_status"] for market, rows in payload["markets"].items()} == expected


def test_guard_and_model_health_reads_leave_archived_state_untouched():
    archive = api_main.DriftGuardrail.STATUS_FILE
    archive.parent.mkdir(parents=True)
    archive.write_text(json.dumps({"status": "GO", "evaluated_at": "2026-04-19T22:37:32Z"}), encoding="utf-8")
    original, mtime = archive.read_bytes(), archive.stat().st_mtime_ns

    guard = api_main.DriftGuardrail()
    assert guard.check_drift() == "OK"
    assert guard.evaluated_at == "2026-04-19T22:37:32Z"
    assert api_main._read_prediction_guard_status().reason == "LEGACY_DRIFT_STALE"
    assert client.get("/model-health").json()["global_drift_reason"] == "LEGACY_DRIFT_STALE"

    assert archive.read_bytes() == original and archive.stat().st_mtime_ns == mtime
    assert list(archive.parent.iterdir()) == [archive]
    assert not api_main.DriftOrchestrator.DEFAULT_ALERTS_FILE.parent.exists()


def test_guard_and_dashboard_use_database_payload_timestamp_then_file_fallback(tmp_path, monkeypatch):
    from sqlalchemy.orm import Session
    from src.db.connection import get_engine
    from src.db.models import Base, DriftEvent

    archive = api_main.DriftGuardrail.STATUS_FILE
    archive.parent.mkdir(parents=True)
    now = datetime.now(timezone.utc)
    file_timestamp = (now - timedelta(hours=1)).isoformat()
    db_timestamp = (now - timedelta(hours=2)).isoformat()
    archive.write_text(json.dumps({"status": "GO", "evaluated_at": file_timestamp}), encoding="utf-8")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{(tmp_path / 'drift.db').as_posix()}")
    engine = get_engine()
    try:
        Base.metadata.create_all(engine)
        with Session(engine) as session:
            session.add(DriftEvent(event_id="test-global-drift", event_type="global_drift_state",
                                   league="GLOBAL", severity="STOP", detected_at=now,
                                   payload_json=json.dumps({"status": "STOP", "evaluated_at": db_timestamp})))
            session.commit()

        guard = api_main._read_prediction_guard_status()
        dashboard = client.get("/model-health").json()
        assert (guard.status, guard.evaluated_at) == ("STOP", db_timestamp)
        assert (dashboard["global_drift_status"], dashboard["global_drift_evaluated_at"],
                dashboard["global_drift_source"]) == ("STOP", db_timestamp, "database")

        with Session(engine) as session:
            session.delete(session.get(DriftEvent, "test-global-drift"))
            session.commit()

        guard = api_main._read_prediction_guard_status()
        dashboard = client.get("/model-health").json()
        assert (guard.status, guard.evaluated_at) == ("GO", file_timestamp)
        assert (dashboard["global_drift_status"], dashboard["global_drift_evaluated_at"],
                dashboard["global_drift_source"]) == ("GO", file_timestamp, "file")
    finally:
        engine.dispose()


def test_configured_database_read_failure_blocks_even_with_fresh_go_files(tmp_path, monkeypatch):
    from src.db.connection import get_engine

    archive = api_main.DriftGuardrail.STATUS_FILE
    archive.parent.mkdir(parents=True)
    fresh = json.dumps({"status": "GO", "evaluated_at": datetime.now(timezone.utc).isoformat()})
    archive.write_text(fresh, encoding="utf-8")
    (archive.parent / "PL_drift_status.json").write_text(fresh, encoding="utf-8")
    confidence = api_main.DriftOrchestrator.DEFAULT_CONFIDENCE_STATE_FILE
    confidence.write_text(json.dumps({"market_status": {"poisson": "GO"}}), encoding="utf-8")
    monkeypatch.setattr(api_main, "DRIFT_STATE_DIR", archive.parent)
    monkeypatch.setattr(api_main, "capture_alert", lambda **event: None)
    db_path = tmp_path / "drift.db"
    db_path.touch()  # An existing database file without the expected table makes SELECT fail.
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path.as_posix()}")

    try:
        decision = api_main._read_prediction_guard_status()
        assert (decision.status, decision.reason, decision.evaluated_at) == ("STOP", "LEGACY_DRIFT_UNAVAILABLE", None)
        guard = api_main.DriftGuardrail()
        assert guard.check_drift() == guard.check_drift(league="PL") == "STOP"
        assert api_main.DriftOrchestrator().get_status("poisson") == "STOP"

        dashboard = client.get("/model-health").json()
        assert dashboard["global_drift_status"] == "UNKNOWN"
        assert dashboard["global_drift_reason"] == "LEGACY_DRIFT_UNAVAILABLE"
        assert dashboard["global_drift_source"] is None
        response = client.post("/api/v1/predictions/trigger", json={"league": "PL"})
        assert response.status_code == 200 and response.json()["blocked"]
    finally:
        get_engine().dispose()


def test_model_health_endpoint_returns_market_entries(monkeypatch, tmp_path):
    model_path = tmp_path / "poisson_home_base.joblib"
    model_path.write_text("stub", encoding="utf-8")
    manifest_path = _write_manifest(
        tmp_path,
        {
            "active_models": {
                "poisson_home_base_pl": "poisson_home_base_pl_v140",
            },
            "poisson_home_base_pl_v140": {
                "name": "poisson_home_base",
                "league": "PL",
                "version": "1.4.0",
                "training_date": "2026-03-21T00:00:00+00:00",
                "metrics": {"brier_score": 0.187},
                "path": str(model_path),
            },
        },
    )
    prediction_cache.invalidate(MODEL_HEALTH_CACHE_KEY)
    monkeypatch.setattr(api_main.ModelRegistry, "MANIFEST_FILE", manifest_path)

    response = client.get("/model-health")
    assert response.status_code == 200

    payload = response.json()
    assert payload["global_drift_status"] == "UNKNOWN"
    assert "poisson_home_base" in payload["markets"]
    entry = payload["markets"]["poisson_home_base"][0]
    assert entry["league"] == "PL"
    assert entry["version"] == "1.4.0"
    assert entry["brier_score"] == 0.187
    assert entry["drift_status"] == "UNKNOWN"
    assert entry["status"] == "ok"

    model = _model_entry(payload, "poisson_home_base")
    assert model["last_trained"] == "2026-03-21T00:00:00+00:00"


def test_model_health_endpoint_uses_registered_at_for_last_trained(monkeypatch, tmp_path):
    manifest_path = _write_manifest(
        tmp_path,
        {
            "active_models": {
                "nb_home_corners_base_pl": "nb_home_corners_base_pl_v200",
            },
            "nb_home_corners_base_pl_v200": {
                "name": "nb_home_corners_base",
                "league": "PL",
                "version": "2.0.0",
                "registered_at": "2026-03-18T00:00:00+00:00",
                "metrics": {},
            },
        },
    )
    prediction_cache.invalidate(MODEL_HEALTH_CACHE_KEY)
    monkeypatch.setattr(api_main.ModelRegistry, "MANIFEST_FILE", manifest_path)

    response = client.get("/model-health")
    assert response.status_code == 200

    payload = response.json()
    entry = payload["markets"]["nb_home_corners_base"][0]
    assert entry["drift_status"] == "UNKNOWN"
    model = _model_entry(payload, "nb_home_corners_base")
    assert model["last_trained"] == "2026-03-18T00:00:00+00:00"


def test_model_health_endpoint_falls_back_to_top_level_manifest_brier(monkeypatch, tmp_path):
    manifest_path = _write_manifest(
        tmp_path,
        {
            "active_models": {
                "poisson_home_base_pl": "poisson_home_base_pl_v140",
            },
            "poisson_home_base_pl_v140": {
                "name": "poisson_home_base",
                "league": "PL",
                "version": "1.4.0",
                "training_date": "2026-03-21T00:00:00+00:00",
                "metrics": {},
                "brier_score": 0.241,
            },
        },
    )
    prediction_cache.invalidate(MODEL_HEALTH_CACHE_KEY)
    monkeypatch.setattr(api_main.ModelRegistry, "MANIFEST_FILE", manifest_path)

    response = client.get("/model-health")
    assert response.status_code == 200

    payload = response.json()
    entry = payload["markets"]["poisson_home_base"][0]
    assert entry["brier_score"] == 0.241
    model = _model_entry(payload, "poisson_home_base")
    assert model["last_trained"] == "2026-03-21T00:00:00+00:00"
