import json

from fastapi.testclient import TestClient

from src.api.cache import prediction_cache
from src.api.cache import prediction_cache_key, slip_cache_key
import src.api.main as api_main
from src.api.main import app
from src.core.exceptions import ConfigurationError
import src.monitoring.drift_orchestrator as drift_orchestrator_module
from src.strategies.drift_guard import DriftGuardrail


client = TestClient(app)


def test_league_slips_endpoint_returns_latest_recommendations(tmp_path, monkeypatch):
    def fake_predict_upcoming(self, league="PL", limit=None):
        assert league == "PL"
        return [
            {
                "match_id": "abc123",
                "home_team": "Home FC",
                "away_team": "Away FC",
                "league": "PL",
                "kickoff_utc": "2026-04-01T12:00:00Z",
                "home": 0.55,
            }
        ]

    def fake_generate(self, predictions, min_probability=0.65, max_selections=4):
        assert len(predictions) == 1
        assert min_probability == 0.65
        assert max_selections == 4
        return [
            {
                "id": "abc123",
                "match": "Home FC vs Away FC",
                "market": "HOME_WIN",
                "probability": 0.72,
                "confidence": 0.72,
                "league": "PL",
                "date": "2026-04-01T12:00:00Z",
            }
        ]

    monkeypatch.setattr(
        "src.predictions.predictor.Predictor.predict_upcoming",
        fake_predict_upcoming,
    )
    monkeypatch.setattr(
        "src.strategies.slip_builder.ForbiddenFruitSlipBuilder.generate",
        fake_generate,
    )
    _use_tmp_drift_state_slips(tmp_path, monkeypatch, league_status="GO")

    response = client.get("/api/v1/slips/PL")

    assert response.status_code == 200
    payload = response.json()
    assert payload["model_state"]
    assert len(payload["slip"]) == 1
    assert payload["slip"][0]["market"] == "HOME_WIN"
    assert payload["slip"][0]["league"] == "PL"


def test_league_slips_endpoint_returns_503_on_model_environment_mismatch(tmp_path, monkeypatch):
    prediction_cache.invalidate(prediction_cache_key("PL"))
    prediction_cache.invalidate(slip_cache_key("PL", 0.65, 4))

    def fake_predict_upcoming(self, league="PL", limit=None):
        raise ConfigurationError("sklearn 1.8.0 required")

    monkeypatch.setattr(
        "src.predictions.predictor.Predictor.predict_upcoming",
        fake_predict_upcoming,
    )
    _use_tmp_drift_state_slips(tmp_path, monkeypatch, league_status="GO")

    response = client.get("/api/v1/slips/PL")

    assert response.status_code == 503
    assert response.json()["detail"] == "Model environment mismatch: sklearn 1.8.0 required"


def test_league_slips_endpoint_returns_blocked_response_when_drift_stops_predictions(monkeypatch):
    prediction_cache.invalidate(slip_cache_key("PL", 0.65, 4))

    def fail_if_called(self, league="PL", limit=None):
        raise AssertionError("predict_upcoming should not run when drift blocks slip generation")

    monkeypatch.setattr(
        "src.api.main._read_prediction_guard_status",
        lambda: "STOP",
    )
    monkeypatch.setattr(
        "src.predictions.predictor.Predictor.predict_upcoming",
        fail_if_called,
    )

    response = client.get("/api/v1/slips/PL")

    assert response.status_code == 200
    payload = response.json()
    assert payload["slip"] == []
    assert payload["drift_status"] == "STOP"
    assert payload["blocked"] is True
    assert "blocked by the drift guardrail" in payload["message"]


# -- Helpers ------------------------------------------------------------------

def _write_drift_state_slips(path, status: str, hit_rate=0.55, ece=0.04, mean_conf=0.52):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({
            "status": status,
            "hit_rate": hit_rate,
            "ece": ece,
            "mean_conf": mean_conf,
        }),
        encoding="utf-8",
    )


def _use_tmp_drift_state_slips(
    tmp_path,
    monkeypatch,
    *,
    global_status: str = "GO",
    league: str = "PL",
    league_status: str | None = "GO",
):
    drift_dir = tmp_path / "drift"
    global_file = drift_dir / "rolling_90d_status.json"
    _write_drift_state_slips(global_file, global_status)
    if league_status is not None:
        _write_drift_state_slips(drift_dir / f"{league}_drift_status.json", league_status)
    monkeypatch.setattr(api_main, "DRIFT_STATE_DIR", drift_dir)
    monkeypatch.setattr(DriftGuardrail, "STATUS_FILE", global_file)
    monkeypatch.setattr(drift_orchestrator_module, "DATA_DIR", tmp_path)
    monkeypatch.setattr(
        drift_orchestrator_module.DriftOrchestrator,
        "DEFAULT_STATUS_FILE",
        global_file,
    )
    monkeypatch.setattr(
        drift_orchestrator_module.DriftOrchestrator,
        "DEFAULT_CONFIDENCE_STATE_FILE",
        drift_dir / "confidence_drift_state.json",
    )
    return drift_dir


def _clear_slip_caches(league="PL"):
    prediction_cache.invalidate(prediction_cache_key(league))
    prediction_cache.invalidate(slip_cache_key(league, 0.65, 4))


# -- (B-1) League drift file STOP blocks slip generation ----------------------

def test_league_slips_stops_when_league_drift_file_is_stop(tmp_path, monkeypatch):
    """
    League drift file = STOP, global = GO -> slips endpoint must return blocked.
    Does NOT monkeypatch _read_prediction_guard_status.
    """
    _clear_slip_caches()
    _use_tmp_drift_state_slips(tmp_path, monkeypatch, league_status="STOP")

    def fail_if_called(self, league="PL", limit=None):
        raise AssertionError("predict_upcoming must not run when league drift is STOP")

    monkeypatch.setattr(
        "src.predictions.predictor.Predictor.predict_upcoming",
        fail_if_called,
    )

    response = client.get("/api/v1/slips/PL")

    assert response.status_code == 200
    payload = response.json()
    assert payload["blocked"] is True
    assert payload["drift_status"] == "STOP"
    assert payload["slip"] == []
    assert "blocked by the drift guardrail" in payload["message"]


# -- (B-2) Corrupt league drift file fails closed -----------------------------

def test_league_slips_fails_closed_on_corrupt_league_drift_file(tmp_path, monkeypatch):
    """
    Corrupt league drift file -> fail closed to STOP.
    Global GO must not override an unreadable league file.
    """
    _clear_slip_caches()
    drift_dir = _use_tmp_drift_state_slips(tmp_path, monkeypatch, league_status=None)
    league_file = drift_dir / "PL_drift_status.json"
    league_file.parent.mkdir(parents=True, exist_ok=True)
    league_file.write_bytes(b"\xff\xfe corrupt garbage \x00")

    def fail_if_called(self, league="PL", limit=None):
        raise AssertionError("predict_upcoming must not run on corrupt drift state")

    monkeypatch.setattr(
        "src.predictions.predictor.Predictor.predict_upcoming",
        fail_if_called,
    )

    response = client.get("/api/v1/slips/PL")

    assert response.status_code == 200
    payload = response.json()
    assert payload["blocked"] is True
    assert payload["drift_status"] == "STOP", (
        "Corrupt league drift file must fail closed - must not fall back to global GO"
    )


# -- (B-3) Missing league drift file fails closed -----------------------------

def test_league_slips_fails_closed_when_league_drift_file_absent(tmp_path, monkeypatch):
    """
    No league drift file present -> fail closed to STOP even with global GO.
    """
    _clear_slip_caches()
    _use_tmp_drift_state_slips(tmp_path, monkeypatch, league_status=None)
    # PL league file deliberately absent

    def fail_if_called(self, league="PL", limit=None):
        raise AssertionError("predict_upcoming must not run when league drift file is absent")

    monkeypatch.setattr(
        "src.predictions.predictor.Predictor.predict_upcoming",
        fail_if_called,
    )

    response = client.get("/api/v1/slips/PL")

    assert response.status_code == 200
    payload = response.json()
    assert payload["blocked"] is True
    assert payload["drift_status"] == "STOP"


# -- (C) Slip cache evicted when league transitions to STOP -------------------

def test_league_slips_cache_invalidated_on_league_stop_global_go(tmp_path, monkeypatch):
    """
    League-aware cache invalidation for slips:
    Stale cached slip from a GO window must be evicted when the league
    drift file transitions to STOP, even when global remains GO.

    Sequence:
      1. Seed prediction cache and slip cache with stale GO-era data.
      2. Flip league PL drift file to STOP (global stays GO).
      3. GET /api/v1/slips/PL.
      4. Assert stale cache was not served - response is blocked.
      5. Assert both caches are now empty for PL.
    """
    _clear_slip_caches()

    prediction_cache.set(prediction_cache_key("PL"), [
        {"match_id": "stale_001", "home_team": "Stale FC", "away_team": "Old United",
         "league": "PL", "kickoff_utc": "2026-04-01T12:00:00Z", "home": 0.55}
    ])
    slip_key = slip_cache_key("PL", 0.65, 4)
    prediction_cache.set(slip_key, [
        {"id": "stale_001", "match": "Stale FC vs Old United", "market": "HOME_WIN",
         "probability": 0.72, "confidence": 0.72, "league": "PL",
         "date": "2026-04-01T12:00:00Z"}
    ])

    _use_tmp_drift_state_slips(tmp_path, monkeypatch, league_status="STOP")

    def fail_if_called(self, league="PL", limit=None):
        raise AssertionError(
            "predict_upcoming must not run - stale caches should have been invalidated"
        )

    monkeypatch.setattr(
        "src.predictions.predictor.Predictor.predict_upcoming",
        fail_if_called,
    )

    response = client.get("/api/v1/slips/PL")

    assert response.status_code == 200
    payload = response.json()
    assert payload["blocked"] is True, "Stale GO-era slip cache must be evicted on league STOP"
    assert payload["slip"] == []
    assert prediction_cache.get(prediction_cache_key("PL")) is None
    assert prediction_cache.get(slip_key) is None
