import json

from fastapi.testclient import TestClient

from src.api.cache import prediction_cache, prediction_cache_key
import src.api.main as api_main
from src.api.main import app
from src.core.exceptions import ConfigurationError
from src.strategies.drift_guard import DriftGuardrail


client = TestClient(app)


def test_trigger_predictions_runs_pipeline(tmp_path, monkeypatch):
    def fake_predict_for_show_predictions(
        self,
        *,
        league=None,
        date="upcoming",
        show_all=True,
        timezone="LOCAL",
        simulate=True,
        limit=None,
    ):
        assert league == "PL"
        assert date == "upcoming"
        assert show_all is True
        assert timezone == "LOCAL"
        assert simulate is True
        assert limit is None
        return [
            {
                "home_team": "Home FC",
                "away_team": "Away FC",
                "home": 0.55,
                "draw": 0.25,
                "away": 0.20,
                "btts": 0.61,
                "o25": 0.57,
                "ensemble_divergence": True,
            }
        ]

    monkeypatch.setattr(
        "src.predictions.predictor.Predictor.predict_for_show_predictions",
        fake_predict_for_show_predictions,
    )
    _use_tmp_drift_state(tmp_path, monkeypatch, league_status="GO")

    response = client.post("/api/v1/predictions/trigger", json={"league": "PL"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["league"] == "PL"
    assert payload["total_predictions"] == 1
    assert len(payload["predictions"]) == 1
    first = payload["predictions"][0]
    assert first["home_team"] == "Home FC"
    assert first["away_team"] == "Away FC"
    assert first["home_win_prob"] == 0.55
    assert first["draw_prob"] == 0.25
    assert first["away_win_prob"] == 0.2
    assert first["btts_prob"] == 0.61
    assert first["over_25_prob"] == 0.57
    assert first["confidence"] == 0.55
    assert first["ensemble_divergence"] is True


def test_trigger_predictions_rejects_invalid_limit():
    response = client.post(
        "/api/v1/predictions/trigger",
        json={"league": "PL", "limit": 0},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "limit must be >= 1"


def test_trigger_predictions_returns_503_on_model_environment_mismatch(tmp_path, monkeypatch):
    def fake_predict_for_show_predictions(self, **kwargs):
        raise ConfigurationError("sklearn 1.8.0 required")

    monkeypatch.setattr(
        "src.predictions.predictor.Predictor.predict_for_show_predictions",
        fake_predict_for_show_predictions,
    )
    _use_tmp_drift_state(tmp_path, monkeypatch, league_status="GO")

    response = client.post("/api/v1/predictions/trigger", json={"league": "PL"})

    assert response.status_code == 503
    assert response.json()["detail"] == "Model environment mismatch: sklearn 1.8.0 required"


def test_trigger_predictions_returns_blocked_response_when_drift_stops_predictions(monkeypatch):
    def fail_if_called(self, **kwargs):
        raise AssertionError("predict_for_show_predictions should not run when drift blocks predictions")

    monkeypatch.setattr(
        "src.api.main._read_prediction_guard_status",
        lambda: "STOP",
    )
    monkeypatch.setattr(
        "src.predictions.predictor.Predictor.predict_for_show_predictions",
        fail_if_called,
    )

    response = client.post("/api/v1/predictions/trigger", json={"league": "PL"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["league"] == "PL"
    assert payload["total_predictions"] == 0
    assert payload["predictions"] == []
    assert payload["drift_status"] == "STOP"
    assert payload["blocked"] is True
    assert "blocked by the drift guardrail" in payload["message"]


# -- Helpers ------------------------------------------------------------------

def _write_drift_state(path, status: str, hit_rate=0.55, ece=0.04, mean_conf=0.52):
    """Write a well-formed drift state JSON file to path."""
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


def _use_tmp_drift_state(
    tmp_path,
    monkeypatch,
    *,
    global_status: str = "GO",
    league: str = "PL",
    league_status: str | None = "GO",
):
    _write_drift_state(tmp_path / "rolling_90d_status.json", global_status)
    if league_status is not None:
        _write_drift_state(tmp_path / f"{league}_drift_status.json", league_status)
    monkeypatch.setattr(api_main, "DRIFT_STATE_DIR", tmp_path)
    monkeypatch.setattr(DriftGuardrail, "STATUS_FILE", tmp_path / "rolling_90d_status.json")


# -- (B-1) Real drift file - league STOP, global GO ---------------------------

def test_trigger_predictions_stops_when_league_drift_file_is_stop(tmp_path, monkeypatch):
    """
    League drift file = STOP, global = GO -> endpoint must block predictions.
    Uses a real tmp_path file - does NOT monkeypatch _read_prediction_guard_status.
    """
    _use_tmp_drift_state(tmp_path, monkeypatch, league_status="STOP")

    def fail_if_called(self, **kwargs):
        raise AssertionError("predict_for_show_predictions must not run when league drift is STOP")

    monkeypatch.setattr(
        "src.predictions.predictor.Predictor.predict_for_show_predictions",
        fail_if_called,
    )

    response = client.post("/api/v1/predictions/trigger", json={"league": "PL"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["blocked"] is True
    assert payload["drift_status"] == "STOP"
    assert payload["total_predictions"] == 0
    assert payload["predictions"] == []
    assert "blocked by the drift guardrail" in payload["message"]


# -- (B-2) Corrupt league drift file fails closed -----------------------------

def test_trigger_predictions_fails_closed_on_corrupt_league_drift_file(tmp_path, monkeypatch):
    """
    Corrupt (non-JSON) league drift file must fail closed to STOP.
    Global GO must not override an unreadable league file.
    """
    _use_tmp_drift_state(tmp_path, monkeypatch, league_status=None)
    league_file = tmp_path / "PL_drift_status.json"
    league_file.parent.mkdir(parents=True, exist_ok=True)
    league_file.write_bytes(b"\xff\xfe corrupt garbage \x00")

    def fail_if_called(self, **kwargs):
        raise AssertionError("predict_for_show_predictions must not run on corrupt drift state")

    monkeypatch.setattr(
        "src.predictions.predictor.Predictor.predict_for_show_predictions",
        fail_if_called,
    )

    response = client.post("/api/v1/predictions/trigger", json={"league": "PL"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["blocked"] is True
    assert payload["drift_status"] == "STOP", (
        "Corrupt drift file must fail closed - unknown state must become STOP"
    )


# -- (B-3) Missing league drift file fails closed -----------------------------

def test_trigger_predictions_fails_closed_when_league_drift_file_absent(tmp_path, monkeypatch):
    """
    Absent league drift file must also fail closed to STOP.
    Regression guard against the PL STOP artifact going undetected.
    """
    _use_tmp_drift_state(tmp_path, monkeypatch, league_status=None)
    # PL league file deliberately not created

    def fail_if_called(self, **kwargs):
        raise AssertionError("predict_for_show_predictions must not run when league drift file is absent")

    monkeypatch.setattr(
        "src.predictions.predictor.Predictor.predict_for_show_predictions",
        fail_if_called,
    )

    response = client.post("/api/v1/predictions/trigger", json={"league": "PL"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["blocked"] is True
    assert payload["drift_status"] == "STOP"


# -- (C) Stale prediction cache evicted on league STOP ------------------------

def test_trigger_predictions_cache_invalidated_on_league_stop_global_go(tmp_path, monkeypatch):
    """
    League-aware cache invalidation: a cached GO-era result must be evicted
    when the league drift file transitions to STOP - global stays GO.

    Sequence:
      1. Seed prediction cache with stale GO-era result for PL.
      2. Flip league drift file to STOP (global stays GO).
      3. POST /api/v1/predictions/trigger.
      4. Assert stale cache was not served and response is blocked.
      5. Assert cache is now empty for PL.
    """
    stale_key = prediction_cache_key("PL")
    prediction_cache.set(stale_key, [
        {
            "home_team": "Stale FC", "away_team": "Old United",
            "home": 0.55, "draw": 0.25, "away": 0.20,
            "btts": 0.60, "o25": 0.58, "ensemble_divergence": False,
        }
    ])

    _use_tmp_drift_state(tmp_path, monkeypatch, league_status="STOP")

    def fail_if_called(self, **kwargs):
        raise AssertionError(
            "predict_for_show_predictions must not run - "
            "stale cache should have been invalidated before reaching predictor"
        )

    monkeypatch.setattr(
        "src.predictions.predictor.Predictor.predict_for_show_predictions",
        fail_if_called,
    )

    response = client.post("/api/v1/predictions/trigger", json={"league": "PL"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["blocked"] is True, "Stale GO-era cache must be evicted; response must be blocked"
    assert prediction_cache.get(stale_key) is None, "Cache must be empty after league STOP invalidation"
