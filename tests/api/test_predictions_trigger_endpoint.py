from fastapi.testclient import TestClient

from src.api.main import app
from src.core.exceptions import ConfigurationError


client = TestClient(app)


def test_trigger_predictions_runs_pipeline(monkeypatch):
    def fake_predict_upcoming(self, league="PL", limit=None):
        assert league == "PL"
        assert limit is None
        return [
            {
                "match_id": "abc123",
                "home_team": "Home FC",
                "away_team": "Away FC",
                "kickoff_utc": "2026-04-01T12:00:00Z",
                "league": "PL",
                "home": 0.55,
                "draw": 0.25,
                "away": 0.20,
            }
        ]

    monkeypatch.setattr(
        "src.predictions.predictor.Predictor.predict_upcoming",
        fake_predict_upcoming,
    )

    response = client.post("/api/v1/predictions/trigger", json={"league": "PL"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["league"] == "PL"
    assert payload["total_predictions"] == 1
    assert len(payload["predictions"]) == 1
    assert payload["predictions"][0]["match_id"] == "abc123"
    assert payload["predictions"][0]["probabilities"]["home"] == 0.55


def test_trigger_predictions_rejects_invalid_limit():
    response = client.post(
        "/api/v1/predictions/trigger",
        json={"league": "PL", "limit": 0},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "limit must be >= 1"


def test_trigger_predictions_returns_503_on_model_environment_mismatch(monkeypatch):
    def fake_predict_upcoming(self, league="PL", limit=None):
        raise ConfigurationError("sklearn 1.8.0 required")

    monkeypatch.setattr(
        "src.predictions.predictor.Predictor.predict_upcoming",
        fake_predict_upcoming,
    )

    response = client.post("/api/v1/predictions/trigger", json={"league": "PL"})

    assert response.status_code == 503
    assert response.json()["detail"] == "Model environment mismatch: sklearn 1.8.0 required"
