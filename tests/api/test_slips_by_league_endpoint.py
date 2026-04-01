from fastapi.testclient import TestClient

from src.api.main import app
from src.core.exceptions import ConfigurationError


client = TestClient(app)


def test_league_slips_endpoint_returns_latest_recommendations(monkeypatch):
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

    response = client.get("/api/v1/slips/PL")

    assert response.status_code == 200
    payload = response.json()
    assert payload["model_state"]
    assert len(payload["slip"]) == 1
    assert payload["slip"][0]["market"] == "HOME_WIN"
    assert payload["slip"][0]["league"] == "PL"


def test_league_slips_endpoint_returns_503_on_model_environment_mismatch(monkeypatch):
    def fake_predict_upcoming(self, league="PL", limit=None):
        raise ConfigurationError("sklearn 1.8.0 required")

    monkeypatch.setattr(
        "src.predictions.predictor.Predictor.predict_upcoming",
        fake_predict_upcoming,
    )

    response = client.get("/api/v1/slips/PL")

    assert response.status_code == 503
    assert response.json()["detail"] == "Model environment mismatch: sklearn 1.8.0 required"
