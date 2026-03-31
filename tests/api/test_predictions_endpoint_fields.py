from fastapi.testclient import TestClient

from src.api.main import app


client = TestClient(app)


def test_predictions_endpoint_exposes_ensemble_fields(monkeypatch):
    def fake_predict_upcoming(self, league="PL", limit=20):
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
                "ensemble_divergence": True,
                "divergence_pct": 27.5,
            }
        ]

    monkeypatch.setattr(
        "src.predictions.predictor.Predictor.predict_upcoming",
        fake_predict_upcoming,
    )

    response = client.get("/api/v1/predictions/PL?limit=1")

    assert response.status_code == 200
    payload = response.json()
    assert len(payload) == 1
    probabilities = payload[0]["probabilities"]
    assert probabilities["ensemble_divergence"] is True
    assert probabilities["divergence_pct"] == 27.5
