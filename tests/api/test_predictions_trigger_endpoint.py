from fastapi.testclient import TestClient

from src.api.main import app
from src.core.exceptions import ConfigurationError


client = TestClient(app)


def test_trigger_predictions_runs_pipeline(monkeypatch):
    def fake_predict_for_show_predictions(
        self,
        *,
        league=None,
        date="today",
        show_all=False,
        timezone="LOCAL",
        simulate=True,
        limit=None,
    ):
        assert league == "PL"
        assert date == "today"
        assert show_all is False
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


def test_trigger_predictions_returns_503_on_model_environment_mismatch(monkeypatch):
    def fake_predict_for_show_predictions(self, **kwargs):
        raise ConfigurationError("sklearn 1.8.0 required")

    monkeypatch.setattr(
        "src.predictions.predictor.Predictor.predict_for_show_predictions",
        fake_predict_for_show_predictions,
    )

    response = client.post("/api/v1/predictions/trigger", json={"league": "PL"})

    assert response.status_code == 503
    assert response.json()["detail"] == "Model environment mismatch: sklearn 1.8.0 required"
