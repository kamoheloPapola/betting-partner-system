from fastapi.testclient import TestClient

from src.api.main import app
from src.core.exceptions import ConfigurationError


client = TestClient(app)


def test_predictions_returns_503_on_model_environment_mismatch(monkeypatch):
    def fake_predict_upcoming(self, league="PL", limit=20):
        raise ConfigurationError("sklearn 1.8.0 required")

    monkeypatch.setattr(
        "src.predictions.predictor.Predictor.predict_upcoming",
        fake_predict_upcoming,
    )

    response = client.get("/api/v1/predictions/PL?limit=1")

    assert response.status_code == 503
    assert response.json()["detail"] == "Model environment mismatch: sklearn 1.8.0 required"


def test_forbidden_fruit_returns_503_on_model_environment_mismatch(monkeypatch):
    def fake_predict_upcoming(self, league="PL", limit=20):
        raise ConfigurationError("sklearn 1.8.0 required")

    monkeypatch.setattr(
        "src.predictions.predictor.Predictor.predict_upcoming",
        fake_predict_upcoming,
    )

    response = client.get("/api/v1/slips/forbidden-fruit")

    assert response.status_code == 503
    assert response.json()["detail"] == "Model environment mismatch: sklearn 1.8.0 required"
