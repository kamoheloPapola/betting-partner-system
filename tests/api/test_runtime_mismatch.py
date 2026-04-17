import json

from fastapi.testclient import TestClient

from src.api.cache import prediction_cache
from src.api.cache import prediction_cache_key, slip_cache_key
from src.api.main import app
from src.core.exceptions import ConfigurationError
from src.strategies.drift_guard import DriftGuardrail


client = TestClient(app)


def _write_global_drift_status(path, status: str = "GO") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({
            "status": status,
            "metrics": {"hit_rate": 0.75, "ece": 0.02, "mean_conf": 0.50},
        }),
        encoding="utf-8",
    )


def test_predictions_returns_503_on_model_environment_mismatch(monkeypatch):
    prediction_cache.invalidate(prediction_cache_key("PL", 1))

    def fake_predict_upcoming(self, league="PL", limit=20):
        raise ConfigurationError("sklearn 1.8.0 required")

    monkeypatch.setattr(
        "src.predictions.predictor.Predictor.predict_upcoming",
        fake_predict_upcoming,
    )

    response = client.get("/api/v1/predictions/PL?limit=1")

    assert response.status_code == 503
    assert response.json()["detail"] == "Model environment mismatch: sklearn 1.8.0 required"


def test_forbidden_fruit_returns_503_on_model_environment_mismatch(tmp_path, monkeypatch):
    status_file = tmp_path / "drift" / "rolling_90d_status.json"
    _write_global_drift_status(status_file, "GO")
    monkeypatch.setattr(DriftGuardrail, "STATUS_FILE", status_file)

    def fake_predict_upcoming(self, league="PL", limit=20):
        raise ConfigurationError("sklearn 1.8.0 required")

    monkeypatch.setattr(
        "src.predictions.predictor.Predictor.predict_upcoming",
        fake_predict_upcoming,
    )

    response = client.get("/api/v1/slips/forbidden-fruit")

    assert response.status_code == 503
    assert response.json()["detail"] == "Model environment mismatch: sklearn 1.8.0 required"
