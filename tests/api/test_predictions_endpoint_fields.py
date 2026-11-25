import json
from datetime import datetime, timezone

from fastapi.testclient import TestClient

from src.api import main as api_main
from src.api.cache import PredictionCache
from src.api.main import app


client = TestClient(app)


def test_predictions_endpoint_exposes_ensemble_fields(tmp_path, monkeypatch):
    global_state = tmp_path / "rolling_90d_status.json"
    state = json.dumps({"status": "GO", "evaluated_at": datetime.now(timezone.utc).isoformat()})
    global_state.write_text(state, encoding="utf-8")
    (tmp_path / "PL_drift_status.json").write_text(state, encoding="utf-8")
    monkeypatch.setattr(api_main.DriftGuardrail, "STATUS_FILE", global_state)
    monkeypatch.setattr(api_main, "DRIFT_STATE_DIR", tmp_path)
    monkeypatch.setattr(api_main, "prediction_cache", PredictionCache())

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
