from fastapi.testclient import TestClient

import src.api.main as api_main


client = TestClient(api_main.app)


def test_model_health_endpoint_returns_market_entries(monkeypatch):
    monkeypatch.setattr(api_main, "DEFAULT_TRAINING_LEAGUES", ["PL"])
    monkeypatch.setattr(api_main, "MODEL_CONFIGS", [{"name": "poisson_home_base"}])

    class FakeRegistry:
        def get_production_model_for_league(self, league, model_type="poisson_home_base"):
            if league == "PL" and model_type == "poisson_home_base":
                return {
                    "league": "PL",
                    "version": "1.4.0",
                    "trained_at": "2026-03-20T00:00:00+00:00",
                    "metrics": {"brier_score": 0.187},
                }
            return None

    class FakeDrift:
        STOP = "STOP"

        def __init__(self):
            self.market_status = {"poisson_home_base": "WATCH"}

        def inspect_global_state(self):
            return {"status": "GO", "evaluated_at": "2026-03-31T00:00:00Z"}

        def load_confidence_state(self):
            return None

    monkeypatch.setattr(api_main, "ModelRegistry", FakeRegistry)
    monkeypatch.setattr(api_main, "DriftOrchestrator", FakeDrift)

    response = client.get("/model-health")
    assert response.status_code == 200

    payload = response.json()
    assert payload["global_drift_status"] == "GO"
    assert "poisson_home_base" in payload["markets"]
    entry = payload["markets"]["poisson_home_base"][0]
    assert entry["league"] == "PL"
    assert entry["version"] == "1.4.0"
    assert entry["brier_score"] == 0.187
    assert entry["drift_status"] == "WATCH"
    assert entry["last_trained"] == "2026-03-20T00:00:00+00:00"


def test_model_health_endpoint_falls_back_to_global_drift_status(monkeypatch):
    monkeypatch.setattr(api_main, "DEFAULT_TRAINING_LEAGUES", ["PL"])
    monkeypatch.setattr(api_main, "MODEL_CONFIGS", [{"name": "nb_home_corners_base"}])

    class FakeRegistry:
        def get_production_model_for_league(self, league, model_type="nb_home_corners_base"):
            if league == "PL" and model_type == "nb_home_corners_base":
                return {
                    "league": "PL",
                    "version": "2.0.0",
                    "registered_at": "2026-03-18T00:00:00+00:00",
                    "metrics": {},
                }
            return None

    class FakeDrift:
        STOP = "STOP"

        def __init__(self):
            self.market_status = {}

        def inspect_global_state(self):
            return {"status": "STOP", "evaluated_at": "2026-03-31T00:00:00Z"}

        def load_confidence_state(self):
            return None

    monkeypatch.setattr(api_main, "ModelRegistry", FakeRegistry)
    monkeypatch.setattr(api_main, "DriftOrchestrator", FakeDrift)

    response = client.get("/model-health")
    assert response.status_code == 200

    payload = response.json()
    entry = payload["markets"]["nb_home_corners_base"][0]
    assert entry["drift_status"] == "STOP"
    assert entry["last_trained"] == "2026-03-18T00:00:00+00:00"

