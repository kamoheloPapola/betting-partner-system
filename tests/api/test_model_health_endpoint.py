import json

from fastapi.testclient import TestClient

from src.api.cache import MODEL_HEALTH_CACHE_KEY, prediction_cache
import src.api.main as api_main


client = TestClient(api_main.app)


def _write_manifest(tmp_path, manifest):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


def _model_entry(payload, market):
    return next(model for model in payload["models"] if model["market"] == market)


def test_model_health_endpoint_returns_market_entries(monkeypatch, tmp_path):
    model_path = tmp_path / "poisson_home_base.joblib"
    model_path.write_text("stub", encoding="utf-8")
    manifest_path = _write_manifest(
        tmp_path,
        {
            "active_models": {
                "poisson_home_base_pl": "poisson_home_base_pl_v140",
            },
            "poisson_home_base_pl_v140": {
                "name": "poisson_home_base",
                "league": "PL",
                "version": "1.4.0",
                "training_date": "2026-03-21T00:00:00+00:00",
                "metrics": {"brier_score": 0.187},
                "path": str(model_path),
            },
        },
    )
    prediction_cache.invalidate(MODEL_HEALTH_CACHE_KEY)
    monkeypatch.setattr(api_main.ModelRegistry, "MANIFEST_FILE", manifest_path)

    response = client.get("/model-health")
    assert response.status_code == 200

    payload = response.json()
    assert payload["global_drift_status"] == "UNKNOWN"
    assert "poisson_home_base" in payload["markets"]
    entry = payload["markets"]["poisson_home_base"][0]
    assert entry["league"] == "PL"
    assert entry["version"] == "1.4.0"
    assert entry["brier_score"] == 0.187
    assert entry["drift_status"] == "UNKNOWN"
    assert entry["status"] == "ok"

    model = _model_entry(payload, "poisson_home_base")
    assert model["last_trained"] == "2026-03-21T00:00:00+00:00"


def test_model_health_endpoint_uses_registered_at_for_last_trained(monkeypatch, tmp_path):
    manifest_path = _write_manifest(
        tmp_path,
        {
            "active_models": {
                "nb_home_corners_base_pl": "nb_home_corners_base_pl_v200",
            },
            "nb_home_corners_base_pl_v200": {
                "name": "nb_home_corners_base",
                "league": "PL",
                "version": "2.0.0",
                "registered_at": "2026-03-18T00:00:00+00:00",
                "metrics": {},
            },
        },
    )
    prediction_cache.invalidate(MODEL_HEALTH_CACHE_KEY)
    monkeypatch.setattr(api_main.ModelRegistry, "MANIFEST_FILE", manifest_path)

    response = client.get("/model-health")
    assert response.status_code == 200

    payload = response.json()
    entry = payload["markets"]["nb_home_corners_base"][0]
    assert entry["drift_status"] == "UNKNOWN"
    model = _model_entry(payload, "nb_home_corners_base")
    assert model["last_trained"] == "2026-03-18T00:00:00+00:00"


def test_model_health_endpoint_falls_back_to_top_level_manifest_brier(monkeypatch, tmp_path):
    manifest_path = _write_manifest(
        tmp_path,
        {
            "active_models": {
                "poisson_home_base_pl": "poisson_home_base_pl_v140",
            },
            "poisson_home_base_pl_v140": {
                "name": "poisson_home_base",
                "league": "PL",
                "version": "1.4.0",
                "training_date": "2026-03-21T00:00:00+00:00",
                "metrics": {},
                "brier_score": 0.241,
            },
        },
    )
    prediction_cache.invalidate(MODEL_HEALTH_CACHE_KEY)
    monkeypatch.setattr(api_main.ModelRegistry, "MANIFEST_FILE", manifest_path)

    response = client.get("/model-health")
    assert response.status_code == 200

    payload = response.json()
    entry = payload["markets"]["poisson_home_base"][0]
    assert entry["brier_score"] == 0.241
    model = _model_entry(payload, "poisson_home_base")
    assert model["last_trained"] == "2026-03-21T00:00:00+00:00"
