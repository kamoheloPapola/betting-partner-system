from __future__ import annotations

from unittest.mock import patch

import joblib
import pytest

from scripts import backtest_simulator
from src.ml import registry as registry_module
from src.ml.artifact_signing import (
    ARTIFACT_INTEGRITY_FIELD,
    ArtifactVerificationError,
    sign_artifact,
)
from src.ml.registry import ModelRegistry
from src.models.train_probability_models import ProbabilityModelTrainer


SIGNING_KEY = "test-signing-key-with-at-least-32-bytes"


class _JoblibModel:
    def __init__(self, value: str) -> None:
        self.value = value


@pytest.fixture
def isolated_registry(tmp_path, monkeypatch):
    monkeypatch.setenv("ARTIFACT_SIGNING_KEY", SIGNING_KEY)
    monkeypatch.setattr(registry_module, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(ModelRegistry, "MANIFEST_FILE", tmp_path / "manifest.json")
    monkeypatch.setattr(ModelRegistry, "BACKUP_FILE", tmp_path / "manifest.json.bak")
    ModelRegistry._instance = None
    ModelRegistry._manifest_cache = None

    registry = ModelRegistry()
    try:
        yield registry
    finally:
        ModelRegistry._instance = None
        ModelRegistry._manifest_cache = None


def _publish_joblib_pkl(registry, tmp_path, *, league):
    artifact = tmp_path / "home_goals_model.pkl"
    joblib.dump(_JoblibModel("trusted"), artifact)
    registry.register_model(
        "home_goals",
        "1.0.0",
        {
            "filename": artifact.name,
            "status": "productive",
            "league": league,
        },
    )
    manifest_key = f"home_goals_v1.0.0_{league if league is not None else 'None'}"
    registry.set_active_model("home_goals", manifest_key, league=league)
    return artifact, registry.manifest[manifest_key]


def _invoke_entrypoint(call_site, registry, artifact, tmp_path, monkeypatch):
    if call_site == "warm_start":
        trainer = ProbabilityModelTrainer(
            features_path=tmp_path / "features.csv",
            models_dir=tmp_path,
        )
        trainer.registry = registry
        return trainer._load_verified_warm_start(
            existing_model_path=artifact,
            model_name="home_goals",
            league="PL",
        )

    monkeypatch.setattr(backtest_simulator, "ModelRegistry", lambda: registry)
    monkeypatch.setattr(backtest_simulator, "MODELS_DIR", tmp_path)
    return backtest_simulator.load_models()


@pytest.mark.parametrize("call_site", ["warm_start", "backtest"])
def test_legacy_entrypoint_loads_joblib_pkl_through_registry(
    call_site,
    isolated_registry,
    tmp_path,
    monkeypatch,
):
    league = "PL" if call_site == "warm_start" else None
    artifact, _metadata = _publish_joblib_pkl(
        isolated_registry,
        tmp_path,
        league=league,
    )

    if call_site == "backtest":
        away_artifact = tmp_path / "away_goals_model.pkl"
        joblib.dump(_JoblibModel("away"), away_artifact)
        isolated_registry.register_model(
            "away_goals",
            "1.0.0",
            {
                "filename": away_artifact.name,
                "status": "productive",
                "league": None,
            },
        )
        isolated_registry.set_active_model(
            "away_goals",
            "away_goals_v1.0.0_None",
            league=None,
        )
        (tmp_path / "feature_columns.json").write_text("[]", encoding="utf-8")

    loaded = _invoke_entrypoint(
        call_site,
        isolated_registry,
        artifact,
        tmp_path,
        monkeypatch,
    )

    if call_site == "warm_start":
        assert loaded.value == "trusted"
    else:
        assert loaded[0].value == "trusted"
        assert loaded[1].value == "away"


@pytest.mark.parametrize("call_site", ["warm_start", "backtest"])
@pytest.mark.parametrize("failure", ["tampered", "unsigned", "version_mismatch"])
def test_legacy_entrypoint_rejects_untrusted_bytes_before_deserialization(
    call_site,
    failure,
    isolated_registry,
    tmp_path,
    monkeypatch,
):
    league = "PL" if call_site == "warm_start" else None
    artifact, metadata = _publish_joblib_pkl(
        isolated_registry,
        tmp_path,
        league=league,
    )

    if failure == "tampered":
        tampered = bytearray(artifact.read_bytes())
        tampered[-1] ^= 0x01
        artifact.write_bytes(tampered)
    elif failure == "unsigned":
        metadata.pop(ARTIFACT_INTEGRITY_FIELD)
    else:
        metadata[ARTIFACT_INTEGRITY_FIELD] = sign_artifact(
            artifact,
            filename=artifact.name,
            model_name="home_goals",
            version="0.9.0",
            league=league,
        )

    with patch.object(registry_module.pickle, "load") as pickle_load, patch.object(
        registry_module.joblib,
        "load",
    ) as joblib_load:
        with pytest.raises(ArtifactVerificationError):
            _invoke_entrypoint(
                call_site,
                isolated_registry,
                artifact,
                tmp_path,
                monkeypatch,
            )

    pickle_load.assert_not_called()
    joblib_load.assert_not_called()


def test_warm_start_rejects_unregistered_existing_artifact_before_deserialization(
    isolated_registry,
    tmp_path,
):
    artifact = tmp_path / "home_goals_model.pkl"
    joblib.dump(_JoblibModel("unsigned"), artifact)
    trainer = ProbabilityModelTrainer(
        features_path=tmp_path / "features.csv",
        models_dir=tmp_path,
    )
    trainer.registry = isolated_registry

    with patch.object(registry_module.pickle, "load") as pickle_load, patch.object(
        registry_module.joblib,
        "load",
    ) as joblib_load:
        with pytest.raises(ArtifactVerificationError) as exc_info:
            trainer._load_verified_warm_start(
                existing_model_path=artifact,
                model_name="home_goals",
                league="PL",
            )

    pickle_load.assert_not_called()
    joblib_load.assert_not_called()
    assert exc_info.value.context["reason"] == "missing_manifest"
