import pickle
import warnings
from unittest.mock import patch

import pytest
from sklearn.exceptions import InconsistentVersionWarning

from src.core.exceptions import ConfigurationError
from src.ml import registry as registry_module
from src.ml.registry import ModelRegistry


class DummyModel:
    pass


def test_load_model_rejects_manifest_sklearn_mismatch(tmp_path) -> None:
    artifact = tmp_path / "stub.pkl"
    artifact.write_bytes(pickle.dumps(DummyModel()))

    registry = ModelRegistry()
    registry.manifest = {}
    meta = {
        "filename": artifact.name,
        "league": "PL",
        "sklearn_version": "0.0.0",
    }

    with patch.object(registry_module, "MODELS_DIR", tmp_path), patch.object(
        registry, "get_production_model_for_league", return_value=meta
    ):
        with pytest.raises(ConfigurationError, match="different sklearn version"):
            registry.load_model("poisson_home_base", league="PL")


def test_load_model_rejects_inconsistent_version_warning(tmp_path) -> None:
    artifact = tmp_path / "legacy.pkl"
    artifact.write_bytes(pickle.dumps(DummyModel()))

    registry = ModelRegistry()
    registry.manifest = {}
    meta = {
        "filename": artifact.name,
        "league": "PL",
    }

    def fake_load(_file):
        warnings.warn(
            InconsistentVersionWarning(
                estimator_name="DummyModel",
                current_sklearn_version="1.7.2",
                original_sklearn_version="0.24.0",
            )
        )
        return DummyModel()

    with patch.object(registry_module, "MODELS_DIR", tmp_path), patch.object(
        registry, "get_production_model_for_league", return_value=meta
    ), patch.object(registry_module.pickle, "load", side_effect=fake_load):
        with pytest.raises(ConfigurationError, match="different sklearn version"):
            registry.load_model("poisson_home_base", league="PL")
