import json
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from src.ml.trainer import ModelTrainer


class StubModel:
    def train(self, X, y):
        self.last_train_shape = X.shape
        self.last_target_size = len(y)

    def predict(self, X):
        return np.ones(len(X), dtype=float)

    def save(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"stub-model")


def _training_frame():
    return pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=12, freq="D"),
            "home_score": [1, 0, 2, 1, 3, 1, 2, 2, 0, 1, 2, 1],
            "feat1": np.linspace(0.1, 1.2, 12),
        }
    )


def test_train_model_exports_drift_baselines(isolated_repo_state, monkeypatch):
    registry = MagicMock()
    registry.get_next_version.return_value = "1.0.0"
    trainer = ModelTrainer(registry)
    monkeypatch.setattr(trainer, "_init_model", lambda model_type, params: StubModel())

    model, metadata = trainer.train_model(
        _training_frame(),
        target_col="home_score",
        league="PL",
        model_type="poisson",
        model_name="poisson_home_base",
        features=["feat1"],
        extra_metadata={
            "drift_baselines": {
                "hit_rate": 0.74,
                "ece": 0.031,
                "mean_conf": 0.56,
                "training_window_days": 365,
            }
        },
        mode="debug",
    )

    baseline_path = isolated_repo_state["drift_baseline_file"]
    payload = json.loads(baseline_path.read_text(encoding="utf-8"))

    assert isinstance(model, StubModel)
    assert payload["baseline_hit_rate"] == pytest.approx(0.74)
    assert payload["baseline_metrics"]["mean_confidence"] == pytest.approx(0.56)
    assert payload["training_window_days"] == 365
    assert payload["league"] == "PL"
    assert payload["source"] == "training_artifact"
    assert metadata["drift_baselines_file"] == str(baseline_path)


def test_train_model_rejects_incomplete_drift_baselines(
    isolated_repo_state,
    monkeypatch,
):
    registry = MagicMock()
    registry.get_next_version.return_value = "1.0.0"
    trainer = ModelTrainer(registry)
    monkeypatch.setattr(trainer, "_init_model", lambda model_type, params: StubModel())

    with pytest.raises(ValueError, match="missing required metrics: mean_conf"):
        trainer.train_model(
            _training_frame(),
            target_col="home_score",
            league="PL",
            model_type="poisson",
            model_name="poisson_home_base",
            features=["feat1"],
            extra_metadata={
                "drift_baselines": {
                    "hit_rate": 0.74,
                    "ece": 0.031,
                }
            },
            mode="debug",
        )
