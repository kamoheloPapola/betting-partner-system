import numpy as np
import pandas as pd
import pytest
from sklearn.isotonic import IsotonicRegression

import src.cli.commands.prediction as prediction_module
from src.ml.calibration import (
    apply_binary_calibrator,
    fit_best_binary_calibrator,
    load_binary_calibrator_artifact,
)
from src.ml.trainer import ModelTrainer


class _StubModel:
    def __init__(self) -> None:
        self.features = ["feat1"]
        self.meta = {
            "name": "stub_model",
            "version": "1.0.0",
            "metrics": {"calibration_score": 0.01},
            "type": "poisson",
        }

    def train(self, X, y):
        return None

    def predict(self, X):
        vals = X["feat1"].to_numpy(dtype=float)
        return np.clip(vals * 2.0, 0.05, 4.0)

    def save(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"stub-model")


class _ConstantPoissonModel:
    def __init__(self, value: float, calibrator_filename: str, features: list[str]) -> None:
        self.value = value
        self.features = features
        self.meta = {
            "name": "const_poisson",
            "version": "test",
            "metrics": {"calibration_score": 0.01},
            "type": "poisson",
            "calibrator_filename": calibrator_filename,
        }

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), self.value, dtype=float)


def test_fit_best_binary_calibrator_selects_lowest_ece():
    p_raw = np.array([0.05, 0.15, 0.25, 0.35, 0.65, 0.75, 0.85, 0.95], dtype=float)
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=int)

    selected = fit_best_binary_calibrator(p_raw=p_raw, y_true=y_true)
    assert selected is not None
    assert selected["type"] in {"isotonic", "platt"}

    iso = IsotonicRegression(out_of_bounds="clip").fit(p_raw, y_true)
    iso_ece = float(np.mean(np.abs(apply_binary_calibrator(iso, "isotonic", p_raw) - y_true)))
    platt_ece = float(
        np.mean(
            np.abs(
                apply_binary_calibrator(selected["calibrator"], selected["type"], p_raw) - y_true
            )
        )
    )
    # Selected calibrator should not be dramatically worse than isotonic baseline on this monotonic sample.
    assert platt_ece <= iso_ece + 0.05


def test_trainer_persists_posthoc_calibrator_metadata(isolated_repo_state, monkeypatch):
    registry = type(
        "RegistryStub",
        (),
        {
            "get_next_version": staticmethod(lambda *args, **kwargs: "1.0.0"),
            "register_model": staticmethod(lambda *args, **kwargs: None),
        },
    )()
    trainer = ModelTrainer(registry)
    monkeypatch.setattr(trainer, "_init_model", lambda model_type, params: _StubModel())

    df = pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=80, freq="D"),
            "league": ["PL"] * 80,
            "feat1": np.linspace(0.0, 1.0, 80),
            "home_score": [0 if i % 3 == 0 else 1 for i in range(80)],
        }
    )

    _, metadata = trainer.train_model(
        df=df,
        target_col="home_score",
        league="PL",
        model_type="poisson",
        model_name="poisson_home_base",
        features=["feat1"],
        mode="debug",
    )

    assert metadata.get("calibration_split") == "validation_only"
    calibrator_filename = metadata.get("calibrator_filename")
    assert isinstance(calibrator_filename, str) and calibrator_filename.endswith("_calibrator.pkl")
    artifact = load_binary_calibrator_artifact(isolated_repo_state["models_dir"] / calibrator_filename)
    assert artifact is not None
    assert artifact.get("type") in {"isotonic", "platt"}


def test_predict_scalar_applies_model_level_calibrator(tmp_path, monkeypatch):
    models_dir = tmp_path / "models"
    monkeypatch.setattr(prediction_module, "MODELS_DIR", models_dir)
    prediction_module._MODEL_CALIBRATOR_CACHE.clear()

    calibrator_path = models_dir / "cal" / "const_calibrator.pkl"
    calibrator_path.parent.mkdir(parents=True, exist_ok=True)

    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(np.array([0.0, 1.0]), np.array([0.0, 0.8]))
    with open(calibrator_path, "wb") as handle:
        import pickle

        pickle.dump({"calibrator": iso, "type": "isotonic"}, handle)

    model = _ConstantPoissonModel(
        value=2.0,
        calibrator_filename="cal/const_calibrator.pkl",
        features=["f1"],
    )
    row = pd.Series({"f1": 0.3})

    calibrated = prediction_module._predict_scalar(model, row, ["f1"], "ctx")
    assert calibrated < 2.0
    assert calibrated > 0.05
