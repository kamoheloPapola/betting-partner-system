from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import joblib
import pandas as pd
import pytest

from src.ml.training_lineage import (
    LEGACY_TRAINING_LINEAGE_UNPROVEN,
    TRAINING_LINEAGE_PARTIAL,
    ModelTrainingLineageRecord,
    TrainingLineageError,
    build_future_training_record,
    build_legacy_unproven_record,
    registry_lineage_metadata,
    seal_record,
    sha256_file,
    verify_lineage_record,
)
from src.models.train_probability_models import ProbabilityModelTrainer


class DummyModel:
    def get_params(self, deep: bool = True):
        return {"random_state": 42, "n_jobs": 1, "depth": 2}


def _artifact(tmp_path: Path) -> Path:
    path = tmp_path / "models" / "goal.pkl"
    path.parent.mkdir(parents=True)
    joblib.dump(DummyModel(), path)
    return path


def _frames():
    fit = pd.DataFrame({"f1": [1.0, 2.0], "f2": [3.0, 4.0]})
    evaluation = pd.DataFrame({"f1": [5.0], "f2": [6.0]})
    return fit, pd.Series([1.0, 2.0]), evaluation, pd.Series([0.0])


def _future_record(tmp_path: Path) -> ModelTrainingLineageRecord:
    artifact = _artifact(tmp_path)
    source = tmp_path / "trainer.py"
    source.write_text("seed = 42\n", encoding="utf-8")
    dataset = tmp_path / "matrix.csv"
    dataset.write_text("id,f1,f2,y\na,1,3,1\nb,2,4,2\nc,5,6,0\n", encoding="utf-8")
    fit, y_fit, evaluation, y_evaluation = _frames()
    return build_future_training_record(
        artifact_path=artifact,
        model_role="home_goals_lgbm",
        manifest_key="home_goals_v1.0.0_PL",
        registry_slot="home_goals_PL",
        model_family="lgbm_regressor_poisson",
        target_name="home_goals",
        fit_features=fit,
        fit_targets=y_fit,
        fit_row_ids=["a", "b"],
        fit_event_times=["2020-01-01", "2020-01-02"],
        evaluation_features=evaluation,
        evaluation_targets=y_evaluation,
        evaluation_row_ids=["c"],
        evaluation_event_times=["2023-01-01"],
        effective_config={"n_estimators": 10, "objective": "poisson"},
        randomness={"random_state": 42, "n_jobs": 1},
        fit_mode="full_retrain",
        warm_start_predecessor=None,
        source_dataset_path=dataset,
        source_files=[source],
        root=tmp_path,
    )


def _legacy_record(tmp_path: Path) -> ModelTrainingLineageRecord:
    artifact = _artifact(tmp_path)
    return build_legacy_unproven_record(
        artifact_path=artifact,
        model_role="home_goals_lgbm",
        manifest_key="home_goals_v1.0.0_PL",
        registry_slot="home_goals_PL",
        model_family="lgbm_regressor_poisson",
        target_name="home_goals",
        ordered_feature_names=["f1", "f2"],
        embedded_config={"n_estimators": 500},
        embedded_randomness={"random_state": 42},
        audit_findings=["No artifact-bound training row identities exist."],
        evidence=[],
        root=tmp_path,
    )


def _reseal(payload: dict) -> dict:
    payload.pop("training_lineage_fingerprint", None)
    payload.pop("record_integrity_sha256", None)
    return seal_record(payload)


def test_legacy_assessment_verifies_but_lineage_remains_unproven(tmp_path: Path):
    record = _legacy_record(tmp_path)
    result = verify_lineage_record(record, root=tmp_path)
    assert result["assessment_verified"] is True
    assert result["lineage_verified"] is False
    assert result["classification"] == LEGACY_TRAINING_LINEAGE_UNPROVEN
    assert result["lineage_trusted_for_promotion"] is False


def test_record_is_canonical_immutable_and_round_trips(tmp_path: Path):
    record = _future_record(tmp_path)
    path = tmp_path / "record.json"
    record.save(path)
    loaded = ModelTrainingLineageRecord.load(path)
    assert loaded.canonical_bytes() == record.canonical_bytes()
    assert loaded.fingerprint == record.fingerprint
    assert verify_lineage_record(loaded, root=tmp_path)["assessment_verified"] is True
    with pytest.raises(Exception):
        record._canonical_record = "changed"


@pytest.mark.parametrize(
    "mutation",
    [
        lambda p: p["training_data"]["fit_partition"]["rows"][0].update(row_id="changed"),
        lambda p: p["training_data"]["evaluation_partition"]["rows"][0].update(row_id="changed"),
        lambda p: p["training_data"]["fit_partition"].update(ordered_feature_names=["f2", "f1"]),
        lambda p: p["training_data"]["fit_partition"]["rows"][0].update(target=99),
        lambda p: p["fit_contract"]["effective_config"].update(n_estimators=11),
        lambda p: p["fit_contract"]["randomness_and_parallelism"].update(random_state=99),
        lambda p: p["source_runtime"].update(runtime_sha256="0" * 64),
    ],
    ids=["fit-row", "evaluation-row", "feature-order", "target", "config", "rng", "runtime"],
)
def test_material_lineage_changes_are_detected(tmp_path: Path, mutation):
    record = _future_record(tmp_path)
    payload = record.to_dict()
    mutation(payload)
    result = verify_lineage_record(payload, root=tmp_path)
    assert result["assessment_verified"] is False


def test_even_resealed_overlap_fails_closed(tmp_path: Path):
    payload = _future_record(tmp_path).to_dict()
    payload["training_data"]["evaluation_partition"]["rows"][0]["row_id"] = "a"
    partition = payload["training_data"]["evaluation_partition"]
    from src.ml.training_lineage import sha256_json

    ids = ["a"]
    partition["row_ids_ordered_sha256"] = sha256_json(ids)
    partition["row_ids_set_sha256"] = sha256_json(ids)
    partition["feature_matrix_sha256"] = sha256_json(
        {
            "ordered_feature_names": partition["ordered_feature_names"],
            "feature_dtypes": partition["feature_dtypes"],
            "rows": [{"row_id": "a", "feature_values": partition["rows"][0]["feature_values"]}],
        }
    )
    partition["target_vector_sha256"] = sha256_json([{"row_id": "a", "target": partition["rows"][0]["target"]}])
    result = verify_lineage_record(_reseal(payload), root=tmp_path)
    assert result["assessment_verified"] is False
    assert "fit/evaluation row identity overlap" in result["errors"]


def test_artifact_substitution_is_detected(tmp_path: Path):
    record = _legacy_record(tmp_path)
    artifact = tmp_path / record.to_dict()["artifact"]["source_location"]
    artifact.write_bytes(b"substituted")
    result = verify_lineage_record(record, root=tmp_path)
    assert result["assessment_verified"] is False
    assert "artifact SHA-256 mismatch" in result["errors"]


def test_partial_future_record_is_not_registry_eligible(tmp_path: Path):
    record = _future_record(tmp_path)
    path = tmp_path / "lineage.json"
    record.save(path)
    metadata = registry_lineage_metadata(record, path, models_dir=tmp_path)
    assert record.to_dict()["classification"] == TRAINING_LINEAGE_PARTIAL
    assert metadata["training_lineage_eligible"] is False
    assert metadata["training_lineage_record_sha256"] == sha256_file(path)


def test_lineage_persistence_failure_prevents_registration(tmp_path: Path, monkeypatch):
    trainer = object.__new__(ProbabilityModelTrainer)
    trainer.models_dir = tmp_path / "models"
    trainer.models_dir.mkdir()
    trainer.features_path = tmp_path / "matrix.csv"
    trainer.features_path.write_text("x\n1\n", encoding="utf-8")
    trainer.registry = Mock()
    register = Mock()
    trainer._register_model = register

    def fail_persist(*args, **kwargs):
        raise TrainingLineageError("simulated persistence failure")

    monkeypatch.setattr("src.models.train_probability_models.persist_verified_record", fail_persist)
    fit, y_fit, evaluation, y_evaluation = _frames()
    with pytest.raises(TrainingLineageError, match="simulated"):
        trainer._persist_goal_model_and_lineage(
            model=DummyModel(),
            model_path=trainer.models_dir / "goal.pkl",
            name="home_goals",
            model_type="lgbm_regressor_poisson",
            target="home_goals",
            metrics={},
            features=["f1", "f2"],
            league="PL",
            X_fit=fit,
            y_fit=y_fit,
            fit_row_ids=pd.Series(["a", "b"]),
            fit_event_times=pd.Series(["2020-01-01", "2020-01-02"]),
            X_evaluation=evaluation,
            y_evaluation=y_evaluation,
            evaluation_row_ids=pd.Series(["c"]),
            evaluation_event_times=pd.Series(["2023-01-01"]),
            full_retrain=True,
        )
    register.assert_not_called()
