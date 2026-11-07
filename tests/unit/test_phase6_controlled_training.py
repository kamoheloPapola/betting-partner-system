from __future__ import annotations

import copy
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from scripts.run_phase6_controlled_retraining import (
    DEFAULT_RUN_DIR,
    EXPECTED_PARTITION_HASHES,
    PHASE4_RUN,
    _same,
    protected_snapshot,
)
from src.ml.phase6_controlled_training import (
    MODEL_ROLE_ORDER,
    Phase6TrainingError,
    assert_evaluation_only_path,
    assert_no_continuation_state,
    compare_predictions,
    explicit_model_configs,
    row_ids_hash,
    seal_specification,
    specification_fingerprint,
    verify_candidate_assessment,
    verify_partition_sets,
)
from src.ml.training_lineage import (
    LINEAGE_REPRODUCED,
    build_future_training_record,
    build_partition_record,
    seal_record,
    verify_lineage_record,
)
from src.predictions.provenance import sha256_file


class FittedLightGBMState:
    booster_ = object()

    def get_params(self, deep=True):
        return {"warm_start": False}


class FittedXGBoostState:
    _Booster = object()

    def get_params(self, deep=True):
        return {}


def _partitions():
    return {
        "training": ["train-a", "train-b"],
        "validation": ["val-a"],
        "calibration_fit": ["cal-a"],
        "final_test": ["test-a"],
    }


def _partition_hashes(partitions):
    return {name: row_ids_hash(values) for name, values in partitions.items()}


def test_legacy_or_fitted_estimator_state_cannot_enter_phase6():
    for model in (FittedLightGBMState(), FittedXGBoostState()):
        with pytest.raises(Phase6TrainingError, match="fitted state"):
            assert_no_continuation_state(model, fit_kwargs={})


@pytest.mark.parametrize("argument", ["init_model", "xgb_model", "process_type", "updater", "warm_start"])
def test_hidden_framework_continuation_argument_is_rejected(argument):
    class Fresh:
        def get_params(self, deep=True):
            return {}

    with pytest.raises(Phase6TrainingError, match="Forbidden continuation"):
        assert_no_continuation_state(Fresh(), fit_kwargs={argument: object()})


@pytest.mark.parametrize("partition", ["training", "validation"])
def test_changed_frozen_fixture_identity_fails(partition):
    partitions = _partitions()
    expected = _partition_hashes(partitions)
    changed = copy.deepcopy(partitions)
    changed[partition][0] = "substituted"
    with pytest.raises(Phase6TrainingError, match="identity mismatch"):
        verify_partition_sets(changed, expected_hashes=expected)


def test_protected_partition_overlap_fails_closed():
    partitions = _partitions()
    expected = _partition_hashes(partitions)
    partitions["calibration_fit"] = ["train-a"]
    expected["calibration_fit"] = row_ids_hash(partitions["calibration_fit"])
    with pytest.raises(Phase6TrainingError, match="overlap"):
        verify_partition_sets(partitions, expected_hashes=expected)


def _partition_record(features=None, target=None):
    frame = features if features is not None else pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    values = target if target is not None else pd.Series([1.0, 2.0])
    return build_partition_record(
        role="model_fit",
        features=frame,
        targets=values,
        row_ids=["one", "two"],
        event_times=["2020-01-01", "2020-01-02"],
        target_name="home_goals",
        missing_value_policy="reject",
    )


def test_changed_matrix_or_target_changes_lineage_material():
    original = _partition_record()
    changed_matrix = _partition_record(pd.DataFrame({"a": [9.0, 2.0], "b": [3.0, 4.0]}))
    changed_target = _partition_record(target=pd.Series([9.0, 2.0]))
    assert changed_matrix["feature_matrix_sha256"] != original["feature_matrix_sha256"]
    assert changed_target["target_vector_sha256"] != original["target_vector_sha256"]


def test_changed_feature_order_changes_schema_and_matrix_identity():
    original = _partition_record()
    changed = _partition_record(pd.DataFrame({"b": [3.0, 4.0], "a": [1.0, 2.0]}))
    assert changed["feature_schema_sha256"] != original["feature_schema_sha256"]
    assert changed["feature_matrix_sha256"] != original["feature_matrix_sha256"]


def test_hyperparameter_or_seed_change_changes_specification_identity():
    base = {
        "schema_version": "ffps.phase6_training_specification.v1",
        "specification_id": "test",
        "created_at": "2026-01-01T00:00:00+00:00",
        "model_roles": list(MODEL_ROLE_ORDER),
        "model_protocols": explicit_model_configs(),
        "seed": 42,
    }
    first = seal_specification(base)
    hyper = copy.deepcopy(base)
    hyper["model_protocols"]["home_goals_lgbm"]["n_estimators"] = 501
    seed = copy.deepcopy(base)
    seed["seed"] = 43
    assert specification_fingerprint(seal_specification(hyper)) != first["specification_sha256"]
    assert specification_fingerprint(seal_specification(seed)) != first["specification_sha256"]


def test_candidate_cannot_become_trusted_without_verified_exclusions(tmp_path: Path):
    artifact = tmp_path / "model.pkl"
    joblib.dump({"model": "dummy"}, artifact)
    dataset = tmp_path / "matrix.csv"
    dataset.write_text("x\n1\n", encoding="utf-8")
    source = tmp_path / "trainer.py"
    source.write_text("seed=42\n", encoding="utf-8")
    record = build_future_training_record(
        artifact_path=artifact,
        model_role="home_goals_lgbm",
        manifest_key="candidate",
        registry_slot="evaluation",
        model_family="lgbm",
        target_name="home_goals",
        fit_features=pd.DataFrame({"x": [1.0]}),
        fit_targets=pd.Series([1.0]),
        fit_row_ids=["train"],
        fit_event_times=["2020-01-01"],
        evaluation_features=pd.DataFrame({"x": [2.0]}),
        evaluation_targets=pd.Series([0.0]),
        evaluation_row_ids=["val"],
        evaluation_event_times=["2023-01-01"],
        effective_config={"n_estimators": 1},
        randomness={"random_state": 42},
        fit_mode="full_retrain",
        warm_start_predecessor=None,
        source_dataset_path=dataset,
        source_files=[source],
        root=tmp_path,
    ).to_dict()
    record.update(
        classification=LINEAGE_REPRODUCED,
        lineage_verified=True,
        lineage_trusted_for_promotion=True,
        reproducibility={"input": "VERIFIED", "fit": "VERIFIED", "prediction": "REPRODUCED", "bytes": "VERIFIED"},
    )
    record["partition_exclusions"] = {"training_vs_final_test": "UNPROVEN"}
    record.pop("training_lineage_fingerprint", None)
    record.pop("record_integrity_sha256", None)
    result = verify_lineage_record(seal_record(record), root=tmp_path)
    assert result["lineage_verified"] is False
    assert "trusted lineage requires verified exclusion disjointness" in result["errors"]


def test_phase6_output_path_cannot_target_production_models(tmp_path: Path):
    production = Path(__file__).resolve().parents[2] / "src" / "ml" / "models" / "candidate.joblib"
    with pytest.raises(Phase6TrainingError, match="escapes Phase 6"):
        assert_evaluation_only_path(production, run_dir=tmp_path)


def test_reproduction_comparison_enforces_predeclared_tolerance():
    result = compare_predictions([1.0], [1.0 + 5e-13], atol=1e-12, rtol=1e-12)
    assert result["reproduced"] is True
    with pytest.raises(Phase6TrainingError, match="tolerance"):
        compare_predictions([1.0], [1.0], atol=1e-11, rtol=0.0)


def test_phase6_frozen_executions_have_identical_input_config_and_seed_identities():
    primary = json.loads((DEFAULT_RUN_DIR / "executions" / "primary_manifest.json").read_text())
    repeat = json.loads((DEFAULT_RUN_DIR / "executions" / "reproduction_manifest.json").read_text())
    assert primary["training_specification_sha256"] == repeat["training_specification_sha256"]
    assert primary["input_contract_sha256"] == repeat["input_contract_sha256"]
    for first, second in zip(primary["models"], repeat["models"], strict=True):
        assert first["role"] == second["role"]
        assert first["config_sha256"] == second["config_sha256"]
        assert first["randomness_sha256"] == second["randomness_sha256"]


def test_phase4_candidate_provenance_records_exact_loaded_candidate_bytes():
    bundle = json.loads((DEFAULT_RUN_DIR / "candidate_bundle.json").read_text())
    provenance = json.loads((DEFAULT_RUN_DIR / "integration" / "prediction_provenance.json").read_text())
    expected = {item["role"]: item["artifact"]["sha256"] for item in bundle["models"]}
    actual = {item["role"]: item["artifact_sha256"] for item in provenance["model_provenance"]["members"]}
    assert actual == expected
    assert provenance["trust_assertions"]["model_training_lineage_proven"] is True


def test_active_model_hashes_and_manifest_remain_unchanged():
    before = json.loads((DEFAULT_RUN_DIR / "protected_before.json").read_text())
    assert _same(before, protected_snapshot())


def test_normal_prediction_routing_still_points_to_active_not_candidate_models():
    phase4 = json.loads((PHASE4_RUN / "prediction_provenance.json").read_text())
    manifest_path = Path(__file__).resolve().parents[2] / "src" / "ml" / "models" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for member in phase4["model_provenance"]["members"]:
        assert manifest["active_models"][member["registry_slot"]] == member["manifest_key"]
        entry = manifest[member["manifest_key"]]
        artifact = manifest_path.parent / entry["filename"]
        assert artifact.name == Path(member["source_location"]).name
        assert sha256_file(artifact) == member["artifact_sha256"]
    assert "phase6-evaluation-only" not in manifest_path.read_text()


def test_candidate_assessment_detects_artifact_substitution(tmp_path: Path):
    assessment_path = DEFAULT_RUN_DIR / "assessments" / "home_goals_lgbm.json"
    assessment = json.loads(assessment_path.read_text())
    fake_root = tmp_path
    for section in ("artifact", "core_lineage", "training_specification", "partition_proof", "reproduction"):
        item = assessment[section]
        location = item.get("source_location") or item.get("path")
        source = Path(__file__).resolve().parents[2] / location
        target = fake_root / location
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
    artifact = fake_root / assessment["artifact"]["source_location"]
    artifact.write_bytes(b"substituted")
    result = verify_candidate_assessment(assessment, root=fake_root)
    assert result["verified"] is False
    assert "artifact reference mismatch" in result["errors"]
