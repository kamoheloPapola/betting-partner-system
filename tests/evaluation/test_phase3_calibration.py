from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from scripts.run_phase3_calibration import verify_frozen
from src.evaluation.phase3_calibration import (
    BOOKMAKER_TOKENS,
    BINARY_MARKETS,
    CausalContextBuilder,
    FourPeriodPolicy,
    Phase3IntegrityError,
    binary_metrics,
    binary_reliability,
    identity_calibrator_payloads,
    load_evaluation_calibrator,
    load_phase3_partitions,
    paired_bootstrap,
    run_fixture_live_path,
    sha256_file,
    sha256_json,
)


class ConstantModel:
    def __init__(self, value: float, role: str, features: list[str]) -> None:
        self.value = value
        self.meta = {"name": role, "type": "lgbm_regressor_poisson", "features": features, "calibrator_filename": None, "metrics": {"calibration_score": 0.05}}
        self.feature_names_in_ = np.asarray(features)

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return np.full(len(frame), self.value)


@pytest.fixture(scope="module")
def partitions_contract():
    return load_phase3_partitions(reveal_final_outcomes=False)


def test_four_period_policy_boundaries() -> None:
    policy = FourPeriodPolicy()
    assert policy.partition_name(pd.Timestamp("2022-07-31")) == "training"
    assert policy.partition_name(pd.Timestamp("2022-08-01")) == "validation"
    assert policy.partition_name(pd.Timestamp("2023-08-01")) == "calibration"
    assert policy.partition_name(pd.Timestamp("2024-08-01")) == "final_test"


def test_partition_isolation_and_final_labels_sealed(partitions_contract) -> None:
    partitions, features, _ = partitions_contract
    sets = [set(getattr(partitions, name)["match_hash"].astype(str)) for name in ("training", "validation", "calibration", "final_test")]
    assert all(not sets[left] & sets[right] for left in range(4) for right in range(left + 1, 4))
    assert not {"home_score", "away_score", "home_corners", "away_corners", "home_cards", "away_cards"} & set(partitions.final_test)
    assert not any(any(token in feature.lower() for token in BOOKMAKER_TOKENS) for feature in features)
    assert partitions.manifest["feature_causality"]["strictly_prior_source_events"] is True


def test_binary_metrics_and_ece_are_deterministic() -> None:
    y = np.asarray([0, 0, 1, 1])
    p = np.asarray([0.0, 0.25, 0.75, 1.0])
    result = binary_metrics(y, p)
    assert result["brier"] == pytest.approx(0.03125)
    assert binary_reliability(y, p) == binary_reliability(y, p)
    assert sum(item["count"] for item in result["reliability"]["bins"]) == 4


def test_paired_comparison_fails_closed_on_bad_alignment() -> None:
    with pytest.raises(Phase3IntegrityError, match="unique"):
        paired_bootstrap(["same", "same"], np.asarray([0, 1]), np.asarray([0.2, 0.8]), np.asarray([0.3, 0.7]), samples=5)
    with pytest.raises(Phase3IntegrityError, match="aligned"):
        paired_bootstrap(["a"], np.asarray([0, 1]), np.asarray([0.2]), np.asarray([0.3]), samples=5)


def test_frozen_experiment_gate_refuses_missing_or_changed_spec(tmp_path: Path) -> None:
    with pytest.raises(Phase3IntegrityError, match="not frozen"):
        verify_frozen(tmp_path)
    (tmp_path / "experiment_spec.json").write_text("{}", encoding="utf-8")
    (tmp_path / "experiment_spec.sha256").write_text("bad", encoding="utf-8")
    with pytest.raises(Phase3IntegrityError, match="hash mismatch"):
        verify_frozen(tmp_path)


def test_calibrator_lineage_mismatch_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "cal.pkl"
    payload = {"evaluation_only": True, "production_eligible": False, "base_bundle_sha256": "bundle", "calibration_predictions_sha256": "calibration", "calibration_input_probability_sha256": "p", "calibration_label_sha256": "y", "calibration_row_ids_sha256": "rows"}
    joblib.dump(payload, path)
    item = {"path": path.as_posix(), "sha256": sha256_file(path), "calibration_input_probability_sha256": "p", "calibration_label_sha256": "y", "calibration_row_ids_sha256": "rows"}
    with pytest.raises(Phase3IntegrityError, match="base-bundle lineage"):
        load_evaluation_calibrator(item, expected_bundle_sha256="wrong", expected_calibration_predictions_sha256="calibration")


def test_identity_hook_exercises_actual_live_probability_stages(partitions_contract) -> None:
    partitions, features, _ = partitions_contract
    roles = ("home_goals", "away_goals", "home_goals_xgb", "away_goals_xgb", "home_corners", "away_corners", "total_corners_lgbm", "total_corners_xgb", "total_cards_lgbm", "total_cards_xgb")
    values = (1.4, 1.1, 1.3, 1.2, 5.2, 4.5, 10.0, 9.8, 4.2, 4.0)
    models = {role: ConstantModel(value, role, features) for role, value in zip(roles, values, strict=True)}
    manifest = {"feature_order": features, "artifacts_aggregate_sha256": sha256_json({"test": "phase3-live-hook"})}
    row = partitions.calibration.iloc[0]
    result = run_fixture_live_path(row, models, manifest, CausalContextBuilder(), calibrator_payloads=identity_calibrator_payloads(), n_simulations=1000)
    assert set(result["probability_stages"]) == set(BINARY_MARKETS)
    for market in ("goals_u25", "goals_u35", "goals_btts_yes", "corners_o75", "corners_u11_5"):
        stages = result["probability_stages"][market]
        assert stages["p_after_calibrator"] == pytest.approx(stages["p_pre_calibration_hook"])
    assert result["temporal_evidence"]["max_source_event_time"] < result["temporal_evidence"]["prediction_time"]
    assert result["rng"]["n_simulations"] == 1000
