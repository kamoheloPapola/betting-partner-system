from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.evaluation.phase1_pl_1x2 import (
    ACTIVE_GOAL_SLOTS,
    BOOKMAKER_TOKENS,
    HOLDOUT_START,
    LEAGUE,
    MATCHES_PATH,
    PHASE0_MANIFEST_PATH,
    TARGET_TOKENS,
    TRAIN_END_EXCLUSIVE,
    Phase1IntegrityError,
    PointInTimeFeatureAuditor,
    TemporalPolicy,
    active_goal_contract,
    assert_feature_contract,
    assert_isolated_training_policy,
    assert_phase0_inputs_unchanged,
    fixture_seed,
    live_path_evidence,
    load_and_partition,
    multiclass_metrics,
    predict_fixture_live_path,
    prediction_records_hash,
    production_goal_snapshot,
    reliability_tables,
    sha256_json,
    verify_prediction_reproduction,
)


class ConstantModel:
    def __init__(self, value: float, feature: str, role: str) -> None:
        self.value = value
        self.features = [feature]
        self.feature_names_in_ = np.asarray([feature])
        self.meta = {
            "name": role,
            "version": "phase1-test",
            "type": "lgbm_regressor_poisson",
            "features": [feature],
            "metrics": {"calibration_score": 0.1},
            "calibrator_filename": None,
            "evaluation_only": True,
        }

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        assert frame.columns.tolist() == self.features
        return np.full(len(frame), self.value, dtype=float)


@pytest.fixture(scope="module")
def phase1_data():
    return load_and_partition()


@pytest.fixture(scope="module")
def raw_auditor() -> PointInTimeFeatureAuditor:
    return PointInTimeFeatureAuditor(pd.read_csv(MATCHES_PATH, low_memory=False))


def _constant_bundle() -> tuple[dict, dict]:
    models = {
        "home_goals": ConstantModel(2.0, "f1", "home_goals"),
        "away_goals": ConstantModel(1.0, "f1", "away_goals"),
        "home_goals_xgb": ConstantModel(1.0, "f1", "home_goals_xgb"),
        "away_goals_xgb": ConstantModel(2.0, "f1", "away_goals_xgb"),
    }
    artifacts = [
        {"role": role, "sha256": sha256_json({"role": role, "value": model.value})}
        for role, model in models.items()
    ]
    bundle = {
        "feature_order": ["f1"],
        "feature_order_sha256": sha256_json(["f1"]),
        "artifacts": artifacts,
        "artifacts_aggregate_sha256": sha256_json(artifacts),
        "evaluation_only": True,
        "production_eligible": False,
        "warm_start": False,
        "calibrators_enabled": False,
    }
    return models, bundle


def _constant_case() -> tuple[pd.Series, dict]:
    row = pd.Series(
        {
            "match_hash": "phase1-live-path-test",
            "date": pd.Timestamp("2025-01-02"),
            "league": LEAGUE,
            "home_team": "Home",
            "away_team": "Away",
            "home_score": 2.0,
            "away_score": 1.0,
            "f1": 0.5,
            "h2h_match_count": 0,
        }
    )
    selected = {"f1": 0.5}
    evidence = {
        "match_hash": row["match_hash"],
        "prediction_time": "2025-01-02T00:00:00+00:00",
        "fixture_date": "2025-01-02",
        "strict_cutoff_passed": True,
        "same_day_sources_excluded": True,
        "source_rows_count": 0,
        "source_rows_sha256": sha256_json([]),
        "max_source_date": None,
        "sides": {},
        "selected_feature_values": selected,
        "selected_feature_values_sha256": sha256_json(selected),
        "missing_features": [],
        "comparison_count": 1,
    }
    return row, evidence


def test_phase0_inputs_and_active_goal_artifacts_are_still_protected() -> None:
    assert PHASE0_MANIFEST_PATH.is_file()
    assert assert_phase0_inputs_unchanged()["passed"] is True
    snapshot = production_goal_snapshot()
    assert len(snapshot["artifacts"]) == len(ACTIVE_GOAL_SLOTS)
    assert all(item["sha256"] for item in snapshot["artifacts"])


def test_active_feature_contract_preserves_order_and_excludes_prohibited_inputs(phase1_data) -> None:
    _parts, features, contracts, _sources = phase1_data
    assert len(features) == 30
    assert all(contract["features"] == features for contract in contracts.values())
    assert all(contract["feature_order_sha256"] == sha256_json(features) for contract in contracts.values())
    lowered = [feature.lower() for feature in features]
    assert not any(token in feature for feature in lowered for token in BOOKMAKER_TOKENS)
    assert not set(lowered) & TARGET_TOKENS
    assert not any("std_" in feature or "global_" in feature or "last_season_" in feature for feature in lowered)


def test_partition_boundaries_are_chronological_disjoint_and_holdout_isolated(phase1_data) -> None:
    parts, _features, _contracts, _sources = phase1_data
    assert parts.training["date"].max().date() < TRAIN_END_EXCLUSIVE
    assert parts.validation["date"].min().date() >= TRAIN_END_EXCLUSIVE
    assert parts.validation["date"].max().date() < HOLDOUT_START
    assert parts.final_holdout["date"].min().date() >= HOLDOUT_START
    train = set(parts.training["match_hash"].astype(str))
    validation = set(parts.validation["match_hash"].astype(str))
    holdout = set(parts.final_holdout["match_hash"].astype(str))
    assert not train & validation
    assert not train & holdout
    assert not validation & holdout


def test_partition_exclusions_are_explicit_and_not_imputed(phase1_data) -> None:
    parts, features, _contracts, _sources = phase1_data
    assert len(parts.exclusions) == 18
    assert all("insufficient_history_or_missing_active_feature" in item["reasons"] for item in parts.exclusions)
    for frame in (parts.training, parts.validation, parts.final_holdout):
        assert not frame[features].isna().any().any()


def test_date_level_policy_excludes_all_same_day_source_events() -> None:
    raw = pd.DataFrame(
        [
            {
                "match_hash": "prior",
                "match_date": "2024-01-01",
                "league": "PL",
                "home_team": "Alpha",
                "away_team": "Beta",
                "home_goals": 1,
                "away_goals": 0,
                "home_corners": 4,
                "away_corners": 3,
            },
            {
                "match_hash": "same-day",
                "match_date": "2024-01-02",
                "league": "PL",
                "home_team": "Alpha",
                "away_team": "Gamma",
                "home_goals": 2,
                "away_goals": 2,
                "home_corners": 5,
                "away_corners": 5,
            },
        ]
    )
    auditor = PointInTimeFeatureAuditor(raw)
    prediction_time = TemporalPolicy().prediction_time(pd.Timestamp("2024-01-02"))
    assert [event.match_hash for event in auditor.prior_events("Alpha", prediction_time)] == ["prior"]


def test_holdout_temporal_provenance_is_before_prediction_time(phase1_data) -> None:
    parts, _features, _contracts, sources = phase1_data
    assert len(sources) == len(parts.final_holdout) + 2  # excluded rows retain evidence too
    for match_hash in parts.final_holdout["match_hash"].astype(str):
        evidence = sources[match_hash]
        assert evidence["strict_cutoff_passed"] is True
        assert evidence["same_day_sources_excluded"] is True
        assert pd.Timestamp(evidence["max_source_date"]) < pd.Timestamp(evidence["fixture_date"])
        assert evidence["missing_features"] == []


def test_feature_auditor_fails_closed_on_value_mismatch(phase1_data, raw_auditor) -> None:
    parts, features, _contracts, _sources = phase1_data
    row = parts.final_holdout.iloc[0].copy()
    row["home_rolling_goals_scored_3"] += 0.25
    with pytest.raises(Phase1IntegrityError, match="Point-in-time feature mismatch"):
        raw_auditor.audit_row(row, features, TemporalPolicy(), include_source_ids=False)


@pytest.mark.parametrize(
    ("warm_start", "calibrators_enabled", "message"),
    [
        (True, False, "Warm start"),
        (False, True, "Calibration"),
    ],
)
def test_isolated_training_rejects_warm_start_and_calibration(
    warm_start: bool,
    calibrators_enabled: bool,
    message: str,
) -> None:
    with pytest.raises(Phase1IntegrityError, match=message):
        assert_isolated_training_policy(
            warm_start=warm_start,
            calibrators_enabled=calibrators_enabled,
        )


def test_feature_contract_rejects_bookmaker_target_and_global_fallback_fields() -> None:
    for feature in ("bookmaker_odds", "home_score", "home_std_goals_scored"):
        with pytest.raises(Phase1IntegrityError):
            assert_feature_contract([feature])


def test_live_path_uses_four_goal_members_simulator_and_not_match_outcome() -> None:
    models, bundle = _constant_bundle()
    row, evidence = _constant_case()
    record = predict_fixture_live_path(row, models, bundle, evidence, n_simulations=2_000)
    assert record["pathway"]["goal_function"].endswith("._calc_goals")
    assert record["pathway"]["simulator"].endswith("MatchSimulator.simulate")
    assert record["pathway"]["member_calls"] == {role: 1 for role in models}
    assert record["pathway"]["match_outcome_classifier_used"] is False
    assert record["pathway"]["goal_calibrators_loaded"] is False
    assert record["output"]["probability_sum"] == pytest.approx(1.0, abs=1e-12)
    assert record["ensemble"]["home_lambda"] == pytest.approx(1.6, abs=1e-12)
    assert record["ensemble"]["away_lambda"] == pytest.approx(1.4, abs=1e-12)


def test_fixture_seed_and_live_prediction_are_deterministic_and_order_independent() -> None:
    models, bundle = _constant_bundle()
    row, evidence = _constant_case()
    first = predict_fixture_live_path(row, models, bundle, evidence, n_simulations=2_000)
    second = predict_fixture_live_path(row, models, bundle, evidence, n_simulations=2_000)
    assert first["rng"]["fixture_seed"] == second["rng"]["fixture_seed"]
    assert prediction_records_hash([first]) == prediction_records_hash([second])
    result = verify_prediction_reproduction([first], [second], tolerance=0.0)
    assert result["passed"] is True
    other_seed = fixture_seed(
        match_hash="different",
        prediction_time=evidence["prediction_time"],
        bundle_hash=bundle["artifacts_aggregate_sha256"],
        n_simulations=2_000,
    )
    assert other_seed != first["rng"]["fixture_seed"]


def test_live_path_source_code_evidence_excludes_match_outcome_classifier() -> None:
    evidence = live_path_evidence()
    assert evidence["passed"] is True
    assert evidence["match_outcome_reference_present"] is False
    assert evidence["probability_source"].startswith("MatchSimulator.simulate")


def test_multiclass_metrics_and_reliability_have_declared_semantics() -> None:
    labels = np.asarray([0, 1, 2, 0], dtype=int)
    probabilities = np.asarray(
        [
            [0.8, 0.1, 0.1],
            [0.2, 0.6, 0.2],
            [0.2, 0.2, 0.6],
            [0.3, 0.4, 0.3],
        ]
    )
    metrics = multiclass_metrics(labels, probabilities)
    expected_one_hot = np.eye(3)[labels]
    assert metrics["accuracy"] == pytest.approx(0.75)
    assert metrics["multiclass_brier"] == pytest.approx(
        np.mean(np.sum((probabilities - expected_one_hot) ** 2, axis=1))
    )
    reliability = reliability_tables(labels, probabilities)
    assert reliability["methodology"]["binning"] == "10 fixed-width bins over [0,1]"
    assert 0.0 <= reliability["top_label"]["ece"] <= 1.0


def test_active_goal_contract_matches_phase1_data(phase1_data) -> None:
    _parts, expected_features, expected_contracts, _sources = phase1_data
    features, contracts = active_goal_contract()
    assert features == expected_features
    assert contracts == expected_contracts

