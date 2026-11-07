from __future__ import annotations

import copy

import numpy as np
import pytest

from src.evaluation.phase7_shadow import (
    AUTHORIZED_DECISIONS,
    CLASS_ORDER,
    PAIR_SCHEMA_VERSION,
    SPEC_SCHEMA_VERSION,
    EvaluationLedger,
    Phase7EvaluationError,
    apply_decision_rule,
    assert_single_labeled_cohort,
    evaluation_only_shadow_call,
    paired_block_bootstrap,
    paired_bootstrap,
    seal_specification,
    validate_paired_prediction,
    verify_specification,
)


def specification() -> dict:
    return seal_specification(
        {
            "schema_version": SPEC_SCHEMA_VERSION,
            "market": "PL_1X2",
            "class_order": list(CLASS_ORDER),
            "active": {"model_bundle_sha256": "active"},
            "candidate": {
                "model_bundle_sha256": "candidate",
                "evaluation_only": True,
            },
            "simulation": {
                "common_random_numbers": True,
                "state_advance_after_pair": "active_post_call_state",
                "n_simulations": 50_000,
            },
            "paired_statistics": {
                "resamples": 10_000,
                "confidence_level": 0.95,
            },
            "prohibited_actions": [
                "training",
                "hyperparameter_tuning",
                "recalibration",
                "ensemble_weight_tuning",
                "registry_write",
                "production_activation",
                "bookmaker_input",
            ],
            "authorized_decisions": sorted(AUTHORIZED_DECISIONS),
        }
    )


def prediction(bundle: str, *, rng_state: int = 7) -> dict:
    return {
        "fixture": {"match_hash": "fixture-1"},
        "feature_provenance": {"feature_bundle_sha256": "features"},
        "data_provenance": {
            "source_identity": {"artifacts_aggregate_sha256": "source"},
        },
        "model_provenance": {"model_bundle_sha256": bundle},
        "method_provenance": {
            "descriptor": {
                "calibrator": {"applied": False},
                "match_outcome_classifier": {"used": False},
                "simulation": {
                    "implementation": "sim",
                    "n_simulations": 50_000,
                    "max_goals": 10,
                    "top_scorelines": 15,
                    "tempo_sigma": 0.1,
                    "league": "PL",
                    "rng_state_before": {"state": rng_state},
                },
            }
        },
        "prediction_output": {
            "class_order": list(CLASS_ORDER),
            "final_probabilities": [0.4, 0.3, 0.3],
        },
        "bookmaker_independence": {
            "calibration_or_post_processing": False,
            "model_calculation": False,
            "model_feature_input": False,
        },
        "trust_assertions": {"temporal_cutoff_verified": True},
    }


def pair() -> dict:
    spec = specification()
    return {
        "schema_version": PAIR_SCHEMA_VERSION,
        "specification_fingerprint": spec["specification_fingerprint"],
        "fixture_id": "fixture-1",
        "cohort": "historical_diagnostic_only",
        "class_order": list(CLASS_ORDER),
        "authoritative_output": "active",
        "feature_bundle_sha256": "features",
        "source_snapshot_sha256": "source",
        "shared_pre_call_rng_state": {"state": 7},
        "active_prediction": prediction("active"),
        "candidate_prediction": prediction("candidate"),
        "active_phase4_verified": True,
        "candidate_phase4_verified": True,
        "candidate_training_lineage_verified": True,
        "prediction_frozen_at": "2026-08-30T00:00:00+00:00",
        "outcome_accessed_at": None,
        "outcome": None,
    }


def test_frozen_specification_verifies() -> None:
    assert verify_specification(specification())["verified"] is True


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda value: value["candidate_prediction"]["feature_provenance"].update(feature_bundle_sha256="other"), "feature-bundle"),
        (lambda value: value["candidate_prediction"]["data_provenance"]["source_identity"].update(artifacts_aggregate_sha256="other"), "source snapshot"),
        (lambda value: value["active_prediction"]["model_provenance"].update(model_bundle_sha256="other"), "active model drift"),
        (lambda value: value["candidate_prediction"]["model_provenance"].update(model_bundle_sha256="other"), "candidate model drift"),
        (lambda value: value.update(candidate_training_lineage_verified=False), "training-lineage"),
        (lambda value: value["candidate_prediction"]["method_provenance"]["descriptor"]["simulation"].update(rng_state_before={"state": 8}), "pre-call RNG"),
        (lambda value: value["candidate_prediction"]["method_provenance"]["descriptor"]["simulation"].update(n_simulations=40_000), "simulator parameter"),
        (lambda value: value["candidate_prediction"]["method_provenance"]["descriptor"]["calibrator"].update(applied=True), "calibration insertion"),
        (lambda value: value["candidate_prediction"]["bookmaker_independence"].update(model_feature_input=True), "bookmaker"),
        (lambda value: value["candidate_prediction"]["prediction_output"].update(class_order=["away_win", "draw", "home_win"]), "class order"),
        (lambda value: value.update(authoritative_output="candidate"), "cannot replace active"),
    ],
)
def test_pair_contract_fails_closed(mutate, message: str) -> None:
    value = pair()
    mutate(value)
    with pytest.raises(Phase7EvaluationError, match=message):
        validate_paired_prediction(value, specification=specification())


def test_outcome_before_prediction_leakage_fails() -> None:
    value = pair()
    value["outcome"] = {"home_goals": 1, "away_goals": 0}
    value["outcome_accessed_at"] = "2026-08-29T23:59:59+00:00"
    with pytest.raises(Phase7EvaluationError, match="outcome-before-prediction"):
        validate_paired_prediction(value, specification=specification())


def test_duplicate_fixture_fails() -> None:
    ledger = EvaluationLedger(specification())
    ledger.add_pair(pair())
    with pytest.raises(Phase7EvaluationError, match="duplicate"):
        ledger.add_pair(pair())


def test_specification_cannot_mutate_after_first_pair() -> None:
    ledger = EvaluationLedger(specification())
    ledger.add_pair(pair())
    changed = copy.deepcopy(ledger.specification)
    changed["simulation"]["n_simulations"] = 10_000
    with pytest.raises(Phase7EvaluationError, match="cannot mutate"):
        ledger.replace_specification(changed)


def test_candidate_failure_preserves_authoritative_active_output() -> None:
    state = {"token": "unchanged"}

    def fail():
        raise RuntimeError("candidate failed")

    result = evaluation_only_shadow_call(
        lambda: {"home": 0.5},
        fail,
        active_state_token=lambda: state["token"],
    )
    assert result["production_output"] == {"home": 0.5}
    assert result["authoritative_output"] == "active"
    assert "candidate failed" in result["candidate_error"]


def test_candidate_active_state_mutation_fails() -> None:
    state = {"token": "before"}

    def mutate():
        state["token"] = "after"

    with pytest.raises(Phase7EvaluationError, match="mutated active model state"):
        evaluation_only_shadow_call(
            lambda: "active",
            mutate,
            active_state_token=lambda: state["token"],
        )


def test_paired_bootstrap_is_deterministic_and_paired() -> None:
    active = {"log_loss": [1.0, 2.0, 3.0], "joint_goal_rmse": [1.0, 4.0, 9.0]}
    candidate = {"log_loss": [0.5, 1.5, 2.5], "joint_goal_rmse": [0.25, 2.25, 6.25]}
    first = paired_bootstrap(
        active,
        candidate,
        resamples=500,
        seed=42,
        confidence_level=0.95,
        root_mean_metrics=("joint_goal_rmse",),
    )
    second = paired_bootstrap(
        active,
        candidate,
        resamples=500,
        seed=42,
        confidence_level=0.95,
        root_mean_metrics=("joint_goal_rmse",),
    )
    assert first == second
    assert first["metrics"]["log_loss"]["point_delta"] == pytest.approx(-0.5)
    assert first["same_fixture_indices_across_bundles_and_metrics"] is True


def test_block_bootstrap_is_deterministic() -> None:
    active = {"log_loss": [1.0, 2.0, 3.0, 4.0]}
    candidate = {"log_loss": [0.9, 1.9, 3.1, 4.1]}
    kwargs = dict(
        blocks=["2026-08", "2026-08", "2026-09", "2026-09"],
        resamples=200,
        seed=17,
        confidence_level=0.95,
    )
    assert paired_block_bootstrap(active, candidate, **kwargs) == paired_block_bootstrap(
        active, candidate, **kwargs
    )


def test_historical_and_prospective_cannot_merge() -> None:
    records = [
        {"cohort": "historical_diagnostic_only"},
        {"cohort": "prospective_primary"},
    ]
    with pytest.raises(Phase7EvaluationError, match="cannot be merged"):
        assert_single_labeled_cohort(records, "historical_diagnostic_only")


def intervals(point: float, lower: float, upper: float) -> dict[str, dict[str, float]]:
    return {
        metric: {"point_delta": point, "lower": lower, "upper": upper}
        for metric in (
            "log_loss",
            "multiclass_brier",
            "joint_goal_rmse",
            "joint_goal_poisson_deviance",
        )
    }


def test_decision_is_inconclusive_before_horizon() -> None:
    result = apply_decision_rule(
        prospective_fixture_count=0,
        horizon_complete=False,
        paired_intervals=None,
        active_reliability=None,
        candidate_reliability=None,
        provenance_verified=True,
    )
    assert result["classification"] == "CANDIDATE_INCONCLUSIVE"


def test_decision_supports_candidate_only_under_frozen_rule() -> None:
    result = apply_decision_rule(
        prospective_fixture_count=100,
        horizon_complete=True,
        paired_intervals=intervals(-0.02, -0.04, -0.001),
        active_reliability={"top_label_ece": 0.1, "classwise_ece": 0.1},
        candidate_reliability={"top_label_ece": 0.09, "classwise_ece": 0.09},
        provenance_verified=True,
    )
    assert result["classification"] == "CANDIDATE_SUPPORTED_FOR_PROMOTION_REVIEW"
    assert result["phase8_promotion_review_authorized"] is True


def test_decision_supports_active_on_credible_primary_harm() -> None:
    result = apply_decision_rule(
        prospective_fixture_count=100,
        horizon_complete=True,
        paired_intervals=intervals(0.02, 0.001, 0.04),
        active_reliability={"top_label_ece": 0.1, "classwise_ece": 0.1},
        candidate_reliability={"top_label_ece": 0.11, "classwise_ece": 0.11},
        provenance_verified=True,
    )
    assert result["classification"] == "ACTIVE_SUPPORTED"


def test_production_rng_instance_is_not_touched_by_offline_contract() -> None:
    production_rng = np.random.default_rng(42)
    expected = np.random.default_rng(42)
    evaluation_rng = np.random.default_rng(42)
    state = copy.deepcopy(evaluation_rng.bit_generator.state)
    evaluation_rng.random(100)
    evaluation_rng.bit_generator.state = state
    evaluation_rng.random(100)
    assert production_rng.random() == expected.random()
