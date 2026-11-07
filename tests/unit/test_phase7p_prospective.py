from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from src.evaluation.phase7_shadow import (
    CLASS_ORDER,
    PAIR_SCHEMA_VERSION,
    evaluation_only_shadow_call,
    seal_specification,
)
from src.evaluation.phase7p_prospective import (
    COHORT,
    ImmutableCohortStore,
    Phase7PError,
    aggregate_eligible,
    build_missed_record,
    build_prediction_record,
    build_settlement_record,
    classify_fixture_state,
    coverage_ratio,
    seal_record,
    verify_prediction_record,
    verify_settlement_record,
)


ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = (
    ROOT
    / "baselines"
    / "phase7"
    / "phase7-active-vs-candidate-shadow-79110f9a-20260829-v2"
    / "evaluation_specification.json"
)


@pytest.fixture()
def specification() -> dict:
    return json.loads(SPEC_PATH.read_text(encoding="utf-8"))


def _prediction(bundle: str, *, fixture_id: str = "fixture-1", rng_state: int = 7) -> dict:
    return {
        "fixture": {"match_hash": fixture_id},
        "feature_provenance": {"feature_bundle_sha256": "features"},
        "data_provenance": {"source_identity": {"artifacts_aggregate_sha256": "source"}},
        "model_provenance": {"model_bundle_sha256": bundle},
        "method_provenance": {
            "descriptor": {
                "calibrator": {"applied": False},
                "match_outcome_classifier": {"used": False},
                "simulation": {
                    "implementation": "src.simulation.match_simulator.MatchSimulator.simulate",
                    "n_simulations": 50_000,
                    "max_goals": 10,
                    "top_scorelines": 15,
                    "tempo_sigma": 0.1,
                    "league": "PL",
                    "rng_state_before": {"state": rng_state},
                    "requested_home_xg": 1.4,
                    "requested_away_xg": 1.1,
                },
            }
        },
        "prediction_output": {
            "class_order": list(CLASS_ORDER),
            "final_probabilities": [0.45, 0.28, 0.27],
        },
        "bookmaker_independence": {
            "calibration_or_post_processing": False,
            "model_calculation": False,
            "model_feature_input": False,
        },
        "trust_assertions": {"temporal_cutoff_verified": True},
    }


def _pair(specification: dict, *, fixture_id: str = "fixture-1") -> dict:
    state = {"state": 7}
    return {
        "schema_version": PAIR_SCHEMA_VERSION,
        "specification_fingerprint": specification["specification_fingerprint"],
        "fixture_id": fixture_id,
        "fixture_date": "2026-09-01",
        "cohort": COHORT,
        "class_order": list(CLASS_ORDER),
        "authoritative_output": "active",
        "feature_bundle_sha256": "features",
        "source_snapshot_sha256": "source",
        "shared_pre_call_rng_state": state,
        "active_post_call_rng_state": {"state": 8},
        "candidate_post_call_rng_state": {"state": 8},
        "stream_advanced_with": "active_post_call_state",
        "active_prediction": _prediction(specification["active"]["model_bundle_sha256"], fixture_id=fixture_id),
        "candidate_prediction": _prediction(specification["candidate"]["model_bundle_sha256"], fixture_id=fixture_id),
        "active_phase4_verified": True,
        "candidate_phase4_verified": True,
        "candidate_training_lineage_verified": True,
        "prediction_frozen_at": "2026-08-31T12:00:00+00:00",
        "outcome_accessed_at": None,
        "outcome": None,
    }


def _prospective(specification: dict, *, pair: dict | None = None, fixture_id: str = "fixture-1") -> dict:
    return build_prediction_record(
        specification=specification,
        paired_evidence=pair or _pair(specification, fixture_id=fixture_id),
        fixture_kickoff_at="2026-09-01T00:00:00+00:00",
        prediction_created_at="2026-08-31T12:00:00+00:00",
        maximum_source_event_at="2026-08-30T18:00:00+00:00",
        causal_source_identity={"artifacts_aggregate_sha256": "source"},
    )


def _settlement(specification: dict, prediction: dict, **overrides) -> dict:
    values = {
        "home_goals": 2,
        "away_goals": 1,
        "outcome_source_identity": {
            "provider": "football-data.org v4",
            "path": "data/processed/matches.csv",
            "sha256": "a" * 64,
        },
        "outcome_observed_at": "2026-09-01T22:00:00+00:00",
        "settlement_created_at": "2026-09-01T22:01:00+00:00",
    }
    values.update(overrides)
    return build_settlement_record(
        specification=specification, prediction_record=prediction, **values
    )


def test_prediction_hashing_is_deterministic(specification: dict) -> None:
    assert _prospective(specification) == _prospective(specification)


def test_prediction_record_verifies(specification: dict) -> None:
    assert verify_prediction_record(_prospective(specification), specification=specification)["verified"]


def test_provenance_wrapper_tamper_fails(specification: dict) -> None:
    prediction = _prospective(specification)
    prediction["active_prediction_provenance"]["model_provenance"]["model_bundle_sha256"] = "tampered"
    prediction = seal_record(prediction)
    result = verify_prediction_record(prediction, specification=specification)
    assert not result["verified"]
    assert "wrapper mismatch" in " ".join(result["errors"])


def test_specification_drift_fails(specification: dict) -> None:
    drifted = copy.deepcopy(specification)
    drifted["simulation"]["n_simulations"] = 10_000
    drifted = seal_specification(drifted)
    with pytest.raises(Phase7PError, match="specification drift"):
        _prospective(drifted)


def test_outcome_cannot_exist_during_prediction(specification: dict) -> None:
    pair = _pair(specification)
    pair["outcome"] = {"home_goals": 1, "away_goals": 0}
    with pytest.raises(Phase7PError, match="outcome cannot be present"):
        _prospective(specification, pair=pair)


def test_prediction_after_cutoff_fails(specification: dict) -> None:
    with pytest.raises(Phase7PError, match="prediction cutoff"):
        build_prediction_record(
            specification=specification,
            paired_evidence=_pair(specification),
            fixture_kickoff_at="2026-09-01T00:00:00+00:00",
            prediction_created_at="2026-09-01T00:00:00+00:00",
            maximum_source_event_at="2026-08-30T00:00:00+00:00",
            causal_source_identity={"artifacts_aggregate_sha256": "source"},
        )


def test_source_cutoff_violation_fails(specification: dict) -> None:
    with pytest.raises(Phase7PError, match="causal cutoff"):
        build_prediction_record(
            specification=specification,
            paired_evidence=_pair(specification),
            fixture_kickoff_at="2026-09-01T00:00:00+00:00",
            prediction_created_at="2026-08-31T12:00:00+00:00",
            maximum_source_event_at="2026-09-01T00:00:00+00:00",
            causal_source_identity={"artifacts_aggregate_sha256": "source"},
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda pair: pair["candidate_prediction"]["feature_provenance"].update(feature_bundle_sha256="other"), "feature-bundle"),
        (lambda pair: pair["candidate_prediction"]["method_provenance"]["descriptor"]["simulation"].update(rng_state_before={"state": 9}), "pre-call RNG"),
        (lambda pair: pair["active_prediction"]["model_provenance"].update(model_bundle_sha256="drift"), "active model drift"),
        (lambda pair: pair.update(candidate_training_lineage_verified=False), "training-lineage"),
        (lambda pair: pair.update(candidate_phase4_verified=False), "Phase 4"),
    ],
)
def test_paired_integrity_failures(specification: dict, mutation, message: str) -> None:
    pair = _pair(specification)
    mutation(pair)
    with pytest.raises(Phase7PError, match=message):
        _prospective(specification, pair=pair)


def test_duplicate_prediction_is_idempotent(specification: dict, tmp_path: Path) -> None:
    store = ImmutableCohortStore(tmp_path)
    prediction = _prospective(specification)
    assert store.persist_prediction(prediction, specification=specification) == "CREATED"
    assert store.persist_prediction(prediction, specification=specification) == "ALREADY_PRESENT"


def test_batch_lock_rejects_concurrent_batch(tmp_path: Path) -> None:
    store = ImmutableCohortStore(tmp_path)
    with store.batch_lock("capture"):
        with pytest.raises(Phase7PError, match="locked"):
            with store.batch_lock("capture"):
                pass


def test_conflicting_prediction_cannot_rewrite_frozen_file(specification: dict, tmp_path: Path) -> None:
    store = ImmutableCohortStore(tmp_path)
    prediction = _prospective(specification)
    store.persist_prediction(prediction, specification=specification)
    before = store.records("prediction")[0]
    changed_pair = _pair(specification)
    changed_pair["active_prediction"]["prediction_output"]["final_probabilities"] = [0.4, 0.3, 0.3]
    changed = _prospective(specification, pair=changed_pair)
    with pytest.raises(Phase7PError, match="conflicting immutable evidence"):
        store.persist_prediction(changed, specification=specification)
    assert store.records("prediction")[0] == before


def test_in_memory_mutation_does_not_change_disk(specification: dict, tmp_path: Path) -> None:
    store = ImmutableCohortStore(tmp_path)
    prediction = _prospective(specification)
    store.persist_prediction(prediction, specification=specification)
    prediction["outcome"] = {"home_goals": 9}
    assert store.records("prediction")[0]["outcome"] is None


def test_settlement_hashing_is_deterministic(specification: dict) -> None:
    prediction = _prospective(specification)
    assert _settlement(specification, prediction) == _settlement(specification, prediction)


def test_settlement_requires_prediction_file(specification: dict, tmp_path: Path) -> None:
    prediction = _prospective(specification)
    settlement = _settlement(specification, prediction)
    with pytest.raises(Phase7PError, match="without a frozen prediction"):
        ImmutableCohortStore(tmp_path).persist_settlement(settlement, specification=specification)


def test_conflicting_settlement_fails(specification: dict, tmp_path: Path) -> None:
    store = ImmutableCohortStore(tmp_path)
    prediction = _prospective(specification)
    store.persist_prediction(prediction, specification=specification)
    store.persist_settlement(_settlement(specification, prediction), specification=specification)
    conflicting = _settlement(specification, prediction, home_goals=3)
    with pytest.raises(Phase7PError, match="conflicting immutable evidence"):
        store.persist_settlement(conflicting, specification=specification)


def test_settlement_must_postdate_kickoff(specification: dict) -> None:
    with pytest.raises(Phase7PError, match="does not postdate"):
        _settlement(
            specification,
            _prospective(specification),
            outcome_observed_at="2026-09-01T00:00:00+00:00",
        )


def test_settlement_class_semantics_tamper_fails(specification: dict) -> None:
    prediction = _prospective(specification)
    settlement = _settlement(specification, prediction)
    settlement["authoritative_outcome"]["class_name"] = "away_win"
    settlement = seal_record(settlement)
    result = verify_settlement_record(
        settlement, prediction_record=prediction, specification=specification
    )
    assert not result["verified"]
    assert "class semantics" in " ".join(result["errors"])


def test_missed_fixture_cannot_later_be_admitted(specification: dict, tmp_path: Path) -> None:
    store = ImmutableCohortStore(tmp_path)
    missed = build_missed_record(
        specification=specification,
        fixture_id="fixture-1",
        fixture_kickoff_at="2026-09-01T00:00:00+00:00",
        detected_at="2026-08-31T23:59:00+00:00",
        reason="causal feature row unavailable",
        affected_components=["common_data"],
        known_before_result=True,
    )
    store.persist_missed(missed)
    with pytest.raises(Phase7PError, match="missed fixture cannot"):
        store.persist_prediction(_prospective(specification), specification=specification)


def test_frozen_prediction_cannot_be_marked_missed(specification: dict, tmp_path: Path) -> None:
    store = ImmutableCohortStore(tmp_path)
    store.persist_prediction(_prospective(specification), specification=specification)
    missed = build_missed_record(
        specification=specification,
        fixture_id="fixture-1",
        fixture_kickoff_at="2026-09-01T00:00:00+00:00",
        detected_at="2026-08-31T23:59:00+00:00",
        reason="late scheduler",
        affected_components=["active", "candidate"],
        known_before_result=True,
    )
    with pytest.raises(Phase7PError, match="cannot be reclassified"):
        store.persist_missed(missed)


def test_historical_record_cannot_enter_prospective_aggregation(specification: dict) -> None:
    prediction = _prospective(specification)
    prediction["cohort"] = "historical_diagnostic_only"
    prediction = seal_record(prediction)
    with pytest.raises(Phase7PError, match="historical evidence"):
        aggregate_eligible([prediction], [], specification=specification)


def test_unsettled_fixture_cannot_enter_metrics(specification: dict) -> None:
    result = aggregate_eligible([_prospective(specification)], [], specification=specification)
    assert result["fixture_count"] == 0
    assert result["metrics"] is None


def test_excluded_fixture_cannot_enter_metrics(specification: dict) -> None:
    prediction = _prospective(specification)
    prediction["eligibility_state"] = "EXCLUDED"
    prediction = seal_record(prediction)
    with pytest.raises(Phase7PError, match="excluded"):
        aggregate_eligible([prediction], [_settlement(specification, _prospective(specification))], specification=specification)


def test_settled_fixture_enters_descriptive_metrics_only(specification: dict) -> None:
    prediction = _prospective(specification)
    result = aggregate_eligible(
        [prediction], [_settlement(specification, prediction)], specification=specification
    )
    assert result["fixture_count"] == 1
    assert result["metrics"] is not None
    assert result["decision"]["classification"] == "CANDIDATE_INCONCLUSIVE"
    assert result["decision"]["final_decision_rule_evaluated"] is False
    assert result["decision"]["phase8_promotion_review_authorized"] is False


def test_candidate_exception_preserves_active_output() -> None:
    result = evaluation_only_shadow_call(
        lambda: {"home": 0.5},
        lambda: (_ for _ in ()).throw(RuntimeError("challenger failed")),
        active_state_token=lambda: "unchanged",
    )
    assert result["production_output"] == {"home": 0.5}
    assert result["authoritative_output"] == "active"


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"discovered": False, "feature_ready": False, "outcome_available": False}, "NOT_YET_ELIGIBLE"),
        ({"discovered": True, "feature_ready": True, "outcome_available": False}, "READY_FOR_PREDICTION"),
        ({"discovered": True, "feature_ready": True, "outcome_available": False, "prediction": {"x": 1}}, "AWAITING_OUTCOME"),
        ({"discovered": True, "feature_ready": True, "outcome_available": True, "prediction": {"x": 1}}, "READY_FOR_SETTLEMENT"),
        ({"discovered": True, "feature_ready": True, "outcome_available": True, "prediction": {"x": 1}, "settlement": {"x": 2}}, "SETTLED"),
        ({"discovered": True, "feature_ready": True, "outcome_available": False, "missed": {"x": 1}}, "MISSED"),
    ],
)
def test_workflow_states(kwargs: dict, expected: str) -> None:
    assert classify_fixture_state(
        now="2026-08-31T12:00:00+00:00",
        kickoff_at="2026-09-01T00:00:00+00:00",
        **kwargs,
    ) == expected


def test_coverage_ratio() -> None:
    assert coverage_ratio(8, 2) == 0.8
    assert coverage_ratio(0, 0) is None


def test_protected_phase0_to_phase7_snapshot_is_unchanged() -> None:
    from scripts.run_phase7p_prospective import DEFAULT_RUN_DIR, protected_equal, protected_snapshot

    before = json.loads((DEFAULT_RUN_DIR / "protected_before.json").read_text(encoding="utf-8"))
    assert protected_equal(before, protected_snapshot())
