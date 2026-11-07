"""Frozen paired active-vs-candidate evaluation contracts for FFPS Phase 7.

This module contains evaluation-only primitives.  It never loads a registry,
trains a model, mutates an artifact, or changes a production response.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, MutableMapping, Sequence, TypeVar, cast

import numpy as np


SPEC_SCHEMA_VERSION = "ffps.phase7_evaluation_specification.v1"
PAIR_SCHEMA_VERSION = "ffps.phase7_paired_fixture.v1"
CLASS_ORDER = ("home_win", "draw", "away_win")
AUTHORIZED_DECISIONS = frozenset(
    {
        "CANDIDATE_SUPPORTED_FOR_PROMOTION_REVIEW",
        "CANDIDATE_INCONCLUSIVE",
        "ACTIVE_SUPPORTED",
    }
)
PRIMARY_METRICS = ("log_loss", "multiclass_brier")


class Phase7EvaluationError(RuntimeError):
    """Raised when the frozen Phase 7 protocol cannot be proven."""


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json(value).encode("utf-8"))


def _spec_payload(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        str(key): copy.deepcopy(item)
        for key, item in value.items()
        if key not in {"specification_fingerprint", "record_integrity_sha256"}
    }


def specification_fingerprint(value: Mapping[str, Any]) -> str:
    return sha256_json(_spec_payload(value))


def seal_specification(value: Mapping[str, Any]) -> dict[str, Any]:
    result = _spec_payload(value)
    result["specification_fingerprint"] = specification_fingerprint(result)
    result["record_integrity_sha256"] = sha256_json(
        {key: item for key, item in result.items() if key != "record_integrity_sha256"}
    )
    return result


def verify_specification(value: Mapping[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    if value.get("schema_version") != SPEC_SCHEMA_VERSION:
        errors.append("unsupported specification schema")
    if tuple(value.get("class_order", ())) != CLASS_ORDER:
        errors.append("class order changed")
    if value.get("market") != "PL_1X2":
        errors.append("market changed")
    if value.get("candidate", {}).get("evaluation_only") is not True:
        errors.append("candidate is not evaluation-only")
    if value.get("prohibited_actions") != [
        "training",
        "hyperparameter_tuning",
        "recalibration",
        "ensemble_weight_tuning",
        "registry_write",
        "production_activation",
        "bookmaker_input",
    ]:
        errors.append("prohibited-action contract changed")
    simulation = cast(Mapping[str, Any], value.get("simulation", {}))
    if simulation.get("common_random_numbers") is not True:
        errors.append("common-random-number mode is not required")
    if simulation.get("state_advance_after_pair") != "active_post_call_state":
        errors.append("paired RNG stream advancement changed")
    bootstrap = cast(Mapping[str, Any], value.get("paired_statistics", {}))
    if bootstrap.get("resamples") != 10_000 or bootstrap.get("confidence_level") != 0.95:
        errors.append("paired-bootstrap contract changed")
    if tuple(value.get("authorized_decisions", ())) != tuple(sorted(AUTHORIZED_DECISIONS)):
        errors.append("authorized decision set changed")
    expected_fingerprint = specification_fingerprint(value)
    if value.get("specification_fingerprint") != expected_fingerprint:
        errors.append("specification fingerprint mismatch")
    expected_integrity = sha256_json(
        {key: item for key, item in value.items() if key != "record_integrity_sha256"}
    )
    if value.get("record_integrity_sha256") != expected_integrity:
        errors.append("specification integrity mismatch")
    return {"verified": not errors, "errors": errors, "fingerprint": expected_fingerprint}


def assert_specification_unchanged(
    frozen: Mapping[str, Any], proposed: Mapping[str, Any], *, evaluation_started: bool
) -> None:
    if not evaluation_started:
        return
    if specification_fingerprint(frozen) != specification_fingerprint(proposed):
        raise Phase7EvaluationError("evaluation specification cannot mutate after evaluation begins")


def validate_probabilities(values: Sequence[float]) -> None:
    array = np.asarray(values, dtype=float)
    if array.shape != (3,) or not np.isfinite(array).all():
        raise Phase7EvaluationError("invalid probability vector")
    if (array < 0.0).any() or (array > 1.0).any():
        raise Phase7EvaluationError("probabilities fall outside [0,1]")
    if not math.isclose(float(array.sum()), 1.0, abs_tol=1e-12, rel_tol=0.0):
        raise Phase7EvaluationError("probabilities do not normalize")


def _record_feature_identity(record: Mapping[str, Any]) -> tuple[str, str]:
    feature = cast(Mapping[str, Any], record["feature_provenance"])
    data = cast(Mapping[str, Any], record["data_provenance"])
    source = cast(Mapping[str, Any], data["source_identity"])
    return str(feature["feature_bundle_sha256"]), str(source["artifacts_aggregate_sha256"])


def _record_rng(record: Mapping[str, Any]) -> Mapping[str, Any]:
    return cast(
        Mapping[str, Any],
        record["method_provenance"]["descriptor"]["simulation"],
    )


def validate_paired_prediction(
    pair: Mapping[str, Any],
    *,
    specification: Mapping[str, Any],
    seen_fixture_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Fail closed unless one paired prediction satisfies the frozen contract."""
    verification = verify_specification(specification)
    if not verification["verified"]:
        raise Phase7EvaluationError(f"invalid frozen specification: {verification['errors']}")
    if pair.get("schema_version") != PAIR_SCHEMA_VERSION:
        raise Phase7EvaluationError("unsupported paired-evidence schema")
    fixture_id = str(pair.get("fixture_id", ""))
    if not fixture_id:
        raise Phase7EvaluationError("fixture identity is missing")
    if seen_fixture_ids is not None and fixture_id in seen_fixture_ids:
        raise Phase7EvaluationError("duplicate fixture evaluation")
    if pair.get("specification_fingerprint") != verification["fingerprint"]:
        raise Phase7EvaluationError("paired evidence uses a different evaluation specification")
    if tuple(pair.get("class_order", ())) != CLASS_ORDER:
        raise Phase7EvaluationError("paired evidence class order changed")
    if pair.get("authoritative_output") != "active":
        raise Phase7EvaluationError("candidate output cannot replace active output")

    active = cast(Mapping[str, Any], pair["active_prediction"])
    candidate = cast(Mapping[str, Any], pair["candidate_prediction"])
    active_fixture = str(active["fixture"]["match_hash"])
    candidate_fixture = str(candidate["fixture"]["match_hash"])
    if active_fixture != fixture_id or candidate_fixture != fixture_id:
        raise Phase7EvaluationError("paired fixture identities differ")

    active_feature, active_source = _record_feature_identity(active)
    candidate_feature, candidate_source = _record_feature_identity(candidate)
    if active_feature != candidate_feature:
        raise Phase7EvaluationError("feature-bundle identity differs between paired executions")
    if active_source != candidate_source:
        raise Phase7EvaluationError("source snapshot differs between paired executions")
    if pair.get("feature_bundle_sha256") != active_feature:
        raise Phase7EvaluationError("paired feature identity sentinel mismatch")
    if pair.get("source_snapshot_sha256") != active_source:
        raise Phase7EvaluationError("paired source identity sentinel mismatch")

    expected_active = str(specification["active"]["model_bundle_sha256"])
    expected_candidate = str(specification["candidate"]["model_bundle_sha256"])
    if active["model_provenance"]["model_bundle_sha256"] != expected_active:
        raise Phase7EvaluationError("active model drift")
    if candidate["model_provenance"]["model_bundle_sha256"] != expected_candidate:
        raise Phase7EvaluationError("candidate model drift")
    if pair.get("candidate_training_lineage_verified") is not True:
        raise Phase7EvaluationError("candidate training-lineage mismatch")
    if pair.get("active_phase4_verified") is not True or pair.get("candidate_phase4_verified") is not True:
        raise Phase7EvaluationError("missing Phase 4 prediction verification")

    active_rng = _record_rng(active)
    candidate_rng = _record_rng(candidate)
    required_simulation = cast(Mapping[str, Any], specification["simulation"])
    fields = ("implementation", "n_simulations", "max_goals", "top_scorelines", "tempo_sigma", "league")
    for field_name in fields:
        if active_rng.get(field_name) != candidate_rng.get(field_name):
            raise Phase7EvaluationError(f"simulator parameter differs: {field_name}")
    if active_rng.get("n_simulations") != required_simulation.get("n_simulations"):
        raise Phase7EvaluationError("simulation count differs from specification")
    if canonical_json(active_rng.get("rng_state_before")) != canonical_json(candidate_rng.get("rng_state_before")):
        raise Phase7EvaluationError("paired executions did not use the same pre-call RNG state")
    if canonical_json(pair.get("shared_pre_call_rng_state")) != canonical_json(active_rng.get("rng_state_before")):
        raise Phase7EvaluationError("shared RNG state sentinel mismatch")

    for label, record in (("active", active), ("candidate", candidate)):
        output = cast(Mapping[str, Any], record["prediction_output"])
        if tuple(output.get("class_order", ())) != CLASS_ORDER:
            raise Phase7EvaluationError(f"{label} class order changed")
        validate_probabilities(cast(Sequence[float], output["final_probabilities"]))
        descriptor = cast(Mapping[str, Any], record["method_provenance"]["descriptor"])
        if descriptor.get("calibrator", {}).get("applied"):
            raise Phase7EvaluationError(f"{label} calibration insertion")
        if descriptor.get("match_outcome_classifier", {}).get("used"):
            raise Phase7EvaluationError(f"{label} match-outcome classifier insertion")
        independence = cast(Mapping[str, Any], record.get("bookmaker_independence", {}))
        if not independence or any(bool(value) for value in independence.values()):
            raise Phase7EvaluationError(f"{label} bookmaker prediction input")
        if record.get("trust_assertions", {}).get("temporal_cutoff_verified") is not True:
            raise Phase7EvaluationError(f"{label} causal-cutoff violation")

    if pair.get("outcome") is not None:
        prediction_frozen_at = str(pair.get("prediction_frozen_at", ""))
        outcome_accessed_at = str(pair.get("outcome_accessed_at", ""))
        if not prediction_frozen_at or not outcome_accessed_at or outcome_accessed_at <= prediction_frozen_at:
            raise Phase7EvaluationError("outcome-before-prediction leakage")
    if seen_fixture_ids is not None:
        seen_fixture_ids.add(fixture_id)
    return {
        "verified": True,
        "fixture_id": fixture_id,
        "feature_bundle_sha256": active_feature,
        "source_snapshot_sha256": active_source,
        "common_random_numbers_verified": True,
    }


def goal_metric_contributions(actual: np.ndarray, prediction: np.ndarray) -> dict[str, np.ndarray]:
    actual = np.asarray(actual, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    if actual.shape != prediction.shape or actual.ndim != 1:
        raise Phase7EvaluationError("goal arrays have incompatible shapes")
    if not np.isfinite(actual).all() or not np.isfinite(prediction).all() or (actual < 0).any():
        raise Phase7EvaluationError("goal arrays contain invalid values")
    prediction = np.clip(prediction, 1e-12, None)
    residual = prediction - actual
    poisson = 2.0 * np.where(
        actual > 0.0,
        actual * np.log(np.where(actual > 0.0, actual / prediction, 1.0)) - (actual - prediction),
        prediction,
    )
    return {
        "absolute_error": np.abs(residual),
        "squared_error": np.square(residual),
        "poisson_deviance": poisson,
        "bias": residual,
        "residual": residual,
    }


def summarize_goal_metrics(actual: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    values = goal_metric_contributions(actual, prediction)
    return {
        "mae": float(values["absolute_error"].mean()),
        "rmse": float(np.sqrt(values["squared_error"].mean())),
        "mean_poisson_deviance": float(values["poisson_deviance"].mean()),
        "mean_prediction_bias": float(values["bias"].mean()),
        "residual_dispersion": float(values["residual"].std(ddof=0)),
    }


def outcome_indices(home_goals: np.ndarray, away_goals: np.ndarray) -> np.ndarray:
    home_goals = np.asarray(home_goals, dtype=float)
    away_goals = np.asarray(away_goals, dtype=float)
    return np.where(home_goals > away_goals, 0, np.where(home_goals == away_goals, 1, 2)).astype(int)


def probability_metric_contributions(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, np.ndarray]:
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.shape != (len(labels), 3):
        raise Phase7EvaluationError("probability matrix must be N x 3")
    for row in probabilities:
        validate_probabilities(row)
    clipped = np.clip(probabilities, 1e-15, 1.0)
    one_hot = np.eye(3, dtype=float)[labels]
    return {
        "log_loss": -np.log(clipped[np.arange(len(labels)), labels]),
        "multiclass_brier": np.sum(np.square(probabilities - one_hot), axis=1),
        "accuracy": (np.argmax(probabilities, axis=1) == labels).astype(float),
        "entropy": -np.sum(np.where(probabilities > 0, probabilities * np.log(probabilities), 0.0), axis=1),
        "maximum_probability": np.max(probabilities, axis=1),
    }


def reliability_metrics(labels: np.ndarray, probabilities: np.ndarray, *, bins: int = 10) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)

    def ece(values: np.ndarray, outcomes: np.ndarray) -> float:
        total = 0.0
        for index in range(bins):
            lower, upper = edges[index], edges[index + 1]
            mask = (values >= lower) & (values < upper if index < bins - 1 else values <= upper)
            if mask.any():
                total += float(mask.mean()) * abs(float(values[mask].mean()) - float(outcomes[mask].mean()))
        return total

    predicted = np.argmax(probabilities, axis=1)
    confidence = np.max(probabilities, axis=1)
    top = ece(confidence, (predicted == labels).astype(float))
    class_values = {
        CLASS_ORDER[index]: ece(probabilities[:, index], (labels == index).astype(float))
        for index in range(3)
    }
    return {
        "top_label_ece": top,
        "classwise_ece": float(np.mean(list(class_values.values()))),
        "per_class_ece": class_values,
        "bins": bins,
    }


def summarize_probability_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    contributions = probability_metric_contributions(labels, probabilities)
    return {
        "log_loss": float(contributions["log_loss"].mean()),
        "multiclass_brier": float(contributions["multiclass_brier"].mean()),
        "accuracy": float(contributions["accuracy"].mean()),
        "average_entropy": float(contributions["entropy"].mean()),
        "mean_maximum_probability": float(contributions["maximum_probability"].mean()),
        **reliability_metrics(labels, probabilities),
    }


def paired_bootstrap(
    active: Mapping[str, Sequence[float]],
    candidate: Mapping[str, Sequence[float]],
    *,
    resamples: int,
    seed: int,
    confidence_level: float,
    root_mean_metrics: Sequence[str] = (),
) -> dict[str, Any]:
    if set(active) != set(candidate) or not active:
        raise Phase7EvaluationError("paired bootstrap metric sets differ")
    arrays: dict[str, np.ndarray] = {}
    size: int | None = None
    for metric in sorted(active):
        left = np.asarray(active[metric], dtype=float)
        right = np.asarray(candidate[metric], dtype=float)
        if left.shape != right.shape or left.ndim != 1:
            raise Phase7EvaluationError(f"paired bootstrap shape mismatch: {metric}")
        if size is None:
            size = len(left)
        if len(left) != size:
            raise Phase7EvaluationError("paired bootstrap fixture counts differ")
        arrays[metric] = right - left
    if not size:
        raise Phase7EvaluationError("paired bootstrap requires at least one fixture")
    rng = np.random.default_rng(seed)
    root_metrics = set(root_mean_metrics)
    unknown_roots = root_metrics - set(arrays)
    if unknown_roots:
        raise Phase7EvaluationError(f"unknown root-mean bootstrap metrics: {sorted(unknown_roots)}")
    distributions = {metric: np.empty(resamples, dtype=float) for metric in arrays}
    for index in range(resamples):
        sampled = rng.integers(0, size, size=size)
        for metric in arrays:
            active_sample = np.asarray(active[metric], dtype=float)[sampled]
            candidate_sample = np.asarray(candidate[metric], dtype=float)[sampled]
            if metric in root_metrics:
                distributions[metric][index] = float(
                    np.sqrt(candidate_sample.mean()) - np.sqrt(active_sample.mean())
                )
            else:
                distributions[metric][index] = float((candidate_sample - active_sample).mean())
    alpha = (1.0 - confidence_level) / 2.0
    return {
        "method": "fixture-level paired nonparametric bootstrap percentile interval",
        "delta_definition": "candidate_minus_active",
        "same_fixture_indices_across_bundles_and_metrics": True,
        "resamples": resamples,
        "seed": seed,
        "confidence_level": confidence_level,
        "metrics": {
            metric: {
                "point_delta": float(
                    np.sqrt(np.asarray(candidate[metric], dtype=float).mean())
                    - np.sqrt(np.asarray(active[metric], dtype=float).mean())
                    if metric in root_metrics
                    else delta.mean()
                ),
                "lower": float(np.quantile(distributions[metric], alpha)),
                "upper": float(np.quantile(distributions[metric], 1.0 - alpha)),
            }
            for metric, delta in arrays.items()
        },
    }


def paired_block_bootstrap(
    active: Mapping[str, Sequence[float]],
    candidate: Mapping[str, Sequence[float]],
    blocks: Sequence[str],
    *,
    resamples: int,
    seed: int,
    confidence_level: float,
    root_mean_metrics: Sequence[str] = (),
) -> dict[str, Any]:
    """Calendar/time-block sensitivity analysis using whole-block resampling."""
    if set(active) != set(candidate) or not active:
        raise Phase7EvaluationError("paired block-bootstrap metric sets differ")
    size = len(next(iter(active.values())))
    if len(blocks) != size:
        raise Phase7EvaluationError("block labels do not match fixture count")
    unique = sorted(set(str(value) for value in blocks))
    if len(unique) < 2:
        raise Phase7EvaluationError("block bootstrap requires at least two blocks")
    positions = {block: np.flatnonzero(np.asarray(blocks, dtype=str) == block) for block in unique}
    roots = set(root_mean_metrics)
    rng = np.random.default_rng(seed)
    distributions = {metric: np.empty(resamples, dtype=float) for metric in active}
    for index in range(resamples):
        sampled_blocks = rng.choice(unique, size=len(unique), replace=True)
        sampled = np.concatenate([positions[str(block)] for block in sampled_blocks])
        for metric in active:
            left = np.asarray(active[metric], dtype=float)[sampled]
            right = np.asarray(candidate[metric], dtype=float)[sampled]
            distributions[metric][index] = float(
                np.sqrt(right.mean()) - np.sqrt(left.mean())
                if metric in roots
                else (right - left).mean()
            )
    alpha = (1.0 - confidence_level) / 2.0
    return {
        "method": "paired whole-calendar-month block bootstrap percentile interval",
        "delta_definition": "candidate_minus_active",
        "blocks": unique,
        "block_count": len(unique),
        "resamples": resamples,
        "seed": seed,
        "confidence_level": confidence_level,
        "metrics": {
            metric: {
                "lower": float(np.quantile(values, alpha)),
                "upper": float(np.quantile(values, 1.0 - alpha)),
            }
            for metric, values in distributions.items()
        },
    }


def apply_decision_rule(
    *,
    prospective_fixture_count: int,
    horizon_complete: bool,
    paired_intervals: Mapping[str, Mapping[str, float]] | None,
    active_reliability: Mapping[str, float] | None,
    candidate_reliability: Mapping[str, float] | None,
    provenance_verified: bool,
) -> dict[str, Any]:
    """Apply the frozen zero-margin rule without human override."""
    reasons: list[str] = []
    if not horizon_complete:
        reasons.append("fixed prospective horizon is not complete")
    if prospective_fixture_count == 0:
        reasons.append("no post-freeze eligible outcomes are available")
    if not provenance_verified:
        reasons.append("execution or lineage evidence is incomplete")
    if reasons:
        return {
            "classification": "CANDIDATE_INCONCLUSIVE",
            "phase8_promotion_review_authorized": False,
            "reasons": reasons,
            "human_override_applied": False,
        }
    if paired_intervals is None or active_reliability is None or candidate_reliability is None:
        raise Phase7EvaluationError("completed prospective horizon lacks decision metrics")

    required = (*PRIMARY_METRICS, "joint_goal_rmse", "joint_goal_poisson_deviance")
    missing = [metric for metric in required if metric not in paired_intervals]
    if missing:
        raise Phase7EvaluationError(f"decision metrics missing: {missing}")
    non_worse = all(
        float(paired_intervals[metric]["point_delta"]) <= 0.0
        and float(paired_intervals[metric]["upper"]) <= 0.0
        for metric in required
    )
    reliability_non_worse = all(
        float(candidate_reliability[name]) <= float(active_reliability[name])
        for name in ("top_label_ece", "classwise_ece")
    )
    if non_worse and reliability_non_worse:
        return {
            "classification": "CANDIDATE_SUPPORTED_FOR_PROMOTION_REVIEW",
            "phase8_promotion_review_authorized": True,
            "reasons": ["zero-margin support rule passed for both primary losses and goal-layer safeguards"],
            "human_override_applied": False,
        }
    credible_primary_harm = all(float(paired_intervals[metric]["lower"]) > 0.0 for metric in PRIMARY_METRICS)
    if credible_primary_harm:
        return {
            "classification": "ACTIVE_SUPPORTED",
            "phase8_promotion_review_authorized": False,
            "reasons": ["both primary proper-scoring-rule intervals show candidate degradation"],
            "human_override_applied": False,
        }
    return {
        "classification": "CANDIDATE_INCONCLUSIVE",
        "phase8_promotion_review_authorized": False,
        "reasons": ["frozen support and active-support rules were both unmet"],
        "human_override_applied": False,
    }


T = TypeVar("T")


@dataclass
class EvaluationLedger:
    """In-memory fail-closed ledger used by offline and operational adapters."""

    specification: dict[str, Any]
    seen_fixture_ids: set[str] = field(default_factory=set)
    evaluation_started: bool = False

    def replace_specification(self, proposed: Mapping[str, Any]) -> None:
        assert_specification_unchanged(
            self.specification,
            proposed,
            evaluation_started=self.evaluation_started,
        )
        self.specification = copy.deepcopy(dict(proposed))

    def add_pair(self, pair: Mapping[str, Any]) -> dict[str, Any]:
        result = validate_paired_prediction(
            pair,
            specification=self.specification,
            seen_fixture_ids=self.seen_fixture_ids,
        )
        self.evaluation_started = True
        return result


def evaluation_only_shadow_call(
    active_call: Callable[[], T],
    candidate_call: Callable[[], Any],
    *,
    active_state_token: Callable[[], str],
) -> dict[str, Any]:
    """Return active output authoritatively and isolate candidate failure.

    The state token lets callers prove that candidate execution did not mutate
    active model state.  This helper is not wired into production serving.
    """
    active_output = active_call()
    state_after_active = active_state_token()
    try:
        candidate_output = candidate_call()
        candidate_error = None
    except Exception as exc:  # evaluation boundary deliberately catches challenger failures
        candidate_output = None
        candidate_error = f"{type(exc).__name__}: {exc}"
    if active_state_token() != state_after_active:
        raise Phase7EvaluationError("candidate execution mutated active model state")
    return {
        "authoritative_output": "active",
        "production_output": active_output,
        "candidate_output": candidate_output,
        "candidate_error": candidate_error,
    }


def append_unique_fixture(
    store: MutableMapping[str, Mapping[str, Any]], fixture_id: str, value: Mapping[str, Any]
) -> None:
    if fixture_id in store:
        raise Phase7EvaluationError("duplicate fixture evaluation")
    store[fixture_id] = copy.deepcopy(dict(value))


def assert_single_labeled_cohort(records: Sequence[Mapping[str, Any]], expected: str) -> None:
    labels = {str(record.get("cohort", "")) for record in records}
    if labels != {expected}:
        raise Phase7EvaluationError(
            f"historical and prospective results cannot be merged or left unlabeled: {sorted(labels)}"
        )
