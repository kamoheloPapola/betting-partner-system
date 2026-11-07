"""Immutable prospective evidence primitives for the frozen Phase 7 protocol.

This module is deliberately storage- and model-agnostic.  The Phase 7P sidecar
uses it to freeze paired predictions before outcomes exist and to attach results
later through separate immutable settlement records.
"""

from __future__ import annotations

import copy
import json
import math
import os
import re
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence, cast

import numpy as np

from src.evaluation.phase7_shadow import (
    CLASS_ORDER,
    Phase7EvaluationError,
    goal_metric_contributions,
    outcome_indices,
    paired_block_bootstrap,
    paired_bootstrap,
    probability_metric_contributions,
    sha256_bytes,
    sha256_json,
    summarize_goal_metrics,
    summarize_probability_metrics,
    validate_paired_prediction,
    verify_specification,
)


PREDICTION_SCHEMA_VERSION = "ffps.phase7p_prospective_prediction.v1"
SETTLEMENT_SCHEMA_VERSION = "ffps.phase7p_prospective_settlement.v1"
MISSED_SCHEMA_VERSION = "ffps.phase7p_prospective_missed_fixture.v1"
STATUS_SCHEMA_VERSION = "ffps.phase7p_prospective_status.v1"
COHORT = "prospective_primary"
UNSETTLED = "UNSETTLED"
ELIGIBLE = "PROSPECTIVE_ELIGIBLE"
FROZEN_SPECIFICATION_FINGERPRINT = (
    "272fc86be883dbf43d41b71485236f44cad8975925ef6a97d0877cff442c7775"
)
WORKFLOW_STATES = (
    "NOT_YET_ELIGIBLE",
    "READY_FOR_PREDICTION",
    "PREDICTION_FROZEN",
    "AWAITING_OUTCOME",
    "READY_FOR_SETTLEMENT",
    "SETTLED",
    "EXCLUDED",
    "MISSED",
    "INTEGRITY_FAILURE",
)


class Phase7PError(Phase7EvaluationError):
    """Raised when prospective evidence cannot be trusted."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _timestamp(value: Any, label: str) -> datetime:
    text = str(value).strip()
    if not text:
        raise Phase7PError(f"{label} is missing")
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise Phase7PError(f"{label} is not an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _record_payload(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        str(key): copy.deepcopy(item)
        for key, item in value.items()
        if key not in {"record_fingerprint", "record_integrity_sha256"}
    }


def record_fingerprint(value: Mapping[str, Any]) -> str:
    return sha256_json(_record_payload(value))


def seal_record(value: Mapping[str, Any]) -> dict[str, Any]:
    sealed = _record_payload(value)
    sealed["record_fingerprint"] = record_fingerprint(sealed)
    sealed["record_integrity_sha256"] = sha256_json(
        {key: item for key, item in sealed.items() if key != "record_integrity_sha256"}
    )
    return sealed


def _verify_seal(value: Mapping[str, Any], errors: list[str]) -> None:
    if value.get("record_fingerprint") != record_fingerprint(value):
        errors.append("record fingerprint mismatch")
    expected = sha256_json(
        {key: item for key, item in value.items() if key != "record_integrity_sha256"}
    )
    if value.get("record_integrity_sha256") != expected:
        errors.append("record integrity mismatch")


def _verified_specification(specification: Mapping[str, Any]) -> dict[str, Any]:
    result = verify_specification(specification)
    if not result["verified"]:
        raise Phase7PError(f"invalid frozen Phase 7 specification: {result['errors']}")
    if result["fingerprint"] != FROZEN_SPECIFICATION_FINGERPRINT:
        raise Phase7PError("Phase 7 specification drift")
    return result


def _fixture_date(value: Any) -> str:
    return _timestamp(value, "fixture kickoff").date().isoformat()


def build_prediction_record(
    *,
    specification: Mapping[str, Any],
    paired_evidence: Mapping[str, Any],
    fixture_kickoff_at: str,
    prediction_created_at: str,
    maximum_source_event_at: str,
    causal_source_identity: Mapping[str, Any],
    outcome_available_at_prediction: bool = False,
) -> dict[str, Any]:
    """Create and seal one pre-outcome prospective prediction record."""
    spec = _verified_specification(specification)
    pair = copy.deepcopy(dict(paired_evidence))
    fixture_id = str(pair.get("fixture_id", ""))
    if not fixture_id:
        raise Phase7PError("fixture identity is missing")
    if outcome_available_at_prediction or pair.get("outcome") is not None:
        raise Phase7PError("outcome cannot be present during prediction creation")
    if pair.get("outcome_accessed_at") is not None:
        raise Phase7PError("outcome access cannot precede prediction freeze")
    if pair.get("cohort") != COHORT:
        raise Phase7PError("prediction evidence is not labeled prospective_primary")

    kickoff = _timestamp(fixture_kickoff_at, "fixture kickoff")
    created = _timestamp(prediction_created_at, "prediction creation")
    source_max = _timestamp(maximum_source_event_at, "maximum source event")
    if created >= kickoff:
        raise Phase7PError("prediction cutoff violation")
    if source_max >= kickoff:
        raise Phase7PError("causal cutoff violation")
    fixture_date = kickoff.date().isoformat()
    eligibility = cast(Mapping[str, Any], specification["eligibility"])
    if not (
        str(eligibility["fixture_date_min_inclusive"])
        <= fixture_date
        <= str(eligibility["fixture_date_max_inclusive"])
    ):
        raise Phase7PError("fixture falls outside the frozen prospective window")
    if pair.get("fixture_date") != fixture_date:
        raise Phase7PError("paired fixture date differs from kickoff")
    pair["prediction_frozen_at"] = created.isoformat()
    try:
        validate_paired_prediction(pair, specification=specification)
    except Phase7EvaluationError as exc:
        raise Phase7PError(str(exc)) from exc

    value = {
        "schema_version": PREDICTION_SCHEMA_VERSION,
        "specification_fingerprint": spec["fingerprint"],
        "cohort": COHORT,
        "fixture": {
            "fixture_id": fixture_id,
            "league": "PL",
            "fixture_date": fixture_date,
            "kickoff_at": kickoff.isoformat(),
        },
        "prediction_created_at": created.isoformat(),
        "eligibility_state": "PREDICTION_VERIFIED",
        "temporal_integrity": {
            "maximum_source_event_at": source_max.isoformat(),
            "maximum_source_event_precedes_kickoff": True,
            "prediction_precedes_kickoff": True,
            "outcome_available_at_prediction": False,
        },
        "causal_source_identity": copy.deepcopy(dict(causal_source_identity)),
        "feature_identity": {
            "feature_bundle_sha256": pair["feature_bundle_sha256"],
            "source_snapshot_sha256": pair["source_snapshot_sha256"],
        },
        "active_prediction_provenance": copy.deepcopy(pair["active_prediction"]),
        "candidate_prediction_provenance": copy.deepcopy(pair["candidate_prediction"]),
        "shared_simulator_rng_evidence": {
            "pre_call": copy.deepcopy(pair["shared_pre_call_rng_state"]),
            "active_post_call": copy.deepcopy(pair.get("active_post_call_rng_state")),
            "candidate_post_call": copy.deepcopy(pair.get("candidate_post_call_rng_state")),
            "stream_advanced_with": pair.get("stream_advanced_with"),
        },
        "paired_evidence": pair,
        "outcome_state": UNSETTLED,
        "outcome": None,
    }
    return seal_record(value)


def verify_prediction_record(
    value: Mapping[str, Any], *, specification: Mapping[str, Any]
) -> dict[str, Any]:
    errors: list[str] = []
    try:
        spec = _verified_specification(specification)
    except Phase7PError as exc:
        return {"verified": False, "errors": [str(exc)]}
    if value.get("schema_version") != PREDICTION_SCHEMA_VERSION:
        errors.append("unsupported prediction schema")
    if value.get("specification_fingerprint") != spec["fingerprint"]:
        errors.append("Phase 7 specification drift")
    if value.get("cohort") != COHORT:
        errors.append("historical evidence cannot enter the prospective cohort")
    if value.get("eligibility_state") != "PREDICTION_VERIFIED":
        errors.append("prediction is excluded or not verified")
    if value.get("outcome_state") != UNSETTLED or value.get("outcome") is not None:
        errors.append("prediction record contains an outcome")
    _verify_seal(value, errors)
    try:
        fixture = cast(Mapping[str, Any], value["fixture"])
        if fixture.get("league") != "PL":
            errors.append("fixture league is not PL")
        kickoff = _timestamp(fixture["kickoff_at"], "fixture kickoff")
        created = _timestamp(value["prediction_created_at"], "prediction creation")
        source_max = _timestamp(
            cast(Mapping[str, Any], value["temporal_integrity"])["maximum_source_event_at"],
            "maximum source event",
        )
        if created >= kickoff:
            errors.append("prediction cutoff violation")
        if source_max >= kickoff:
            errors.append("causal cutoff violation")
        timing = cast(Mapping[str, Any], value["temporal_integrity"])
        if timing.get("outcome_available_at_prediction") is not False:
            errors.append("outcome was available at prediction")
        pair = cast(Mapping[str, Any], value["paired_evidence"])
        if pair.get("fixture_id") != fixture.get("fixture_id"):
            errors.append("paired fixture identity mismatch")
        if value.get("active_prediction_provenance") != pair.get("active_prediction"):
            errors.append("active provenance wrapper mismatch")
        if value.get("candidate_prediction_provenance") != pair.get("candidate_prediction"):
            errors.append("candidate provenance wrapper mismatch")
        identity = cast(Mapping[str, Any], value["feature_identity"])
        if identity.get("feature_bundle_sha256") != pair.get("feature_bundle_sha256"):
            errors.append("feature identity wrapper mismatch")
        if identity.get("source_snapshot_sha256") != pair.get("source_snapshot_sha256"):
            errors.append("source identity wrapper mismatch")
        validate_paired_prediction(pair, specification=specification)
    except (KeyError, TypeError, Phase7EvaluationError, Phase7PError) as exc:
        errors.append(str(exc))
    return {
        "verified": not errors,
        "errors": errors,
        "record_fingerprint": value.get("record_fingerprint"),
    }


def _goal(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) < 0:
        raise Phase7PError(f"{label} must be a non-negative integer")
    return int(value)


def build_settlement_record(
    *,
    specification: Mapping[str, Any],
    prediction_record: Mapping[str, Any],
    home_goals: int,
    away_goals: int,
    outcome_source_identity: Mapping[str, Any],
    outcome_observed_at: str,
    settlement_created_at: str,
) -> dict[str, Any]:
    prediction_verification = verify_prediction_record(
        prediction_record, specification=specification
    )
    if not prediction_verification["verified"]:
        raise Phase7PError(
            f"settlement requires a valid frozen prediction: {prediction_verification['errors']}"
        )
    fixture = cast(Mapping[str, Any], prediction_record["fixture"])
    observed = _timestamp(outcome_observed_at, "outcome observed")
    settled = _timestamp(settlement_created_at, "settlement creation")
    kickoff = _timestamp(fixture["kickoff_at"], "fixture kickoff")
    predicted = _timestamp(prediction_record["prediction_created_at"], "prediction creation")
    if observed <= kickoff or observed <= predicted:
        raise Phase7PError("outcome observation does not postdate prediction and kickoff")
    if settled < observed:
        raise Phase7PError("settlement precedes outcome observation")
    home = _goal(home_goals, "home goals")
    away = _goal(away_goals, "away goals")
    if not outcome_source_identity.get("provider") or not outcome_source_identity.get("sha256"):
        raise Phase7PError("authoritative outcome source identity is incomplete")
    label = int(outcome_indices(np.asarray([home]), np.asarray([away]))[0])
    value = {
        "schema_version": SETTLEMENT_SCHEMA_VERSION,
        "specification_fingerprint": prediction_record["specification_fingerprint"],
        "cohort": COHORT,
        "prediction_record_fingerprint": prediction_record["record_fingerprint"],
        "fixture_id": fixture["fixture_id"],
        "authoritative_outcome": {
            "home_goals": home,
            "away_goals": away,
            "class_index": label,
            "class_name": CLASS_ORDER[label],
            "class_order": list(CLASS_ORDER),
        },
        "outcome_source_identity": copy.deepcopy(dict(outcome_source_identity)),
        "outcome_observed_at": observed.isoformat(),
        "settlement_created_at": settled.isoformat(),
        "integrity_assertions": {
            "prediction_record_verified": True,
            "fixture_identity_matches": True,
            "outcome_postdates_prediction": True,
            "prediction_record_was_not_mutated": True,
        },
    }
    return seal_record(value)


def verify_settlement_record(
    value: Mapping[str, Any], *, prediction_record: Mapping[str, Any], specification: Mapping[str, Any]
) -> dict[str, Any]:
    errors: list[str] = []
    pred = verify_prediction_record(prediction_record, specification=specification)
    if not pred["verified"]:
        errors.append(f"invalid prediction record: {pred['errors']}")
    if value.get("schema_version") != SETTLEMENT_SCHEMA_VERSION:
        errors.append("unsupported settlement schema")
    if value.get("cohort") != COHORT:
        errors.append("settlement is not prospective")
    if value.get("prediction_record_fingerprint") != prediction_record.get("record_fingerprint"):
        errors.append("prediction fingerprint mismatch")
    source = cast(Mapping[str, Any], value.get("outcome_source_identity", {}))
    if not source.get("provider") or not source.get("sha256"):
        errors.append("authoritative outcome source identity is incomplete")
    fixture = cast(Mapping[str, Any], prediction_record.get("fixture", {}))
    if value.get("fixture_id") != fixture.get("fixture_id"):
        errors.append("settlement fixture identity mismatch")
    _verify_seal(value, errors)
    try:
        outcome = cast(Mapping[str, Any], value["authoritative_outcome"])
        home = _goal(outcome["home_goals"], "home goals")
        away = _goal(outcome["away_goals"], "away goals")
        expected = int(outcome_indices(np.asarray([home]), np.asarray([away]))[0])
        if tuple(outcome.get("class_order", ())) != CLASS_ORDER:
            errors.append("settlement class order changed")
        if outcome.get("class_index") != expected or outcome.get("class_name") != CLASS_ORDER[expected]:
            errors.append("settlement class semantics mismatch")
        observed = _timestamp(value["outcome_observed_at"], "outcome observed")
        settled = _timestamp(value["settlement_created_at"], "settlement creation")
        kickoff = _timestamp(fixture["kickoff_at"], "fixture kickoff")
        predicted = _timestamp(prediction_record["prediction_created_at"], "prediction creation")
        if observed <= kickoff or observed <= predicted or settled < observed:
            errors.append("settlement timing violation")
    except (KeyError, TypeError, Phase7PError) as exc:
        errors.append(str(exc))
    return {"verified": not errors, "errors": errors}


def build_missed_record(
    *, specification: Mapping[str, Any], fixture_id: str, fixture_kickoff_at: str,
    detected_at: str, reason: str, affected_components: Sequence[str], known_before_result: bool
) -> dict[str, Any]:
    spec = _verified_specification(specification)
    if not fixture_id or not reason:
        raise Phase7PError("missed fixture identity and reason are required")
    kickoff = _timestamp(fixture_kickoff_at, "fixture kickoff")
    detected = _timestamp(detected_at, "miss detection")
    return seal_record(
        {
            "schema_version": MISSED_SCHEMA_VERSION,
            "specification_fingerprint": spec["fingerprint"],
            "cohort": COHORT,
            "fixture_id": fixture_id,
            "fixture_date": kickoff.date().isoformat(),
            "fixture_kickoff_at": kickoff.isoformat(),
            "detected_at": detected.isoformat(),
            "reason": reason,
            "affected_components": sorted({str(value) for value in affected_components}),
            "known_before_result": bool(known_before_result),
            "state": "PROSPECTIVE_FIXTURE_MISSED",
        }
    )


def _filename(fixture_id: str) -> str:
    prefix = re.sub(r"[^A-Za-z0-9_.-]+", "-", fixture_id).strip("-.")[:48] or "fixture"
    return f"{prefix}-{sha256_bytes(fixture_id.encode('utf-8'))[:16]}.json"


class ImmutableCohortStore:
    """Per-fixture immutable files with lock-protected idempotent creation."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.predictions = self.root / "predictions"
        self.settlements = self.root / "settlements"
        self.missed = self.root / "missed"
        self.locks = self.root / ".locks"
        for path in (self.predictions, self.settlements, self.missed, self.locks):
            path.mkdir(parents=True, exist_ok=True)

    def path_for(self, kind: str, fixture_id: str) -> Path:
        directory = {"prediction": self.predictions, "settlement": self.settlements, "missed": self.missed}[kind]
        return directory / _filename(fixture_id)

    @contextmanager
    def _lock(self, fixture_id: str) -> Iterator[None]:
        lock = self.locks / f"{sha256_bytes(fixture_id.encode('utf-8'))}.lock"
        try:
            descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as exc:
            raise Phase7PError(f"fixture evidence is locked: {fixture_id}") from exc
        try:
            os.write(descriptor, f"pid={os.getpid()}\n".encode("ascii"))
            os.close(descriptor)
            yield
        finally:
            try:
                os.close(descriptor)
            except OSError:
                pass
            lock.unlink(missing_ok=True)

    @contextmanager
    def batch_lock(self, name: str) -> Iterator[None]:
        """Serialize operations whose RNG or source snapshot spans fixtures."""
        with self._lock(f"__phase7p_batch__:{name}"):
            yield

    @staticmethod
    def _read(path: Path) -> dict[str, Any]:
        return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))

    @staticmethod
    def _persist(path: Path, value: Mapping[str, Any]) -> str:
        payload = json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n"
        if path.exists():
            existing = ImmutableCohortStore._read(path)
            if existing.get("record_fingerprint") == value.get("record_fingerprint") and existing == value:
                return "ALREADY_PRESENT"
            raise Phase7PError(f"conflicting immutable evidence: {path.name}")
        temporary = path.with_suffix(f"{path.suffix}.{os.getpid()}.tmp")
        temporary.write_text(payload, encoding="utf-8", newline="\n")
        if json.loads(temporary.read_text(encoding="utf-8")) != value:
            temporary.unlink(missing_ok=True)
            raise Phase7PError("persisted evidence verification failed")
        os.replace(temporary, path)
        if ImmutableCohortStore._read(path) != value:
            raise Phase7PError("immutable evidence changed during persistence")
        return "CREATED"

    def persist_prediction(self, value: Mapping[str, Any], *, specification: Mapping[str, Any]) -> str:
        result = verify_prediction_record(value, specification=specification)
        if not result["verified"]:
            raise Phase7PError(f"prediction record did not verify: {result['errors']}")
        fixture_id = str(cast(Mapping[str, Any], value["fixture"])["fixture_id"])
        with self._lock(fixture_id):
            if self.path_for("missed", fixture_id).exists():
                raise Phase7PError("missed fixture cannot later be admitted as prospective")
            return self._persist(self.path_for("prediction", fixture_id), value)

    def persist_settlement(
        self, value: Mapping[str, Any], *, specification: Mapping[str, Any]
    ) -> str:
        fixture_id = str(value.get("fixture_id", ""))
        with self._lock(fixture_id):
            prediction_path = self.path_for("prediction", fixture_id)
            if not prediction_path.exists():
                raise Phase7PError("settlement cannot exist without a frozen prediction")
            prediction = self._read(prediction_path)
            result = verify_settlement_record(value, prediction_record=prediction, specification=specification)
            if not result["verified"]:
                raise Phase7PError(f"settlement record did not verify: {result['errors']}")
            return self._persist(self.path_for("settlement", fixture_id), value)

    def persist_missed(self, value: Mapping[str, Any]) -> str:
        fixture_id = str(value.get("fixture_id", ""))
        if value.get("schema_version") != MISSED_SCHEMA_VERSION or not fixture_id:
            raise Phase7PError("invalid missed-fixture record")
        errors: list[str] = []
        _verify_seal(value, errors)
        if errors:
            raise Phase7PError(f"missed-fixture record did not verify: {errors}")
        with self._lock(fixture_id):
            if self.path_for("prediction", fixture_id).exists():
                raise Phase7PError("frozen prediction cannot be reclassified as missed")
            return self._persist(self.path_for("missed", fixture_id), value)

    def records(self, kind: str) -> list[dict[str, Any]]:
        directory = {"prediction": self.predictions, "settlement": self.settlements, "missed": self.missed}[kind]
        return [self._read(path) for path in sorted(directory.glob("*.json"))]


def classify_fixture_state(
    *, now: str, kickoff_at: str, discovered: bool, feature_ready: bool,
    outcome_available: bool, prediction: Mapping[str, Any] | None = None,
    settlement: Mapping[str, Any] | None = None, missed: Mapping[str, Any] | None = None,
    excluded: bool = False, integrity_failure: bool = False,
) -> str:
    if integrity_failure:
        return "INTEGRITY_FAILURE"
    if excluded:
        return "EXCLUDED"
    if missed is not None:
        return "MISSED"
    if settlement is not None:
        return "SETTLED"
    if prediction is not None:
        return "READY_FOR_SETTLEMENT" if outcome_available else "AWAITING_OUTCOME"
    if not discovered or not feature_ready:
        return "NOT_YET_ELIGIBLE"
    if _timestamp(now, "current time") >= _timestamp(kickoff_at, "fixture kickoff"):
        return "MISSED"
    return "READY_FOR_PREDICTION"


def eligible_pairs(
    predictions: Sequence[Mapping[str, Any]], settlements: Sequence[Mapping[str, Any]],
    *, specification: Mapping[str, Any]
) -> list[dict[str, Any]]:
    settlement_by_fixture = {str(value.get("fixture_id")): value for value in settlements}
    if len(settlement_by_fixture) != len(settlements):
        raise Phase7PError("duplicate settlement fixture")
    eligible: list[dict[str, Any]] = []
    seen: set[str] = set()
    for prediction in predictions:
        pred = verify_prediction_record(prediction, specification=specification)
        if not pred["verified"]:
            raise Phase7PError(f"prediction integrity failure: {pred['errors']}")
        fixture = cast(Mapping[str, Any], prediction["fixture"])
        fixture_id = str(fixture["fixture_id"])
        if fixture_id in seen:
            raise Phase7PError("duplicate prospective fixture")
        seen.add(fixture_id)
        settlement = settlement_by_fixture.get(fixture_id)
        if settlement is None:
            continue
        verified = verify_settlement_record(
            settlement, prediction_record=prediction, specification=specification
        )
        if not verified["verified"]:
            raise Phase7PError(f"settlement integrity failure: {verified['errors']}")
        pair = copy.deepcopy(cast(Mapping[str, Any], prediction["paired_evidence"]))
        outcome = settlement["authoritative_outcome"]
        pair["outcome_accessed_at"] = settlement["outcome_observed_at"]
        pair["outcome"] = {
            "home_goals": outcome["home_goals"],
            "away_goals": outcome["away_goals"],
            "class_index": outcome["class_index"],
        }
        pair["cohort"] = COHORT
        validate_paired_prediction(pair, specification=specification)
        eligible.append(
            {
                "eligibility_state": ELIGIBLE,
                "prediction_record_fingerprint": prediction["record_fingerprint"],
                "settlement_record_fingerprint": settlement["record_fingerprint"],
                "pair": pair,
            }
        )
    return eligible


def aggregate_eligible(
    predictions: Sequence[Mapping[str, Any]], settlements: Sequence[Mapping[str, Any]],
    *, specification: Mapping[str, Any]
) -> dict[str, Any]:
    """Compute descriptive frozen Phase 7 metrics without running its final decision rule."""
    _verified_specification(specification)
    evidence = eligible_pairs(predictions, settlements, specification=specification)
    if not evidence:
        return {
            "schema_version": "ffps.phase7p_interim_aggregate.v1",
            "cohort": COHORT,
            "fixture_count": 0,
            "metrics": None,
            "paired_intervals": None,
            "time_block_sensitivity": None,
            "decision": {
                "classification": "CANDIDATE_INCONCLUSIVE",
                "phase8_promotion_review_authorized": False,
                "final_decision_rule_evaluated": False,
                "reason": "fixed prospective horizon is not complete",
            },
        }

    pairs = [item["pair"] for item in evidence]
    active_probs: list[np.ndarray] = []
    candidate_probs: list[np.ndarray] = []
    actual_home: list[float] = []
    actual_away: list[float] = []
    active_home: list[float] = []
    active_away: list[float] = []
    candidate_home: list[float] = []
    candidate_away: list[float] = []
    for pair in pairs:
        active = pair["active_prediction"]
        candidate = pair["candidate_prediction"]
        active_probs.append(np.asarray(active["prediction_output"]["final_probabilities"], dtype=float))
        candidate_probs.append(np.asarray(candidate["prediction_output"]["final_probabilities"], dtype=float))
        actual_home.append(float(pair["outcome"]["home_goals"]))
        actual_away.append(float(pair["outcome"]["away_goals"]))
        active_sim = active["method_provenance"]["descriptor"]["simulation"]
        candidate_sim = candidate["method_provenance"]["descriptor"]["simulation"]
        active_home.append(float(active_sim["requested_home_xg"]))
        active_away.append(float(active_sim["requested_away_xg"]))
        candidate_home.append(float(candidate_sim["requested_home_xg"]))
        candidate_away.append(float(candidate_sim["requested_away_xg"]))
    y_home, y_away = np.asarray(actual_home), np.asarray(actual_away)
    a_home, a_away = np.asarray(active_home), np.asarray(active_away)
    c_home, c_away = np.asarray(candidate_home), np.asarray(candidate_away)
    a_probs, c_probs = np.vstack(active_probs), np.vstack(candidate_probs)
    labels = outcome_indices(y_home, y_away)
    a_pl = probability_metric_contributions(labels, a_probs)
    c_pl = probability_metric_contributions(labels, c_probs)
    a_h, c_h = goal_metric_contributions(y_home, a_home), goal_metric_contributions(y_home, c_home)
    a_a, c_a = goal_metric_contributions(y_away, a_away), goal_metric_contributions(y_away, c_away)

    def contributions(pl: Mapping[str, np.ndarray], home: Mapping[str, np.ndarray], away: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        return {
            "log_loss": pl["log_loss"], "multiclass_brier": pl["multiclass_brier"], "accuracy": pl["accuracy"],
            "home_goal_mae": home["absolute_error"], "home_goal_rmse": home["squared_error"],
            "home_goal_poisson_deviance": home["poisson_deviance"],
            "away_goal_mae": away["absolute_error"], "away_goal_rmse": away["squared_error"],
            "away_goal_poisson_deviance": away["poisson_deviance"],
            "joint_goal_rmse": (home["squared_error"] + away["squared_error"]) / 2.0,
            "joint_goal_poisson_deviance": (home["poisson_deviance"] + away["poisson_deviance"]) / 2.0,
        }

    active_contrib = contributions(a_pl, a_h, a_a)
    candidate_contrib = contributions(c_pl, c_h, c_a)
    statistics = cast(Mapping[str, Any], specification["paired_statistics"])
    root_metrics = ("home_goal_rmse", "away_goal_rmse", "joint_goal_rmse")
    paired = paired_bootstrap(
        active_contrib, candidate_contrib, resamples=int(statistics["resamples"]),
        seed=int(statistics["seed"]), confidence_level=float(statistics["confidence_level"]),
        root_mean_metrics=root_metrics,
    )
    blocks = [str(pair["fixture_date"])[:7] for pair in pairs]
    block_result = None
    if len(set(blocks)) >= 2:
        sensitivity = cast(Mapping[str, Any], statistics["sensitivity"])
        block_result = paired_block_bootstrap(
            active_contrib, candidate_contrib, blocks, resamples=int(statistics["resamples"]),
            seed=int(sensitivity["seed"]), confidence_level=float(statistics["confidence_level"]),
            root_mean_metrics=root_metrics,
        )
    return {
        "schema_version": "ffps.phase7p_interim_aggregate.v1", "cohort": COHORT,
        "fixture_count": len(pairs), "date_min": min(pair["fixture_date"] for pair in pairs),
        "date_max": max(pair["fixture_date"] for pair in pairs),
        "metrics": {
            "pl_1x2": {"active": summarize_probability_metrics(labels, a_probs), "candidate": summarize_probability_metrics(labels, c_probs)},
            "joint_goals": {
                "active": summarize_goal_metrics(np.concatenate([y_home, y_away]), np.concatenate([a_home, a_away])),
                "candidate": summarize_goal_metrics(np.concatenate([y_home, y_away]), np.concatenate([c_home, c_away])),
            },
        },
        "paired_intervals": paired, "time_block_sensitivity": block_result,
        "decision": {
            "classification": "CANDIDATE_INCONCLUSIVE", "phase8_promotion_review_authorized": False,
            "final_decision_rule_evaluated": False, "reason": "descriptive interim only; no optional stopping",
        },
    }


def coverage_ratio(prediction_count: int, missed_count: int) -> float | None:
    denominator = prediction_count + missed_count
    return None if denominator == 0 else prediction_count / denominator
