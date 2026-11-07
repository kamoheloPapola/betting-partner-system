"""Build and execute the frozen FFPS Phase 7 offline shadow evaluation."""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import os
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence, cast

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_phase4_prediction_provenance import _input_frames
from scripts.run_phase6_controlled_retraining import protected_snapshot as phase6_protected_snapshot
from src.cli.commands.prediction import _calc_goals, _load_prediction_models
from src.evaluation.phase3_calibration import aggregate_directory
from src.evaluation.phase7_shadow import (
    AUTHORIZED_DECISIONS,
    CLASS_ORDER,
    PAIR_SCHEMA_VERSION,
    SPEC_SCHEMA_VERSION,
    EvaluationLedger,
    Phase7EvaluationError,
    apply_decision_rule,
    assert_single_labeled_cohort,
    canonical_json,
    goal_metric_contributions,
    outcome_indices,
    paired_block_bootstrap,
    paired_bootstrap,
    probability_metric_contributions,
    seal_specification,
    sha256_bytes,
    sha256_json,
    summarize_goal_metrics,
    summarize_probability_metrics,
    validate_paired_prediction,
    verify_specification,
)
from src.ml.phase6_controlled_training import verify_candidate_assessment
from src.predictions.provenance import (
    PredictionCapture,
    build_file_snapshot,
    data_provenance_for_fixture,
    mark_output_reproduced,
    replay_pl_1x2,
    sha256_file,
    source_event_index,
    verify_record,
)
from src.simulation.match_simulator import DEFAULT_N_SIMULATIONS, MatchSimulator


RUN_ID = "phase7-active-vs-candidate-shadow-79110f9a-20260829-v2"
DEFAULT_RUN_DIR = ROOT / "baselines" / "phase7" / RUN_ID
PHASE1_RUN = ROOT / "baselines" / "phase1" / "phase1-pl-1x2-79110f9a-20240801-v1"
PHASE3_RUN = ROOT / "baselines" / "phase3" / "phase3-pl-1x2-temperature-79110f9a-20260827-v1"
PHASE4_RUN = ROOT / "baselines" / "phase4" / "phase4-pl-1x2-provenance-79110f9a-20260827-v1"
PHASE6_RUN = ROOT / "baselines" / "phase6" / "phase6-controlled-pl-goals-79110f9a-20260829-v1"
FEATURE_MATRIX = ROOT / "data" / "features" / "feature_matrix.csv"
RAW_MATCHES = ROOT / "data" / "processed" / "matches.csv"
ACTIVE_MANIFEST = ROOT / "src" / "ml" / "models" / "manifest.json"

PROSPECTIVE_START = "2026-08-30"
PROSPECTIVE_END = "2026-12-31"
BOOTSTRAP_SEED = 7_911_007
BLOCK_BOOTSTRAP_SEED = 7_911_008
BOOTSTRAP_RESAMPLES = 10_000


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_jsonl(path: Path, values: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for value in values:
            handle.write(canonical_json(value) + "\n")
    os.replace(temporary, path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [cast(dict[str, Any], json.loads(line)) for line in handle if line.strip()]


def relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def protected_equal(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    ignored = {"captured_at"}
    return {key: value for key, value in left.items() if key not in ignored} == {
        key: value for key, value in right.items() if key not in ignored
    }


def _active_reference() -> dict[str, Any]:
    record = read_json(PHASE4_RUN / "prediction_provenance.json")
    return {
        "bundle_id": "active-production-pl-goals-phase4-frozen",
        "model_bundle_sha256": record["model_provenance"]["model_bundle_sha256"],
        "manifest_path": relative(ACTIVE_MANIFEST),
        "manifest_sha256": sha256_file(ACTIVE_MANIFEST),
        "training_lineage": "LEGACY_TRAINING_LINEAGE_UNPROVEN",
        "execution_provenance": "VERIFIED",
        "members": {
            str(item["role"]): {
                "path": str(item["source_location"]),
                "sha256": str(item["artifact_sha256"]),
                "size_bytes": int(item["size_bytes"]),
            }
            for item in record["model_provenance"]["members"]
        },
    }


def _candidate_reference() -> dict[str, Any]:
    bundle = read_json(PHASE6_RUN / "candidate_bundle.json")
    record = read_json(PHASE6_RUN / "integration" / "prediction_provenance.json")
    return {
        "bundle_id": bundle["bundle_id"],
        "bundle_aggregate_sha256": bundle["bundle_aggregate_sha256"],
        "model_bundle_sha256": record["model_provenance"]["model_bundle_sha256"],
        "candidate_classification": bundle["candidate_classification"],
        "evaluation_only": True,
        "production_activation_eligible": False,
        "members": {
            str(item["role"]): {
                "path": str(item["artifact"]["path"]),
                "sha256": str(item["artifact"]["sha256"]),
                "size_bytes": int(item["artifact"]["size_bytes"]),
                "assessment_path": str(item["assessment"]["path"]),
                "assessment_sha256": str(item["assessment"]["sha256"]),
            }
            for item in bundle["models"]
        },
    }


def _partition_contract() -> dict[str, Any]:
    phase1 = read_json(PHASE1_RUN / "partition_manifest.json")
    phase3 = read_json(PHASE3_RUN / "partition_manifest.json")
    return {
        "model_training_sha256": phase1["partitions"]["training"]["match_hashes_sha256"],
        "model_validation_sha256": phase1["partitions"]["validation"]["match_hashes_sha256"],
        "calibration_fit_sha256": phase3["partitions"]["calibration_fit"]["fixture_ids_sha256"],
        "historical_final_test_sha256": phase3["partitions"]["final_test"]["fixture_ids_sha256"],
        "historical_final_test_count": len(phase3["partitions"]["final_test"]["fixture_ids"]),
    }


def protected_snapshot() -> dict[str, Any]:
    snapshot = phase6_protected_snapshot()
    snapshot["captured_at"] = utc_now()
    snapshot["phase6_evidence_aggregate_sha256"] = aggregate_directory(PHASE6_RUN)
    snapshot["phase6_candidate_bundle_sha256"] = sha256_file(PHASE6_RUN / "candidate_bundle.json")
    snapshot["phase6_run_manifest_sha256"] = sha256_file(PHASE6_RUN / "run_manifest.json")
    snapshot["phase6_offline_verification_sha256"] = sha256_file(PHASE6_RUN / "offline_verification.json")
    snapshot["active_manifest_sha256"] = sha256_file(ACTIVE_MANIFEST)
    snapshot["active_member_artifacts"] = {
        role: sha256_file(ROOT / item["path"])
        for role, item in _active_reference()["members"].items()
    }
    snapshot["phase6_candidate_artifacts"] = {
        role: sha256_file(ROOT / item["path"])
        for role, item in _candidate_reference()["members"].items()
    }
    return snapshot


def _safe_new_run_dir(run_dir: Path) -> None:
    allowed = (ROOT / "baselines" / "phase7").resolve()
    resolved = run_dir.resolve()
    try:
        resolved.relative_to(allowed)
    except ValueError as exc:
        raise Phase7EvaluationError("Phase 7 evidence must remain beneath baselines/phase7") from exc
    if run_dir.exists():
        raise Phase7EvaluationError(f"Phase 7 evidence directory already exists: {run_dir}")
    run_dir.mkdir(parents=True)


def _historical_fixture_ids() -> list[str]:
    phase3 = read_json(PHASE3_RUN / "partition_manifest.json")
    return [str(value) for value in phase3["partitions"]["final_test"]["fixture_ids"]]


def _phase4_features() -> list[str]:
    record = read_json(PHASE4_RUN / "prediction_provenance.json")
    schemas = [list(item["ordered_feature_names"]) for item in record["feature_provenance"]["model_inputs"]]
    if not schemas or any(schema != schemas[0] for schema in schemas[1:]):
        raise Phase7EvaluationError("Phase 4 ordered feature schemas differ")
    return [str(value) for value in schemas[0]]


def build_specification() -> dict[str, Any]:
    active = _active_reference()
    candidate = _candidate_reference()
    partitions = _partition_contract()
    source_files = [
        ROOT / "src" / "evaluation" / "phase7_shadow.py",
        ROOT / "scripts" / "run_phase7_shadow_evaluation.py",
        ROOT / "src" / "predictions" / "provenance.py",
        ROOT / "src" / "simulation" / "match_simulator.py",
        ROOT / "src" / "cli" / "commands" / "prediction.py",
    ]
    value = {
        "schema_version": SPEC_SCHEMA_VERSION,
        "frozen_at": utc_now(),
        "phase": 7,
        "run_id": RUN_ID,
        "active": active,
        "candidate": candidate,
        "market": "PL_1X2",
        "class_order": list(CLASS_ORDER),
        "eligibility": {
            "league": "PL",
            "fixture_date_min_inclusive": PROSPECTIVE_START,
            "fixture_date_max_inclusive": PROSPECTIVE_END,
            "outcome_required_for_scoring": True,
            "prediction_must_be_frozen_before_phase7_outcome_access": True,
            "causal_rule": "maximum_source_event_timestamp < fixture kickoff/date cutoff",
            "date_only_policy": "start-of-day cutoff; same-day sources excluded",
            "duplicate_fixture_policy": "reject",
        },
        "horizon": {
            "type": "fixed_calendar_horizon",
            "start_date": PROSPECTIVE_START,
            "end_date": PROSPECTIVE_END,
            "stopping_rule": "score every eligible PL fixture; do not stop early",
            "minimum_fixture_count": None,
            "decision_before_horizon_complete": "CANDIDATE_INCONCLUSIVE",
        },
        "excluded_partitions": partitions,
        "historical_diagnostic": {
            "cohort": "Phase 3 final-test fixtures",
            "fixture_ids_sha256": partitions["historical_final_test_sha256"],
            "fixture_count": partitions["historical_final_test_count"],
            "decision_weight": "diagnostic_only",
            "promotion_quality_evidence": False,
            "reason": "prior aggregate reporting means the cohort is not globally pristine",
        },
        "feature_contract": {
            "ordered_feature_names": _phase4_features(),
            "ordered_feature_schema_sha256": sha256_json(_phase4_features()),
            "same_feature_bundle_required": True,
            "same_source_snapshot_required": True,
        },
        "simulation": {
            "implementation": "src.simulation.match_simulator.MatchSimulator.simulate",
            "n_simulations": DEFAULT_N_SIMULATIONS,
            "seed": 42,
            "seed_scope": "phase7_paired_shared_active_stream",
            "common_random_numbers": True,
            "per_fixture_protocol": "snapshot; active; restore; candidate",
            "state_advance_after_pair": "active_post_call_state",
            "production_rng_behavior_changed": False,
            "ensemble_lgbm_weight": 0.6,
            "ensemble_xgb_weight": 0.4,
            "rl_weights": None,
            "calibrator": None,
            "match_outcome_classifier": None,
        },
        "goal_metrics": [
            "mae",
            "rmse",
            "mean_poisson_deviance",
            "mean_prediction_bias",
            "residual_dispersion",
        ],
        "pl_1x2_metrics": [
            "log_loss",
            "multiclass_brier",
            "accuracy",
            "top_label_ece",
            "classwise_ece",
            "average_entropy",
            "mean_maximum_probability",
        ],
        "primary_metrics": ["log_loss", "multiclass_brier"],
        "paired_statistics": {
            "method": "fixture-level paired nonparametric bootstrap percentile interval",
            "resamples": BOOTSTRAP_RESAMPLES,
            "seed": BOOTSTRAP_SEED,
            "confidence_level": 0.95,
            "delta": "candidate_minus_active",
            "same_fixture_indices_across_bundles_and_metrics": True,
            "sensitivity": {
                "method": "whole-calendar-month paired block bootstrap",
                "seed": BLOCK_BOOTSTRAP_SEED,
                "run_when_at_least_two_blocks": True,
            },
        },
        "decision_rule": {
            "non_inferiority_margin": 0.0,
            "margin_basis": "strict zero-degradation fallback; no defensible historical margin was available",
            "candidate_support": (
                "horizon complete; non-worse point estimates and upper 95% paired bounds <= 0 for both primary "
                "losses, joint goal RMSE and joint goal Poisson deviance; top-label and classwise ECE non-worse"
            ),
            "active_support": "lower 95% paired bounds > 0 for both primary losses",
            "otherwise": "CANDIDATE_INCONCLUSIVE",
            "human_override": "forbidden",
        },
        "missing_data_policy": "exclude before prediction with a recorded reason; never impute or merge cohorts",
        "abandonment_conditions": [
            "bundle hash drift",
            "feature or source identity mismatch",
            "specification mutation after first prediction",
            "causal cutoff violation",
            "prediction cannot be frozen before outcome access",
        ],
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
        "implementation_sources": [
            {"path": relative(path), "sha256": sha256_file(path)} for path in source_files
        ],
    }
    return seal_specification(value)


def prepare(run_dir: Path) -> dict[str, Any]:
    _safe_new_run_dir(run_dir)
    before = protected_snapshot()
    write_json(run_dir / "protected_before.json", before)
    coverage = {
        "inspected_at": utc_now(),
        "phase6_frozen_at": read_json(PHASE6_RUN / "run_manifest.json")["finalized_at"],
        "available_pl_rows_after_phase6_freeze": 0,
        "available_pl_outcomes_after_phase6_freeze": 0,
        "strongest_promotion_evidence_available": False,
        "historical_diagnostic_fixture_count": len(_historical_fixture_ids()),
        "historical_diagnostic_label": "supporting_only_not_globally_pristine",
    }
    write_json(run_dir / "prospective_coverage_at_freeze.json", coverage)
    specification = build_specification()
    verification = verify_specification(specification)
    if not verification["verified"]:
        raise Phase7EvaluationError(f"specification did not verify: {verification['errors']}")
    write_json(run_dir / "evaluation_specification.json", specification)
    result = {
        "status": "FROZEN",
        "run_id": RUN_ID,
        "specification_fingerprint": verification["fingerprint"],
        "protected_phase0_to_phase6_captured": True,
        "prospective_fixture_count": 0,
    }
    write_json(run_dir / "preparation_result.json", result)
    return result


def _verify_candidate_lineage() -> dict[str, Any]:
    bundle = read_json(PHASE6_RUN / "candidate_bundle.json")
    results: list[dict[str, Any]] = []
    for model in bundle["models"]:
        assessment = read_json(ROOT / str(model["assessment"]["path"]))
        result = verify_candidate_assessment(assessment, root=ROOT)
        results.append(result)
    errors = [result for result in results if not result["verified"]]
    return {"verified": not errors, "classification": bundle["candidate_classification"], "results": results}


def _candidate_suite() -> dict[str, Any]:
    bundle = read_json(PHASE6_RUN / "candidate_bundle.json")
    features = _phase4_features()
    guard_scores = read_json(PHASE6_RUN / "integration" / "integration_result.json")[
        "descriptive_guard_calibration_scores"
    ]
    models: dict[str, Any] = {}
    for item in bundle["models"]:
        role = str(item["role"])
        artifact = item["artifact"]
        path = ROOT / str(artifact["path"])
        if sha256_file(path) != artifact["sha256"]:
            raise Phase7EvaluationError(f"candidate artifact drift: {role}")
        model = joblib.load(path)
        model.meta = {
            "name": f"phase7:{role}",
            "version": bundle["bundle_id"],
            "features": features,
            "type": "lgbm_regressor_poisson" if role.endswith("lgbm") else "xgb_regressor_count_poisson",
            "metrics": {"calibration_score": float(guard_scores[role])},
            "candidate_bundle_id": bundle["bundle_id"],
        }
        model._ffps_provenance = {
            "artifact_sha256": artifact["sha256"],
            "source_location": artifact["path"],
            "size_bytes": artifact["size_bytes"],
            "manifest_key": f"{bundle['bundle_id']}:{role}",
            "registry_slot": f"evaluation_only:{role}",
            "training_lineage": {
                "classification": "proven",
                "phase6_classification": bundle["candidate_classification"],
                "lineage_fingerprint": item["lineage"]["fingerprint"],
                "lineage_path": item["lineage"]["path"],
            },
        }
        models[role] = model
    return {
        "mh_goals": models["home_goals_lgbm"],
        "ma_goals": models["away_goals_lgbm"],
        "meta_goals": {"features": features},
        "mh_goals_xgb": models["home_goals_xgb"],
        "ma_goals_xgb": models["away_goals_xgb"],
        "meta_goals_xgb": {"features": features},
    }


def _assert_protected(run_dir: Path) -> None:
    before = read_json(run_dir / "protected_before.json")
    current = protected_snapshot()
    if not protected_equal(before, current):
        raise Phase7EvaluationError("protected Phase 0-6 or active identity drift")


def _prediction_record(
    fixture: pd.Series,
    suite: Mapping[str, Any],
    simulator: MatchSimulator,
    matrix: pd.DataFrame,
    *,
    label: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    fixture_id = str(fixture["match_hash"])
    prediction_timestamp = pd.Timestamp(fixture["date"]).isoformat()
    with PredictionCapture(
        prediction_timestamp=prediction_timestamp,
        data_provenance=data_provenance_for_fixture(matrix, fixture, prediction_timestamp),
    ) as capture:
        output = _calc_goals(
            fixture,
            cast(Any, suite),
            f"Phase7:{label}:{fixture_id}",
            simulator=simulator,
            league="PL",
            rl_weights=None,
            use_simulator=True,
        )
        record = capture.finalize(
            fixture=fixture,
            league="PL",
            probabilities={key: float(output[key]) for key in ("home", "draw", "away")},
            simulator=simulator,
        )
    replay = replay_pl_1x2(record, root=ROOT)
    record = mark_output_reproduced(record, replay)
    verification = verify_record(record, root=ROOT, verify_referenced_artifacts=False)
    if not verification["overall_phase4_trusted"]:
        raise Phase7EvaluationError(f"{label} Phase 4 verification failed: {verification['errors']}")
    return record.to_dict(), verification


def predict_historical(run_dir: Path) -> dict[str, Any]:
    specification = read_json(run_dir / "evaluation_specification.json")
    verified_spec = verify_specification(specification)
    if not verified_spec["verified"]:
        raise Phase7EvaluationError(f"frozen specification invalid: {verified_spec['errors']}")
    _assert_protected(run_dir)
    output_path = run_dir / "historical_diagnostic" / "paired_predictions.jsonl"
    if output_path.exists():
        raise Phase7EvaluationError("historical predictions are already frozen")
    lineage = _verify_candidate_lineage()
    if not lineage["verified"]:
        raise Phase7EvaluationError(f"candidate training lineage failed: {lineage}")

    matrix, _ = _input_frames()
    ids = _historical_fixture_ids()
    indexed = matrix.set_index(matrix["match_hash"].astype(str), drop=False)
    missing = sorted(set(ids) - set(indexed.index))
    if missing:
        raise Phase7EvaluationError(f"historical feature rows missing: {missing[:5]}")
    rows = indexed.loc[ids].copy().reset_index(drop=True).sort_values(["date", "match_hash"])
    features = list(specification["feature_contract"]["ordered_feature_names"])
    prediction_columns = ["match_hash", "date", "league", "home_team", "away_team", *features]
    active_suite = _load_prediction_models("PL")
    candidate_suite = _candidate_suite()
    simulator = MatchSimulator(
        n_simulations=int(specification["simulation"]["n_simulations"]),
        league="PL",
        seed=int(specification["simulation"]["seed"]),
        seed_scope=str(specification["simulation"]["seed_scope"]),
    )
    ledger = EvaluationLedger(specification)
    pairs: list[dict[str, Any]] = []
    logging.getLogger("src.cli.commands.prediction").setLevel(logging.ERROR)
    for index, (_, full_row) in enumerate(rows.iterrows(), start=1):
        fixture = full_row[prediction_columns].copy()
        fixture_id = str(fixture["match_hash"])
        shared_before = copy.deepcopy(simulator.rng.bit_generator.state)
        active_record, active_verification = _prediction_record(
            fixture, active_suite, simulator, matrix, label="active"
        )
        active_after = copy.deepcopy(simulator.rng.bit_generator.state)
        simulator.rng.bit_generator.state = copy.deepcopy(shared_before)
        candidate_record, candidate_verification = _prediction_record(
            fixture, candidate_suite, simulator, matrix, label="candidate"
        )
        candidate_after = copy.deepcopy(simulator.rng.bit_generator.state)
        simulator.rng.bit_generator.state = copy.deepcopy(active_after)
        pair = {
            "schema_version": PAIR_SCHEMA_VERSION,
            "specification_fingerprint": verified_spec["fingerprint"],
            "fixture_id": fixture_id,
            "fixture_date": pd.Timestamp(fixture["date"]).date().isoformat(),
            "cohort": "historical_diagnostic_only",
            "class_order": list(CLASS_ORDER),
            "authoritative_output": "active",
            "feature_bundle_sha256": active_record["feature_provenance"]["feature_bundle_sha256"],
            "source_snapshot_sha256": active_record["data_provenance"]["source_identity"]
            ["artifacts_aggregate_sha256"],
            "shared_pre_call_rng_state": shared_before,
            "active_post_call_rng_state": active_after,
            "candidate_post_call_rng_state": candidate_after,
            "stream_advanced_with": "active_post_call_state",
            "active_prediction": active_record,
            "candidate_prediction": candidate_record,
            "active_phase4_verified": bool(active_verification["overall_phase4_trusted"]),
            "candidate_phase4_verified": bool(candidate_verification["overall_phase4_trusted"]),
            "candidate_training_lineage_verified": True,
            "prediction_frozen_at": None,
            "outcome_accessed_at": None,
            "outcome": None,
        }
        ledger.add_pair(pair)
        pairs.append(pair)
        if index % 25 == 0:
            print(f"Phase 7 historical predictions: {index}/{len(rows)}", flush=True)
    frozen_at = utc_now()
    for pair in pairs:
        pair["prediction_frozen_at"] = frozen_at
    write_jsonl(output_path, pairs)
    result = {
        "status": "PREDICTIONS_FROZEN_BEFORE_PHASE7_SCORING",
        "cohort": "historical_diagnostic_only",
        "fixture_count": len(pairs),
        "prediction_frozen_at": frozen_at,
        "paired_predictions_path": relative(output_path),
        "paired_predictions_sha256": sha256_file(output_path),
        "specification_fingerprint": verified_spec["fingerprint"],
        "candidate_training_lineage": lineage,
        "production_output_mutated": False,
        "production_rng_behavior_changed": False,
        "active_stream_advance_contract": "active_post_call_state",
        "outcomes_consumed_by_scoring_stage": False,
    }
    write_json(run_dir / "historical_diagnostic" / "prediction_manifest.json", result)
    return result


def _prediction_arrays(pair: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, float, float, float, float]:
    active = pair["active_prediction"]
    candidate = pair["candidate_prediction"]
    active_probs = np.asarray(active["prediction_output"]["final_probabilities"], dtype=float)
    candidate_probs = np.asarray(candidate["prediction_output"]["final_probabilities"], dtype=float)
    active_sim = active["method_provenance"]["descriptor"]["simulation"]
    candidate_sim = candidate["method_provenance"]["descriptor"]["simulation"]
    return (
        active_probs,
        candidate_probs,
        float(active_sim["requested_home_xg"]),
        float(active_sim["requested_away_xg"]),
        float(candidate_sim["requested_home_xg"]),
        float(candidate_sim["requested_away_xg"]),
    )


def score_historical(run_dir: Path) -> dict[str, Any]:
    _assert_protected(run_dir)
    specification = read_json(run_dir / "evaluation_specification.json")
    prediction_manifest = read_json(run_dir / "historical_diagnostic" / "prediction_manifest.json")
    predictions_path = ROOT / prediction_manifest["paired_predictions_path"]
    if sha256_file(predictions_path) != prediction_manifest["paired_predictions_sha256"]:
        raise Phase7EvaluationError("frozen prediction evidence changed before scoring")
    if (run_dir / "historical_diagnostic" / "comparison_metrics.json").exists():
        raise Phase7EvaluationError("historical diagnostic has already been scored")
    pairs = read_jsonl(predictions_path)
    assert_single_labeled_cohort(pairs, "historical_diagnostic_only")
    outcome_accessed_at = utc_now()
    if outcome_accessed_at <= str(prediction_manifest["prediction_frozen_at"]):
        raise Phase7EvaluationError("outcome scoring did not occur after prediction freeze")
    outcomes = pd.read_csv(
        FEATURE_MATRIX,
        usecols=["match_hash", "date", "league", "home_score", "away_score"],
        low_memory=False,
    )
    outcomes = outcomes.loc[outcomes["league"].astype(str).eq("PL")].copy()
    outcomes["match_hash"] = outcomes["match_hash"].astype(str)
    indexed = outcomes.set_index("match_hash", drop=False)

    active_probs: list[np.ndarray] = []
    candidate_probs: list[np.ndarray] = []
    actual_home: list[float] = []
    actual_away: list[float] = []
    active_home: list[float] = []
    active_away: list[float] = []
    candidate_home: list[float] = []
    candidate_away: list[float] = []
    scored: list[dict[str, Any]] = []
    ledger = EvaluationLedger(specification)
    for pair in pairs:
        fixture_id = str(pair["fixture_id"])
        if fixture_id not in indexed.index:
            raise Phase7EvaluationError(f"outcome missing: {fixture_id}")
        row = indexed.loc[fixture_id]
        if isinstance(row, pd.DataFrame) or pd.isna(row["home_score"]) or pd.isna(row["away_score"]):
            raise Phase7EvaluationError(f"outcome is invalid or duplicated: {fixture_id}")
        ah = float(row["home_score"])
        aa = float(row["away_score"])
        ap, cp, alh, ala, clh, cla = _prediction_arrays(pair)
        scored_pair = copy.deepcopy(pair)
        scored_pair["outcome_accessed_at"] = outcome_accessed_at
        scored_pair["outcome"] = {
            "home_goals": ah,
            "away_goals": aa,
            "class_index": int(outcome_indices(np.asarray([ah]), np.asarray([aa]))[0]),
        }
        ledger.add_pair(scored_pair)
        active_probs.append(ap)
        candidate_probs.append(cp)
        actual_home.append(ah)
        actual_away.append(aa)
        active_home.append(alh)
        active_away.append(ala)
        candidate_home.append(clh)
        candidate_away.append(cla)
        scored.append(scored_pair)

    y_home = np.asarray(actual_home)
    y_away = np.asarray(actual_away)
    a_home = np.asarray(active_home)
    a_away = np.asarray(active_away)
    c_home = np.asarray(candidate_home)
    c_away = np.asarray(candidate_away)
    a_probs = np.vstack(active_probs)
    c_probs = np.vstack(candidate_probs)
    labels = outcome_indices(y_home, y_away)

    goal_metrics = {
        "home": {
            "active": summarize_goal_metrics(y_home, a_home),
            "candidate": summarize_goal_metrics(y_home, c_home),
        },
        "away": {
            "active": summarize_goal_metrics(y_away, a_away),
            "candidate": summarize_goal_metrics(y_away, c_away),
        },
        "joint": {
            "active": summarize_goal_metrics(np.concatenate([y_home, y_away]), np.concatenate([a_home, a_away])),
            "candidate": summarize_goal_metrics(
                np.concatenate([y_home, y_away]), np.concatenate([c_home, c_away])
            ),
        },
    }
    pl_metrics = {
        "active": summarize_probability_metrics(labels, a_probs),
        "candidate": summarize_probability_metrics(labels, c_probs),
    }
    active_pl = probability_metric_contributions(labels, a_probs)
    candidate_pl = probability_metric_contributions(labels, c_probs)
    active_h = goal_metric_contributions(y_home, a_home)
    candidate_h = goal_metric_contributions(y_home, c_home)
    active_a = goal_metric_contributions(y_away, a_away)
    candidate_a = goal_metric_contributions(y_away, c_away)
    active_contributions = {
        "log_loss": active_pl["log_loss"],
        "multiclass_brier": active_pl["multiclass_brier"],
        "accuracy": active_pl["accuracy"],
        "home_goal_mae": active_h["absolute_error"],
        "home_goal_rmse": active_h["squared_error"],
        "home_goal_poisson_deviance": active_h["poisson_deviance"],
        "away_goal_mae": active_a["absolute_error"],
        "away_goal_rmse": active_a["squared_error"],
        "away_goal_poisson_deviance": active_a["poisson_deviance"],
        "joint_goal_rmse": (active_h["squared_error"] + active_a["squared_error"]) / 2.0,
        "joint_goal_poisson_deviance": (
            active_h["poisson_deviance"] + active_a["poisson_deviance"]
        )
        / 2.0,
    }
    candidate_contributions = {
        "log_loss": candidate_pl["log_loss"],
        "multiclass_brier": candidate_pl["multiclass_brier"],
        "accuracy": candidate_pl["accuracy"],
        "home_goal_mae": candidate_h["absolute_error"],
        "home_goal_rmse": candidate_h["squared_error"],
        "home_goal_poisson_deviance": candidate_h["poisson_deviance"],
        "away_goal_mae": candidate_a["absolute_error"],
        "away_goal_rmse": candidate_a["squared_error"],
        "away_goal_poisson_deviance": candidate_a["poisson_deviance"],
        "joint_goal_rmse": (candidate_h["squared_error"] + candidate_a["squared_error"]) / 2.0,
        "joint_goal_poisson_deviance": (
            candidate_h["poisson_deviance"] + candidate_a["poisson_deviance"]
        )
        / 2.0,
    }
    root_metrics = ("home_goal_rmse", "away_goal_rmse", "joint_goal_rmse")
    bootstrap = paired_bootstrap(
        active_contributions,
        candidate_contributions,
        resamples=BOOTSTRAP_RESAMPLES,
        seed=BOOTSTRAP_SEED,
        confidence_level=0.95,
        root_mean_metrics=root_metrics,
    )
    blocks = [str(pair["fixture_date"])[:7] for pair in pairs]
    block_sensitivity = paired_block_bootstrap(
        active_contributions,
        candidate_contributions,
        blocks,
        resamples=BOOTSTRAP_RESAMPLES,
        seed=BLOCK_BOOTSTRAP_SEED,
        confidence_level=0.95,
        root_mean_metrics=root_metrics,
    )
    anomalies = [
        {
            "fixture_id": pair["fixture_id"],
            "fixture_date": pair["fixture_date"],
            "active_log_loss": float(active_pl["log_loss"][index]),
            "candidate_log_loss": float(candidate_pl["log_loss"][index]),
            "delta": float(candidate_pl["log_loss"][index] - active_pl["log_loss"][index]),
        }
        for index, pair in enumerate(pairs)
    ]
    anomalies.sort(key=lambda item: abs(float(item["delta"])), reverse=True)
    comparison = {
        "schema_version": "ffps.phase7_comparison_metrics.v1",
        "cohort": "historical_diagnostic_only",
        "promotion_quality_evidence": False,
        "fixture_count": len(pairs),
        "date_min": min(str(pair["fixture_date"]) for pair in pairs),
        "date_max": max(str(pair["fixture_date"]) for pair in pairs),
        "goal_layer": goal_metrics,
        "pl_1x2_layer": pl_metrics,
        "paired_bootstrap": bootstrap,
        "time_block_sensitivity": block_sensitivity,
        "largest_fixture_log_loss_anomalies": anomalies[:10],
        "outcome_timing": {
            "prediction_frozen_at": prediction_manifest["prediction_frozen_at"],
            "phase7_outcome_accessed_at": outcome_accessed_at,
            "prediction_preceded_phase7_scoring": True,
            "historical_outcomes_were_not_prospectively_unavailable": True,
        },
        "bookmaker_metrics": None,
        "roi_metrics": None,
    }
    evidence_path = run_dir / "historical_diagnostic" / "paired_fixture_evidence.jsonl"
    write_jsonl(evidence_path, scored)
    write_json(run_dir / "historical_diagnostic" / "comparison_metrics.json", comparison)
    result = {
        "status": "HISTORICAL_DIAGNOSTIC_COMPLETE",
        "fixture_count": len(pairs),
        "paired_fixture_evidence_sha256": sha256_file(evidence_path),
        "comparison_metrics_sha256": sha256_file(
            run_dir / "historical_diagnostic" / "comparison_metrics.json"
        ),
        "promotion_quality_evidence": False,
        "outcomes_consumed_after_prediction_freeze": True,
    }
    write_json(run_dir / "historical_diagnostic" / "scoring_result.json", result)
    return result


def _parse_pytest_xml(path: Path) -> dict[str, Any]:
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    counts = {
        "tests": sum(int(suite.attrib.get("tests", 0)) for suite in suites),
        "failures": sum(int(suite.attrib.get("failures", 0)) for suite in suites),
        "errors": sum(int(suite.attrib.get("errors", 0)) for suite in suites),
        "skipped": sum(int(suite.attrib.get("skipped", 0)) for suite in suites),
    }
    failures = []
    for case in root.iter("testcase"):
        if case.find("failure") is not None or case.find("error") is not None:
            failures.append(f"{case.attrib.get('classname')}::{case.attrib.get('name')}")
    return {"path": relative(path), **counts, "failure_ids": sorted(failures)}


def _test_summary(run_dir: Path) -> dict[str, Any]:
    focused = _parse_pytest_xml(run_dir / "focused_test_results.xml")
    protected = _parse_pytest_xml(run_dir / "protected_regression_results.xml")
    full = _parse_pytest_xml(run_dir / "full_test_results.xml")
    prior = read_json(PHASE6_RUN / "test_results.json")
    inherited = set(prior["inherited_failure_allowlist"])
    new_failures = sorted(set(full["failure_ids"]) - inherited)
    return {
        "focused": focused,
        "protected_regressions": protected,
        "full_suite": full,
        "phase6_inherited_failure_ids": sorted(inherited),
        "new_unexplained_failure_ids": new_failures,
        "no_new_unexplained_failures": not new_failures,
    }


def _report_text(
    specification: Mapping[str, Any],
    comparison: Mapping[str, Any],
    decision: Mapping[str, Any],
    tests: Mapping[str, Any],
    protected_unchanged: bool,
) -> str:
    active = comparison["pl_1x2_layer"]["active"]
    candidate = comparison["pl_1x2_layer"]["candidate"]
    intervals = comparison["paired_bootstrap"]["metrics"]
    goal = comparison["goal_layer"]
    lines = [
        "# FFPS Phase 7 — Candidate vs Active Shadow Evaluation",
        "",
        "## Outcome",
        "",
        f"**{decision['classification']}**",
        "",
        "Phase 8 production-promotion review is **not authorized**. No post-Phase-6 outcomes exist yet, "
        "so the fixed prospective horizon is neither populated nor complete. The comparison below is supporting "
        "historical diagnostic evidence only.",
        "",
        "## Frozen protocol",
        "",
        f"- Specification fingerprint: `{specification['specification_fingerprint']}`.",
        f"- Active calculation bundle: `{specification['active']['model_bundle_sha256']}`.",
        f"- Candidate: `{specification['candidate']['bundle_id']}` / `{specification['candidate']['bundle_aggregate_sha256']}`.",
        f"- Candidate calculation bundle: `{specification['candidate']['model_bundle_sha256']}`.",
        f"- Prospective window: {PROSPECTIVE_START} through {PROSPECTIVE_END}; fixed horizon, no early stopping.",
        "- Common random numbers: snapshot RNG state, run active, restore, run candidate; advance the evaluation stream with the active post-call state.",
        "- No quantitative historical non-inferiority margin was defensible; the frozen fallback is a strict zero-degradation margin.",
        "",
        "## Prospective coverage",
        "",
        "- Eligible post-freeze fixtures with outcomes: 0.",
        "- Prospective metrics and paired intervals: unavailable by construction.",
        "- Final decision is therefore mechanically `CANDIDATE_INCONCLUSIVE`.",
        "",
        "## Historical diagnostic (not promotion-quality evidence)",
        "",
        f"- Coverage: {comparison['fixture_count']} fixtures, {comparison['date_min']} to {comparison['date_max']}.",
        f"- Active log loss: {active['log_loss']:.8f}; candidate: {candidate['log_loss']:.8f}; delta: {intervals['log_loss']['point_delta']:.8f} "
        f"(95% {intervals['log_loss']['lower']:.8f} to {intervals['log_loss']['upper']:.8f}).",
        f"- Active Brier: {active['multiclass_brier']:.8f}; candidate: {candidate['multiclass_brier']:.8f}; delta: "
        f"{intervals['multiclass_brier']['point_delta']:.8f} (95% {intervals['multiclass_brier']['lower']:.8f} to "
        f"{intervals['multiclass_brier']['upper']:.8f}).",
        f"- Accuracy: active {active['accuracy']:.4f}; candidate {candidate['accuracy']:.4f}.",
        f"- Top-label ECE: active {active['top_label_ece']:.6f}; candidate {candidate['top_label_ece']:.6f}.",
        f"- Classwise ECE: active {active['classwise_ece']:.6f}; candidate {candidate['classwise_ece']:.6f}.",
        f"- Average entropy: active {active['average_entropy']:.6f}; candidate {candidate['average_entropy']:.6f}.",
        f"- Mean maximum probability: active {active['mean_maximum_probability']:.6f}; candidate {candidate['mean_maximum_probability']:.6f}.",
        "",
        "### Goal layer",
        "",
        f"- Home RMSE: active {goal['home']['active']['rmse']:.6f}; candidate {goal['home']['candidate']['rmse']:.6f}.",
        f"- Away RMSE: active {goal['away']['active']['rmse']:.6f}; candidate {goal['away']['candidate']['rmse']:.6f}.",
        f"- Joint Poisson deviance: active {goal['joint']['active']['mean_poisson_deviance']:.6f}; candidate "
        f"{goal['joint']['candidate']['mean_poisson_deviance']:.6f}.",
        "- MAE, bias, residual dispersion, all paired intervals, reliability tables, and fixture anomalies are in `historical_diagnostic/comparison_metrics.json`.",
        "- Calendar-month whole-block bootstrap sensitivity was run exactly as predeclared.",
        "",
        "## Integrity and tests",
        "",
        f"- Protected Phase 0–6 and active identities unchanged: {str(protected_unchanged).lower()}.",
        "- Both predictions on every diagnostic fixture passed the Phase 4 verifier and exact replay.",
        "- Candidate model hashes resolved through all four Phase 6 verified assessments.",
        "- No calibration, outcome classifier, bookmaker input, retraining, registry write, or production activation occurred.",
        f"- Focused tests: {tests['focused']['tests']} tests, {tests['focused']['failures'] + tests['focused']['errors']} failures.",
        f"- Protected regressions: {tests['protected_regressions']['tests']} tests, "
        f"{tests['protected_regressions']['failures'] + tests['protected_regressions']['errors']} failures.",
        f"- Full suite: {tests['full_suite']['tests']} tests, {tests['full_suite']['failures'] + tests['full_suite']['errors']} inherited failures, "
        f"{tests['full_suite']['skipped']} skipped; zero new unexplained failures: {str(tests['no_new_unexplained_failures']).lower()}.",
        "",
        "## Deviations and limitations",
        "",
        "- No post-freeze fixtures can exist in the local snapshot: Phase 6 finalized on 2026-08-29 and the newest PL feature row is 2025-12-28.",
        "- The Phase 3 final-test cohort has appeared in prior aggregate reporting and is intentionally labeled diagnostic-only.",
        "- Historical diagnostic predictions were persisted before the Phase 7 scoring process read their labels, but the historical outcomes were not prospectively unavailable.",
        "- This phase implemented an offline adapter because adding live production shadow execution would create unnecessary serving risk.",
        "- A v1 specification was superseded before any prediction or outcome was incorporated after its first prediction command failed closed on an ambiguous DataFrame index; v2 is the executed frozen protocol.",
        "- A later collection/finalization run must use the unchanged specification and candidate bytes; any change requires a new challenger and specification.",
    ]
    return "\n".join(lines) + "\n"


def finalize(run_dir: Path) -> dict[str, Any]:
    specification = read_json(run_dir / "evaluation_specification.json")
    comparison = read_json(run_dir / "historical_diagnostic" / "comparison_metrics.json")
    before = read_json(run_dir / "protected_before.json")
    after = protected_snapshot()
    unchanged = protected_equal(before, after)
    if not unchanged:
        raise Phase7EvaluationError("protected Phase 0-6 or active identities changed")
    write_json(run_dir / "protected_after_final.json", after)
    tests = _test_summary(run_dir)
    if not tests["no_new_unexplained_failures"]:
        raise Phase7EvaluationError(f"new full-suite failures: {tests['new_unexplained_failure_ids']}")
    if tests["focused"]["failures"] or tests["focused"]["errors"]:
        raise Phase7EvaluationError("focused Phase 7 tests failed")
    if tests["protected_regressions"]["failures"] or tests["protected_regressions"]["errors"]:
        raise Phase7EvaluationError("protected regressions failed")
    write_json(run_dir / "test_results.json", tests)
    decision = apply_decision_rule(
        prospective_fixture_count=0,
        horizon_complete=False,
        paired_intervals=None,
        active_reliability=None,
        candidate_reliability=None,
        provenance_verified=True,
    )
    write_json(
        run_dir / "prospective_result.json",
        {
            "status": "AWAITING_FIXED_HORIZON",
            "fixture_count": 0,
            "date_min": None,
            "date_max": None,
            "metrics": None,
            "paired_intervals": None,
            "decision": decision,
        },
    )
    report_path = run_dir / "PHASE7_IMPLEMENTATION_REPORT.md"
    report_path.write_text(
        _report_text(specification, comparison, decision, tests, unchanged),
        encoding="utf-8",
        newline="\n",
    )
    excluded = {"artifact_index.json", "run_manifest.json", "offline_verification.json"}
    artifacts = [
        {
            "path": relative(path),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for path in sorted(run_dir.rglob("*"))
        if path.is_file() and path.name not in excluded
    ]
    index = {
        "schema_version": "ffps.phase7_artifact_index.v1",
        "artifacts": artifacts,
        "artifact_count": len(artifacts),
        "aggregate_sha256": sha256_json(artifacts),
    }
    write_json(run_dir / "artifact_index.json", index)
    manifest = {
        "schema_version": "ffps.phase7_run_manifest.v1",
        "phase": 7,
        "run_id": RUN_ID,
        "status": "complete",
        "finalized_at": utc_now(),
        "specification_fingerprint": specification["specification_fingerprint"],
        "active_bundle_sha256": specification["active"]["model_bundle_sha256"],
        "candidate_bundle_id": specification["candidate"]["bundle_id"],
        "candidate_bundle_aggregate_sha256": specification["candidate"]["bundle_aggregate_sha256"],
        "prospective_fixture_count": 0,
        "historical_diagnostic_fixture_count": comparison["fixture_count"],
        "final_classification": decision["classification"],
        "phase8_promotion_review_authorized": decision["phase8_promotion_review_authorized"],
        "production_activation_performed": False,
        "protected_phase0_to_phase6_unchanged": unchanged,
        "no_new_unexplained_test_failures": tests["no_new_unexplained_failures"],
        "artifact_aggregate_sha256": index["aggregate_sha256"],
        "artifact_index_sha256": sha256_file(run_dir / "artifact_index.json"),
        "report_sha256": sha256_file(report_path),
    }
    write_json(run_dir / "run_manifest.json", manifest)
    return manifest


def verify(run_dir: Path, *, persist: bool = True) -> dict[str, Any]:
    errors: list[str] = []
    required = (
        "evaluation_specification.json",
        "protected_before.json",
        "protected_after_final.json",
        "historical_diagnostic/paired_predictions.jsonl",
        "historical_diagnostic/paired_fixture_evidence.jsonl",
        "historical_diagnostic/comparison_metrics.json",
        "prospective_result.json",
        "test_results.json",
        "PHASE7_IMPLEMENTATION_REPORT.md",
        "artifact_index.json",
        "run_manifest.json",
    )
    for name in required:
        if not (run_dir / name).is_file():
            errors.append(f"missing artifact: {name}")
    if errors:
        result = {"status": "FAILED", "errors": errors}
        if persist:
            write_json(run_dir / "offline_verification.json", result)
        return result
    specification = read_json(run_dir / "evaluation_specification.json")
    spec_verification = verify_specification(specification)
    errors.extend(spec_verification["errors"])
    for source in specification.get("implementation_sources", []):
        source_path = ROOT / str(source["path"])
        if not source_path.is_file() or sha256_file(source_path) != source["sha256"]:
            errors.append(f"implementation source mismatch: {source['path']}")
    before = read_json(run_dir / "protected_before.json")
    after = read_json(run_dir / "protected_after_final.json")
    current = protected_snapshot()
    if not protected_equal(before, after) or not protected_equal(before, current):
        errors.append("protected identity mismatch")
    index = read_json(run_dir / "artifact_index.json")
    for item in index["artifacts"]:
        path = ROOT / str(item["path"])
        if not path.is_file() or sha256_file(path) != item["sha256"] or path.stat().st_size != item["size_bytes"]:
            errors.append(f"artifact mismatch: {item['path']}")
    if sha256_json(index["artifacts"]) != index["aggregate_sha256"]:
        errors.append("artifact index aggregate mismatch")
    manifest = read_json(run_dir / "run_manifest.json")
    decision = read_json(run_dir / "prospective_result.json")["decision"]
    if decision["classification"] not in AUTHORIZED_DECISIONS:
        errors.append("unauthorized Phase 7 decision")
    if decision["classification"] != "CANDIDATE_INCONCLUSIVE":
        errors.append("zero-outcome prospective decision must be CANDIDATE_INCONCLUSIVE")
    if manifest["phase8_promotion_review_authorized"] or manifest["production_activation_performed"]:
        errors.append("promotion or activation was improperly authorized")
    prediction_manifest = read_json(run_dir / "historical_diagnostic" / "prediction_manifest.json")
    predictions_path = ROOT / prediction_manifest["paired_predictions_path"]
    if sha256_file(predictions_path) != prediction_manifest["paired_predictions_sha256"]:
        errors.append("historical paired prediction hash mismatch")
    seen: set[str] = set()
    for pair in read_jsonl(run_dir / "historical_diagnostic" / "paired_fixture_evidence.jsonl"):
        try:
            validate_paired_prediction(pair, specification=specification, seen_fixture_ids=seen)
        except Phase7EvaluationError as exc:
            errors.append(f"paired fixture {pair.get('fixture_id')}: {exc}")
    tests = read_json(run_dir / "test_results.json")
    if not tests["no_new_unexplained_failures"]:
        errors.append("new unexplained test failures recorded")
    result = {
        "status": "VERIFIED" if not errors else "FAILED",
        "errors": errors,
        "specification_fingerprint": specification["specification_fingerprint"],
        "final_classification": decision["classification"],
        "prospective_fixture_count": 0,
        "historical_diagnostic_fixture_count": len(seen),
        "protected_phase0_to_phase6_unchanged": not any("protected" in error for error in errors),
        "production_activation_performed": False,
    }
    if persist:
        write_json(run_dir / "offline_verification.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("prepare", "predict-historical", "score-historical", "finalize", "verify"),
    )
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            result = prepare(args.run_dir)
        elif args.command == "predict-historical":
            result = predict_historical(args.run_dir)
        elif args.command == "score-historical":
            result = score_historical(args.run_dir)
        elif args.command == "finalize":
            result = finalize(args.run_dir)
        else:
            result = verify(args.run_dir)
    except Phase7EvaluationError as exc:
        print(f"Phase 7 error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
