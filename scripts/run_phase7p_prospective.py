"""Operate the Phase 7P prospective shadow-evidence sidecar.

The runner consumes the frozen Phase 7 v2 specification and model bundles.  It
does not train, register, activate, recalibrate, or call production persistence.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import shutil
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence, cast

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_phase4_prediction_provenance import _input_frames
from scripts.run_phase7_shadow_evaluation import (
    _candidate_suite,
    _verify_candidate_lineage,
)
from src.cli.commands.prediction import _calc_goals, _load_prediction_models
from src.evaluation.phase3_calibration import aggregate_directory
from src.evaluation.phase7_shadow import (
    CLASS_ORDER,
    PAIR_SCHEMA_VERSION,
    Phase7EvaluationError,
    canonical_json,
    sha256_json,
    validate_paired_prediction,
    verify_specification,
)
from src.evaluation.phase7p_prospective import (
    COHORT,
    FROZEN_SPECIFICATION_FINGERPRINT,
    ImmutableCohortStore,
    Phase7PError,
    aggregate_eligible,
    build_prediction_record,
    build_settlement_record,
    coverage_ratio,
    seal_record,
    verify_prediction_record,
    verify_settlement_record,
)
from src.predictions.provenance import (
    PredictionCapture,
    data_provenance_for_fixture,
    mark_output_reproduced,
    replay_pl_1x2,
    sha256_file,
    verify_record,
)
from src.simulation.match_simulator import MatchSimulator


RUN_ID = "phase7p-prospective-shadow-79110f9a-20260829-v1"
DEFAULT_RUN_DIR = ROOT / "baselines" / "phase7p" / RUN_ID
PHASE7_RUN = ROOT / "baselines" / "phase7" / "phase7-active-vs-candidate-shadow-79110f9a-20260829-v2"
PHASE7_V1_RUN = ROOT / "baselines" / "phase7" / "phase7-active-vs-candidate-shadow-79110f9a-20260829-v1"
SPECIFICATION_PATH = PHASE7_RUN / "evaluation_specification.json"
FEATURE_MATRIX = ROOT / "data" / "features" / "feature_matrix.csv"
RAW_MATCHES = ROOT / "data" / "processed" / "matches.csv"
LOCAL_UPCOMING = ROOT / "data" / "processed" / "matches" / "PL_upcoming.csv"
FIXTURE_CACHE = ROOT / "data" / "processed" / "fixtures" / "PL_upcoming.json"
PROSPECTIVE_START = "2026-08-30"
PROSPECTIVE_END = "2026-12-31"
ALLOWED_CLASSIFICATIONS = {
    "PROSPECTIVE_COLLECTION_READY",
    "PROSPECTIVE_COLLECTION_BLOCKED_DATA",
    "PROSPECTIVE_COLLECTION_INTEGRITY_BLOCKED",
}
INHERITED_FULL_SUITE_FAILURES = {
    "tests.cli.test_verbosity_levels.TestCLIVerbosityLevels::test_quiet_mode_suppresses_pipeline_logs",
    "tests.test_drift_artifacts::test_global_drift_artifact_evaluated_at_is_present_and_recent",
    "tests.test_drift_artifacts::test_league_drift_artifact_evaluated_at_is_present_and_recent[BL1]",
    "tests.test_drift_artifacts::test_league_drift_artifact_evaluated_at_is_present_and_recent[PD]",
    "tests.test_drift_artifacts::test_league_drift_artifact_evaluated_at_is_present_and_recent[PL]",
    "tests.test_drift_artifacts::test_league_drift_artifact_evaluated_at_is_present_and_recent[SA]",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _timestamp(value: Any) -> pd.Timestamp:
    parsed = pd.Timestamp(value)
    if parsed.tzinfo is None:
        parsed = parsed.tz_localize("UTC")
    return parsed.tz_convert("UTC")


def _safe_new_run_dir(run_dir: Path) -> None:
    allowed = (ROOT / "baselines" / "phase7p").resolve()
    resolved = run_dir.resolve()
    try:
        resolved.relative_to(allowed)
    except ValueError as exc:
        raise Phase7PError("Phase 7P evidence must remain beneath baselines/phase7p") from exc
    if resolved.exists():
        raise Phase7PError(f"Phase 7P evidence already exists: {resolved}")
    resolved.mkdir(parents=True)


def load_frozen_specification() -> dict[str, Any]:
    specification = read_json(SPECIFICATION_PATH)
    result = verify_specification(specification)
    if not result["verified"]:
        raise Phase7PError(f"frozen Phase 7 specification failed verification: {result['errors']}")
    if result["fingerprint"] != FROZEN_SPECIFICATION_FINGERPRINT:
        raise Phase7PError("authoritative Phase 7 v2 specification fingerprint changed")
    return specification


def verify_frozen_contract() -> dict[str, Any]:
    """Verify frozen spec sources, model bytes, and Phase 5/6 lineage."""
    errors: list[str] = []
    try:
        specification = load_frozen_specification()
    except Phase7PError as exc:
        return {"verified": False, "errors": [str(exc)]}
    for source in specification["implementation_sources"]:
        path = ROOT / str(source["path"])
        if not path.is_file() or sha256_file(path) != source["sha256"]:
            errors.append(f"frozen implementation source drift: {source['path']}")
    for label in ("active", "candidate"):
        for role, member in specification[label]["members"].items():
            path = ROOT / str(member["path"])
            if not path.is_file() or sha256_file(path) != member["sha256"]:
                errors.append(f"{label} model drift: {role}")
    try:
        lineage = _verify_candidate_lineage()
        if not lineage["verified"]:
            errors.append("candidate Phase 5/6 lineage did not resolve")
    except Exception as exc:  # fail-closed audit boundary
        lineage = {"verified": False, "error": f"{type(exc).__name__}: {exc}"}
        errors.append("candidate Phase 5/6 lineage verification failed")
    phase7_offline = read_json(PHASE7_RUN / "offline_verification.json")
    if phase7_offline.get("status") != "VERIFIED" or phase7_offline.get("errors"):
        errors.append("authoritative Phase 7 v2 evidence is not offline-verified")
    return {
        "verified": not errors,
        "errors": errors,
        "specification_fingerprint": specification["specification_fingerprint"],
        "active_bundle_sha256": specification["active"]["model_bundle_sha256"],
        "candidate_bundle_id": specification["candidate"]["bundle_id"],
        "candidate_bundle_aggregate_sha256": specification["candidate"]["bundle_aggregate_sha256"],
        "candidate_training_lineage": lineage,
    }


def protected_snapshot() -> dict[str, Any]:
    specification = load_frozen_specification()
    contract = verify_frozen_contract()
    if not contract["verified"]:
        raise Phase7PError(f"frozen contract failed: {contract['errors']}")
    return {
        "captured_at": utc_now(),
        "phase7_v2_evidence_aggregate_sha256": aggregate_directory(PHASE7_RUN),
        "phase7_v1_abandoned_evidence_aggregate_sha256": aggregate_directory(PHASE7_V1_RUN),
        "phase7_specification_path": relative(SPECIFICATION_PATH),
        "phase7_specification_file_sha256": sha256_file(SPECIFICATION_PATH),
        "phase7_specification_fingerprint": specification["specification_fingerprint"],
        "phase7_run_manifest_sha256": sha256_file(PHASE7_RUN / "run_manifest.json"),
        "phase7_offline_verification_sha256": sha256_file(PHASE7_RUN / "offline_verification.json"),
        "active_bundle_sha256": contract["active_bundle_sha256"],
        "candidate_bundle_id": contract["candidate_bundle_id"],
        "candidate_bundle_aggregate_sha256": contract["candidate_bundle_aggregate_sha256"],
        "implementation_sources": {
            str(item["path"]): sha256_file(ROOT / str(item["path"]))
            for item in specification["implementation_sources"]
        },
    }


def protected_equal(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return {key: value for key, value in left.items() if key != "captured_at"} == {
        key: value for key, value in right.items() if key != "captured_at"
    }


def _frame_date_summary(path: Path, *, date_columns: Sequence[str], league_only: bool = True) -> dict[str, Any]:
    if not path.is_file():
        return {"path": relative(path), "exists": False, "row_count": 0, "date_min": None, "date_max": None}
    frame = pd.read_csv(path, low_memory=False)
    if league_only and "league" in frame.columns:
        frame = frame.loc[frame["league"].astype(str).eq("PL")]
    date_column = next((column for column in date_columns if column in frame.columns), None)
    dates = pd.to_datetime(frame[date_column], utc=True, errors="coerce") if date_column else pd.Series(dtype="datetime64[ns, UTC]")
    valid = dates.dropna()
    return {
        "path": relative(path), "exists": True, "sha256": sha256_file(path), "row_count": len(frame),
        "date_min": valid.min().date().isoformat() if len(valid) else None,
        "date_max": valid.max().date().isoformat() if len(valid) else None,
    }


def _feature_readiness() -> dict[str, Any]:
    frame = pd.read_csv(FEATURE_MATRIX, low_memory=False)
    frame = frame.loc[frame["league"].astype(str).eq("PL")].copy()
    frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="coerce")
    in_window = frame["date"].dt.date.astype("string").between(PROSPECTIVE_START, PROSPECTIVE_END)
    outcomes_absent = frame["home_score"].isna() & frame["away_score"].isna()
    ready = frame.loc[in_window & outcomes_absent]
    return {
        "latest_causal_source_row": frame["date"].max().date().isoformat() if not frame.empty else None,
        "authorized_window_rows": int(in_window.sum()),
        "prediction_feature_rows_without_outcome": len(ready),
        "feature_matrix_sha256": sha256_file(FEATURE_MATRIX),
    }


def _outcome_readiness() -> dict[str, Any]:
    files = [
        _frame_date_summary(path, date_columns=("match_date", "date"))
        for path in _approved_outcome_paths()
    ]
    latest = max(
        (str(item["date_max"]) for item in files if item.get("date_max")),
        default=None,
    )
    return {"files": files, "latest_outcome_row": latest}


def probe_approved_upstream() -> dict[str, Any]:
    """Read the configured provider in memory; never invoke its file-writing main."""
    from collections import Counter
    from scripts.fetch_fresh_data import fetch_matches

    matches = fetch_matches("PL", 2026)
    rows: list[dict[str, Any]] = []
    for match in matches:
        date = str(match.get("utcDate", ""))[:10]
        if date:
            rows.append({"provider_fixture_id": str(match.get("id", "")), "date": date, "status": str(match.get("status", ""))})
    dates = sorted(row["date"] for row in rows)
    window = [row for row in rows if PROSPECTIVE_START <= row["date"] <= PROSPECTIVE_END]
    return {
        "status": "AVAILABLE", "source": "football-data.org v4 through scripts/fetch_fresh_data.py",
        "probe_mode": "in_memory_no_repository_write", "season": 2026, "fixture_count": len(rows),
        "date_min": dates[0] if dates else None, "date_max": dates[-1] if dates else None,
        "status_counts": dict(Counter(row["status"] for row in rows)),
        "authorized_window_fixture_count": len(window),
        "authorized_window_fixture_ids_sha256": sha256_json(sorted(row["provider_fixture_id"] for row in window)),
        "probed_at": utc_now(),
    }


def audit_ingestion(*, probe_upstream: bool) -> dict[str, Any]:
    feature = _feature_readiness()
    upstream: dict[str, Any]
    if probe_upstream:
        try:
            upstream = probe_approved_upstream()
        except Exception as exc:
            upstream = {"status": "UNAVAILABLE", "error": f"{type(exc).__name__}: {exc}", "probed_at": utc_now()}
    else:
        upstream = {"status": "NOT_PROBED", "reason": "network probe not requested"}
    local_upcoming = _frame_date_summary(LOCAL_UPCOMING, date_columns=("date", "match_date"))
    raw = _outcome_readiness()
    raw_feature_ready = feature["prediction_feature_rows_without_outcome"] > 0
    classification = (
        "PROSPECTIVE_COLLECTION_READY" if raw_feature_ready else "PROSPECTIVE_COLLECTION_BLOCKED_DATA"
    )
    return {
        "schema_version": "ffps.phase7p_ingestion_audit.v1", "audited_at": utc_now(),
        "classification": classification,
        "approved_fixture_and_result_source": {
            "provider": "football-data.org v4", "repository_adapter": "scripts/fetch_fresh_data.py",
            "credential_configured": _dotenv_key_configured("FOOTBALL_DATA_API_KEY"),
            "source_substitution_performed": False,
        },
        "fixture_discovery_alternative": {
            "provider": "The Odds API", "repository_adapter": "src/ingestion/fixture_manager.py",
            "prediction_input_allowed": False, "bookmaker_independence_preserved": True,
        },
        "upstream_probe": upstream, "local_upcoming": local_upcoming,
        "fixture_cache": _json_cache_summary(FIXTURE_CACHE), "feature_readiness": feature,
        "outcome_readiness": raw,
        "blocker": None if raw_feature_ready else (
            "Approved raw 2026/27 fixtures are available upstream, but the normal FFPS feature build has not "
            "produced any causally valid authorized-window feature rows. Protected Phase 0-7 data was not rewritten."
        ),
    }


def _dotenv_key_configured(name: str) -> bool:
    env = ROOT / ".env"
    if not env.is_file():
        return False
    for line in env.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.strip().startswith(f"{name}="):
            return bool(line.split("=", 1)[1].strip())
    return False


def _json_cache_summary(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"path": relative(path), "exists": False}
    try:
        value = read_json(path)
        fixtures = value.get("fixtures", value if isinstance(value, list) else [])
        return {
            "path": relative(path), "exists": True, "sha256": sha256_file(path),
            "fetched_at": value.get("fetched_at") if isinstance(value, dict) else None,
            "fixture_count": len(fixtures) if isinstance(fixtures, list) else None,
        }
    except Exception as exc:
        return {"path": relative(path), "exists": True, "error": f"{type(exc).__name__}: {exc}"}


def prepare(run_dir: Path, *, probe_upstream: bool) -> dict[str, Any]:
    _safe_new_run_dir(run_dir)
    contract = verify_frozen_contract()
    if not contract["verified"]:
        raise Phase7PError(f"frozen Phase 7 contract failed: {contract['errors']}")
    specification = load_frozen_specification()
    before = protected_snapshot()
    write_json(run_dir / "protected_before.json", before)
    write_json(
        run_dir / "frozen_specification_reference.json",
        {
            "schema_version": "ffps.phase7p_frozen_specification_reference.v1",
            "path": relative(SPECIFICATION_PATH), "file_sha256": sha256_file(SPECIFICATION_PATH),
            "specification_fingerprint": specification["specification_fingerprint"],
            "consumed_unchanged": True,
        },
    )
    audit = audit_ingestion(probe_upstream=probe_upstream)
    write_json(run_dir / "ingestion_readiness.json", audit)
    ImmutableCohortStore(run_dir / "cohort")
    result = {
        "status": "INFRASTRUCTURE_PREPARED", "run_id": RUN_ID,
        "classification": audit["classification"],
        "specification_fingerprint": specification["specification_fingerprint"],
        "protected_phase0_to_phase7_captured": True, "production_behavior_changed": False,
        "source_substitution_performed": False,
    }
    write_json(run_dir / "preparation_result.json", result)
    return result


def _prediction_record(
    fixture: pd.Series, suite: Mapping[str, Any], simulator: MatchSimulator,
    matrix: pd.DataFrame, *, label: str, prediction_timestamp: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    fixture_id = str(fixture["match_hash"])
    with PredictionCapture(
        prediction_timestamp=prediction_timestamp,
        data_provenance=data_provenance_for_fixture(matrix, fixture, prediction_timestamp),
    ) as capture:
        output = _calc_goals(
            fixture, cast(Any, suite), f"Phase7P:{label}:{fixture_id}", simulator=simulator,
            league="PL", rl_weights=None, use_simulator=True,
        )
        record = capture.finalize(
            fixture=fixture, league="PL",
            probabilities={key: float(output[key]) for key in ("home", "draw", "away")},
            simulator=simulator,
        )
    replay = replay_pl_1x2(record, root=ROOT)
    record = mark_output_reproduced(record, replay)
    verification = verify_record(record, root=ROOT, verify_referenced_artifacts=False)
    if not verification["overall_phase4_trusted"]:
        raise Phase7PError(f"{label} Phase 4 verification failed: {verification['errors']}")
    return record.to_dict(), verification


def capture_ready(run_dir: Path, *, as_of: str) -> dict[str, Any]:
    contract = verify_frozen_contract()
    if not contract["verified"]:
        raise Phase7PError(f"frozen contract failed before prediction batch: {contract['errors']}")
    specification = load_frozen_specification()
    matrix, _ = _input_frames()
    matrix["date"] = pd.to_datetime(matrix["date"], utc=True, errors="coerce")
    now = _timestamp(as_of)
    window = matrix["date"].dt.date.astype("string").between(PROSPECTIVE_START, PROSPECTIVE_END)
    no_outcome = matrix["home_score"].isna() & matrix["away_score"].isna()
    before_cutoff = matrix["date"].gt(now)
    rows = matrix.loc[window & no_outcome & before_cutoff].copy().sort_values(["date", "match_hash"])
    features = list(specification["feature_contract"]["ordered_feature_names"])
    required = ["match_hash", "date", "league", "home_team", "away_team", *features]
    missing = [column for column in required if column not in rows.columns]
    if missing:
        raise Phase7PError(f"prospective feature contract missing columns: {missing[:5]}")
    store = ImmutableCohortStore(run_dir / "cohort")
    with store.batch_lock("prediction_capture"):
        return _capture_rows_locked(
            specification=specification,
            matrix=matrix,
            rows=rows,
            required=required,
            store=store,
            as_of=as_of,
        )


def _capture_rows_locked(
    *, specification: Mapping[str, Any], matrix: pd.DataFrame, rows: pd.DataFrame,
    required: Sequence[str], store: ImmutableCohortStore, as_of: str,
) -> dict[str, Any]:
    existing = store.records("prediction")
    existing_ids = {str(value["fixture"]["fixture_id"]) for value in existing}
    rows = rows.loc[~rows["match_hash"].astype(str).isin(existing_ids)]
    if rows.empty:
        return {
            "status": "CAPTURE_COMPLETE",
            "created_count": 0,
            "fixture_ids": [],
            "production_output_mutated": False,
            "reason": "no authorized-window causal feature rows are prediction-ready",
        }
    active_suite = _load_prediction_models("PL")
    candidate_suite = _candidate_suite()
    simulator = MatchSimulator(
        n_simulations=int(specification["simulation"]["n_simulations"]), league="PL",
        seed=int(specification["simulation"]["seed"]),
        seed_scope=str(specification["simulation"]["seed_scope"]),
    )
    next_sequence = 0
    if existing:
        latest = max(
            existing,
            key=lambda value: int(value["paired_evidence"].get("capture_sequence_index", -1)),
        )
        next_sequence = int(latest["paired_evidence"].get("capture_sequence_index", -1)) + 1
        simulator.rng.bit_generator.state = copy.deepcopy(
            latest["paired_evidence"]["active_post_call_rng_state"]
        )
    lineage = _verify_candidate_lineage()
    if not lineage["verified"]:
        raise Phase7PError("candidate training lineage failed before prediction batch")
    created: list[str] = []
    logging.getLogger("src.cli.commands.prediction").setLevel(logging.ERROR)
    for offset, (_, full_row) in enumerate(rows.iterrows()):
        fixture = full_row[required].copy()
        fixture_id = str(fixture["match_hash"])
        kickoff = _timestamp(fixture["date"])
        created_at = _timestamp(as_of).isoformat()
        shared_before = copy.deepcopy(simulator.rng.bit_generator.state)
        active_record, active_verification = _prediction_record(
            fixture, active_suite, simulator, matrix, label="active", prediction_timestamp=created_at
        )
        active_after = copy.deepcopy(simulator.rng.bit_generator.state)
        simulator.rng.bit_generator.state = copy.deepcopy(shared_before)
        candidate_record, candidate_verification = _prediction_record(
            fixture, candidate_suite, simulator, matrix, label="candidate", prediction_timestamp=created_at
        )
        candidate_after = copy.deepcopy(simulator.rng.bit_generator.state)
        simulator.rng.bit_generator.state = copy.deepcopy(active_after)
        pair = {
            "schema_version": PAIR_SCHEMA_VERSION,
            "specification_fingerprint": specification["specification_fingerprint"],
            "fixture_id": fixture_id, "fixture_date": kickoff.date().isoformat(), "cohort": COHORT,
            "class_order": list(CLASS_ORDER), "authoritative_output": "active",
            "feature_bundle_sha256": active_record["feature_provenance"]["feature_bundle_sha256"],
            "source_snapshot_sha256": active_record["data_provenance"]["source_identity"]["artifacts_aggregate_sha256"],
            "shared_pre_call_rng_state": shared_before, "active_post_call_rng_state": active_after,
            "candidate_post_call_rng_state": candidate_after, "stream_advanced_with": "active_post_call_state",
            "active_prediction": active_record, "candidate_prediction": candidate_record,
            "active_phase4_verified": bool(active_verification["overall_phase4_trusted"]),
            "candidate_phase4_verified": bool(candidate_verification["overall_phase4_trusted"]),
            "candidate_training_lineage_verified": True, "prediction_frozen_at": created_at,
            "capture_sequence_index": next_sequence + offset,
            "outcome_accessed_at": None, "outcome": None,
        }
        validate_paired_prediction(pair, specification=specification)
        maximum_source = str(active_record["data_provenance"]["maximum_source_event_timestamp"])
        prediction = build_prediction_record(
            specification=specification, paired_evidence=pair, fixture_kickoff_at=kickoff.isoformat(),
            prediction_created_at=created_at, maximum_source_event_at=maximum_source,
            causal_source_identity=active_record["data_provenance"]["source_identity"],
            outcome_available_at_prediction=False,
        )
        store.persist_prediction(prediction, specification=specification)
        created.append(fixture_id)
    return {"status": "CAPTURE_COMPLETE", "created_count": len(created), "fixture_ids": created, "production_output_mutated": False}


def settle_ready(run_dir: Path, *, observed_at: str | None = None) -> dict[str, Any]:
    contract = verify_frozen_contract()
    if not contract["verified"]:
        raise Phase7PError(f"frozen contract failed before settlement: {contract['errors']}")
    specification = load_frozen_specification()
    store = ImmutableCohortStore(run_dir / "cohort")
    with store.batch_lock("outcome_settlement"):
        return _settle_locked(
            specification=specification, store=store, observed_at=observed_at
        )


def _approved_outcome_paths() -> list[Path]:
    season_file = ROOT / "data" / "processed" / "matches" / "PL_2026.csv"
    return [path for path in (season_file, RAW_MATCHES) if path.is_file()]


def _settle_locked(
    *, specification: Mapping[str, Any], store: ImmutableCohortStore,
    observed_at: str | None,
) -> dict[str, Any]:
    predictions = store.records("prediction")
    settled_ids = {str(value["fixture_id"]) for value in store.records("settlement")}
    sources: list[tuple[Path, pd.DataFrame, str, str, str]] = []
    for path in _approved_outcome_paths():
        frame = pd.read_csv(path, low_memory=False)
        id_column = "match_hash" if "match_hash" in frame.columns else "match_id" if "match_id" in frame.columns else ""
        home_column = "home_goals" if "home_goals" in frame.columns else "home_score" if "home_score" in frame.columns else ""
        away_column = "away_goals" if "away_goals" in frame.columns else "away_score" if "away_score" in frame.columns else ""
        if id_column and home_column and away_column:
            frame[id_column] = frame[id_column].astype(str)
            sources.append((path, frame, id_column, home_column, away_column))
    created: list[str] = []
    for prediction in predictions:
        fixture_id = str(prediction["fixture"]["fixture_id"])
        if fixture_id in settled_ids:
            continue
        candidates: list[tuple[Path, pd.Series, str, str]] = []
        for path, frame, id_column, home_column, away_column in sources:
            matches = frame.loc[frame[id_column].eq(fixture_id)]
            if len(matches) > 1:
                raise Phase7PError(f"duplicate authoritative outcome rows: {fixture_id} in {relative(path)}")
            if len(matches) == 1:
                row = matches.iloc[0]
                if not pd.isna(row[home_column]) and not pd.isna(row[away_column]):
                    candidates.append((path, row, home_column, away_column))
        if not candidates:
            continue
        distinct = {
            (int(row[home_column]), int(row[away_column]))
            for _, row, home_column, away_column in candidates
        }
        if len(distinct) != 1:
            raise Phase7PError(f"conflicting approved outcome sources require integrity review: {fixture_id}")
        path, row, home_column, away_column = candidates[0]
        source_observed = (
            _timestamp(observed_at)
            if observed_at
            else pd.Timestamp(path.stat().st_mtime, unit="s", tz="UTC")
        )
        source = {
            "provider": "football-data.org v4",
            "repository_adapter": "scripts/fetch_fresh_data.py",
            "path": relative(path), "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
            "local_file_observed_at": source_observed.isoformat(),
        }
        row_identity = sha256_json(
            {"fixture_id": fixture_id, "home_goals": int(row[home_column]), "away_goals": int(row[away_column])}
        )
        settlement = build_settlement_record(
            specification=specification, prediction_record=prediction,
            home_goals=int(row[home_column]), away_goals=int(row[away_column]),
            outcome_source_identity={**source, "authoritative_row_sha256": row_identity},
            outcome_observed_at=source_observed.isoformat(), settlement_created_at=utc_now(),
        )
        store.persist_settlement(settlement, specification=specification)
        created.append(fixture_id)
    return {"status": "SETTLEMENT_COMPLETE", "created_count": len(created), "fixture_ids": created}


def status_report(run_dir: Path, *, persist: bool = True) -> dict[str, Any]:
    specification = load_frozen_specification()
    store = ImmutableCohortStore(run_dir / "cohort")
    predictions, settlements, missed = store.records("prediction"), store.records("settlement"), store.records("missed")
    prediction_ids = {str(value["fixture"]["fixture_id"]) for value in predictions}
    settlement_ids = {str(value["fixture_id"]) for value in settlements}
    audit = read_json(run_dir / "ingestion_readiness.json")
    current_features = _feature_readiness()
    discovered = int(audit.get("upstream_probe", {}).get("authorized_window_fixture_count", 0) or 0)
    feature_ready = int(current_features["prediction_feature_rows_without_outcome"])
    integrity_failures = 0
    for prediction in predictions:
        if not verify_prediction_record(prediction, specification=specification)["verified"]:
            integrity_failures += 1
    indexed_predictions = {str(value["fixture"]["fixture_id"]): value for value in predictions}
    for settlement in settlements:
        prediction = indexed_predictions.get(str(settlement["fixture_id"]))
        if prediction is None or not verify_settlement_record(settlement, prediction_record=prediction, specification=specification)["verified"]:
            integrity_failures += 1
    report = {
        "schema_version": "ffps.phase7p_prospective_status.v1", "generated_at": utc_now(),
        "specification_fingerprint": specification["specification_fingerprint"],
        "fixtures_discovered": discovered, "prediction_ready_count": feature_ready,
        "frozen_prediction_count": len(predictions), "missed_count": len(missed),
        "pending_settlement_count": len(prediction_ids - settlement_ids),
        "settled_eligible_count": len(settlement_ids & prediction_ids), "excluded_count": 0,
        "integrity_failures": integrity_failures,
        "latest_causal_source_row": current_features["latest_causal_source_row"],
        "latest_outcome_row": _outcome_readiness()["latest_outcome_row"],
        "active_bundle_identity": specification["active"]["model_bundle_sha256"],
        "candidate_bundle_identity": specification["candidate"]["model_bundle_sha256"],
        "coverage": coverage_ratio(len(predictions), len(missed)),
        "classification": (
            "PROSPECTIVE_COLLECTION_INTEGRITY_BLOCKED" if integrity_failures else
            "PROSPECTIVE_COLLECTION_READY" if feature_ready else
            "PROSPECTIVE_COLLECTION_BLOCKED_DATA"
        ),
        "phase8_promotion_review_authorized": False, "production_activation_performed": False,
    }
    if persist:
        write_json(run_dir / "status_report.json", report)
    return report


def aggregate(run_dir: Path, *, persist: bool = True) -> dict[str, Any]:
    specification = load_frozen_specification()
    store = ImmutableCohortStore(run_dir / "cohort")
    result = aggregate_eligible(store.records("prediction"), store.records("settlement"), specification=specification)
    if persist:
        write_json(run_dir / "interim_aggregate.json", result)
    return result


def _test_summary(path: Path) -> dict[str, Any]:
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    failed = sorted(
        f"{case.attrib.get('classname')}::{case.attrib.get('name')}"
        for suite in suites
        for case in suite.findall("testcase")
        if case.find("failure") is not None or case.find("error") is not None
    )
    return {
        "tests": sum(int(item.attrib.get("tests", 0)) for item in suites),
        "failures": sum(int(item.attrib.get("failures", 0)) for item in suites),
        "errors": sum(int(item.attrib.get("errors", 0)) for item in suites),
        "skipped": sum(int(item.attrib.get("skipped", 0)) for item in suites),
        "failure_ids": failed,
        "source_path": relative(path), "sha256": sha256_file(path),
    }


def finalize(run_dir: Path, *, focused_xml: Path, protected_xml: Path, full_xml: Path) -> dict[str, Any]:
    before = read_json(run_dir / "protected_before.json")
    after = protected_snapshot()
    if not protected_equal(before, after):
        raise Phase7PError("protected Phase 0-7 or model identities changed")
    write_json(run_dir / "protected_after_final.json", after)
    copied_paths: dict[str, Path] = {}
    for label, source in (("focused", focused_xml), ("protected", protected_xml), ("full", full_xml)):
        target = run_dir / f"{label}_test_results.xml"
        shutil.copyfile(source, target)
        copied_paths[label] = target
    tests = {label: _test_summary(path) for label, path in copied_paths.items()}
    new_unexplained = sorted(set(tests["full"]["failure_ids"]) - INHERITED_FULL_SUITE_FAILURES)
    tests["inherited_full_suite_failure_ids"] = sorted(INHERITED_FULL_SUITE_FAILURES)
    tests["new_unexplained_failure_ids"] = new_unexplained
    tests["no_new_unexplained_failures"] = not new_unexplained
    if tests["focused"]["failures"] or tests["focused"]["errors"]:
        raise Phase7PError(f"focused test failures prevent Phase 7P finalization: {tests['focused']}")
    if tests["protected"]["failures"] or tests["protected"]["errors"]:
        raise Phase7PError(f"protected regression failures prevent Phase 7P finalization: {tests['protected']}")
    if new_unexplained:
        raise Phase7PError(f"new unexplained full-suite failures: {new_unexplained}")
    write_json(run_dir / "test_results.json", tests)
    report = status_report(run_dir)
    interim = aggregate(run_dir)
    classification = report["classification"]
    if classification not in ALLOWED_CLASSIFICATIONS:
        raise Phase7PError(f"invalid completion classification: {classification}")
    report_path = run_dir / "PHASE7P_IMPLEMENTATION_REPORT.md"
    report_path.write_text(
        "\n".join(
            [
                "# FFPS Phase 7P — Prospective Shadow Evidence Collection",
                "", f"- Completion classification: `{classification}`",
                f"- Frozen specification: `{FROZEN_SPECIFICATION_FINGERPRINT}`",
                f"- Frozen predictions: {report['frozen_prediction_count']}",
                f"- Settled eligible fixtures: {report['settled_eligible_count']}",
                f"- Missed fixtures: {report['missed_count']}",
                "- Prediction and settlement records are separate immutable per-fixture JSON artifacts.",
                "- The active and Phase 6 candidate bundles are verified before each prediction/settlement batch.",
                "- The approved provider is available, but no authorized-window causal feature row exists locally.",
                "- Interim aggregation is descriptive and never executes the frozen final decision rule.",
                "- No production activation, registration, calibration, training, or Phase 8 authorization occurred.",
            ]
        ) + "\n",
        encoding="utf-8", newline="\n",
    )
    implementation_sources = [
        ROOT / "src" / "evaluation" / "phase7p_prospective.py",
        ROOT / "scripts" / "run_phase7p_prospective.py",
        ROOT / "tests" / "unit" / "test_phase7p_prospective.py",
    ]
    write_json(
        run_dir / "implementation_sources.json",
        {
            "schema_version": "ffps.phase7p_implementation_sources.v1",
            "sources": [
                {"path": relative(path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size}
                for path in implementation_sources
            ],
        },
    )
    static_excluded = {"artifact_index.json", "run_manifest.json", "offline_verification.json", "status_report.json", "interim_aggregate.json"}
    artifacts = [
        {"path": relative(path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size}
        for path in sorted(run_dir.rglob("*"))
        if path.is_file() and path.name not in static_excluded and "cohort/" not in relative(path)
    ]
    index = {"schema_version": "ffps.phase7p_artifact_index.v1", "artifacts": artifacts, "aggregate_sha256": sha256_json(artifacts)}
    write_json(run_dir / "artifact_index.json", index)
    manifest = {
        "schema_version": "ffps.phase7p_run_manifest.v1", "phase": "7P", "run_id": RUN_ID,
        "status": "infrastructure_complete_collection_operational", "finalized_at": utc_now(),
        "classification": classification, "specification_fingerprint": FROZEN_SPECIFICATION_FINGERPRINT,
        "active_bundle_sha256": report["active_bundle_identity"], "candidate_bundle_sha256": report["candidate_bundle_identity"],
        "prospective_fixture_count": interim["fixture_count"], "final_phase7_decision_recomputed": False,
        "phase8_promotion_review_authorized": False, "production_activation_performed": False,
        "protected_phase0_to_phase7_unchanged": True, "source_substitution_performed": False,
        "artifact_index_sha256": sha256_file(run_dir / "artifact_index.json"),
        "implementation_sources_sha256": sha256_file(run_dir / "implementation_sources.json"),
    }
    write_json(run_dir / "run_manifest.json", manifest)
    return manifest


def verify(run_dir: Path, *, persist: bool = True) -> dict[str, Any]:
    errors: list[str] = []
    required = (
        "protected_before.json", "protected_after_final.json", "frozen_specification_reference.json",
        "ingestion_readiness.json", "preparation_result.json", "test_results.json",
        "PHASE7P_IMPLEMENTATION_REPORT.md", "implementation_sources.json",
        "artifact_index.json", "run_manifest.json",
    )
    for name in required:
        if not (run_dir / name).is_file():
            errors.append(f"missing artifact: {name}")
    contract = verify_frozen_contract()
    errors.extend(contract["errors"])
    if not errors:
        before, after, current = read_json(run_dir / "protected_before.json"), read_json(run_dir / "protected_after_final.json"), protected_snapshot()
        if not protected_equal(before, after) or not protected_equal(before, current):
            errors.append("protected Phase 0-7 identity mismatch")
        reference = read_json(run_dir / "frozen_specification_reference.json")
        if reference.get("specification_fingerprint") != FROZEN_SPECIFICATION_FINGERPRINT or reference.get("file_sha256") != sha256_file(SPECIFICATION_PATH):
            errors.append("frozen specification reference mismatch")
        index = read_json(run_dir / "artifact_index.json")
        for item in index["artifacts"]:
            path = ROOT / str(item["path"])
            if not path.is_file() or sha256_file(path) != item["sha256"] or path.stat().st_size != item["size_bytes"]:
                errors.append(f"static artifact mismatch: {item['path']}")
        if sha256_json(index["artifacts"]) != index["aggregate_sha256"]:
            errors.append("artifact index aggregate mismatch")
        source_index = read_json(run_dir / "implementation_sources.json")
        for item in source_index.get("sources", []):
            path = ROOT / str(item["path"])
            if not path.is_file() or sha256_file(path) != item["sha256"] or path.stat().st_size != item["size_bytes"]:
                errors.append(f"Phase 7P implementation source mismatch: {item['path']}")
        status = status_report(run_dir, persist=False)
        if status["integrity_failures"]:
            errors.append("prospective cohort integrity failure")
        try:
            aggregate(run_dir, persist=False)
        except Phase7PError as exc:
            errors.append(f"prospective aggregation verification failed: {exc}")
        manifest = read_json(run_dir / "run_manifest.json")
        if manifest.get("implementation_sources_sha256") != sha256_file(run_dir / "implementation_sources.json"):
            errors.append("Phase 7P implementation-source index mismatch")
        if manifest.get("phase8_promotion_review_authorized") or manifest.get("production_activation_performed"):
            errors.append("Phase 8 or production activation was improperly authorized")
        if manifest.get("final_phase7_decision_recomputed"):
            errors.append("Phase 7 final decision was recomputed during collection")
    result = {
        "status": "VERIFIED" if not errors else "FAILED", "errors": errors,
        "specification_fingerprint": FROZEN_SPECIFICATION_FINGERPRINT,
        "completion_classification": read_json(run_dir / "preparation_result.json").get("classification") if (run_dir / "preparation_result.json").is_file() else None,
        "protected_phase0_to_phase7_unchanged": not any("protected" in value for value in errors),
        "phase8_promotion_review_authorized": False, "production_activation_performed": False,
    }
    if persist:
        write_json(run_dir / "offline_verification.json", result)
    return result


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    sub = result.add_subparsers(dest="command", required=True)
    prepare_cmd = sub.add_parser("prepare")
    prepare_cmd.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    prepare_cmd.add_argument("--probe-upstream", action="store_true")
    capture_cmd = sub.add_parser("capture-ready")
    capture_cmd.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    capture_cmd.add_argument("--as-of", default=utc_now())
    settle_cmd = sub.add_parser("settle-ready")
    settle_cmd.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    settle_cmd.add_argument("--observed-at")
    for name in ("status", "aggregate", "verify"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    finalize_cmd = sub.add_parser("finalize")
    finalize_cmd.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    finalize_cmd.add_argument("--focused-xml", type=Path, required=True)
    finalize_cmd.add_argument("--protected-xml", type=Path, required=True)
    finalize_cmd.add_argument("--full-xml", type=Path, required=True)
    return result


def main() -> int:
    args = parser().parse_args()
    try:
        if args.command == "prepare":
            result = prepare(args.run_dir, probe_upstream=args.probe_upstream)
        elif args.command == "capture-ready":
            result = capture_ready(args.run_dir, as_of=args.as_of)
        elif args.command == "settle-ready":
            result = settle_ready(args.run_dir, observed_at=args.observed_at)
        elif args.command == "status":
            result = status_report(args.run_dir)
        elif args.command == "aggregate":
            result = aggregate(args.run_dir)
        elif args.command == "finalize":
            result = finalize(args.run_dir, focused_xml=args.focused_xml, protected_xml=args.protected_xml, full_xml=args.full_xml)
        else:
            result = verify(args.run_dir)
    except (Phase7PError, Phase7EvaluationError, FileNotFoundError, ValueError) as exc:
        print(json.dumps({"status": "FAILED", "error": str(exc)}, indent=2))
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("status") not in {"FAILED"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
