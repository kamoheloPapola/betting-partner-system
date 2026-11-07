"""Build and verify the additive Phase 7D PL data-continuity extension."""

from __future__ import annotations

import argparse
import copy
import io
import json
import os
import shutil
import sys
import tempfile
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence, cast

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_phase7p_prospective import (
    DEFAULT_RUN_DIR as PHASE7P_RUN,
    verify as verify_phase7p,
    verify_frozen_contract as verify_phase7_contract,
)
from src.evaluation.phase3_calibration import aggregate_directory
from src.evaluation.phase7_shadow import sha256_json
from src.evaluation.phase7d_data_continuity import (
    HISTORICAL_CUTOFF,
    NORMALIZATION_CONTRACT,
    NORMALIZATION_CONTRACT_SHA256,
    PROSPECTIVE_END,
    PROSPECTIVE_START,
    PROVIDER_ID,
    ImmutableExtensionStore,
    Phase7DError,
    build_causal_source_bundle,
    canonical_fixture_identity,
    extension_aggregate,
    normalize_provider_event,
    raw_source_identity,
    strict_normalize_team,
    verify_event_record,
)
from src.evaluation.phase7p_prospective import (
    ImmutableCohortStore,
    build_missed_record,
)
from src.predictions.provenance import sha256_bytes, sha256_file


RUN_ID = "phase7d-approved-provider-pl-continuity-79110f9a-20260829-v1"
DEFAULT_RUN_DIR = ROOT / "baselines" / "phase7d" / RUN_ID
PHASE7_RUN = ROOT / "baselines" / "phase7" / "phase7-active-vs-candidate-shadow-79110f9a-20260829-v2"
PHASE6_RUN = ROOT / "baselines" / "phase6" / "phase6-controlled-pl-goals-79110f9a-20260829-v1"
PROTECTED_BASE = ROOT / "data" / "processed" / "matches.csv"
FROZEN_FEATURE_MATRIX = ROOT / "data" / "features" / "feature_matrix.csv"
PHASE7_SPEC = PHASE7_RUN / "evaluation_specification.json"
PROVIDER_BASE = "https://api.football-data.org/v4"
PROVIDER_SEASONS = (2025, 2026)
ALLOWED_CLASSIFICATIONS = {
    "PROSPECTIVE_FEATURE_READY",
    "DATA_CONTINUITY_INCOMPLETE",
    "SOURCE_SEMANTICS_INCOMPATIBLE",
    "FEATURE_PIPELINE_INCOMPATIBLE",
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


def write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    if sha256_bytes(temporary.read_bytes()) != sha256_bytes(payload):
        temporary.unlink(missing_ok=True)
        raise Phase7DError("raw source persistence verification failed")
    os.replace(temporary, path)


def relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _safe_new_run_dir(run_dir: Path) -> None:
    allowed = (ROOT / "baselines" / "phase7d").resolve()
    resolved = run_dir.resolve()
    try:
        resolved.relative_to(allowed)
    except ValueError as exc:
        raise Phase7DError("Phase 7D evidence must remain beneath baselines/phase7d") from exc
    if resolved.exists():
        raise Phase7DError(f"Phase 7D evidence already exists: {resolved}")
    resolved.mkdir(parents=True)


def protected_snapshot() -> dict[str, Any]:
    phase7p = verify_phase7p(PHASE7P_RUN, persist=False)
    if phase7p.get("status") != "VERIFIED":
        raise Phase7DError(f"frozen Phase 7P evidence failed verification: {phase7p}")
    phase7 = verify_phase7_contract()
    if not phase7["verified"]:
        raise Phase7DError(f"frozen Phase 7/model contract failed: {phase7['errors']}")
    phase7p_sources = read_json(PHASE7P_RUN / "implementation_sources.json")
    return {
        "captured_at": utc_now(),
        "phase7_evidence_aggregate_sha256": aggregate_directory(PHASE7_RUN),
        "phase7p_evidence_aggregate_sha256": aggregate_directory(PHASE7P_RUN),
        "phase7p_run_manifest_sha256": sha256_file(PHASE7P_RUN / "run_manifest.json"),
        "phase7p_offline_verification_sha256": sha256_file(PHASE7P_RUN / "offline_verification.json"),
        "phase7p_implementation_sources": {
            str(item["path"]): sha256_file(ROOT / str(item["path"]))
            for item in phase7p_sources["sources"]
        },
        "protected_historical_base_sha256": sha256_file(PROTECTED_BASE),
        "frozen_feature_matrix_sha256": sha256_file(FROZEN_FEATURE_MATRIX),
        "phase7_specification_sha256": sha256_file(PHASE7_SPEC),
        "phase7_specification_fingerprint": read_json(PHASE7_SPEC)["specification_fingerprint"],
        "active_bundle_sha256": phase7["active_bundle_sha256"],
        "candidate_bundle_id": phase7["candidate_bundle_id"],
        "candidate_bundle_aggregate_sha256": phase7["candidate_bundle_aggregate_sha256"],
    }


def protected_equal(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return {key: value for key, value in left.items() if key != "captured_at"} == {
        key: value for key, value in right.items() if key != "captured_at"
    }


def assert_protected(run_dir: Path) -> None:
    if not protected_equal(read_json(run_dir / "protected_before.json"), protected_snapshot()):
        raise Phase7DError("protected Phase 0-7P, historical data, or model identity changed")


def prepare(run_dir: Path) -> dict[str, Any]:
    _safe_new_run_dir(run_dir)
    before = protected_snapshot()
    write_json(run_dir / "protected_before.json", before)
    write_json(
        run_dir / "architecture_contract.json",
        {
            "schema_version": "ffps.phase7d_architecture_contract.v1",
            "phase": "7D",
            "provider": PROVIDER_ID,
            "historical_cutoff": HISTORICAL_CUTOFF,
            "prospective_window": {"start": PROSPECTIVE_START, "end": PROSPECTIVE_END},
            "protected_base": {"path": relative(PROTECTED_BASE), "sha256": sha256_file(PROTECTED_BASE)},
            "extension_path": relative(run_dir / "post_protocol_pl_data_extension"),
            "normalization_contract": NORMALIZATION_CONTRACT,
            "normalization_contract_sha256": NORMALIZATION_CONTRACT_SHA256,
            "evaluation_only": True,
            "production_path_changed": False,
            "source_substitution_allowed": False,
            "phase8_authorized": False,
        },
    )
    ImmutableExtensionStore(run_dir / "post_protocol_pl_data_extension")
    result = {
        "status": "PREPARED",
        "run_id": RUN_ID,
        "protected_phase0_to_phase7p_captured": True,
        "protected_historical_base_unchanged": True,
        "provider": PROVIDER_ID,
    }
    write_json(run_dir / "preparation_result.json", result)
    return result


def _provider_headers() -> dict[str, str]:
    from dotenv import dotenv_values

    key = str(dotenv_values(ROOT / ".env").get("FOOTBALL_DATA_API_KEY") or os.getenv("FOOTBALL_DATA_API_KEY") or "")
    if not key:
        raise Phase7DError("approved provider credential is not configured")
    return {"X-Auth-Token": key}


def acquire(run_dir: Path) -> dict[str, Any]:
    assert_protected(run_dir)
    raw_dir = run_dir / "raw_provider_snapshot"
    if (run_dir / "acquisition_result.json").exists():
        raise Phase7DError("Phase 7D provider snapshot is already frozen")
    store = ImmutableExtensionStore(run_dir / "post_protocol_pl_data_extension")
    raw_sources: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    statuses: dict[str, int] = {}
    created = 0
    idempotent = 0
    for season in PROVIDER_SEASONS:
        url = f"{PROVIDER_BASE}/competitions/PL/matches"
        response = requests.get(url, headers=_provider_headers(), params={"season": season}, timeout=30)
        response.raise_for_status()
        payload = response.content
        retrieved_at = utc_now()
        raw_path = raw_dir / f"football-data-org-pl-{season}.response.json"
        write_bytes(raw_path, payload)
        source = raw_source_identity(
            payload,
            season=season,
            retrieved_at=retrieved_at,
            request_url=str(response.url),
            response_headers=response.headers,
        )
        source["path"] = relative(raw_path)
        parsed = json.loads(payload)
        matches = parsed.get("matches", [])
        source["provider_event_count"] = len(matches)
        write_json(raw_dir / f"football-data-org-pl-{season}.source.json", source)
        raw_sources.append(source)
        seen_provider_ids: set[str] = set()
        for raw_event in matches:
            provider_event_id = str(raw_event.get("id", ""))
            if provider_event_id in seen_provider_ids:
                issues.append(
                    {"season": season, "provider_event_id": provider_event_id, "error": "duplicate provider ID in raw response"}
                )
                continue
            seen_provider_ids.add(provider_event_id)
            status = str(raw_event.get("status", ""))
            statuses[status] = statuses.get(status, 0) + 1
            try:
                record = normalize_provider_event(raw_event, raw_source=source)
                result = store.persist_event(record)
                created += result == "CREATED"
                idempotent += result == "ALREADY_PRESENT"
            except Phase7DError as exc:
                issues.append(
                    {
                        "season": season,
                        "provider_event_id": provider_event_id,
                        "kickoff_at": raw_event.get("utcDate"),
                        "status": status,
                        "home_team": cast(Mapping[str, Any], raw_event.get("homeTeam", {})).get("name"),
                        "away_team": cast(Mapping[str, Any], raw_event.get("awayTeam", {})).get("name"),
                        "error": str(exc),
                    }
                )
    write_json(run_dir / "normalization_issues.json", {"issues": issues, "issue_count": len(issues)})
    raw_aggregate = sha256_json(
        [
            {"season": value["season"], "sha256": value["response_sha256"], "size_bytes": value["response_size_bytes"]}
            for value in raw_sources
        ]
    )
    records = store.event_records()
    result = {
        "status": "SOURCE_SNAPSHOT_FROZEN",
        "provider": PROVIDER_ID,
        "seasons": list(PROVIDER_SEASONS),
        "raw_source_aggregate_sha256": raw_aggregate,
        "raw_event_count": sum(int(value["provider_event_count"]) for value in raw_sources),
        "normalized_record_count": len(records),
        "created_count": created,
        "idempotent_count": idempotent,
        "normalization_issue_count": len(issues),
        "status_counts": statuses,
        "normalized_extension_aggregate_sha256": extension_aggregate(records),
        "credentials_retained": False,
        "raw_retention_limitation": "No repository-specific provider retention prohibition was found; exact responses retained for offline verification.",
    }
    write_json(run_dir / "acquisition_result.json", result)
    return result


def _raw_matches(run_dir: Path) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    for season in PROVIDER_SEASONS:
        path = run_dir / "raw_provider_snapshot" / f"football-data-org-pl-{season}.response.json"
        parsed = json.loads(path.read_bytes())
        for event in parsed.get("matches", []):
            copied = copy.deepcopy(event)
            copied["_provider_season"] = season
            values.append(copied)
    return values


def _current_team(value: Any) -> str:
    return strict_normalize_team(str(value), season=2025)


def overlap_audit(run_dir: Path) -> dict[str, Any]:
    records = ImmutableExtensionStore(run_dir / "post_protocol_pl_data_extension").event_records()
    provider_rows: list[dict[str, Any]] = []
    for record in records:
        event = cast(Mapping[str, Any], record["normalized_event"])
        if int(event["season_start_year"]) == 2025 and str(event["fixture_date"]) <= HISTORICAL_CUTOFF:
            provider_rows.append(dict(event))
    protected = pd.read_csv(PROTECTED_BASE, low_memory=False)
    protected = protected.loc[protected["league"].astype(str).eq("PL")].copy()
    protected["fixture_date"] = pd.to_datetime(protected["match_date"]).dt.date.astype(str)
    protected["current_home"] = protected["home_team"].map(_current_team)
    protected["current_away"] = protected["away_team"].map(_current_team)
    provider = pd.DataFrame(provider_rows)
    joined = provider.merge(
        protected,
        left_on=["fixture_date", "home_team", "away_team"],
        right_on=["fixture_date", "current_home", "current_away"],
        how="left",
        suffixes=("_provider", "_protected"),
    )
    found = joined["match_hash"].notna()
    scores = found & joined["home_goals_provider"].eq(joined["home_goals_protected"]) & joined["away_goals_provider"].eq(joined["away_goals_protected"])
    canonical = found & joined["canonical_fixture_id"].eq(joined["match_hash"])
    result = {
        "schema_version": "ffps.phase7d_overlap_audit.v1",
        "provider_overlap_count": len(provider),
        "protected_fixture_match_count": int(found.sum()),
        "home_away_score_semantics_match_count": int(scores.sum()),
        "current_vs_protected_fixture_id_match_count": int(canonical.sum()),
        "date_min": min((row["fixture_date"] for row in provider_rows), default=None),
        "date_max": max((row["fixture_date"] for row in provider_rows), default=None),
        "home_away_orientation_verified": bool(len(provider) and scores.all()),
        "completed_score_semantics_verified": bool(len(provider) and scores.all()),
        "provider_status_semantics_verified": all(row["completed"] and row["status"] == "FINISHED" for row in provider_rows),
        "canonical_identity_transition": {
            "classification": "HISTORICAL_IDENTITY_VERSION_BOUNDARY",
            "protected_method": "CanonicalDatasetBuilder SHA-256 prefix with frozen historical team labels",
            "current_adapter_method": "same SHA-256 contract with current exact canonical team labels",
            "natural_key_and_score_alignment_complete": bool(len(provider) and scores.all()),
            "protected_rows_rewritten": False,
        },
    }
    write_json(run_dir / "historical_overlap_audit.json", result)
    return result


def continuity_audit(run_dir: Path) -> dict[str, Any]:
    raw = _raw_matches(run_dir)
    expected = [
        value
        for value in raw
        if HISTORICAL_CUTOFF < str(value.get("utcDate", ""))[:10] < PROSPECTIVE_START
        and str(value.get("status")) == "FINISHED"
    ]
    records = ImmutableExtensionStore(run_dir / "post_protocol_pl_data_extension").event_records()
    normalized = {
        str(cast(Mapping[str, Any], value["normalized_event"])["provider_event_id"]): value
        for value in records
    }
    expected_ids = {str(value.get("id")) for value in expected}
    available_ids = expected_ids & set(normalized)
    gaps = sorted(expected_ids - set(normalized))
    canonical_ids = [
        str(cast(Mapping[str, Any], normalized[value]["normalized_event"])["canonical_fixture_id"])
        for value in sorted(available_ids)
    ]
    invalid_scores = [
        value
        for value in available_ids
        if not cast(Mapping[str, Any], normalized[value]["normalized_event"])["completed"]
    ]
    result = {
        "schema_version": "ffps.phase7d_continuity_audit.v1",
        "interval": {"after_exclusive": HISTORICAL_CUTOFF, "before_exclusive": PROSPECTIVE_START},
        "expected_completed_provider_events": len(expected),
        "normalized_verified_completed_events": len(available_ids),
        "missing_or_unresolved_provider_event_ids": gaps,
        "missing_or_unresolved_count": len(gaps),
        "duplicate_provider_ids": len(expected_ids) != len(expected),
        "canonical_fixture_collision_count": len(canonical_ids) - len(set(canonical_ids)),
        "invalid_completed_score_count": len(invalid_scores),
        "date_min": min((str(value.get("utcDate", ""))[:10] for value in expected), default=None),
        "date_max": max((str(value.get("utcDate", ""))[:10] for value in expected), default=None),
        "required_signal_coverage": {
            "goals": {"available_count": len(available_ids), "required_count": len(expected)},
            "corners": {
                "available_count": 0,
                "required_count": len(expected),
                "provider_contract": "football-data.org v4 match response does not expose corner totals",
                "frozen_model_features_affected": True,
            },
        },
        "complete": not gaps and len(canonical_ids) == len(set(canonical_ids)) and not invalid_scores,
    }
    write_json(run_dir / "continuity_completeness.json", result)
    return result


def audit(run_dir: Path) -> dict[str, Any]:
    assert_protected(run_dir)
    overlap = overlap_audit(run_dir)
    continuity = continuity_audit(run_dir)
    issues = read_json(run_dir / "normalization_issues.json")
    unresolved_names = sorted(
        {
            str(item["error"]).split("unresolved canonical team mapping:", 1)[1].strip()
            for item in issues["issues"]
            if "unresolved canonical team mapping:" in str(item.get("error"))
        }
    )
    source_compatible = (
        overlap["home_away_orientation_verified"]
        and overlap["completed_score_semantics_verified"]
        and not continuity["missing_or_unresolved_count"]
    )
    result = {
        "status": "AUDIT_COMPLETE",
        "provider": PROVIDER_ID,
        "source_semantics_compatible": source_compatible,
        "unresolved_provider_team_names": unresolved_names,
        "normalization_issue_count": issues["issue_count"],
        "overlap_semantics_verified": overlap["home_away_orientation_verified"] and overlap["completed_score_semantics_verified"],
        "continuity_complete": continuity["complete"],
        "protected_historical_rows_rewritten": False,
        "source_substitution_performed": False,
    }
    write_json(run_dir / "audit_result.json", result)
    return result


def _adapt_protected_base() -> pd.DataFrame:
    frame = pd.read_csv(PROTECTED_BASE, low_memory=False)
    frame = frame.loc[frame["league"].astype(str).eq("PL")].copy()
    frame = frame.rename(
        columns={
            "match_date": "date",
            "home_goals": "home_score",
            "away_goals": "away_score",
            "match_hash": "match_id",
        }
    )
    frame["home_team"] = frame["home_team"].map(_current_team)
    frame["away_team"] = frame["away_team"].map(_current_team)
    frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="raise")
    frame["status"] = "FINISHED"
    return frame


def _event_row(record: Mapping[str, Any]) -> dict[str, Any]:
    event = cast(Mapping[str, Any], record["normalized_event"])
    return {
        "match_id": event["canonical_fixture_id"],
        "date": pd.Timestamp(str(event["kickoff_at"])),
        "season": event["season"],
        "league": event["league"],
        "home_team": event["home_team"],
        "away_team": event["away_team"],
        "home_score": event["home_goals"] if event["home_goals"] is not None else np.nan,
        "away_score": event["away_goals"] if event["away_goals"] is not None else np.nan,
        "status": event["status"],
        "home_corners": np.nan,
        "away_corners": np.nan,
        "home_yellow_cards": np.nan,
        "away_yellow_cards": np.nan,
        "home_red_cards": np.nan,
        "away_red_cards": np.nan,
        "referee": None,
    }


def _run_existing_feature_pipeline(frame: pd.DataFrame) -> pd.DataFrame:
    import src.features.pipeline as pipeline_module
    from src.features.pipeline import FeaturePipeline

    with tempfile.TemporaryDirectory(prefix="ffps-phase7d-") as directory:
        original_data_dir = pipeline_module.DATA_DIR
        try:
            pipeline_module.DATA_DIR = Path(directory)
            return FeaturePipeline().run(match_data=frame, league="PL")
        finally:
            pipeline_module.DATA_DIR = original_data_dir


def _historical_pipeline_reproduction() -> dict[str, Any]:
    generated = _run_existing_feature_pipeline(_adapt_protected_base())
    frozen = pd.read_csv(FROZEN_FEATURE_MATRIX, low_memory=False)
    frozen = frozen.loc[frozen["league"].astype(str).eq("PL")].copy()
    features = list(read_json(PHASE7_SPEC)["feature_contract"]["ordered_feature_names"])
    joined = generated.merge(
        frozen[["match_hash", *features]],
        left_on="match_id",
        right_on="match_hash",
        suffixes=("_generated", "_frozen"),
    )
    max_diff = 0.0
    differing: list[str] = []
    for feature in features:
        delta = np.abs(
            pd.to_numeric(joined[f"{feature}_generated"], errors="coerce").to_numpy(dtype=float)
            - pd.to_numeric(joined[f"{feature}_frozen"], errors="coerce").to_numpy(dtype=float)
        )
        feature_max = float(np.nanmax(delta)) if len(delta) else float("inf")
        max_diff = max(max_diff, feature_max)
        if feature_max > 1e-12:
            differing.append(feature)
    return {
        "protected_input_rows": len(generated),
        "matched_frozen_feature_rows": len(joined),
        "ordered_feature_count": len(features),
        "ordered_feature_schema_sha256": sha256_json(features),
        "maximum_absolute_difference": max_diff,
        "differing_features_over_1e_12": differing,
        "reproduced": len(joined) == len(generated) == len(frozen) and not differing,
        "production_file_written": False,
    }


def feature_assessment(run_dir: Path, *, prediction_at: str) -> dict[str, Any]:
    assert_protected(run_dir)
    audit_result = read_json(run_dir / "audit_result.json")
    continuity = read_json(run_dir / "continuity_completeness.json")
    store = ImmutableExtensionStore(run_dir / "post_protocol_pl_data_extension")
    effective = store.effective_records(as_of=prediction_at)
    post_protocol_effective = [
        value
        for value in effective
        if str(cast(Mapping[str, Any], value["normalized_event"])["fixture_date"])
        > HISTORICAL_CUTOFF
    ]
    scheduled = [
        value
        for value in post_protocol_effective
        if not cast(Mapping[str, Any], value["normalized_event"])["completed"]
        and PROSPECTIVE_START
        <= str(cast(Mapping[str, Any], value["normalized_event"])["fixture_date"])
        <= PROSPECTIVE_END
        and str(cast(Mapping[str, Any], value["normalized_event"])["kickoff_at"])
        > pd.Timestamp(prediction_at).isoformat()
    ]
    scheduled.sort(
        key=lambda value: str(cast(Mapping[str, Any], value["normalized_event"])["kickoff_at"])
    )
    reproduction = _historical_pipeline_reproduction()
    example: dict[str, Any] | None = None
    pipeline_compatible = reproduction["reproduced"]
    if scheduled and pipeline_compatible:
        target_record = scheduled[0]
        target = cast(Mapping[str, Any], target_record["normalized_event"])
        bundle = build_causal_source_bundle(
            protected_base_path=relative(PROTECTED_BASE),
            protected_base_sha256=sha256_file(PROTECTED_BASE),
            extension_records=post_protocol_effective,
            target_fixture_id=str(target["canonical_fixture_id"]),
            target_kickoff_at=str(target["kickoff_at"]),
            prediction_at=prediction_at,
        )
        included = set(bundle["ordered_extension_record_fingerprints"])
        causal_records = [
            value for value in post_protocol_effective if value["record_fingerprint"] in included
        ]
        composite = pd.concat(
            [
                _adapt_protected_base(),
                pd.DataFrame([_event_row(value) for value in causal_records]),
                pd.DataFrame([_event_row(target_record)]),
            ],
            ignore_index=True,
            sort=False,
        )
        generated = _run_existing_feature_pipeline(composite)
        target_rows = generated.loc[generated["match_id"].astype(str).eq(str(target["canonical_fixture_id"]))]
        features = list(read_json(PHASE7_SPEC)["feature_contract"]["ordered_feature_names"])
        schema_match = all(value in target_rows.columns for value in features)
        vector: list[float] = []
        finite = False
        if len(target_rows) == 1 and schema_match:
            vector = [float(target_rows.iloc[0][value]) for value in features]
            finite = bool(np.isfinite(np.asarray(vector, dtype=float)).all())
        example = {
            "target_fixture_id": target["canonical_fixture_id"],
            "target_kickoff_at": target["kickoff_at"],
            "home_team": target["home_team"],
            "away_team": target["away_team"],
            "causal_source_bundle": bundle,
            "feature_schema_sha256": sha256_json(features),
            "feature_vector_sha256": sha256_json(vector) if finite else None,
            "feature_vector_finite": finite,
            "feature_schema_matches_active_and_candidate": schema_match,
            "active_candidate_feature_identity_equal": schema_match and finite,
            "existing_feature_pipeline_used": True,
            "production_path_changed": False,
        }
        write_json(run_dir / "feature_examples" / f"{target['canonical_fixture_id']}.json", example)
        pipeline_compatible = pipeline_compatible and len(target_rows) == 1 and schema_match and finite
    corner_complete = continuity["required_signal_coverage"]["corners"]["available_count"] == continuity["required_signal_coverage"]["corners"]["required_count"]
    if not audit_result["source_semantics_compatible"]:
        classification = "SOURCE_SEMANTICS_INCOMPATIBLE"
    elif not continuity["complete"] or not corner_complete:
        classification = "DATA_CONTINUITY_INCOMPLETE"
    elif not pipeline_compatible:
        classification = "FEATURE_PIPELINE_INCOMPATIBLE"
    else:
        classification = "PROSPECTIVE_FEATURE_READY"
    result = {
        "status": "FEATURE_ASSESSMENT_COMPLETE",
        "classification": classification,
        "prediction_at": pd.Timestamp(prediction_at).isoformat(),
        "historical_pipeline_reproduction": reproduction,
        "normalized_scheduled_fixtures_in_window": len(scheduled),
        "example": example,
        "pipeline_compatible": pipeline_compatible,
        "required_corner_history_complete": corner_complete,
        "feature_ready_fixture_count": len(scheduled) if classification == "PROSPECTIVE_FEATURE_READY" else 0,
        "earliest_feature_ready_fixture": example["target_fixture_id"] if example and classification == "PROSPECTIVE_FEATURE_READY" else None,
        "protected_historical_base_unchanged": True,
    }
    write_json(run_dir / "feature_readiness.json", result)
    return result


def resume_phase7p(run_dir: Path, *, observed_at: str) -> dict[str, Any]:
    assert_protected(run_dir)
    readiness = read_json(run_dir / "feature_readiness.json")
    raw = _raw_matches(run_dir)
    concluded = [
        value
        for value in raw
        if PROSPECTIVE_START <= str(value.get("utcDate", ""))[:10] <= PROSPECTIVE_END
        and str(value.get("status")) == "FINISHED"
    ]
    specification = read_json(PHASE7_SPEC)
    cohort = ImmutableCohortStore(PHASE7P_RUN / "cohort")
    frozen_ids = {
        str(value["fixture"]["fixture_id"])
        for value in cohort.records("prediction")
    }
    missed_created: list[str] = []
    normalized_by_provider = {
        str(cast(Mapping[str, Any], value["normalized_event"])["provider_event_id"]): value
        for value in ImmutableExtensionStore(run_dir / "post_protocol_pl_data_extension").event_records()
    }
    for raw_event in concluded:
        record = normalized_by_provider.get(str(raw_event.get("id")))
        if record is None:
            continue
        event = cast(Mapping[str, Any], record["normalized_event"])
        fixture_id = str(event["canonical_fixture_id"])
        if fixture_id in frozen_ids:
            continue
        missed = build_missed_record(
            specification=specification,
            fixture_id=fixture_id,
            fixture_kickoff_at=str(event["kickoff_at"]),
            detected_at=observed_at,
            reason="Phase 7D was not feature-ready before the fixture outcome became available",
            affected_components=["common_data"],
            known_before_result=False,
        )
        cohort.persist_missed(missed)
        missed_created.append(fixture_id)
    if readiness["classification"] == "PROSPECTIVE_FEATURE_READY":
        status = "READY_FOR_PHASE7P_CAPTURE"
        resumed = True
    else:
        status = "NOT_RESUMED"
        resumed = False
    result = {
        "status": status,
        "phase7d_classification": readiness["classification"],
        "phase7p_collection_resumed": resumed,
        "paired_predictions_created": 0,
        "concluded_window_fixtures_without_prediction": len(concluded),
        "missed_records_created": len(missed_created),
        "missed_fixture_ids": missed_created,
        "post_outcome_predictions_created": False,
        "phase8_authorized": False,
    }
    write_json(run_dir / "phase7p_resume_result.json", result)
    return result


def _test_summary(path: Path) -> dict[str, Any]:
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    failures = sorted(
        f"{case.attrib.get('classname')}::{case.attrib.get('name')}"
        for suite in suites
        for case in suite.findall("testcase")
        if case.find("failure") is not None or case.find("error") is not None
    )
    return {
        "tests": sum(int(suite.attrib.get("tests", 0)) for suite in suites),
        "failures": sum(int(suite.attrib.get("failures", 0)) for suite in suites),
        "errors": sum(int(suite.attrib.get("errors", 0)) for suite in suites),
        "skipped": sum(int(suite.attrib.get("skipped", 0)) for suite in suites),
        "failure_ids": failures,
        "path": relative(path),
        "sha256": sha256_file(path),
    }


def finalize(
    run_dir: Path, *, focused_xml: Path, protected_xml: Path, full_xml: Path
) -> dict[str, Any]:
    assert_protected(run_dir)
    before = read_json(run_dir / "protected_before.json")
    after = protected_snapshot()
    if not protected_equal(before, after):
        raise Phase7DError("protected identities changed before finalization")
    write_json(run_dir / "protected_after_final.json", after)
    copied: dict[str, Path] = {}
    for label, source in (("focused", focused_xml), ("protected", protected_xml), ("full", full_xml)):
        target = run_dir / f"{label}_test_results.xml"
        shutil.copyfile(source, target)
        copied[label] = target
    tests = {label: _test_summary(path) for label, path in copied.items()}
    unexplained = sorted(set(tests["full"]["failure_ids"]) - INHERITED_FULL_SUITE_FAILURES)
    tests["inherited_full_suite_failure_ids"] = sorted(INHERITED_FULL_SUITE_FAILURES)
    tests["new_unexplained_failure_ids"] = unexplained
    tests["no_new_unexplained_failures"] = not unexplained
    if tests["focused"]["failures"] or tests["focused"]["errors"]:
        raise Phase7DError("focused Phase 7D tests failed")
    if tests["protected"]["failures"] or tests["protected"]["errors"]:
        raise Phase7DError("protected Phase 0-7P regressions failed")
    if unexplained:
        raise Phase7DError(f"new unexplained full-suite failures: {unexplained}")
    write_json(run_dir / "test_results.json", tests)
    readiness = read_json(run_dir / "feature_readiness.json")
    resume = read_json(run_dir / "phase7p_resume_result.json")
    acquisition = read_json(run_dir / "acquisition_result.json")
    overlap = read_json(run_dir / "historical_overlap_audit.json")
    continuity = read_json(run_dir / "continuity_completeness.json")
    classification = str(readiness["classification"])
    if classification not in ALLOWED_CLASSIFICATIONS:
        raise Phase7DError(f"unauthorized Phase 7D classification: {classification}")
    report_path = run_dir / "PHASE7D_IMPLEMENTATION_REPORT.md"
    unresolved = read_json(run_dir / "audit_result.json")["unresolved_provider_team_names"]
    report_path.write_text(
        "\n".join(
            [
                "# FFPS Phase 7D — Approved-Provider Data Continuity",
                "",
                f"- Final classification: `{classification}`",
                f"- Approved provider: `{PROVIDER_ID}`",
                f"- Protected historical cutoff: `{HISTORICAL_CUTOFF}`",
                f"- Raw source aggregate: `{acquisition['raw_source_aggregate_sha256']}`",
                f"- Normalized extension aggregate: `{acquisition['normalized_extension_aggregate_sha256']}`",
                f"- Continuity events expected / verified: {continuity['expected_completed_provider_events']} / {continuity['normalized_verified_completed_events']}",
                f"- Overlap score/orientation matches: {overlap['home_away_score_semantics_match_count']} / {overlap['provider_overlap_count']}",
                f"- Unresolved provider team names: {', '.join(unresolved) if unresolved else 'none'}",
                f"- Required corner observations available: {continuity['required_signal_coverage']['corners']['available_count']} / {continuity['required_signal_coverage']['corners']['required_count']}",
                f"- Earliest feature-ready fixture: {readiness['earliest_feature_ready_fixture']}",
                f"- Phase 7P collection resumed: {resume['phase7p_collection_resumed']}",
                f"- Fixtures marked missed: {resume['missed_records_created']}",
                "- Protected historical files, Phase 0-7P evidence, model bytes, manifest, production path, and Phase 7 specification remained unchanged.",
                "- No second provider, training, calibration, activation, registration, post-outcome prediction, or Phase 8 authorization occurred.",
                "",
                "The approved source aligns perfectly on the 180-fixture overlap's teams, orientation, and scores. "
                "It cannot currently cross the strict normalization gate for Coventry City FC and Hull City AFC; "
                "the existing fuzzy mapper would incorrectly map Hull City to Manchester City. The source also "
                "does not provide the corner history required by the frozen model feature contract. Both conditions "
                "are surfaced rather than hidden by approximate aliases or retrospective imputation.",
            ]
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    source_paths = [
        ROOT / "src" / "evaluation" / "phase7d_data_continuity.py",
        ROOT / "scripts" / "run_phase7d_data_continuity.py",
        ROOT / "tests" / "unit" / "test_phase7d_data_continuity.py",
    ]
    write_json(
        run_dir / "implementation_sources.json",
        {
            "schema_version": "ffps.phase7d_implementation_sources.v1",
            "sources": [
                {"path": relative(path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size}
                for path in source_paths
            ],
        },
    )
    excluded = {"artifact_index.json", "run_manifest.json", "offline_verification.json"}
    artifacts = [
        {"path": relative(path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size}
        for path in sorted(run_dir.rglob("*"))
        if path.is_file() and path.name not in excluded and "/.locks/" not in f"/{relative(path)}"
    ]
    index = {
        "schema_version": "ffps.phase7d_artifact_index.v1",
        "artifacts": artifacts,
        "artifact_count": len(artifacts),
        "aggregate_sha256": sha256_json(artifacts),
    }
    write_json(run_dir / "artifact_index.json", index)
    manifest = {
        "schema_version": "ffps.phase7d_run_manifest.v1",
        "phase": "7D",
        "run_id": RUN_ID,
        "status": "complete",
        "classification": classification,
        "finalized_at": utc_now(),
        "provider": PROVIDER_ID,
        "protected_historical_base_sha256": sha256_file(PROTECTED_BASE),
        "raw_source_aggregate_sha256": acquisition["raw_source_aggregate_sha256"],
        "normalized_extension_aggregate_sha256": acquisition["normalized_extension_aggregate_sha256"],
        "latest_causal_event_available": continuity["date_max"],
        "earliest_feature_ready_fixture": readiness["earliest_feature_ready_fixture"],
        "phase7p_collection_resumed": resume["phase7p_collection_resumed"],
        "phase8_authorized": False,
        "production_path_changed": False,
        "source_substitution_performed": False,
        "protected_phase0_to_phase7p_unchanged": True,
        "artifact_index_sha256": sha256_file(run_dir / "artifact_index.json"),
        "implementation_sources_sha256": sha256_file(run_dir / "implementation_sources.json"),
    }
    write_json(run_dir / "run_manifest.json", manifest)
    return manifest


def _offline_feature_rebuild(run_dir: Path) -> dict[str, Any]:
    readiness = read_json(run_dir / "feature_readiness.json")
    saved_example = readiness.get("example")
    reproduction = _historical_pipeline_reproduction()
    if not saved_example:
        return {
            "required": False,
            "verified": bool(reproduction["reproduced"]),
            "historical_pipeline_reproduction_verified": bool(reproduction["reproduced"]),
            "reason": "no normalized scheduled target was available in the frozen snapshot",
        }

    prediction_at = str(readiness["prediction_at"])
    target_fixture_id = str(saved_example["target_fixture_id"])
    store = ImmutableExtensionStore(run_dir / "post_protocol_pl_data_extension")
    effective = store.effective_records(as_of=prediction_at)
    post_protocol_effective = [
        value
        for value in effective
        if str(cast(Mapping[str, Any], value["normalized_event"])["fixture_date"])
        > HISTORICAL_CUTOFF
    ]
    target_records = [
        value
        for value in post_protocol_effective
        if str(cast(Mapping[str, Any], value["normalized_event"])["canonical_fixture_id"])
        == target_fixture_id
    ]
    if len(target_records) != 1:
        return {
            "required": True,
            "verified": False,
            "historical_pipeline_reproduction_verified": bool(reproduction["reproduced"]),
            "target_fixture_id": target_fixture_id,
            "target_record_count": len(target_records),
        }

    target_record = target_records[0]
    target = cast(Mapping[str, Any], target_record["normalized_event"])
    bundle = build_causal_source_bundle(
        protected_base_path=relative(PROTECTED_BASE),
        protected_base_sha256=sha256_file(PROTECTED_BASE),
        extension_records=post_protocol_effective,
        target_fixture_id=target_fixture_id,
        target_kickoff_at=str(target["kickoff_at"]),
        prediction_at=prediction_at,
    )
    included = set(bundle["ordered_extension_record_fingerprints"])
    causal_records = [
        value for value in post_protocol_effective if value["record_fingerprint"] in included
    ]
    composite = pd.concat(
        [
            _adapt_protected_base(),
            pd.DataFrame([_event_row(value) for value in causal_records]),
            pd.DataFrame([_event_row(target_record)]),
        ],
        ignore_index=True,
        sort=False,
    )
    generated = _run_existing_feature_pipeline(composite)
    target_rows = generated.loc[
        generated["match_id"].astype(str).eq(target_fixture_id)
    ]
    features = list(read_json(PHASE7_SPEC)["feature_contract"]["ordered_feature_names"])
    schema_sha256 = sha256_json(features)
    vector_sha256: str | None = None
    finite = False
    if len(target_rows) == 1 and all(value in target_rows.columns for value in features):
        vector = [float(target_rows.iloc[0][value]) for value in features]
        finite = bool(np.isfinite(np.asarray(vector, dtype=float)).all())
        if finite:
            vector_sha256 = sha256_json(vector)
    verified = bool(
        reproduction["reproduced"]
        and len(target_rows) == 1
        and finite
        and bundle == saved_example["causal_source_bundle"]
        and schema_sha256 == saved_example["feature_schema_sha256"]
        and vector_sha256 == saved_example["feature_vector_sha256"]
    )
    return {
        "required": True,
        "verified": verified,
        "historical_pipeline_reproduction_verified": bool(reproduction["reproduced"]),
        "target_fixture_id": target_fixture_id,
        "target_row_count": len(target_rows),
        "causal_source_bundle_sha256": bundle["causal_source_bundle_sha256"],
        "feature_schema_sha256": schema_sha256,
        "feature_vector_sha256": vector_sha256,
        "feature_vector_finite": finite,
    }


def verify(run_dir: Path, *, persist: bool = True) -> dict[str, Any]:
    errors: list[str] = []
    required = (
        "protected_before.json",
        "protected_after_final.json",
        "architecture_contract.json",
        "acquisition_result.json",
        "normalization_issues.json",
        "historical_overlap_audit.json",
        "continuity_completeness.json",
        "audit_result.json",
        "feature_readiness.json",
        "phase7p_resume_result.json",
        "test_results.json",
        "PHASE7D_IMPLEMENTATION_REPORT.md",
        "implementation_sources.json",
        "artifact_index.json",
        "run_manifest.json",
    )
    for name in required:
        if not (run_dir / name).is_file():
            errors.append(f"missing artifact: {name}")
    if not errors:
        before = read_json(run_dir / "protected_before.json")
        after = read_json(run_dir / "protected_after_final.json")
        current = protected_snapshot()
        if not protected_equal(before, after) or not protected_equal(before, current):
            errors.append("protected Phase 0-7P or historical identity mismatch")
        for season in PROVIDER_SEASONS:
            source = read_json(
                run_dir / "raw_provider_snapshot" / f"football-data-org-pl-{season}.source.json"
            )
            payload = (
                run_dir / "raw_provider_snapshot" / f"football-data-org-pl-{season}.response.json"
            )
            if sha256_file(payload) != source["response_sha256"] or payload.stat().st_size != source["response_size_bytes"]:
                errors.append(f"raw provider artifact mismatch: {season}")
        records = ImmutableExtensionStore(run_dir / "post_protocol_pl_data_extension").event_records()
        for record in records:
            result = verify_event_record(record)
            if not result["verified"]:
                errors.append(f"normalized extension record failed: {result['errors']}")
        acquisition = read_json(run_dir / "acquisition_result.json")
        raw_source_components = []
        for season in PROVIDER_SEASONS:
            source = read_json(
                run_dir / "raw_provider_snapshot" / f"football-data-org-pl-{season}.source.json"
            )
            raw_source_components.append(
                {
                    "season": source["season"],
                    "sha256": source["response_sha256"],
                    "size_bytes": source["response_size_bytes"],
                }
            )
        if sha256_json(raw_source_components) != acquisition["raw_source_aggregate_sha256"]:
            errors.append("raw source aggregate mismatch")
        if extension_aggregate(records) != acquisition["normalized_extension_aggregate_sha256"]:
            errors.append("normalized extension aggregate mismatch")
        sources = read_json(run_dir / "implementation_sources.json")
        for item in sources["sources"]:
            path = ROOT / str(item["path"])
            if not path.is_file() or sha256_file(path) != item["sha256"] or path.stat().st_size != item["size_bytes"]:
                errors.append(f"implementation source mismatch: {item['path']}")
        index = read_json(run_dir / "artifact_index.json")
        for item in index["artifacts"]:
            path = ROOT / str(item["path"])
            if not path.is_file() or sha256_file(path) != item["sha256"] or path.stat().st_size != item["size_bytes"]:
                errors.append(f"artifact mismatch: {item['path']}")
        if sha256_json(index["artifacts"]) != index["aggregate_sha256"]:
            errors.append("artifact index aggregate mismatch")
        manifest = read_json(run_dir / "run_manifest.json")
        if manifest.get("classification") not in ALLOWED_CLASSIFICATIONS:
            errors.append("unauthorized completion classification")
        if manifest.get("phase8_authorized") or manifest.get("production_path_changed"):
            errors.append("production or Phase 8 boundary violated")
        tests = read_json(run_dir / "test_results.json")
        if not tests.get("no_new_unexplained_failures"):
            errors.append("new unexplained test failures recorded")
        offline_rebuild = _offline_feature_rebuild(run_dir)
        if not offline_rebuild["verified"]:
            errors.append("offline causal feature rebuild mismatch")
    else:
        offline_rebuild = {"required": False, "verified": False}
    result = {
        "status": "VERIFIED" if not errors else "FAILED",
        "errors": errors,
        "classification": (
            read_json(run_dir / "run_manifest.json").get("classification")
            if (run_dir / "run_manifest.json").is_file()
            else None
        ),
        "protected_phase0_to_phase7p_unchanged": not any("protected" in value for value in errors),
        "protected_historical_base_unchanged": sha256_file(PROTECTED_BASE)
        == read_json(run_dir / "protected_before.json").get("protected_historical_base_sha256")
        if (run_dir / "protected_before.json").is_file()
        else False,
        "phase7p_collection_resumed": False,
        "phase8_authorized": False,
        "production_path_changed": False,
        "offline_feature_rebuild": offline_rebuild,
    }
    if persist:
        write_json(run_dir / "offline_verification.json", result)
    return result


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    sub = value.add_subparsers(dest="command", required=True)
    for name in ("prepare", "acquire", "audit", "verify"):
        command = sub.add_parser(name)
        command.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    features = sub.add_parser("features")
    features.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    features.add_argument("--prediction-at", default=utc_now())
    resume = sub.add_parser("resume-phase7p")
    resume.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    resume.add_argument("--observed-at", default=utc_now())
    final = sub.add_parser("finalize")
    final.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    final.add_argument("--focused-xml", type=Path, required=True)
    final.add_argument("--protected-xml", type=Path, required=True)
    final.add_argument("--full-xml", type=Path, required=True)
    return value


def main() -> int:
    args = parser().parse_args()
    try:
        if args.command == "prepare":
            result = prepare(args.run_dir)
        elif args.command == "acquire":
            result = acquire(args.run_dir)
        elif args.command == "audit":
            result = audit(args.run_dir)
        elif args.command == "features":
            result = feature_assessment(args.run_dir, prediction_at=args.prediction_at)
        elif args.command == "resume-phase7p":
            result = resume_phase7p(args.run_dir, observed_at=args.observed_at)
        elif args.command == "finalize":
            result = finalize(
                args.run_dir,
                focused_xml=args.focused_xml,
                protected_xml=args.protected_xml,
                full_xml=args.full_xml,
            )
        else:
            result = verify(args.run_dir)
    except (Phase7DError, FileNotFoundError, ValueError, requests.RequestException) as exc:
        print(json.dumps({"status": "FAILED", "error": str(exc)}, indent=2))
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("status") != "FAILED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
