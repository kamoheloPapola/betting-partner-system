"""Qualify football-data.co.uk corners and build Phase 7E causal continuity."""

from __future__ import annotations

import argparse
import copy
import json
import logging
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
    _candidate_suite,
    _capture_rows_locked,
    _prediction_record,
    _verify_candidate_lineage,
    load_frozen_specification,
    verify_frozen_contract,
)
from src.cli.commands.prediction import _load_prediction_models
from src.evaluation.phase3_calibration import aggregate_directory
from src.evaluation.phase7_shadow import sha256_json
from src.evaluation.phase7d_data_continuity import strict_normalize_team
from src.evaluation.phase7e_multi_source_continuity import (
    FIELD_AUTHORITY_MATRIX,
    FIELD_AUTHORITY_MATRIX_SHA256,
    HISTORICAL_CUTOFF,
    PRIMARY_PROVIDER,
    PROSPECTIVE_END,
    PROSPECTIVE_START,
    SUPPLEMENTAL_NORMALIZATION_CONTRACT,
    SUPPLEMENTAL_NORMALIZATION_CONTRACT_SHA256,
    SUPPLEMENTAL_PROVIDER,
    TEAM_ALIAS_EXTENSION,
    TEAM_ALIAS_EXTENSION_SHA256,
    ImmutableObservationStore,
    Phase7EError,
    build_causal_bundle,
    compose_observation,
    normalize_primary_event,
    normalize_supplemental_csv,
    observation_aggregate,
    raw_source_identity,
    unique_fixture_index,
    verify_observation,
    verify_seal,
)
from src.evaluation.phase7p_prospective import ImmutableCohortStore, verify_prediction_record
from src.predictions.provenance import (
    build_file_snapshot,
    sha256_bytes,
    sha256_file,
    source_event_index,
)
from src.simulation.match_simulator import MatchSimulator


RUN_ID = "phase7e-football-data-co-uk-corners-79110f9a-20260830-v1"
DEFAULT_RUN_DIR = ROOT / "baselines" / "phase7e" / RUN_ID
PHASE7D_RUN = ROOT / "baselines" / "phase7d" / "phase7d-approved-provider-pl-continuity-79110f9a-20260829-v1"
PHASE7P_RUN = ROOT / "baselines" / "phase7p" / "phase7p-prospective-shadow-79110f9a-20260829-v1"
PHASE7_RUN = ROOT / "baselines" / "phase7" / "phase7-active-vs-candidate-shadow-79110f9a-20260829-v2"
PHASE6_RUN = ROOT / "baselines" / "phase6" / "phase6-controlled-pl-goals-79110f9a-20260829-v1"
PROTECTED_BASE = ROOT / "data" / "processed" / "matches.csv"
FROZEN_FEATURE_MATRIX = ROOT / "data" / "features" / "feature_matrix.csv"
PHASE7_SPEC = PHASE7_RUN / "evaluation_specification.json"
SUPPLEMENTAL_BASE_URL = "https://www.football-data.co.uk/mmz4281"
SEASON_TOKENS = ("1920", "2021", "2122", "2223", "2324", "2425", "2526", "2627")
ALLOWED_CLASSIFICATIONS = {
    "PROSPECTIVE_FEATURE_READY",
    "SUPPLEMENTAL_SOURCE_PARTIAL",
    "SUPPLEMENTAL_SOURCE_SEMANTICS_INCOMPATIBLE",
    "SUPPLEMENTAL_SOURCE_UNAVAILABLE",
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
        raise Phase7EError("supplemental raw persistence verification failed")
    os.replace(temporary, path)


def relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _safe_new_run_dir(run_dir: Path) -> None:
    root = (ROOT / "baselines" / "phase7e").resolve()
    resolved = run_dir.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise Phase7EError("Phase 7E evidence must remain beneath baselines/phase7e") from exc
    if resolved.exists():
        raise Phase7EError(f"Phase 7E evidence already exists: {resolved}")
    resolved.mkdir(parents=True)


def protected_snapshot() -> dict[str, Any]:
    from scripts.run_phase7d_data_continuity import verify as verify_phase7d

    verification = verify_phase7d(PHASE7D_RUN, persist=False)
    if verification.get("status") != "VERIFIED":
        raise Phase7EError(f"frozen Phase 7D verification failed: {verification}")
    phase7_contract = verify_frozen_contract()
    if not phase7_contract["verified"]:
        raise Phase7EError(f"frozen Phase 7 contract failed: {phase7_contract['errors']}")
    phase7d_manifest = read_json(PHASE7D_RUN / "run_manifest.json")
    phase7d_source = read_json(PHASE7D_RUN / "implementation_sources.json")
    return {
        "captured_at": utc_now(),
        "protected_historical_base_sha256": sha256_file(PROTECTED_BASE),
        "frozen_feature_matrix_sha256": sha256_file(FROZEN_FEATURE_MATRIX),
        "phase7_specification_fingerprint": read_json(PHASE7_SPEC)["specification_fingerprint"],
        "phase7_specification_sha256": sha256_file(PHASE7_SPEC),
        "phase7d_classification": phase7d_manifest["classification"],
        "phase7d_evidence_aggregate_sha256": aggregate_directory(PHASE7D_RUN),
        "phase7d_run_manifest_sha256": sha256_file(PHASE7D_RUN / "run_manifest.json"),
        "phase7d_implementation_sources_sha256": sha256_json(phase7d_source["sources"]),
        "phase7p_evidence_aggregate_sha256": aggregate_directory(PHASE7P_RUN),
        "active_bundle_sha256": phase7_contract["active_bundle_sha256"],
        "candidate_bundle_sha256": phase7_contract["candidate_bundle_aggregate_sha256"],
    }


def protected_equal(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return {key: value for key, value in left.items() if key != "captured_at"} == {
        key: value for key, value in right.items() if key != "captured_at"
    }


def assert_protected(run_dir: Path) -> None:
    if not protected_equal(read_json(run_dir / "protected_before.json"), protected_snapshot()):
        raise Phase7EError("protected Phase 0-7D, Phase 7P, data, or model identity changed")


def prepare(run_dir: Path) -> dict[str, Any]:
    _safe_new_run_dir(run_dir)
    before = protected_snapshot()
    write_json(run_dir / "protected_before.json", before)
    protocol = {
        "schema_version": "ffps.phase7e_qualification_protocol.v1",
        "frozen_before_provider_evaluation": True,
        "selected_supplemental_provider": SUPPLEMENTAL_PROVIDER,
        "primary_provider": PRIMARY_PROVIDER,
        "selection_basis": [
            "already integrated by src.fetch.football_data and src.cli.commands.data",
            "HC and AC are the direct raw origin of protected home_corners and away_corners",
            "no credential is required and repository practice retains exact season CSV bytes",
            "one supplemental provider is sufficient",
        ],
        "provider_inventory": {
            "football-data.co.uk:csv": "selected; direct historical corner source",
            "api-football:v3": "configured integration exists but not selected; multiple new authorities forbidden",
        },
        "qualification_overlap": "all protected PL fixtures from 2019-08-09 through 2025-12-28",
        "required_exact_fields": ["home_goals", "away_goals", "home_corners", "away_corners"],
        "fixture_join": "date + exact canonical home + exact canonical away; orientation cannot reverse",
        "corner_tolerance": 0,
        "feature_reproduction_tolerance": 1e-12,
        "retention": "exact raw CSV retention follows the repository's existing football-data.co.uk fetcher practice",
        "production_path_changed": False,
    }
    protocol["qualification_protocol_sha256"] = sha256_json(protocol)
    write_json(run_dir / "qualification_protocol.json", protocol)
    write_json(run_dir / "field_authority_matrix.json", FIELD_AUTHORITY_MATRIX)
    write_json(run_dir / "team_alias_extension.json", TEAM_ALIAS_EXTENSION)
    write_json(run_dir / "supplemental_normalization_contract.json", SUPPLEMENTAL_NORMALIZATION_CONTRACT)
    result = {
        "status": "QUALIFICATION_PROTOCOL_FROZEN",
        "provider": SUPPLEMENTAL_PROVIDER,
        "qualification_protocol_sha256": protocol["qualification_protocol_sha256"],
        "field_authority_matrix_sha256": FIELD_AUTHORITY_MATRIX_SHA256,
        "team_alias_extension_sha256": TEAM_ALIAS_EXTENSION_SHA256,
        "normalization_contract_sha256": SUPPLEMENTAL_NORMALIZATION_CONTRACT_SHA256,
        "protected_phase0_to_phase7d_captured": True,
        "phase7d_remains_frozen": True,
    }
    write_json(run_dir / "preparation_result.json", result)
    return result


def _source_path(run_dir: Path, token: str) -> Path:
    return run_dir / "raw_supplemental_snapshot" / f"football-data-co-uk-E0-{token}.source.json"


def _payload_path(run_dir: Path, token: str) -> Path:
    return run_dir / "raw_supplemental_snapshot" / f"football-data-co-uk-E0-{token}.csv"


def _events_path(run_dir: Path, token: str) -> Path:
    return run_dir / "normalized_supplemental_events" / f"E0-{token}.json"


def acquire(run_dir: Path) -> dict[str, Any]:
    assert_protected(run_dir)
    if (run_dir / "acquisition_result.json").exists():
        raise Phase7EError("supplemental source snapshot is already frozen")
    sources: list[dict[str, Any]] = []
    normalized_all: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    for token in SEASON_TOKENS:
        url = f"{SUPPLEMENTAL_BASE_URL}/{token}/E0.csv"
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        payload = response.content
        retrieved_at = utc_now()
        write_bytes(_payload_path(run_dir, token), payload)
        source = raw_source_identity(
            payload,
            season_token=token,
            retrieved_at=retrieved_at,
            request_url=str(response.url),
        )
        source["path"] = relative(_payload_path(run_dir, token))
        try:
            events = normalize_supplemental_csv(payload, raw_source=source)
        except Phase7EError as exc:
            issues.append({"season_token": token, "error": str(exc)})
            events = []
        source["normalized_completed_event_count"] = len(events)
        write_json(_source_path(run_dir, token), source)
        write_json(_events_path(run_dir, token), events)
        sources.append(source)
        normalized_all.extend(events)
    raw_components = [
        {
            "season_token": source["season_token"],
            "sha256": source["response_sha256"],
            "size_bytes": source["response_size_bytes"],
        }
        for source in sources
    ]
    result = {
        "status": "SUPPLEMENTAL_SNAPSHOT_FROZEN" if normalized_all else "SUPPLEMENTAL_SOURCE_UNAVAILABLE",
        "provider": SUPPLEMENTAL_PROVIDER,
        "season_tokens": list(SEASON_TOKENS),
        "raw_source_count": len(sources),
        "raw_supplemental_aggregate_sha256": sha256_json(raw_components),
        "normalized_completed_event_count": len(normalized_all),
        "normalized_supplemental_aggregate_sha256": sha256_json(
            sorted(str(value["record_fingerprint"]) for value in normalized_all)
        ),
        "normalization_issue_count": len(issues),
        "credentials_retained": False,
        "raw_retention_limitation": "none found in repository; exact CSV bytes retained under existing fetcher practice",
    }
    write_json(run_dir / "supplemental_normalization_issues.json", {"issues": issues})
    write_json(run_dir / "acquisition_result.json", result)
    return result


def _supplemental_events(run_dir: Path) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    for token in SEASON_TOKENS:
        payload = json.loads(_events_path(run_dir, token).read_text(encoding="utf-8"))
        values.extend(cast(list[dict[str, Any]], payload))
    return values


def _current_team(value: Any) -> str:
    return strict_normalize_team(str(value), season=2025)


def _key(value: Mapping[str, Any]) -> tuple[str, str, str]:
    return (str(value["fixture_date"]), str(value["home_team"]), str(value["away_team"]))


def _primary_sources(run_dir: Path | None = None) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    values: list[tuple[dict[str, Any], dict[str, Any]]] = []
    raw_dir = PHASE7D_RUN / "raw_provider_snapshot"
    for season in (2025, 2026):
        source = read_json(raw_dir / f"football-data-org-pl-{season}.source.json")
        payload = read_json(raw_dir / f"football-data-org-pl-{season}.response.json")
        for event in payload["matches"]:
            values.append((cast(dict[str, Any], event), source))
    return values


def qualify(run_dir: Path) -> dict[str, Any]:
    assert_protected(run_dir)
    acquisition = read_json(run_dir / "acquisition_result.json")
    if acquisition["status"] == "SUPPLEMENTAL_SOURCE_UNAVAILABLE":
        result = {"classification": "SUPPLEMENTAL_SOURCE_UNAVAILABLE", "qualified": False}
        write_json(run_dir / "qualification_result.json", result)
        return result
    supplemental = _supplemental_events(run_dir)
    supplemental_index = unique_fixture_index(supplemental)
    protected = pd.read_csv(PROTECTED_BASE, low_memory=False)
    protected = protected.loc[protected["league"].astype(str).eq("PL")].copy()
    protected["fixture_date"] = pd.to_datetime(protected["match_date"]).dt.date.astype(str)
    protected["current_home"] = protected["home_team"].map(_current_team)
    protected["current_away"] = protected["away_team"].map(_current_team)
    mismatch_rows: list[dict[str, Any]] = []
    unmatched: list[dict[str, Any]] = []
    matched = 0
    home_corner_exact = 0
    away_corner_exact = 0
    score_exact = 0
    for row in protected.to_dict(orient="records"):
        key = (str(row["fixture_date"]), str(row["current_home"]), str(row["current_away"]))
        event = supplemental_index.get(key)
        if event is None:
            unmatched.append({"fixture_date": key[0], "home_team": key[1], "away_team": key[2]})
            continue
        matched += 1
        home_ok = int(row["home_corners"]) == int(event["home_corners"])
        away_ok = int(row["away_corners"]) == int(event["away_corners"])
        score_ok = int(row["home_goals"]) == int(event["home_goals"]) and int(row["away_goals"]) == int(event["away_goals"])
        home_corner_exact += home_ok
        away_corner_exact += away_ok
        score_exact += score_ok
        if not home_ok or not away_ok or not score_ok:
            mismatch_rows.append(
                {
                    "fixture_date": key[0],
                    "home_team": key[1],
                    "away_team": key[2],
                    "protected_home_goals": int(row["home_goals"]),
                    "protected_away_goals": int(row["away_goals"]),
                    "supplemental_home_goals": event["home_goals"],
                    "supplemental_away_goals": event["away_goals"],
                    "protected_home_corners": int(row["home_corners"]),
                    "protected_away_corners": int(row["away_corners"]),
                    "supplemental_home_corners": event["home_corners"],
                    "supplemental_away_corners": event["away_corners"],
                    "raw_source_sha256": event["raw_source_sha256"],
                    "raw_row_sha256": event["raw_row_sha256"],
                    "classification": "UNEXPLAINED_SOURCE_VALUE_DIFFERENCE",
                }
            )
    overlap = {
        "protected_fixture_count": len(protected),
        "matched_fixture_count": matched,
        "unmatched_fixture_count": len(unmatched),
        "home_corner_exact_match_count": home_corner_exact,
        "away_corner_exact_match_count": away_corner_exact,
        "final_score_exact_match_count": score_exact,
        "mismatch_count": len(mismatch_rows),
        "home_away_orientation_verified": matched == len(protected),
        "corner_semantics": "HC/AC are full-time, home/away-oriented league-match corner-kick counts",
        "all_practical_overlap_used": True,
        "mismatches": mismatch_rows,
        "unmatched": unmatched,
    }
    write_json(run_dir / "historical_corner_overlap_audit.json", overlap)

    primary_2026: list[dict[str, Any]] = []
    for raw, source in _primary_sources():
        if int(source["season"]) == 2026:
            primary_2026.append(normalize_primary_event(raw, raw_source=source))
    supplemental_2026 = unique_fixture_index(
        [value for value in supplemental if int(value["season_start_year"]) == 2026]
    )
    cross_links: dict[str, int] = {"COVENTRY": 0, "HULL": 0}
    primary_matches = 0
    for event in primary_2026:
        linked = supplemental_2026.get(_key(event))
        if linked is None:
            continue
        primary_matches += 1
        for team in cross_links:
            if team in {event["home_team"], event["away_team"]}:
                cross_links[team] += 1
    alias_qualification = {
        "team_alias_extension_sha256": TEAM_ALIAS_EXTENSION_SHA256,
        "primary_2026_completed_and_scheduled_count": len(primary_2026),
        "exact_cross_provider_fixture_link_count": primary_matches,
        "cross_provider_fixture_links_by_alias": cross_links,
        "qualified": all(value > 0 for value in cross_links.values()),
        "evidence_rule": "primary provider team ID/name plus exact date, orientation and opponent relationship to supplemental row",
        "name_similarity_only": False,
        "production_mapping_changed": False,
    }
    write_json(run_dir / "team_alias_qualification.json", alias_qualification)
    if mismatch_rows:
        classification = "SUPPLEMENTAL_SOURCE_SEMANTICS_INCOMPATIBLE"
    elif unmatched or not alias_qualification["qualified"]:
        classification = "SUPPLEMENTAL_SOURCE_PARTIAL"
    else:
        classification = "SUPPLEMENTAL_SOURCE_QUALIFIED"
    result = {
        "status": "QUALIFICATION_COMPLETE",
        "classification": classification,
        "qualified": classification == "SUPPLEMENTAL_SOURCE_QUALIFIED",
        "supplemental_provider": SUPPLEMENTAL_PROVIDER,
        "qualification_protocol_sha256": read_json(run_dir / "qualification_protocol.json")["qualification_protocol_sha256"],
        "overlap_fixture_count": matched,
        "home_corner_exact_match_count": home_corner_exact,
        "away_corner_exact_match_count": away_corner_exact,
        "mismatch_count": len(mismatch_rows),
        "team_alias_extension_qualified": alias_qualification["qualified"],
    }
    write_json(run_dir / "qualification_result.json", result)
    return result


def compose(run_dir: Path) -> dict[str, Any]:
    assert_protected(run_dir)
    qualification = read_json(run_dir / "qualification_result.json")
    if not qualification["qualified"]:
        result = {
            "status": "COMPOSITION_BLOCKED",
            "classification": qualification["classification"],
            "causal_history_complete_count": 0,
        }
        write_json(run_dir / "composition_result.json", result)
        return result
    supplement = unique_fixture_index(
        [
            value
            for value in _supplemental_events(run_dir)
            if HISTORICAL_CUTOFF < str(value["fixture_date"]) < PROSPECTIVE_START
        ]
    )
    primary: list[dict[str, Any]] = []
    for raw, source in _primary_sources():
        date = str(raw.get("utcDate", ""))[:10]
        if str(raw.get("status")) == "FINISHED" and HISTORICAL_CUTOFF < date < PROSPECTIVE_START:
            primary.append(normalize_primary_event(raw, raw_source=source))
    if len({str(value["provider_event_id"]) for value in primary}) != len(primary):
        raise Phase7EError("duplicate primary provider event ID")
    store = ImmutableObservationStore(run_dir / "canonical_match_observations")
    unresolved: list[dict[str, Any]] = []
    for event in primary:
        supplemental = supplement.get(_key(event))
        if supplemental is None:
            unresolved.append(
                {
                    "provider_event_id": event["provider_event_id"],
                    "fixture_date": event["fixture_date"],
                    "home_team": event["home_team"],
                    "away_team": event["away_team"],
                    "reason": "supplemental fixture unavailable",
                }
            )
            continue
        try:
            store.persist(compose_observation(event, supplemental))
        except Phase7EError as exc:
            unresolved.append(
                {
                    "provider_event_id": event["provider_event_id"],
                    "fixture_date": event["fixture_date"],
                    "reason": str(exc),
                }
            )
    observations = store.records()
    acquisition = read_json(run_dir / "acquisition_result.json")
    phase7d_acquisition = read_json(PHASE7D_RUN / "acquisition_result.json")
    primary_aggregate = sha256_json(sorted(str(value["record_fingerprint"]) for value in primary))
    observation_sha = observation_aggregate(observations)
    multi_source = sha256_json(
        {
            "protected_base_sha256": sha256_file(PROTECTED_BASE),
            "primary_resolved_aggregate_sha256": primary_aggregate,
            "supplemental_raw_aggregate_sha256": acquisition["raw_supplemental_aggregate_sha256"],
            "observation_aggregate_sha256": observation_sha,
            "field_authority_matrix_sha256": FIELD_AUTHORITY_MATRIX_SHA256,
            "team_alias_extension_sha256": TEAM_ALIAS_EXTENSION_SHA256,
        }
    )
    resolved_primary_ids = sorted(
        value["provider_event_id"]
        for value in primary
        if value["provider_event_id"] in {"560542", "560543", "560561"}
    )
    complete = len(primary) == len(observations) == 215 and not unresolved
    result = {
        "status": "MULTI_SOURCE_CONTINUITY_FROZEN" if complete else "SUPPLEMENTAL_SOURCE_PARTIAL",
        "classification": "SUPPLEMENTAL_SOURCE_QUALIFIED" if complete else "SUPPLEMENTAL_SOURCE_PARTIAL",
        "continuity_interval": {"after_exclusive": HISTORICAL_CUTOFF, "before_exclusive": PROSPECTIVE_START},
        "continuity_before": {
            "completed_events": 212,
            "required_events": 215,
            "corner_complete_events": 0,
        },
        "primary_completed_event_count": len(primary),
        "canonical_match_observation_count": len(observations),
        "causal_history_complete_count": sum(verify_observation(value)["verified"] for value in observations),
        "unresolved_count": len(unresolved),
        "unresolved": unresolved,
        "primary_source_present_alias_extension_resolved_ids": resolved_primary_ids,
        "primary_source_absent_supplemental_fallback_ids": [],
        "all_home_away_corners_present": all(
            cast(Mapping[str, Any], value["fields"])["home_corners"]["value"] is not None
            and cast(Mapping[str, Any], value["fields"])["away_corners"]["value"] is not None
            for value in observations
        ),
        "primary_phase7d_extension_sha256": phase7d_acquisition["normalized_extension_aggregate_sha256"],
        "primary_resolved_aggregate_sha256": primary_aggregate,
        "supplemental_raw_aggregate_sha256": acquisition["raw_supplemental_aggregate_sha256"],
        "observation_aggregate_sha256": observation_sha,
        "multi_source_continuity_sha256": multi_source,
        "protected_base_rewritten": False,
        "production_path_changed": False,
    }
    write_json(run_dir / "composition_result.json", result)
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


def _observation_row(value: Mapping[str, Any]) -> dict[str, Any]:
    fields = cast(Mapping[str, Any], value["fields"])
    return {
        "match_id": value["canonical_fixture_id"],
        "date": pd.Timestamp(str(value["kickoff_at"])),
        "season": value["season_start_year"],
        "league": "PL",
        "home_team": value["home_team"],
        "away_team": value["away_team"],
        "home_score": fields["home_goals"]["value"],
        "away_score": fields["away_goals"]["value"],
        "home_corners": fields["home_corners"]["value"],
        "away_corners": fields["away_corners"]["value"],
        "status": "FINISHED",
        "referee": np.nan,
    }


def _target_row(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "match_id": value["canonical_fixture_id"],
        "date": pd.Timestamp(str(value["kickoff_at"])),
        "season": value["season_start_year"],
        "league": "PL",
        "home_team": value["home_team"],
        "away_team": value["away_team"],
        "home_score": np.nan,
        "away_score": np.nan,
        "home_corners": np.nan,
        "away_corners": np.nan,
        "status": value["status"],
        "referee": np.nan,
    }


def _run_pipeline(frame: pd.DataFrame) -> pd.DataFrame:
    import src.features.pipeline as pipeline_module
    from src.features.pipeline import FeaturePipeline

    with tempfile.TemporaryDirectory(prefix="ffps-phase7e-") as directory:
        original = pipeline_module.DATA_DIR
        try:
            pipeline_module.DATA_DIR = Path(directory)
            return FeaturePipeline().run(match_data=frame, league="PL")
        finally:
            pipeline_module.DATA_DIR = original


def _compare_to_frozen(generated: pd.DataFrame) -> dict[str, Any]:
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
        "generated_rows": len(generated),
        "matched_rows": len(joined),
        "feature_count": len(features),
        "feature_schema_sha256": sha256_json(features),
        "maximum_absolute_difference": max_diff,
        "differing_features_over_1e_12": differing,
        "reproduced": len(generated) == len(joined) == len(frozen) and not differing,
    }


def _historical_reproduction(run_dir: Path) -> dict[str, Any]:
    supplemental = unique_fixture_index(
        [
            value
            for value in _supplemental_events(run_dir)
            if str(value["fixture_date"]) <= HISTORICAL_CUTOFF
        ]
    )
    reconstructed = _adapt_protected_base()
    missing: list[str] = []
    for index, row in reconstructed.iterrows():
        key = (row["date"].date().isoformat(), str(row["home_team"]), str(row["away_team"]))
        event = supplemental.get(key)
        if event is None:
            missing.append("|".join(key))
            continue
        reconstructed.at[index, "home_corners"] = event["home_corners"]
        reconstructed.at[index, "away_corners"] = event["away_corners"]
    if missing:
        return {"reproduced": False, "missing_fixture_count": len(missing), "missing": missing[:20]}
    result = _compare_to_frozen(_run_pipeline(reconstructed))
    result["supplemental_corner_substitution_rows"] = len(reconstructed)
    result["protected_file_written"] = False
    return result


def _scheduled_primary(prediction_at: str) -> list[dict[str, Any]]:
    now = pd.Timestamp(prediction_at)
    values: list[dict[str, Any]] = []
    for raw, source in _primary_sources():
        if int(source["season"]) != 2026:
            continue
        event = normalize_primary_event(raw, raw_source=source)
        if (
            event["completed"] is False
            and PROSPECTIVE_START <= event["fixture_date"] <= PROSPECTIVE_END
            and pd.Timestamp(event["kickoff_at"]) > now
        ):
            values.append(event)
    return sorted(values, key=lambda value: (str(value["kickoff_at"]), str(value["canonical_fixture_id"])))


def _build_target_matrix(
    run_dir: Path, *, prediction_at: str, target_fixture_id: str | None = None
) -> tuple[pd.DataFrame, pd.Series, dict[str, Any]]:
    composition = read_json(run_dir / "composition_result.json")
    observations = ImmutableObservationStore(
        run_dir / "canonical_match_observations"
    ).effective_records(as_of=prediction_at)
    scheduled = _scheduled_primary(prediction_at)
    if target_fixture_id:
        scheduled = [value for value in scheduled if value["canonical_fixture_id"] == target_fixture_id]
    if not scheduled:
        raise Phase7EError("no still-unsettled Phase 7 fixture is available")
    target = scheduled[0]
    acquisition = read_json(run_dir / "acquisition_result.json")
    bundle = build_causal_bundle(
        protected_base_sha256=sha256_file(PROTECTED_BASE),
        primary_extension_sha256=composition["primary_resolved_aggregate_sha256"],
        supplemental_source_sha256=acquisition["raw_supplemental_aggregate_sha256"],
        observations=observations,
        target_fixture_id=str(target["canonical_fixture_id"]),
        target_kickoff_at=str(target["kickoff_at"]),
        prediction_at=prediction_at,
    )
    included = set(bundle["ordered_observation_fingerprints"])
    causal = [value for value in observations if value["record_fingerprint"] in included]
    composite = pd.concat(
        [
            _adapt_protected_base(),
            pd.DataFrame([_observation_row(value) for value in causal]),
            pd.DataFrame([_target_row(target)]),
        ],
        ignore_index=True,
        sort=False,
    )
    matrix = _run_pipeline(composite)
    artifacts = [
        {"role": "protected_historical_base", "sha256": sha256_file(PROTECTED_BASE)},
        {"role": "primary_resolved_extension", "sha256": composition["primary_resolved_aggregate_sha256"]},
        {"role": "supplemental_raw_snapshot", "sha256": acquisition["raw_supplemental_aggregate_sha256"]},
        {"role": "composed_observations", "sha256": composition["observation_aggregate_sha256"]},
        {"role": "field_authority_matrix", "sha256": FIELD_AUTHORITY_MATRIX_SHA256},
        {"role": "team_alias_extension", "sha256": TEAM_ALIAS_EXTENSION_SHA256},
        {"role": "target_causal_bundle", "sha256": bundle["multi_source_continuity_sha256"]},
    ]
    snapshot = build_file_snapshot(
        frame=matrix,
        artifacts=artifacts,
        source_type="phase7e_evaluation_only_multi_source_causal_history",
    )
    snapshot["multi_source_continuity_sha256"] = bundle["multi_source_continuity_sha256"]
    snapshot["causal_bundle"] = bundle
    matrix.attrs["ffps_data_snapshot"] = snapshot
    matrix.attrs["ffps_source_events"] = source_event_index(composite)
    target_rows = matrix.loc[matrix["match_id"].astype(str).eq(str(target["canonical_fixture_id"]))]
    if len(target_rows) != 1:
        raise Phase7EError("target feature row is not unique")
    return matrix, target_rows.iloc[0], bundle


def feature_assessment(run_dir: Path, *, prediction_at: str) -> dict[str, Any]:
    assert_protected(run_dir)
    qualification = read_json(run_dir / "qualification_result.json")
    composition = read_json(run_dir / "composition_result.json")
    reproduction = _historical_reproduction(run_dir)
    if not qualification["qualified"]:
        classification = qualification["classification"]
        result = {
            "status": "FEATURE_ASSESSMENT_BLOCKED",
            "classification": classification,
            "historical_feature_reproduction": reproduction,
            "earliest_feature_ready_fixture": None,
        }
        write_json(run_dir / "feature_readiness.json", result)
        return result
    if composition["causal_history_complete_count"] != 215:
        classification = "SUPPLEMENTAL_SOURCE_PARTIAL"
        result = {
            "status": "FEATURE_ASSESSMENT_BLOCKED",
            "classification": classification,
            "historical_feature_reproduction": reproduction,
            "earliest_feature_ready_fixture": None,
        }
        write_json(run_dir / "feature_readiness.json", result)
        return result
    matrix, fixture, bundle = _build_target_matrix(run_dir, prediction_at=prediction_at)
    features = list(read_json(PHASE7_SPEC)["feature_contract"]["ordered_feature_names"])
    vector = [float(fixture[value]) for value in features]
    finite = bool(np.isfinite(np.asarray(vector, dtype=float)).all())
    active_suite = _load_prediction_models("PL")
    candidate_suite = _candidate_suite()
    specification = load_frozen_specification()
    simulator = MatchSimulator(
        n_simulations=int(specification["simulation"]["n_simulations"]),
        league="PL",
        seed=int(specification["simulation"]["seed"]),
        seed_scope=str(specification["simulation"]["seed_scope"]),
    )
    state = copy.deepcopy(simulator.rng.bit_generator.state)
    active, active_verification = _prediction_record(
        fixture, active_suite, simulator, matrix, label="active-dry-run", prediction_timestamp=prediction_at
    )
    simulator.rng.bit_generator.state = copy.deepcopy(state)
    candidate, candidate_verification = _prediction_record(
        fixture, candidate_suite, simulator, matrix, label="candidate-dry-run", prediction_timestamp=prediction_at
    )
    lineage = _verify_candidate_lineage()
    feature_identity_equal = (
        active["feature_provenance"]["feature_bundle_sha256"]
        == candidate["feature_provenance"]["feature_bundle_sha256"]
    )
    dry_run = {
        "target_fixture_id": str(fixture["match_id"]),
        "target_kickoff_at": pd.Timestamp(fixture["date"]).isoformat(),
        "prediction_at": pd.Timestamp(prediction_at).isoformat(),
        "target_outcome_present": False,
        "causal_source_bundle": bundle,
        "feature_schema_sha256": sha256_json(features),
        "feature_vector_sha256": sha256_json(vector) if finite else None,
        "feature_vector_finite": finite,
        "active_candidate_feature_identity_equal": feature_identity_equal,
        "active_bundle_accepted": bool(active_verification["overall_phase4_trusted"]),
        "candidate_bundle_accepted": bool(candidate_verification["overall_phase4_trusted"]),
        "candidate_training_lineage_verified": bool(lineage["verified"]),
        "active_probabilities": active["prediction_output"]["final_probabilities"],
        "candidate_probabilities": candidate["prediction_output"]["final_probabilities"],
        "prediction_record_persisted": False,
    }
    write_json(run_dir / "prospective_dry_run.json", dry_run)
    ready = bool(
        reproduction["reproduced"]
        and finite
        and feature_identity_equal
        and dry_run["active_bundle_accepted"]
        and dry_run["candidate_bundle_accepted"]
        and dry_run["candidate_training_lineage_verified"]
    )
    classification = "PROSPECTIVE_FEATURE_READY" if ready else "SUPPLEMENTAL_SOURCE_PARTIAL"
    result = {
        "status": "FEATURE_ASSESSMENT_COMPLETE",
        "classification": classification,
        "prediction_at": pd.Timestamp(prediction_at).isoformat(),
        "historical_feature_reproduction": reproduction,
        "ordered_feature_count": len(features),
        "ordered_feature_schema_sha256": sha256_json(features),
        "earliest_feature_ready_fixture": str(fixture["match_id"]) if ready else None,
        "earliest_feature_ready_kickoff": pd.Timestamp(fixture["date"]).isoformat() if ready else None,
        "prospective_dry_run_verified": ready,
        "production_path_changed": False,
    }
    write_json(run_dir / "feature_readiness.json", result)
    return result


def resume_phase7p(run_dir: Path, *, as_of: str) -> dict[str, Any]:
    assert_protected(run_dir)
    readiness = read_json(run_dir / "feature_readiness.json")
    if readiness["classification"] != "PROSPECTIVE_FEATURE_READY":
        result = {
            "status": "NOT_RESUMED",
            "phase7p_collection_resumed": False,
            "created_count": 0,
            "phase8_authorized": False,
        }
        write_json(run_dir / "phase7p_resume_result.json", result)
        return result
    fixture_id = str(readiness["earliest_feature_ready_fixture"])
    matrix, fixture, bundle = _build_target_matrix(
        run_dir, prediction_at=as_of, target_fixture_id=fixture_id
    )
    specification = load_frozen_specification()
    features = list(specification["feature_contract"]["ordered_feature_names"])
    required = ["match_hash", "date", "league", "home_team", "away_team", *features]
    capture_rows = matrix.loc[matrix["match_id"].astype(str).eq(fixture_id)].copy()
    capture_rows["match_hash"] = capture_rows["match_id"].astype(str)
    matrix = matrix.copy()
    matrix["match_hash"] = matrix["match_id"].astype(str)
    store = ImmutableCohortStore(run_dir / "phase7p_resumed_cohort")
    with store.batch_lock("prediction_capture"):
        capture = _capture_rows_locked(
            specification=specification,
            matrix=matrix,
            rows=capture_rows,
            required=required,
            store=store,
            as_of=as_of,
        )
    predictions = store.records("prediction")
    result = {
        "status": "RESUMED" if capture["created_count"] else "NO_NEW_PREDICTION",
        "phase7p_collection_resumed": bool(capture["created_count"]),
        "created_count": capture["created_count"],
        "fixture_ids": capture["fixture_ids"],
        "prediction_record_fingerprints": [value["record_fingerprint"] for value in predictions],
        "causal_source_bundle_sha256": bundle["multi_source_continuity_sha256"],
        "production_output_mutated": False,
        "post_outcome_predictions_created": False,
        "phase7_specification_changed": False,
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


def finalize(run_dir: Path, *, focused_xml: Path, protected_xml: Path, full_xml: Path) -> dict[str, Any]:
    assert_protected(run_dir)
    before = read_json(run_dir / "protected_before.json")
    after = protected_snapshot()
    if not protected_equal(before, after):
        raise Phase7EError("protected identities changed before finalization")
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
        raise Phase7EError("focused Phase 7E tests failed")
    if tests["protected"]["failures"] or tests["protected"]["errors"]:
        raise Phase7EError("protected Phase 0-7D regressions failed")
    if unexplained:
        raise Phase7EError(f"new unexplained full-suite failures: {unexplained}")
    write_json(run_dir / "test_results.json", tests)
    qualification = read_json(run_dir / "qualification_result.json")
    composition = read_json(run_dir / "composition_result.json")
    readiness = read_json(run_dir / "feature_readiness.json")
    resume = read_json(run_dir / "phase7p_resume_result.json")
    acquisition = read_json(run_dir / "acquisition_result.json")
    overlap = read_json(run_dir / "historical_corner_overlap_audit.json")
    classification = str(readiness["classification"])
    if classification not in ALLOWED_CLASSIFICATIONS:
        raise Phase7EError(f"unauthorized Phase 7E classification: {classification}")
    report = [
        "# FFPS Phase 7E — Supplemental Source Qualification",
        "",
        f"- Final classification: `{classification}`",
        f"- Supplemental provider: `{SUPPLEMENTAL_PROVIDER}`",
        f"- Qualification identity: `{read_json(run_dir / 'qualification_protocol.json')['qualification_protocol_sha256']}`",
        f"- Corner semantics: {overlap['corner_semantics']}",
        f"- Historical overlap: {overlap['matched_fixture_count']} / {overlap['protected_fixture_count']}",
        f"- Exact home corners: {overlap['home_corner_exact_match_count']} / {overlap['protected_fixture_count']}",
        f"- Exact away corners: {overlap['away_corner_exact_match_count']} / {overlap['protected_fixture_count']}",
        f"- Mismatches: {overlap['mismatch_count']}",
        f"- Continuity complete before / after: 0 / {composition['causal_history_complete_count']}",
        f"- Raw supplemental aggregate: `{acquisition['raw_supplemental_aggregate_sha256']}`",
        f"- Field-authority matrix: `{FIELD_AUTHORITY_MATRIX_SHA256}`",
        f"- Multi-source continuity: `{composition['multi_source_continuity_sha256']}`",
        f"- Earliest feature-ready fixture: `{readiness['earliest_feature_ready_fixture']}`",
        f"- Phase 7P resumed: {resume['phase7p_collection_resumed']}",
        f"- Frozen predictions created: {resume['created_count']}",
        "- Protected history, Phase 0-7D/7P evidence, model artifacts, feature schema, Phase 7 v2 specification and production ingestion remained unchanged.",
        "- No retraining, calibration, candidate activation, production provider migration, post-outcome prediction or Phase 8 authorization occurred.",
    ]
    (run_dir / "PHASE7E_IMPLEMENTATION_REPORT.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8", newline="\n"
    )
    source_paths = [
        ROOT / "src" / "evaluation" / "phase7e_multi_source_continuity.py",
        ROOT / "scripts" / "run_phase7e_multi_source_continuity.py",
        ROOT / "tests" / "unit" / "test_phase7e_multi_source_continuity.py",
    ]
    write_json(
        run_dir / "implementation_sources.json",
        {
            "schema_version": "ffps.phase7e_implementation_sources.v1",
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
        "schema_version": "ffps.phase7e_artifact_index.v1",
        "artifacts": artifacts,
        "artifact_count": len(artifacts),
        "aggregate_sha256": sha256_json(artifacts),
    }
    write_json(run_dir / "artifact_index.json", index)
    manifest = {
        "schema_version": "ffps.phase7e_run_manifest.v1",
        "phase": "7E",
        "run_id": RUN_ID,
        "status": "complete",
        "classification": classification,
        "finalized_at": utc_now(),
        "supplemental_provider": SUPPLEMENTAL_PROVIDER,
        "qualification_state": qualification["classification"],
        "raw_supplemental_aggregate_sha256": acquisition["raw_supplemental_aggregate_sha256"],
        "field_authority_matrix_sha256": FIELD_AUTHORITY_MATRIX_SHA256,
        "multi_source_continuity_sha256": composition["multi_source_continuity_sha256"],
        "earliest_feature_ready_fixture": readiness["earliest_feature_ready_fixture"],
        "phase7p_collection_resumed": resume["phase7p_collection_resumed"],
        "phase8_authorized": False,
        "production_path_changed": False,
        "protected_phase0_to_phase7d_unchanged": True,
        "artifact_index_sha256": sha256_file(run_dir / "artifact_index.json"),
        "implementation_sources_sha256": sha256_file(run_dir / "implementation_sources.json"),
    }
    write_json(run_dir / "run_manifest.json", manifest)
    return manifest


def verify(run_dir: Path, *, persist: bool = True) -> dict[str, Any]:
    errors: list[str] = []
    required = (
        "protected_before.json", "protected_after_final.json", "qualification_protocol.json",
        "acquisition_result.json", "historical_corner_overlap_audit.json", "qualification_result.json",
        "team_alias_qualification.json", "composition_result.json", "feature_readiness.json",
        "phase7p_resume_result.json", "test_results.json",
        "implementation_sources.json", "artifact_index.json", "run_manifest.json",
    )
    for name in required:
        if not (run_dir / name).is_file():
            errors.append(f"missing artifact: {name}")
    if not errors:
        before = read_json(run_dir / "protected_before.json")
        after = read_json(run_dir / "protected_after_final.json")
        if not protected_equal(before, after) or not protected_equal(before, protected_snapshot()):
            errors.append("protected Phase 0-7D identity mismatch")
        acquisition = read_json(run_dir / "acquisition_result.json")
        raw_components = []
        for token in SEASON_TOKENS:
            source = read_json(_source_path(run_dir, token))
            payload = _payload_path(run_dir, token)
            if sha256_file(payload) != source["response_sha256"] or payload.stat().st_size != source["response_size_bytes"]:
                errors.append(f"supplemental raw artifact mismatch: {token}")
            raw_components.append(
                {"season_token": token, "sha256": source["response_sha256"], "size_bytes": source["response_size_bytes"]}
            )
            events = cast(list[dict[str, Any]], json.loads(_events_path(run_dir, token).read_text(encoding="utf-8")))
            if any(verify_seal(value) for value in events):
                errors.append(f"normalized supplemental event mismatch: {token}")
            reconstructed_events = normalize_supplemental_csv(payload.read_bytes(), raw_source=source)
            if reconstructed_events != events:
                errors.append(f"offline supplemental normalization mismatch: {token}")
        if sha256_json(raw_components) != acquisition["raw_supplemental_aggregate_sha256"]:
            errors.append("raw supplemental aggregate mismatch")
        observations = ImmutableObservationStore(run_dir / "canonical_match_observations").records()
        if any(not verify_observation(value)["verified"] for value in observations):
            errors.append("canonical observation verification failed")
        composition = read_json(run_dir / "composition_result.json")
        if observation_aggregate(observations) != composition["observation_aggregate_sha256"]:
            errors.append("observation aggregate mismatch")
        readiness = read_json(run_dir / "feature_readiness.json")
        historical_rebuild = _historical_reproduction(run_dir)
        if (
            not historical_rebuild.get("reproduced")
            or historical_rebuild != readiness["historical_feature_reproduction"]
        ):
            errors.append("offline historical feature reproduction mismatch")
        if readiness["classification"] == "PROSPECTIVE_FEATURE_READY":
            if not (run_dir / "prospective_dry_run.json").is_file():
                errors.append("missing prospective dry-run artifact")
            else:
                try:
                    matrix, fixture, bundle = _build_target_matrix(
                        run_dir,
                        prediction_at=readiness["prediction_at"],
                        target_fixture_id=readiness["earliest_feature_ready_fixture"],
                    )
                    features = list(read_json(PHASE7_SPEC)["feature_contract"]["ordered_feature_names"])
                    vector = [float(fixture[value]) for value in features]
                    dry_run = read_json(run_dir / "prospective_dry_run.json")
                    if (
                        sha256_json(vector) != dry_run["feature_vector_sha256"]
                        or bundle != dry_run["causal_source_bundle"]
                    ):
                        errors.append("offline causal feature reconstruction mismatch")
                except Exception as exc:
                    errors.append(f"offline causal feature reconstruction failed: {exc}")
        specification = load_frozen_specification()
        cohort = ImmutableCohortStore(run_dir / "phase7p_resumed_cohort")
        predictions = cohort.records("prediction")
        if any(not verify_prediction_record(value, specification=specification)["verified"] for value in predictions):
            errors.append("resumed Phase 7P prediction verification failed")
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
        if manifest["classification"] not in ALLOWED_CLASSIFICATIONS:
            errors.append("unauthorized classification")
        if manifest["phase8_authorized"] or manifest["production_path_changed"]:
            errors.append("production or Phase 8 boundary violated")
        if not read_json(run_dir / "test_results.json")["no_new_unexplained_failures"]:
            errors.append("new unexplained test failures recorded")
    result = {
        "status": "VERIFIED" if not errors else "FAILED",
        "errors": errors,
        "classification": read_json(run_dir / "run_manifest.json").get("classification") if (run_dir / "run_manifest.json").is_file() else None,
        "protected_phase0_to_phase7d_unchanged": not any("protected" in value for value in errors),
        "offline_historical_feature_reproduction_verified": not any(
            "historical feature reproduction" in value for value in errors
        ),
        "offline_causal_feature_rebuild_required": (
            read_json(run_dir / "feature_readiness.json").get("classification")
            == "PROSPECTIVE_FEATURE_READY"
            if (run_dir / "feature_readiness.json").is_file()
            else False
        ),
        "offline_causal_feature_rebuild_verified": not any("feature reconstruction" in value for value in errors),
        "phase7p_prediction_records_verified": not any("Phase 7P prediction" in value for value in errors),
        "phase8_authorized": False,
        "production_path_changed": False,
    }
    if persist:
        write_json(run_dir / "offline_verification.json", result)
    return result


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    commands = value.add_subparsers(dest="command", required=True)
    for name in ("prepare", "acquire", "qualify", "compose", "verify"):
        command = commands.add_parser(name)
        command.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    feature = commands.add_parser("features")
    feature.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    feature.add_argument("--prediction-at", default=utc_now())
    resume = commands.add_parser("resume-phase7p")
    resume.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    resume.add_argument("--as-of", default=utc_now())
    final = commands.add_parser("finalize")
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
        elif args.command == "qualify":
            result = qualify(args.run_dir)
        elif args.command == "compose":
            result = compose(args.run_dir)
        elif args.command == "features":
            result = feature_assessment(args.run_dir, prediction_at=args.prediction_at)
        elif args.command == "resume-phase7p":
            result = resume_phase7p(args.run_dir, as_of=args.as_of)
        elif args.command == "finalize":
            result = finalize(
                args.run_dir,
                focused_xml=args.focused_xml,
                protected_xml=args.protected_xml,
                full_xml=args.full_xml,
            )
        else:
            result = verify(args.run_dir)
    except Exception as exc:
        logging.exception("Phase 7E failed")
        print(json.dumps({"status": "FAILED", "error": str(exc)}, indent=2))
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
