"""Refresh the qualified Phase 7E supplemental source without changing its contracts."""

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

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.run_phase7e_multi_source_continuity as phase7e
from scripts.run_phase7p_prospective import (
    _candidate_suite,
    _capture_rows_locked,
    _prediction_record,
    _verify_candidate_lineage,
    load_frozen_specification,
)
from src.cli.commands.prediction import _load_prediction_models
from src.evaluation.phase3_calibration import aggregate_directory
from src.evaluation.phase7_shadow import sha256_json
from src.evaluation.phase7e_multi_source_continuity import (
    FIELD_AUTHORITY_MATRIX_SHA256,
    HISTORICAL_CUTOFF,
    PROSPECTIVE_END,
    PROSPECTIVE_START,
    SUPPLEMENTAL_NORMALIZATION_CONTRACT_SHA256,
    SUPPLEMENTAL_PROVIDER,
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
from src.evaluation.phase7er_source_refresh import (
    BLOCKING_FIXTURES,
    REFRESH_SCHEMA_VERSION,
    assess_semantic_drift,
    audit_blocking_fixtures,
    continuity_classification,
    fixture_key,
    refresh_snapshot_identity,
    replace_current_season,
)
from src.evaluation.phase7p_prospective import ImmutableCohortStore, verify_prediction_record
from src.predictions.provenance import build_file_snapshot, sha256_bytes, sha256_file, source_event_index
from src.simulation.match_simulator import MatchSimulator


RUN_ID = "phase7er-qualified-source-refresh-79110f9a-20260830-v1"
DEFAULT_RUN_DIR = ROOT / "baselines" / "phase7er" / RUN_ID
PHASE7E_RUN = phase7e.DEFAULT_RUN_DIR
SOURCE_URL = "https://www.football-data.co.uk/mmz4281/2627/E0.csv"
SEASON_TOKEN = "2627"
ALLOWED_CLASSIFICATIONS = {
    "PROSPECTIVE_FEATURE_READY",
    "SUPPLEMENTAL_SOURCE_PARTIAL",
    "SUPPLEMENTAL_SOURCE_REFRESH_INCOMPATIBLE",
    "FEATURE_PIPELINE_INCOMPATIBLE",
    "PROSPECTIVE_WINDOW_MISSED",
}
INHERITED_FULL_SUITE_FAILURES = phase7e.INHERITED_FULL_SUITE_FAILURES


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
        raise Phase7EError("refresh raw persistence verification failed")
    os.replace(temporary, path)


def relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _safe_new_run_dir(run_dir: Path) -> None:
    root = (ROOT / "baselines" / "phase7er").resolve()
    resolved = run_dir.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise Phase7EError("Phase 7E-R evidence must remain beneath baselines/phase7er") from exc
    if resolved.exists():
        raise Phase7EError(f"Phase 7E-R evidence already exists: {resolved}")
    resolved.mkdir(parents=True)


def _phase7e_source_state() -> list[dict[str, Any]]:
    recorded = read_json(PHASE7E_RUN / "implementation_sources.json")["sources"]
    state: list[dict[str, Any]] = []
    for item in recorded:
        path = ROOT / str(item["path"])
        state.append(
            {
                "path": item["path"],
                "recorded_sha256": item["sha256"],
                "current_sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
                "unchanged": sha256_file(path) == item["sha256"] and path.stat().st_size == item["size_bytes"],
            }
        )
    return state


def protected_snapshot(*, deep_verify: bool = False) -> dict[str, Any]:
    source_state = _phase7e_source_state()
    if not all(value["unchanged"] for value in source_state):
        raise Phase7EError("frozen Phase 7E implementation source changed")
    if deep_verify:
        verification = phase7e.verify(PHASE7E_RUN, persist=False)
        if verification.get("status") != "VERIFIED":
            raise Phase7EError(f"frozen Phase 7E verification failed: {verification}")
    manifest = read_json(PHASE7E_RUN / "run_manifest.json")
    protected = read_json(PHASE7E_RUN / "protected_before.json")
    return {
        "captured_at": utc_now(),
        "phase7e_run_id": manifest["run_id"],
        "phase7e_classification": manifest["classification"],
        "phase7e_evidence_aggregate_sha256": aggregate_directory(PHASE7E_RUN),
        "phase7e_run_manifest_sha256": sha256_file(PHASE7E_RUN / "run_manifest.json"),
        "phase7e_artifact_index_sha256": sha256_file(PHASE7E_RUN / "artifact_index.json"),
        "phase7e_old_multi_source_continuity_sha256": manifest["multi_source_continuity_sha256"],
        "phase7e_old_raw_supplemental_aggregate_sha256": manifest["raw_supplemental_aggregate_sha256"],
        "phase7e_implementation_source_state_sha256": sha256_json(source_state),
        "protected_historical_base_sha256": protected["protected_historical_base_sha256"],
        "frozen_feature_matrix_sha256": protected["frozen_feature_matrix_sha256"],
        "phase7_specification_fingerprint": protected["phase7_specification_fingerprint"],
        "phase7_specification_sha256": protected["phase7_specification_sha256"],
        "phase7d_evidence_aggregate_sha256": protected["phase7d_evidence_aggregate_sha256"],
        "phase7p_evidence_aggregate_sha256": protected["phase7p_evidence_aggregate_sha256"],
        "active_bundle_sha256": protected["active_bundle_sha256"],
        "candidate_bundle_sha256": protected["candidate_bundle_sha256"],
        "deep_phase7e_verification_performed": deep_verify,
    }


def protected_equal(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    ignored = {"captured_at", "deep_phase7e_verification_performed"}
    return {key: value for key, value in left.items() if key not in ignored} == {
        key: value for key, value in right.items() if key not in ignored
    }


def assert_protected(run_dir: Path) -> None:
    before = read_json(run_dir / "protected_before.json")
    if not protected_equal(before, protected_snapshot()):
        raise Phase7EError("protected Phase 0-7E identity changed")


def _old_payload_path() -> Path:
    return PHASE7E_RUN / "raw_supplemental_snapshot" / "football-data-co-uk-E0-2627.csv"


def _old_source_path() -> Path:
    return PHASE7E_RUN / "raw_supplemental_snapshot" / "football-data-co-uk-E0-2627.source.json"


def _old_events_path() -> Path:
    return PHASE7E_RUN / "normalized_supplemental_events" / "E0-2627.json"


def _payload_path(run_dir: Path) -> Path:
    return run_dir / "raw_supplemental_refresh" / "football-data-co-uk-E0-2627.csv"


def _source_path(run_dir: Path) -> Path:
    return run_dir / "raw_supplemental_refresh" / "football-data-co-uk-E0-2627.source.json"


def _events_path(run_dir: Path) -> Path:
    return run_dir / "normalized_supplemental_refresh" / "E0-2627.json"


def _read_events(path: Path) -> list[dict[str, Any]]:
    return cast(list[dict[str, Any]], json.loads(path.read_text(encoding="utf-8")))


def prepare(run_dir: Path) -> dict[str, Any]:
    _safe_new_run_dir(run_dir)
    before = protected_snapshot(deep_verify=True)
    write_json(run_dir / "protected_before.json", before)
    old_source = read_json(_old_source_path())
    contract = {
        "schema_version": REFRESH_SCHEMA_VERSION,
        "frozen_before_refresh": True,
        "phase7e_run_id": phase7e.RUN_ID,
        "phase7e_classification": "SUPPLEMENTAL_SOURCE_PARTIAL",
        "phase7e_qualification_state": "SUPPLEMENTAL_SOURCE_QUALIFIED",
        "same_provider": SUPPLEMENTAL_PROVIDER,
        "same_product_url": SOURCE_URL,
        "season_token": SEASON_TOKEN,
        "prior_response_sha256": old_source["response_sha256"],
        "prior_multi_source_continuity_sha256": read_json(PHASE7E_RUN / "run_manifest.json")[
            "multi_source_continuity_sha256"
        ],
        "blocking_fixtures": list(BLOCKING_FIXTURES),
        "required_continuity_count": 215,
        "semantic_drift_checks": [
            "exact CSV header equality",
            "required Div/Date/HomeTeam/AwayTeam/FTHG/FTAG/HC/AC fields",
            "E0/2627 product identity",
            "unchanged qualified values for every prior row",
            "unchanged home/away orientation",
            "unchanged Phase 7E normalization contract",
        ],
        "full_requalification_required_only_on_drift": True,
        "normalization_contract_sha256": SUPPLEMENTAL_NORMALIZATION_CONTRACT_SHA256,
        "field_authority_matrix_sha256": FIELD_AUTHORITY_MATRIX_SHA256,
        "team_alias_extension_sha256": TEAM_ALIAS_EXTENSION_SHA256,
        "feature_schema_sha256": sha256_json(
            read_json(phase7e.PHASE7_SPEC)["feature_contract"]["ordered_feature_names"]
        ),
        "production_path_changed": False,
        "phase8_authorized": False,
    }
    contract["refresh_contract_sha256"] = sha256_json(contract)
    write_json(run_dir / "refresh_contract.json", contract)
    result = {
        "status": "REFRESH_CONTRACT_FROZEN",
        "refresh_contract_sha256": contract["refresh_contract_sha256"],
        "phase7e_evidence_frozen": True,
        "same_provider_product_only": True,
        "full_requalification_performed": False,
    }
    write_json(run_dir / "preparation_result.json", result)
    return result


def _full_source_components(new_source: Mapping[str, Any]) -> list[dict[str, Any]]:
    components: list[dict[str, Any]] = []
    for token in phase7e.SEASON_TOKENS:
        source = (
            new_source
            if token == SEASON_TOKEN
            else read_json(
                PHASE7E_RUN
                / "raw_supplemental_snapshot"
                / f"football-data-co-uk-E0-{token}.source.json"
            )
        )
        components.append(
            {
                "season_token": token,
                "sha256": source["response_sha256"],
                "size_bytes": source["response_size_bytes"],
            }
        )
    return components


def acquire(run_dir: Path) -> dict[str, Any]:
    assert_protected(run_dir)
    if (run_dir / "acquisition_result.json").exists():
        raise Phase7EError("Phase 7E-R source snapshot is already frozen")
    response = requests.get(SOURCE_URL, timeout=30)
    response.raise_for_status()
    payload = response.content
    retrieved_at = utc_now()
    write_bytes(_payload_path(run_dir), payload)
    source = raw_source_identity(
        payload,
        season_token=SEASON_TOKEN,
        retrieved_at=retrieved_at,
        request_url=str(response.url),
    )
    source["path"] = relative(_payload_path(run_dir))
    old_source = read_json(_old_source_path())
    source["prior_response_sha256"] = old_source["response_sha256"]
    source["refresh_snapshot_sha256"] = refresh_snapshot_identity(
        prior_response_sha256=old_source["response_sha256"],
        response_sha256=source["response_sha256"],
        response_size_bytes=source["response_size_bytes"],
        retrieved_at=source["retrieved_at"],
        request_url=source["request_url"],
    )
    normalization_error: str | None = None
    try:
        events = normalize_supplemental_csv(payload, raw_source=source)
    except Exception as exc:
        events = []
        normalization_error = str(exc)
    source["normalized_completed_event_count"] = len(events)
    write_json(_source_path(run_dir), source)
    write_json(_events_path(run_dir), events)
    components = _full_source_components(source)
    result = {
        "status": "REFRESH_SNAPSHOT_FROZEN",
        "provider": SUPPLEMENTAL_PROVIDER,
        "product_url": SOURCE_URL,
        "season_token": SEASON_TOKEN,
        "prior_response_sha256": old_source["response_sha256"],
        "response_sha256": source["response_sha256"],
        "response_size_bytes": source["response_size_bytes"],
        "retrieved_at": source["retrieved_at"],
        "refresh_snapshot_sha256": source["refresh_snapshot_sha256"],
        "full_supplemental_source_aggregate_sha256": sha256_json(components),
        "full_supplemental_source_components": components,
        "normalized_completed_event_count": len(events),
        "normalized_event_aggregate_sha256": sha256_json(
            sorted(str(value["record_fingerprint"]) for value in events)
        ),
        "normalization_error": normalization_error,
        "same_provider_product": str(response.url).endswith("/2627/E0.csv"),
        "prior_phase7e_snapshot_overwritten": False,
        "credentials_retained": False,
    }
    write_json(run_dir / "acquisition_result.json", result)
    return result


def _primary_completed() -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    for raw, source in phase7e._primary_sources():
        date = str(raw.get("utcDate", ""))[:10]
        if str(raw.get("status")) == "FINISHED" and HISTORICAL_CUTOFF < date < PROSPECTIVE_START:
            values.append(normalize_primary_event(raw, raw_source=source))
    return values


def assess(run_dir: Path) -> dict[str, Any]:
    assert_protected(run_dir)
    old_payload = _old_payload_path().read_bytes()
    new_payload = _payload_path(run_dir).read_bytes()
    old_events = _read_events(_old_events_path())
    new_events = _read_events(_events_path(run_dir))
    source = read_json(_source_path(run_dir))
    sentinel = assess_semantic_drift(
        old_payload=old_payload,
        new_payload=new_payload,
        old_events=old_events,
        new_events=new_events,
        new_source=source,
    )
    acquisition = read_json(run_dir / "acquisition_result.json")
    if acquisition.get("normalization_error"):
        sentinel["errors"].append(
            f"refresh normalization failed: {acquisition['normalization_error']}"
        )
        sentinel["status"] = "INCOMPATIBLE"
        sentinel["classification"] = "SUPPLEMENTAL_SOURCE_REFRESH_INCOMPATIBLE"
        sentinel["sentinel_sha256"] = sha256_json(
            {key: value for key, value in sentinel.items() if key != "sentinel_sha256"}
        )
    write_json(run_dir / "semantic_drift_sentinel.json", sentinel)
    primary = _primary_completed()
    primary_by_id = {str(value["provider_event_id"]): value for value in primary}
    supplemental_by_key = {fixture_key(value): value for value in new_events}
    audit = audit_blocking_fixtures(
        primary_by_id=primary_by_id,
        supplemental_by_key=supplemental_by_key,
        raw_snapshot_sha256=source["response_sha256"],
    )
    write_json(run_dir / "blocking_fixture_audit.json", audit)
    result = {
        "status": "REFRESH_ASSESSED",
        "classification": (
            "SUPPLEMENTAL_SOURCE_REFRESH_INCOMPATIBLE"
            if sentinel["status"] != "COMPATIBLE"
            else (
                "BLOCKING_FIXTURES_AVAILABLE"
                if audit["all_complete"]
                else "SUPPLEMENTAL_SOURCE_PARTIAL"
            )
        ),
        "semantic_drift_compatible": sentinel["status"] == "COMPATIBLE",
        "sentinel_sha256": sentinel["sentinel_sha256"],
        "blocking_fixture_audit_sha256": audit["audit_sha256"],
        "blocking_fixture_complete_count": audit["complete_fixture_count"],
        "blocking_fixture_required_count": audit["required_fixture_count"],
        "full_phase7e_requalification_performed": False,
    }
    write_json(run_dir / "refresh_assessment.json", result)
    return result


def _supplemental_events(run_dir: Path) -> list[dict[str, Any]]:
    frozen = phase7e._supplemental_events(PHASE7E_RUN)
    refreshed = _read_events(_events_path(run_dir))
    return replace_current_season(frozen, refreshed)


def compose(run_dir: Path) -> dict[str, Any]:
    assert_protected(run_dir)
    assessment = read_json(run_dir / "refresh_assessment.json")
    acquisition = read_json(run_dir / "acquisition_result.json")
    if not assessment["semantic_drift_compatible"]:
        result = {
            "status": "COMPOSITION_BLOCKED",
            "classification": "SUPPLEMENTAL_SOURCE_REFRESH_INCOMPATIBLE",
            "primary_completed_event_count": 215,
            "canonical_match_observation_count": 0,
            "causal_history_complete_count": 0,
            "unresolved_count": 215,
        }
        write_json(run_dir / "composition_result.json", result)
        return result
    supplemental = unique_fixture_index(
        [
            value
            for value in _supplemental_events(run_dir)
            if HISTORICAL_CUTOFF < str(value["fixture_date"]) < PROSPECTIVE_START
        ]
    )
    primary = _primary_completed()
    if len(primary) != len({str(value["provider_event_id"]) for value in primary}):
        raise Phase7EError("duplicate primary provider event ID")
    store = ImmutableObservationStore(run_dir / "canonical_match_observations")
    unresolved: list[dict[str, Any]] = []
    for event in primary:
        supplement = supplemental.get(fixture_key(event))
        if supplement is None:
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
            store.persist(compose_observation(event, supplement))
        except Phase7EError as exc:
            unresolved.append(
                {
                    "provider_event_id": event["provider_event_id"],
                    "fixture_date": event["fixture_date"],
                    "home_team": event["home_team"],
                    "away_team": event["away_team"],
                    "reason": str(exc),
                }
            )
    observations = store.records()
    primary_aggregate = sha256_json(sorted(str(value["record_fingerprint"]) for value in primary))
    observation_sha = observation_aggregate(observations)
    snapshot = {
        "schema_version": "ffps.phase7er_multi_source_continuity_snapshot.v1",
        "protected_historical_base_sha256": sha256_file(phase7e.PROTECTED_BASE),
        "primary_provider": "football-data.org:v4",
        "primary_snapshot_state": "FROZEN_AUTHORITATIVE_PHASE7D_INPUT",
        "primary_resolved_aggregate_sha256": primary_aggregate,
        "supplemental_provider": SUPPLEMENTAL_PROVIDER,
        "supplemental_refresh_snapshot_sha256": acquisition["refresh_snapshot_sha256"],
        "supplemental_full_source_aggregate_sha256": acquisition[
            "full_supplemental_source_aggregate_sha256"
        ],
        "field_authority_matrix_sha256": FIELD_AUTHORITY_MATRIX_SHA256,
        "team_alias_extension_sha256": TEAM_ALIAS_EXTENSION_SHA256,
        "normalization_contract_sha256": SUPPLEMENTAL_NORMALIZATION_CONTRACT_SHA256,
        "observation_aggregate_sha256": observation_sha,
        "ordered_observation_fingerprints": sorted(
            str(value["record_fingerprint"]) for value in observations
        ),
        "prior_phase7e_multi_source_continuity_sha256": read_json(
            PHASE7E_RUN / "run_manifest.json"
        )["multi_source_continuity_sha256"],
    }
    snapshot["multi_source_continuity_sha256"] = sha256_json(snapshot)
    write_json(run_dir / "multi_source_continuity_snapshot.json", snapshot)
    audit = read_json(run_dir / "blocking_fixture_audit.json")
    verified_count = sum(bool(verify_observation(value)["verified"]) for value in observations)
    complete = (
        len(primary) == len(observations) == verified_count == 215
        and not unresolved
        and audit["all_complete"]
    )
    classification = continuity_classification(
        drift_compatible=True,
        complete_count=verified_count if complete else min(verified_count, 214),
    )
    result = {
        "status": "MULTI_SOURCE_CONTINUITY_FROZEN" if complete else "SUPPLEMENTAL_SOURCE_PARTIAL",
        "classification": classification,
        "continuity_interval": {
            "after_exclusive": HISTORICAL_CUTOFF,
            "before_exclusive": PROSPECTIVE_START,
        },
        "required_continuity_count": 215,
        "primary_completed_event_count": len(primary),
        "canonical_match_observation_count": len(observations),
        "causal_history_complete_count": verified_count,
        "unresolved_count": len(unresolved),
        "unresolved": unresolved,
        "all_five_blocking_fixtures_complete": audit["all_complete"],
        "primary_event_560561_has_valid_supplemental_corners": audit[
            "primary_event_560561_has_valid_supplemental_corners"
        ],
        "all_home_away_corners_present": all(
            cast(Mapping[str, Any], value["fields"])["home_corners"]["value"] is not None
            and cast(Mapping[str, Any], value["fields"])["away_corners"]["value"] is not None
            for value in observations
        ),
        "primary_resolved_aggregate_sha256": primary_aggregate,
        "supplemental_refresh_snapshot_sha256": acquisition["refresh_snapshot_sha256"],
        "supplemental_full_source_aggregate_sha256": acquisition[
            "full_supplemental_source_aggregate_sha256"
        ],
        "observation_aggregate_sha256": observation_sha,
        "multi_source_continuity_sha256": snapshot["multi_source_continuity_sha256"],
        "prior_phase7e_multi_source_continuity_sha256": snapshot[
            "prior_phase7e_multi_source_continuity_sha256"
        ],
        "prior_phase7e_snapshot_overwritten": False,
        "protected_base_rewritten": False,
        "production_path_changed": False,
    }
    write_json(run_dir / "composition_result.json", result)
    return result


def _eligible_scheduled(prediction_at: str) -> list[dict[str, Any]]:
    return phase7e._scheduled_primary(prediction_at)


def _build_target_matrix(
    run_dir: Path, *, prediction_at: str, target_fixture_id: str | None = None
) -> tuple[pd.DataFrame, pd.Series, dict[str, Any]]:
    composition = read_json(run_dir / "composition_result.json")
    observations = ImmutableObservationStore(
        run_dir / "canonical_match_observations"
    ).effective_records(as_of=prediction_at)
    scheduled = _eligible_scheduled(prediction_at)
    if target_fixture_id:
        scheduled = [
            value for value in scheduled if str(value["canonical_fixture_id"]) == target_fixture_id
        ]
    if not scheduled:
        raise Phase7EError("no still-unsettled Phase 7 fixture is available")
    target = scheduled[0]
    acquisition = read_json(run_dir / "acquisition_result.json")
    bundle = build_causal_bundle(
        protected_base_sha256=sha256_file(phase7e.PROTECTED_BASE),
        primary_extension_sha256=composition["primary_resolved_aggregate_sha256"],
        supplemental_source_sha256=acquisition["full_supplemental_source_aggregate_sha256"],
        observations=observations,
        target_fixture_id=str(target["canonical_fixture_id"]),
        target_kickoff_at=str(target["kickoff_at"]),
        prediction_at=prediction_at,
    )
    included = set(bundle["ordered_observation_fingerprints"])
    causal = [value for value in observations if value["record_fingerprint"] in included]
    composite = pd.concat(
        [
            phase7e._adapt_protected_base(),
            pd.DataFrame([phase7e._observation_row(value) for value in causal]),
            pd.DataFrame([phase7e._target_row(target)]),
        ],
        ignore_index=True,
        sort=False,
    )
    matrix = phase7e._run_pipeline(composite)
    artifacts = [
        {"role": "protected_historical_base", "sha256": sha256_file(phase7e.PROTECTED_BASE)},
        {"role": "primary_resolved_extension", "sha256": composition["primary_resolved_aggregate_sha256"]},
        {"role": "supplemental_refresh", "sha256": acquisition["refresh_snapshot_sha256"]},
        {"role": "supplemental_full_source", "sha256": acquisition["full_supplemental_source_aggregate_sha256"]},
        {"role": "composed_observations", "sha256": composition["observation_aggregate_sha256"]},
        {"role": "field_authority_matrix", "sha256": FIELD_AUTHORITY_MATRIX_SHA256},
        {"role": "team_alias_extension", "sha256": TEAM_ALIAS_EXTENSION_SHA256},
        {"role": "target_causal_bundle", "sha256": bundle["multi_source_continuity_sha256"]},
    ]
    snapshot = build_file_snapshot(
        frame=matrix,
        artifacts=artifacts,
        source_type="phase7er_evaluation_only_refreshed_multi_source_causal_history",
    )
    snapshot["multi_source_continuity_sha256"] = bundle["multi_source_continuity_sha256"]
    snapshot["causal_bundle"] = bundle
    matrix.attrs["ffps_data_snapshot"] = snapshot
    matrix.attrs["ffps_source_events"] = source_event_index(composite)
    target_rows = matrix.loc[
        matrix["match_id"].astype(str).eq(str(target["canonical_fixture_id"]))
    ]
    if len(target_rows) != 1:
        raise Phase7EError("target feature row is not unique")
    return matrix, target_rows.iloc[0], bundle


def feature_assessment(run_dir: Path, *, prediction_at: str) -> dict[str, Any]:
    assert_protected(run_dir)
    prediction_at = pd.Timestamp(prediction_at).isoformat()
    assessment = read_json(run_dir / "refresh_assessment.json")
    composition = read_json(run_dir / "composition_result.json")
    reproduction = phase7e._historical_reproduction(PHASE7E_RUN)
    features = list(read_json(phase7e.PHASE7_SPEC)["feature_contract"]["ordered_feature_names"])
    base = {
        "prediction_at": prediction_at,
        "historical_feature_reproduction": reproduction,
        "ordered_feature_count": len(features),
        "ordered_feature_schema_sha256": sha256_json(features),
        "earliest_feature_ready_fixture": None,
        "earliest_feature_ready_kickoff": None,
        "prospective_dry_run_verified": False,
        "production_path_changed": False,
    }
    if not assessment["semantic_drift_compatible"]:
        result = {
            **base,
            "status": "FEATURE_ASSESSMENT_BLOCKED",
            "classification": "SUPPLEMENTAL_SOURCE_REFRESH_INCOMPATIBLE",
        }
        write_json(run_dir / "feature_readiness.json", result)
        return result
    if composition["causal_history_complete_count"] != 215:
        result = {
            **base,
            "status": "FEATURE_ASSESSMENT_BLOCKED",
            "classification": "SUPPLEMENTAL_SOURCE_PARTIAL",
        }
        write_json(run_dir / "feature_readiness.json", result)
        return result
    if (
        not reproduction.get("reproduced")
        or len(features) != 30
        or reproduction.get("maximum_absolute_difference", float("inf")) > 1e-12
    ):
        result = {
            **base,
            "status": "FEATURE_ASSESSMENT_FAILED",
            "classification": "FEATURE_PIPELINE_INCOMPATIBLE",
        }
        write_json(run_dir / "feature_readiness.json", result)
        return result
    scheduled = _eligible_scheduled(prediction_at)
    if not scheduled:
        result = {
            **base,
            "status": "PROSPECTIVE_WINDOW_CLOSED",
            "classification": "PROSPECTIVE_WINDOW_MISSED",
        }
        write_json(run_dir / "feature_readiness.json", result)
        return result
    try:
        matrix, fixture, bundle = _build_target_matrix(run_dir, prediction_at=prediction_at)
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
            fixture,
            active_suite,
            simulator,
            matrix,
            label="active-dry-run",
            prediction_timestamp=prediction_at,
        )
        simulator.rng.bit_generator.state = copy.deepcopy(state)
        candidate, candidate_verification = _prediction_record(
            fixture,
            candidate_suite,
            simulator,
            matrix,
            label="candidate-dry-run",
            prediction_timestamp=prediction_at,
        )
        lineage = _verify_candidate_lineage()
        feature_identity_equal = (
            active["feature_provenance"]["feature_bundle_sha256"]
            == candidate["feature_provenance"]["feature_bundle_sha256"]
        )
        dry_run = {
            "target_fixture_id": str(fixture["match_id"]),
            "target_kickoff_at": pd.Timestamp(fixture["date"]).isoformat(),
            "prediction_at": prediction_at,
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
            finite
            and feature_identity_equal
            and dry_run["active_bundle_accepted"]
            and dry_run["candidate_bundle_accepted"]
            and dry_run["candidate_training_lineage_verified"]
        )
        result = {
            **base,
            "status": "FEATURE_ASSESSMENT_COMPLETE" if ready else "FEATURE_ASSESSMENT_FAILED",
            "classification": (
                "PROSPECTIVE_FEATURE_READY" if ready else "FEATURE_PIPELINE_INCOMPATIBLE"
            ),
            "earliest_feature_ready_fixture": str(fixture["match_id"]) if ready else None,
            "earliest_feature_ready_kickoff": (
                pd.Timestamp(fixture["date"]).isoformat() if ready else None
            ),
            "prospective_dry_run_verified": ready,
        }
    except Exception as exc:
        result = {
            **base,
            "status": "FEATURE_ASSESSMENT_FAILED",
            "classification": "FEATURE_PIPELINE_INCOMPATIBLE",
            "error": str(exc),
        }
    write_json(run_dir / "feature_readiness.json", result)
    return result


def resume_phase7p(run_dir: Path, *, as_of: str) -> dict[str, Any]:
    assert_protected(run_dir)
    as_of = pd.Timestamp(as_of).isoformat()
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
    try:
        matrix, _, bundle = _build_target_matrix(
            run_dir, prediction_at=as_of, target_fixture_id=fixture_id
        )
    except Phase7EError as exc:
        readiness["classification"] = "PROSPECTIVE_WINDOW_MISSED"
        readiness["status"] = "PROSPECTIVE_WINDOW_CLOSED_BEFORE_CAPTURE"
        readiness["earliest_feature_ready_fixture"] = None
        readiness["earliest_feature_ready_kickoff"] = None
        write_json(run_dir / "feature_readiness.json", readiness)
        result = {
            "status": "NOT_RESUMED_WINDOW_MISSED",
            "phase7p_collection_resumed": False,
            "created_count": 0,
            "reason": str(exc),
            "post_outcome_predictions_created": False,
            "phase8_authorized": False,
        }
        write_json(run_dir / "phase7p_resume_result.json", result)
        return result
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
    after = protected_snapshot(deep_verify=True)
    if not protected_equal(before, after):
        raise Phase7EError("protected Phase 0-7E identities changed before finalization")
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
        raise Phase7EError("focused Phase 7E-R tests failed")
    if tests["protected"]["failures"] or tests["protected"]["errors"]:
        raise Phase7EError("protected Phase 0-7E regressions failed")
    if unexplained:
        raise Phase7EError(f"new unexplained full-suite failures: {unexplained}")
    write_json(run_dir / "test_results.json", tests)
    acquisition = read_json(run_dir / "acquisition_result.json")
    sentinel = read_json(run_dir / "semantic_drift_sentinel.json")
    audit = read_json(run_dir / "blocking_fixture_audit.json")
    composition = read_json(run_dir / "composition_result.json")
    readiness = read_json(run_dir / "feature_readiness.json")
    resume = read_json(run_dir / "phase7p_resume_result.json")
    classification = str(readiness["classification"])
    if classification not in ALLOWED_CLASSIFICATIONS:
        raise Phase7EError(f"unauthorized Phase 7E-R classification: {classification}")
    report = [
        "# FFPS Phase 7E-R — Qualified Source Refresh",
        "",
        f"- Final classification: `{classification}`",
        f"- Supplemental provider/product: `{SUPPLEMENTAL_PROVIDER}` / `{SOURCE_URL}`",
        f"- Refresh snapshot: `{acquisition['refresh_snapshot_sha256']}`",
        f"- Raw response: `{acquisition['response_sha256']}`",
        f"- Semantic drift sentinel: `{sentinel['status']}`",
        f"- Blocking fixtures complete: {audit['complete_fixture_count']} / {audit['required_fixture_count']}",
        f"- Continuity complete: {composition['causal_history_complete_count']} / 215",
        f"- Primary 560561 has supplemental corners: {composition.get('primary_event_560561_has_valid_supplemental_corners', False)}",
        f"- New multi-source continuity: `{composition.get('multi_source_continuity_sha256')}`",
        f"- Frozen Phase 7E continuity remains: `{read_json(PHASE7E_RUN / 'run_manifest.json')['multi_source_continuity_sha256']}`",
        f"- Historical maximum feature difference: {readiness['historical_feature_reproduction'].get('maximum_absolute_difference')}",
        f"- Earliest feature-ready fixture: `{readiness['earliest_feature_ready_fixture']}`",
        f"- Phase 7P resumed: {resume['phase7p_collection_resumed']}",
        f"- Frozen predictions created: {resume['created_count']}",
        "- No provider qualification, source substitution, manual data, feature change, fitting, calibration, activation, production mutation, or Phase 8 authorization occurred.",
    ]
    (run_dir / "PHASE7ER_IMPLEMENTATION_REPORT.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8", newline="\n"
    )
    source_paths = [
        ROOT / "src" / "evaluation" / "phase7e_multi_source_continuity.py",
        ROOT / "src" / "evaluation" / "phase7er_source_refresh.py",
        ROOT / "scripts" / "run_phase7e_multi_source_continuity.py",
        ROOT / "scripts" / "run_phase7er_source_refresh.py",
        ROOT / "tests" / "unit" / "test_phase7er_source_refresh.py",
    ]
    write_json(
        run_dir / "implementation_sources.json",
        {
            "schema_version": "ffps.phase7er_implementation_sources.v1",
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
        "schema_version": "ffps.phase7er_artifact_index.v1",
        "artifacts": artifacts,
        "artifact_count": len(artifacts),
        "aggregate_sha256": sha256_json(artifacts),
    }
    write_json(run_dir / "artifact_index.json", index)
    manifest = {
        "schema_version": "ffps.phase7er_run_manifest.v1",
        "phase": "7E-R",
        "run_id": RUN_ID,
        "status": "complete",
        "classification": classification,
        "finalized_at": utc_now(),
        "supplemental_provider": SUPPLEMENTAL_PROVIDER,
        "refresh_snapshot_sha256": acquisition["refresh_snapshot_sha256"],
        "raw_response_sha256": acquisition["response_sha256"],
        "semantic_drift_compatible": sentinel["status"] == "COMPATIBLE",
        "causal_history_complete_count": composition["causal_history_complete_count"],
        "multi_source_continuity_sha256": composition.get("multi_source_continuity_sha256"),
        "prior_phase7e_multi_source_continuity_sha256": read_json(PHASE7E_RUN / "run_manifest.json")[
            "multi_source_continuity_sha256"
        ],
        "earliest_feature_ready_fixture": readiness["earliest_feature_ready_fixture"],
        "phase7p_collection_resumed": resume["phase7p_collection_resumed"],
        "phase8_authorized": False,
        "production_path_changed": False,
        "protected_phase0_to_phase7e_unchanged": True,
        "artifact_index_sha256": sha256_file(run_dir / "artifact_index.json"),
        "implementation_sources_sha256": sha256_file(run_dir / "implementation_sources.json"),
    }
    write_json(run_dir / "run_manifest.json", manifest)
    return manifest


def verify(run_dir: Path, *, persist: bool = True) -> dict[str, Any]:
    errors: list[str] = []
    required = (
        "protected_before.json",
        "protected_after_final.json",
        "refresh_contract.json",
        "acquisition_result.json",
        "semantic_drift_sentinel.json",
        "blocking_fixture_audit.json",
        "refresh_assessment.json",
        "composition_result.json",
        "multi_source_continuity_snapshot.json",
        "feature_readiness.json",
        "phase7p_resume_result.json",
        "test_results.json",
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
        try:
            current = protected_snapshot(deep_verify=True)
            if not protected_equal(before, after) or not protected_equal(before, current):
                errors.append("protected Phase 0-7E identity mismatch")
        except Exception as exc:
            errors.append(f"protected Phase 0-7E verification failed: {exc}")
        source = read_json(_source_path(run_dir))
        payload = _payload_path(run_dir).read_bytes()
        if sha256_bytes(payload) != source["response_sha256"] or len(payload) != source["response_size_bytes"]:
            errors.append("refreshed raw source mismatch")
        events = _read_events(_events_path(run_dir))
        try:
            rebuilt = normalize_supplemental_csv(payload, raw_source=source)
            if rebuilt != events or any(verify_seal(value) for value in events):
                errors.append("offline refresh normalization mismatch")
        except Exception as exc:
            if not read_json(run_dir / "acquisition_result.json").get("normalization_error"):
                errors.append(f"offline refresh normalization failed: {exc}")
        rebuilt_sentinel = assess_semantic_drift(
            old_payload=_old_payload_path().read_bytes(),
            new_payload=payload,
            old_events=_read_events(_old_events_path()),
            new_events=events,
            new_source=source,
        )
        recorded_sentinel = read_json(run_dir / "semantic_drift_sentinel.json")
        if not read_json(run_dir / "acquisition_result.json").get("normalization_error") and rebuilt_sentinel != recorded_sentinel:
            errors.append("offline semantic drift sentinel mismatch")
        primary = _primary_completed()
        rebuilt_audit = audit_blocking_fixtures(
            primary_by_id={str(value["provider_event_id"]): value for value in primary},
            supplemental_by_key={fixture_key(value): value for value in events},
            raw_snapshot_sha256=source["response_sha256"],
        )
        if rebuilt_audit != read_json(run_dir / "blocking_fixture_audit.json"):
            errors.append("offline blocking-fixture audit mismatch")
        observations = ImmutableObservationStore(run_dir / "canonical_match_observations").records()
        composition = read_json(run_dir / "composition_result.json")
        if any(not verify_observation(value)["verified"] for value in observations):
            errors.append("canonical observation verification failed")
        if observations and observation_aggregate(observations) != composition.get("observation_aggregate_sha256"):
            errors.append("observation aggregate mismatch")
        readiness = read_json(run_dir / "feature_readiness.json")
        reproduction = phase7e._historical_reproduction(PHASE7E_RUN)
        if reproduction != readiness["historical_feature_reproduction"]:
            errors.append("offline historical feature reproduction mismatch")
        if readiness["classification"] == "PROSPECTIVE_FEATURE_READY":
            try:
                _, fixture, bundle = _build_target_matrix(
                    run_dir,
                    prediction_at=readiness["prediction_at"],
                    target_fixture_id=readiness["earliest_feature_ready_fixture"],
                )
                features = list(read_json(phase7e.PHASE7_SPEC)["feature_contract"]["ordered_feature_names"])
                vector = [float(fixture[value]) for value in features]
                dry = read_json(run_dir / "prospective_dry_run.json")
                if sha256_json(vector) != dry["feature_vector_sha256"] or bundle != dry["causal_source_bundle"]:
                    errors.append("offline causal feature reconstruction mismatch")
            except Exception as exc:
                errors.append(f"offline causal feature reconstruction failed: {exc}")
        specification = load_frozen_specification()
        predictions = ImmutableCohortStore(run_dir / "phase7p_resumed_cohort").records("prediction")
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
        "classification": (
            read_json(run_dir / "run_manifest.json").get("classification")
            if (run_dir / "run_manifest.json").is_file()
            else None
        ),
        "protected_phase0_to_phase7e_unchanged": not any("protected" in value for value in errors),
        "old_phase7e_evidence_unchanged": not any("Phase 7E" in value for value in errors),
        "offline_refresh_normalization_verified": not any("normalization" in value for value in errors),
        "offline_semantic_drift_sentinel_verified": not any("sentinel" in value for value in errors),
        "offline_historical_feature_reproduction_verified": not any("historical feature" in value for value in errors),
        "offline_causal_feature_rebuild_verified": not any("causal feature" in value for value in errors),
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
    for name in ("prepare", "acquire", "assess", "compose", "verify"):
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
        elif args.command == "assess":
            result = assess(args.run_dir)
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
        logging.exception("Phase 7E-R failed")
        print(json.dumps({"status": "FAILED", "error": str(exc)}, indent=2))
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
