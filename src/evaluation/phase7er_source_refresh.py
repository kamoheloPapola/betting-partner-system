"""Fail-closed helpers for the Phase 7E qualified-source refresh."""

from __future__ import annotations

import csv
import io
from typing import Any, Mapping, Sequence, cast

from src.evaluation.phase7_shadow import sha256_json
from src.evaluation.phase7e_multi_source_continuity import (
    SUPPLEMENTAL_NORMALIZATION_CONTRACT_SHA256,
    SUPPLEMENTAL_PROVIDER,
)


REFRESH_SCHEMA_VERSION = "ffps.phase7er_refresh_contract.v1"
REFRESH_SENTINEL_SCHEMA_VERSION = "ffps.phase7er_semantic_drift_sentinel.v1"
REQUIRED_COLUMNS = ("Div", "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "HC", "AC")
BLOCKING_FIXTURES: tuple[dict[str, str], ...] = (
    {
        "provider_event_id": "560555",
        "fixture_date": "2026-08-28",
        "home_team": "CRYSTAL PALACE",
        "away_team": "MANCHESTER CITY",
    },
    {
        "provider_event_id": "560552",
        "fixture_date": "2026-08-29",
        "home_team": "LIVERPOOL",
        "away_team": "NOTTINGHAM FOREST",
    },
    {
        "provider_event_id": "560560",
        "fixture_date": "2026-08-29",
        "home_team": "BOURNEMOUTH",
        "away_team": "EVERTON",
    },
    {
        "provider_event_id": "560561",
        "fixture_date": "2026-08-29",
        "home_team": "COVENTRY",
        "away_team": "HULL",
    },
    {
        "provider_event_id": "560558",
        "fixture_date": "2026-08-29",
        "home_team": "TOTTENHAM",
        "away_team": "NEWCASTLE UNITED",
    },
)


def csv_header(payload: bytes) -> list[str]:
    """Return the provider header without altering the frozen normalizer."""
    try:
        text = payload.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = payload.decode("cp1252")
    reader = csv.reader(io.StringIO(text))
    try:
        return [str(value) for value in next(reader)]
    except StopIteration:
        return []


def fixture_key(value: Mapping[str, Any]) -> tuple[str, str, str]:
    return (
        str(value["fixture_date"]),
        str(value["home_team"]),
        str(value["away_team"]),
    )


def semantic_projection(value: Mapping[str, Any]) -> dict[str, Any]:
    """Fields whose meanings were qualified in Phase 7E, excluding provenance."""
    return {
        "fixture_date": str(value["fixture_date"]),
        "season_start_year": int(value["season_start_year"]),
        "league": str(value["league"]),
        "home_team": str(value["home_team"]),
        "away_team": str(value["away_team"]),
        "home_goals": value.get("home_goals"),
        "away_goals": value.get("away_goals"),
        "home_corners": value.get("home_corners"),
        "away_corners": value.get("away_corners"),
        "completed": value.get("completed"),
        "normalization_contract_sha256": value.get("normalization_contract_sha256"),
    }


def refresh_snapshot_identity(
    *,
    prior_response_sha256: str,
    response_sha256: str,
    response_size_bytes: int,
    retrieved_at: str,
    request_url: str,
) -> str:
    """Identify the retrieval, even when publication bytes have not changed."""
    return sha256_json(
        {
            "schema_version": REFRESH_SCHEMA_VERSION,
            "provider": SUPPLEMENTAL_PROVIDER,
            "product": "mmz4281/2627/E0.csv",
            "prior_response_sha256": prior_response_sha256,
            "response_sha256": response_sha256,
            "response_size_bytes": response_size_bytes,
            "retrieved_at": retrieved_at,
            "request_url": request_url,
        }
    )


def assess_semantic_drift(
    *,
    old_payload: bytes,
    new_payload: bytes,
    old_events: Sequence[Mapping[str, Any]],
    new_events: Sequence[Mapping[str, Any]],
    new_source: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate that a publication refresh still has the qualified semantics."""
    errors: list[str] = []
    old_header = csv_header(old_payload)
    new_header = csv_header(new_payload)
    if old_header != new_header:
        errors.append("provider CSV header changed")
    missing_columns = [value for value in REQUIRED_COLUMNS if value not in new_header]
    if missing_columns:
        errors.append(f"required columns missing: {missing_columns}")
    if new_source.get("provider") != SUPPLEMENTAL_PROVIDER:
        errors.append("supplemental provider identity changed")
    if str(new_source.get("season_token")) != "2627":
        errors.append("league/season product identity changed")
    if not str(new_source.get("request_url", "")).endswith("/2627/E0.csv"):
        errors.append("supplemental dataset/product changed")

    old_by_key = {fixture_key(value): value for value in old_events}
    new_by_key = {fixture_key(value): value for value in new_events}
    if len(old_by_key) != len(old_events):
        errors.append("old snapshot contains duplicate fixture identities")
    if len(new_by_key) != len(new_events):
        errors.append("refreshed snapshot contains duplicate fixture identities")
    reversed_keys = sorted(
        key for key in new_by_key if (key[0], key[2], key[1]) in new_by_key
    )
    if reversed_keys:
        errors.append("refreshed snapshot contains reversed home/away identities")

    missing_old = sorted(key for key in old_by_key if key not in new_by_key)
    if missing_old:
        errors.append("previously published fixtures disappeared")
    changed: list[dict[str, Any]] = []
    for key, old in old_by_key.items():
        new = new_by_key.get(key)
        if new is None:
            continue
        old_semantics = semantic_projection(old)
        new_semantics = semantic_projection(new)
        if old_semantics != new_semantics:
            changed.append(
                {"fixture_key": list(key), "old": old_semantics, "new": new_semantics}
            )
    if changed:
        errors.append("previously published rows changed qualified values")

    invalid_contract = [
        list(fixture_key(value))
        for value in new_events
        if value.get("normalization_contract_sha256")
        != SUPPLEMENTAL_NORMALIZATION_CONTRACT_SHA256
    ]
    if invalid_contract:
        errors.append("normalization contract fingerprint changed")
    invalid_league = [
        list(fixture_key(value))
        for value in new_events
        if value.get("league") != "PL" or int(value.get("season_start_year", -1)) != 2026
    ]
    if invalid_league:
        errors.append("normalized league/season identity changed")

    new_keys = sorted(key for key in new_by_key if key not in old_by_key)
    result = {
        "schema_version": REFRESH_SENTINEL_SCHEMA_VERSION,
        "status": "COMPATIBLE" if not errors else "INCOMPATIBLE",
        "classification": (
            "REFRESH_SEMANTICS_UNCHANGED"
            if not errors
            else "SUPPLEMENTAL_SOURCE_REFRESH_INCOMPATIBLE"
        ),
        "errors": errors,
        "old_header": old_header,
        "new_header": new_header,
        "required_columns": list(REQUIRED_COLUMNS),
        "required_columns_present": not missing_columns,
        "old_completed_event_count": len(old_events),
        "new_completed_event_count": len(new_events),
        "unchanged_event_count": len(old_by_key) - len(changed) - len(missing_old),
        "new_event_count": len(new_keys),
        "new_fixture_keys": [list(value) for value in new_keys],
        "missing_old_fixture_keys": [list(value) for value in missing_old],
        "changed_events": changed,
        "invalid_contract_fixture_keys": invalid_contract,
        "invalid_league_fixture_keys": invalid_league,
        "home_away_orientation_unchanged": not reversed_keys,
        "normalization_contract_sha256": SUPPLEMENTAL_NORMALIZATION_CONTRACT_SHA256,
        "full_phase7e_requalification_performed": False,
    }
    result["sentinel_sha256"] = sha256_json(result)
    return result


def audit_blocking_fixtures(
    *,
    primary_by_id: Mapping[str, Mapping[str, Any]],
    supplemental_by_key: Mapping[tuple[str, str, str], Mapping[str, Any]],
    raw_snapshot_sha256: str,
) -> dict[str, Any]:
    """Reconcile the five frozen blockers using field-authority rules."""
    rows: list[dict[str, Any]] = []
    for expected in BLOCKING_FIXTURES:
        provider_id = expected["provider_event_id"]
        primary = primary_by_id.get(provider_id)
        key = (
            expected["fixture_date"],
            expected["home_team"],
            expected["away_team"],
        )
        supplemental = supplemental_by_key.get(key)
        errors: list[str] = []
        if primary is None:
            errors.append("authoritative primary fixture unavailable")
        else:
            for name, expected_value in (
                ("fixture_date", expected["fixture_date"]),
                ("home_team", expected["home_team"]),
                ("away_team", expected["away_team"]),
            ):
                if str(primary.get(name)) != expected_value:
                    errors.append(f"primary {name} mismatch")
            if primary.get("completed") is not True:
                errors.append("authoritative primary result is incomplete")
        if supplemental is None:
            errors.append("supplemental fixture unavailable")
        else:
            if supplemental.get("raw_source_sha256") != raw_snapshot_sha256:
                errors.append("supplemental source identity mismatch")
            if supplemental.get("normalization_contract_sha256") != SUPPLEMENTAL_NORMALIZATION_CONTRACT_SHA256:
                errors.append("supplemental normalization contract mismatch")
            for name in ("home_corners", "away_corners"):
                value = supplemental.get(name)
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    errors.append(f"valid supplemental {name} unavailable")
            if primary is not None:
                if (
                    supplemental.get("home_goals") != primary.get("home_goals")
                    or supplemental.get("away_goals") != primary.get("away_goals")
                ):
                    errors.append("cross-provider final score mismatch")
                if (
                    supplemental.get("home_team") != primary.get("home_team")
                    or supplemental.get("away_team") != primary.get("away_team")
                ):
                    errors.append("cross-provider home/away orientation mismatch")
        rows.append(
            {
                **expected,
                "complete": not errors,
                "errors": errors,
                "primary_record_fingerprint": primary.get("record_fingerprint") if primary else None,
                "supplemental_record_fingerprint": (
                    supplemental.get("record_fingerprint") if supplemental else None
                ),
                "supplemental_raw_source_sha256": (
                    supplemental.get("raw_source_sha256") if supplemental else None
                ),
                "home_goals": primary.get("home_goals") if primary else None,
                "away_goals": primary.get("away_goals") if primary else None,
                "home_corners": supplemental.get("home_corners") if supplemental else None,
                "away_corners": supplemental.get("away_corners") if supplemental else None,
            }
        )
    result = {
        "schema_version": "ffps.phase7er_blocking_fixture_audit.v1",
        "required_fixture_count": len(BLOCKING_FIXTURES),
        "complete_fixture_count": sum(bool(value["complete"]) for value in rows),
        "all_complete": all(bool(value["complete"]) for value in rows),
        "primary_event_560561_has_valid_supplemental_corners": next(
            bool(value["complete"])
            for value in rows
            if value["provider_event_id"] == "560561"
        ),
        "fixtures": rows,
    }
    result["audit_sha256"] = sha256_json(result)
    return result


def continuity_classification(*, drift_compatible: bool, complete_count: int) -> str:
    if not drift_compatible:
        return "SUPPLEMENTAL_SOURCE_REFRESH_INCOMPATIBLE"
    if complete_count != 215:
        return "SUPPLEMENTAL_SOURCE_PARTIAL"
    return "CONTINUITY_COMPLETE"


def replace_current_season(
    frozen_events: Sequence[Mapping[str, Any]],
    refreshed_events: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Reuse frozen history and replace, never append to, the refreshed season."""
    combined = [
        dict(value)
        for value in frozen_events
        if int(cast(Any, value.get("season_start_year", -1))) != 2026
    ]
    combined.extend(dict(value) for value in refreshed_events)
    by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    for value in combined:
        key = fixture_key(value)
        existing = by_key.get(key)
        if existing is not None and semantic_projection(existing) != semantic_projection(value):
            raise ValueError(f"conflicting refreshed fixture: {key}")
        by_key[key] = value
    return [by_key[key] for key in sorted(by_key)]
