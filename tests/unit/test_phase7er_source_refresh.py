from __future__ import annotations

import copy
import json
from pathlib import Path

from src.evaluation.phase7e_multi_source_continuity import (
    SUPPLEMENTAL_NORMALIZATION_CONTRACT_SHA256,
    normalize_supplemental_csv,
    raw_source_identity,
)
from src.evaluation.phase7er_source_refresh import (
    BLOCKING_FIXTURES,
    assess_semantic_drift,
    audit_blocking_fixtures,
    continuity_classification,
    fixture_key,
    refresh_snapshot_identity,
    replace_current_season,
)


ROOT = Path(__file__).resolve().parents[2]
PHASE7E = (
    ROOT
    / "baselines"
    / "phase7e"
    / "phase7e-football-data-co-uk-corners-79110f9a-20260830-v1"
)


HEADER = b"Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,HC,AC\n"
OLD_ROW = b"E0,21/08/2026,Arsenal,Coventry,3,0,8,2\n"
NEW_ROW = b"E0,29/08/2026,Coventry,Hull,1,0,5,4\n"


def source(payload: bytes, retrieved_at: str = "2026-08-30T07:00:00+00:00") -> dict:
    return raw_source_identity(
        payload,
        season_token="2627",
        retrieved_at=retrieved_at,
        request_url="https://www.football-data.co.uk/mmz4281/2627/E0.csv",
    )


def events(payload: bytes, retrieved_at: str = "2026-08-30T07:00:00+00:00") -> list[dict]:
    return normalize_supplemental_csv(payload, raw_source=source(payload, retrieved_at))


def test_refresh_snapshot_identity_is_new_even_when_bytes_are_unchanged() -> None:
    first = refresh_snapshot_identity(
        prior_response_sha256="a" * 64,
        response_sha256="a" * 64,
        response_size_bytes=10,
        retrieved_at="2026-08-30T07:00:00+00:00",
        request_url="https://www.football-data.co.uk/mmz4281/2627/E0.csv",
    )
    second = refresh_snapshot_identity(
        prior_response_sha256="a" * 64,
        response_sha256="a" * 64,
        response_size_bytes=10,
        retrieved_at="2026-08-30T08:00:00+00:00",
        request_url="https://www.football-data.co.uk/mmz4281/2627/E0.csv",
    )
    assert first != second


def test_unchanged_rows_and_appended_rows_pass_drift_sentinel() -> None:
    old_payload = HEADER + OLD_ROW
    new_payload = HEADER + OLD_ROW + NEW_ROW
    result = assess_semantic_drift(
        old_payload=old_payload,
        new_payload=new_payload,
        old_events=events(old_payload),
        new_events=events(new_payload, "2026-08-30T08:00:00+00:00"),
        new_source=source(new_payload, "2026-08-30T08:00:00+00:00"),
    )
    assert result["status"] == "COMPATIBLE"
    assert result["unchanged_event_count"] == 1
    assert result["new_event_count"] == 1
    assert result["full_phase7e_requalification_performed"] is False


def test_schema_drift_fails_closed() -> None:
    old_payload = HEADER + OLD_ROW
    new_payload = b"Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,HC\nE0,21/08/2026,Arsenal,Coventry,3,0,8\n"
    result = assess_semantic_drift(
        old_payload=old_payload,
        new_payload=new_payload,
        old_events=events(old_payload),
        new_events=[],
        new_source=source(new_payload),
    )
    assert result["classification"] == "SUPPLEMENTAL_SOURCE_REFRESH_INCOMPATIBLE"
    assert any("header" in value for value in result["errors"])
    assert result["required_columns_present"] is False


def test_changed_qualified_corner_value_fails_closed() -> None:
    old_payload = HEADER + OLD_ROW
    changed_payload = HEADER + OLD_ROW.replace(b",8,2", b",9,2")
    result = assess_semantic_drift(
        old_payload=old_payload,
        new_payload=changed_payload,
        old_events=events(old_payload),
        new_events=events(changed_payload),
        new_source=source(changed_payload),
    )
    assert result["status"] == "INCOMPATIBLE"
    assert result["changed_events"][0]["old"]["home_corners"] == 8
    assert result["changed_events"][0]["new"]["home_corners"] == 9


def _primary(expected: dict[str, str]) -> dict:
    return {
        **expected,
        "completed": True,
        "home_goals": 1,
        "away_goals": 0,
        "record_fingerprint": "p" + expected["provider_event_id"],
    }


def _supplemental(expected: dict[str, str], raw_hash: str) -> dict:
    return {
        **expected,
        "home_goals": 1,
        "away_goals": 0,
        "home_corners": 5,
        "away_corners": 4,
        "raw_source_sha256": raw_hash,
        "normalization_contract_sha256": SUPPLEMENTAL_NORMALIZATION_CONTRACT_SHA256,
        "record_fingerprint": "s" + expected["provider_event_id"],
    }


def test_all_five_blockers_reconcile_by_primary_id_and_exact_orientation() -> None:
    raw_hash = "a" * 64
    primary = {value["provider_event_id"]: _primary(value) for value in BLOCKING_FIXTURES}
    supplemental_values = [_supplemental(value, raw_hash) for value in BLOCKING_FIXTURES]
    result = audit_blocking_fixtures(
        primary_by_id=primary,
        supplemental_by_key={fixture_key(value): value for value in supplemental_values},
        raw_snapshot_sha256=raw_hash,
    )
    assert result["complete_fixture_count"] == 5
    assert result["all_complete"]
    assert result["primary_event_560561_has_valid_supplemental_corners"]


def test_primary_560561_cannot_complete_without_both_corner_fields() -> None:
    raw_hash = "a" * 64
    expected = next(value for value in BLOCKING_FIXTURES if value["provider_event_id"] == "560561")
    supplemental = _supplemental(expected, raw_hash)
    supplemental["away_corners"] = None
    result = audit_blocking_fixtures(
        primary_by_id={"560561": _primary(expected)},
        supplemental_by_key={fixture_key(supplemental): supplemental},
        raw_snapshot_sha256=raw_hash,
    )
    row = next(value for value in result["fixtures"] if value["provider_event_id"] == "560561")
    assert not row["complete"]
    assert "valid supplemental away_corners unavailable" in row["errors"]
    assert result["primary_event_560561_has_valid_supplemental_corners"] is False


def test_continuity_gate_requires_exactly_215_complete_rows() -> None:
    assert continuity_classification(drift_compatible=True, complete_count=214) == "SUPPLEMENTAL_SOURCE_PARTIAL"
    assert continuity_classification(drift_compatible=True, complete_count=215) == "CONTINUITY_COMPLETE"
    assert (
        continuity_classification(drift_compatible=False, complete_count=215)
        == "SUPPLEMENTAL_SOURCE_REFRESH_INCOMPATIBLE"
    )


def test_current_season_refresh_replaces_instead_of_duplicating_rows() -> None:
    old_payload = HEADER + OLD_ROW
    new_payload = HEADER + OLD_ROW + NEW_ROW
    old = events(old_payload)
    refreshed = events(new_payload, "2026-08-30T08:00:00+00:00")
    combined = replace_current_season(old, refreshed)
    assert len(combined) == 2
    assert len({fixture_key(value) for value in combined}) == 2


def test_future_fixture_cannot_be_selected_after_its_kickoff() -> None:
    from scripts.run_phase7er_source_refresh import _eligible_scheduled

    assert _eligible_scheduled("2027-01-01T00:00:00+00:00") == []


def test_frozen_phase7e_snapshot_remains_partial_and_verified() -> None:
    manifest = json.loads((PHASE7E / "run_manifest.json").read_text(encoding="utf-8"))
    verification = json.loads((PHASE7E / "offline_verification.json").read_text(encoding="utf-8"))
    assert manifest["classification"] == "SUPPLEMENTAL_SOURCE_PARTIAL"
    assert manifest["multi_source_continuity_sha256"] == "d59ec33d9a8e93f98cf6075721fc2cbcc6458f9a8804a27d62c86359613ce589"
    assert verification["status"] == "VERIFIED"


def test_actual_refresh_evidence_respects_all_gates() -> None:
    from scripts.run_phase7er_source_refresh import DEFAULT_RUN_DIR, protected_equal, protected_snapshot

    before = json.loads((DEFAULT_RUN_DIR / "protected_before.json").read_text(encoding="utf-8"))
    assert protected_equal(before, protected_snapshot())
    sentinel = json.loads((DEFAULT_RUN_DIR / "semantic_drift_sentinel.json").read_text(encoding="utf-8"))
    composition = json.loads((DEFAULT_RUN_DIR / "composition_result.json").read_text(encoding="utf-8"))
    readiness = json.loads((DEFAULT_RUN_DIR / "feature_readiness.json").read_text(encoding="utf-8"))
    resume = json.loads((DEFAULT_RUN_DIR / "phase7p_resume_result.json").read_text(encoding="utf-8"))
    assert sentinel["status"] in {"COMPATIBLE", "INCOMPATIBLE"}
    if composition["causal_history_complete_count"] < 215:
        assert readiness["classification"] == "SUPPLEMENTAL_SOURCE_PARTIAL"
        assert resume["created_count"] == 0
    elif readiness["classification"] == "PROSPECTIVE_FEATURE_READY":
        assert resume["created_count"] == 1
        assert resume["post_outcome_predictions_created"] is False


def test_actual_refresh_historical_reproduction_remains_within_tolerance() -> None:
    from scripts.run_phase7er_source_refresh import DEFAULT_RUN_DIR

    readiness = json.loads((DEFAULT_RUN_DIR / "feature_readiness.json").read_text(encoding="utf-8"))
    reproduction = readiness["historical_feature_reproduction"]
    assert reproduction["reproduced"]
    assert reproduction["matched_rows"] == 2460
    assert reproduction["maximum_absolute_difference"] <= 1e-12


def test_actual_ready_dry_run_uses_identical_active_candidate_features() -> None:
    from scripts.run_phase7er_source_refresh import DEFAULT_RUN_DIR

    readiness = json.loads((DEFAULT_RUN_DIR / "feature_readiness.json").read_text(encoding="utf-8"))
    dry_path = DEFAULT_RUN_DIR / "prospective_dry_run.json"
    if readiness["classification"] == "PROSPECTIVE_FEATURE_READY":
        dry = json.loads(dry_path.read_text(encoding="utf-8"))
        assert dry["feature_vector_finite"]
        assert dry["active_candidate_feature_identity_equal"]
        assert dry["target_outcome_present"] is False
    else:
        assert not dry_path.exists()
