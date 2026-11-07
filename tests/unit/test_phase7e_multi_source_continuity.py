from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from src.evaluation.phase7_shadow import sha256_json
from src.evaluation.phase7e_multi_source_continuity import (
    FIELD_AUTHORITY_MATRIX,
    FIELD_AUTHORITY_MATRIX_SHA256,
    PRIMARY_PROVIDER,
    SUPPLEMENTAL_NORMALIZATION_CONTRACT_SHA256,
    SUPPLEMENTAL_PROVIDER,
    TEAM_ALIAS_EXTENSION_SHA256,
    ImmutableObservationStore,
    Phase7EError,
    build_causal_bundle,
    causal_observations,
    compose_observation,
    normalize_primary_event,
    normalize_supplemental_csv,
    normalize_team,
    raw_source_identity,
    seal_record,
    unique_fixture_index,
    verify_observation,
    verify_seal,
)


ROOT = Path(__file__).resolve().parents[2]


def raw_source(payload: bytes = b"Div,Date\nE0,21/08/2026\n") -> dict:
    return raw_source_identity(
        payload,
        season_token="2627",
        retrieved_at="2026-08-30T06:00:00+00:00",
        request_url="https://www.football-data.co.uk/mmz4281/2627/E0.csv",
    )


def primary_event(*, completed: bool = True) -> dict:
    return {
        "id": 560542,
        "utcDate": "2026-08-21T19:00:00Z",
        "status": "FINISHED" if completed else "TIMED",
        "homeTeam": {"id": 57, "name": "Arsenal FC"},
        "awayTeam": {"id": 1076, "name": "Coventry City FC"},
        "score": {"fullTime": {"home": 3 if completed else None, "away": 0 if completed else None}},
    }


def primary_record(
    *, fixture_id: str = "fixture-1", fixture_date: str = "2026-08-29", observed: str = "2026-08-30T06:00:00+00:00"
) -> dict:
    return seal_record(
        {
            "provider": PRIMARY_PROVIDER,
            "provider_event_id": "p-1",
            "canonical_fixture_id": fixture_id,
            "fixture_date": fixture_date,
            "kickoff_at": f"{fixture_date}T14:00:00+00:00",
            "season_start_year": 2026,
            "home_team": "ARSENAL",
            "away_team": "COVENTRY",
            "status": "FINISHED",
            "completed": True,
            "home_goals": 3,
            "away_goals": 0,
            "raw_source_sha256": "a" * 64,
            "observed_at": observed,
            "team_alias_extension_sha256": TEAM_ALIAS_EXTENSION_SHA256,
        }
    )


def supplemental_record(
    *, fixture_id: str = "fixture-1", fixture_date: str = "2026-08-29", observed: str = "2026-08-30T06:00:00+00:00"
) -> dict:
    return seal_record(
        {
            "provider": SUPPLEMENTAL_PROVIDER,
            "provider_event_id": "s-1",
            "canonical_fixture_id": fixture_id,
            "fixture_date": fixture_date,
            "season_start_year": 2026,
            "home_team": "ARSENAL",
            "away_team": "COVENTRY",
            "completed": True,
            "home_goals": 3,
            "away_goals": 0,
            "home_corners": 7,
            "away_corners": 2,
            "raw_source_sha256": "b" * 64,
            "raw_row_sha256": "c" * 64,
            "observed_at": observed,
            "normalization_contract_sha256": SUPPLEMENTAL_NORMALIZATION_CONTRACT_SHA256,
        }
    )


def observation(**kwargs) -> dict:
    return compose_observation(primary_record(**kwargs), supplemental_record(**kwargs))


def test_source_authority_matrix_hash_is_deterministic() -> None:
    assert sha256_json(FIELD_AUTHORITY_MATRIX) == FIELD_AUTHORITY_MATRIX_SHA256


def test_raw_source_hash_is_deterministic_and_sensitive_to_bytes() -> None:
    payload = b"Div,Date,HC,AC\nE0,21/08/2026,7,2\n"
    assert raw_source(payload) == raw_source(payload)
    assert raw_source(payload)["response_sha256"] != raw_source(payload + b"x")["response_sha256"]


def test_supplemental_csv_normalizes_full_time_oriented_corners() -> None:
    payload = (
        b"Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,HC,AC\n"
        b"E0,21/08/2026,Arsenal,Coventry,3,0,7,2\n"
    )
    events = normalize_supplemental_csv(payload, raw_source=raw_source(payload))
    assert len(events) == 1
    assert events[0]["home_team"] == "ARSENAL"
    assert events[0]["away_team"] == "COVENTRY"
    assert events[0]["home_corners"] == 7
    assert events[0]["away_corners"] == 2


def test_supplemental_raw_tampering_fails() -> None:
    payload = b"Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,HC,AC\n"
    with pytest.raises(Phase7EError, match="hash mismatch"):
        normalize_supplemental_csv(payload + b"x", raw_source=raw_source(payload))


@pytest.mark.parametrize(
    ("provider_id", "provider_name", "canonical"),
    [("1076", "Coventry City FC", "COVENTRY"), ("322", "Hull City AFC", "HULL")],
)
def test_primary_team_alias_extension_is_id_bound(
    provider_id: str, provider_name: str, canonical: str
) -> None:
    assert normalize_team(
        provider_name,
        provider=PRIMARY_PROVIDER,
        provider_team_id=provider_id,
        season_start_year=2026,
    ) == canonical


def test_primary_team_alias_wrong_id_fails() -> None:
    with pytest.raises(Phase7EError, match="unresolved"):
        normalize_team(
            "Coventry City FC",
            provider=PRIMARY_PROVIDER,
            provider_team_id="wrong",
            season_start_year=2026,
        )


@pytest.mark.parametrize(("name", "canonical"), [("Coventry", "COVENTRY"), ("Hull", "HULL")])
def test_supplemental_team_aliases_are_deterministic(name: str, canonical: str) -> None:
    assert normalize_team(name, provider=SUPPLEMENTAL_PROVIDER, season_start_year=2026) == canonical


def test_future_primary_result_is_rejected() -> None:
    raw = primary_event(completed=False)
    raw["score"]["fullTime"]["home"] = 9
    source = {
        "provider": PRIMARY_PROVIDER,
        "season": 2026,
        "response_sha256": "a" * 64,
        "retrieved_at": "2026-08-30T06:00:00+00:00",
    }
    with pytest.raises(Phase7EError, match="future primary fixture"):
        normalize_primary_event(raw, raw_source=source)


def test_composed_observation_preserves_field_level_authority() -> None:
    value = observation()
    assert verify_observation(value)["verified"]
    assert value["fields"]["home_goals"]["provider"] == PRIMARY_PROVIDER
    assert value["fields"]["home_corners"]["provider"] == SUPPLEMENTAL_PROVIDER


def test_reversed_home_away_mapping_fails() -> None:
    supplemental = supplemental_record()
    supplemental["home_team"], supplemental["away_team"] = (
        supplemental["away_team"],
        supplemental["home_team"],
    )
    supplemental = seal_record(supplemental)
    with pytest.raises(Phase7EError, match="home_team mismatch"):
        compose_observation(primary_record(), supplemental)


def test_cross_provider_score_mismatch_fails() -> None:
    supplemental = supplemental_record()
    supplemental["home_goals"] = 2
    supplemental = seal_record(supplemental)
    with pytest.raises(Phase7EError, match="final-score mismatch"):
        compose_observation(primary_record(), supplemental)


def test_unauthorized_provider_cannot_override_primary_field() -> None:
    value = observation()
    value["fields"]["home_goals"]["provider"] = SUPPLEMENTAL_PROVIDER
    value = seal_record(value)
    result = verify_observation(value)
    assert not result["verified"]
    assert "unauthorized provider for field home_goals" in result["errors"]


def test_corner_value_tampering_breaks_seal() -> None:
    value = observation()
    value["fields"]["home_corners"]["value"] = 99
    assert "record fingerprint mismatch" in verify_observation(value)["errors"]


def test_missing_required_corner_prevents_complete_observation() -> None:
    value = observation()
    value["fields"]["away_corners"]["value"] = None
    value = seal_record(value)
    result = verify_observation(value)
    assert not result["verified"]
    assert "required field missing: away_corners" in result["errors"]


def test_unique_fixture_index_rejects_ambiguous_and_reversed_joins() -> None:
    first = supplemental_record()
    with pytest.raises(Phase7EError, match="ambiguous"):
        unique_fixture_index([first, first])
    reversed_value = copy.deepcopy(first)
    reversed_value["home_team"], reversed_value["away_team"] = (
        reversed_value["away_team"],
        reversed_value["home_team"],
    )
    with pytest.raises(Phase7EError, match="reversed"):
        unique_fixture_index([first, reversed_value])


def test_future_self_and_late_observation_cannot_enter_causal_history() -> None:
    earlier = observation(fixture_id="earlier", fixture_date="2026-08-29")
    self_event = observation(fixture_id="target", fixture_date="2026-08-31")
    late = observation(
        fixture_id="late",
        fixture_date="2026-08-28",
        observed="2026-08-30T09:00:00+00:00",
    )
    selected = causal_observations(
        [earlier, self_event, late],
        target_kickoff_at="2026-08-31T13:00:00+00:00",
        prediction_at="2026-08-30T08:00:00+00:00",
    )
    assert [value["canonical_fixture_id"] for value in selected] == ["earlier"]


def test_causal_bundle_changes_only_for_legitimate_earlier_history() -> None:
    earlier = observation(fixture_id="earlier", fixture_date="2026-08-29")
    future = observation(fixture_id="future", fixture_date="2026-09-01")
    kwargs = {
        "protected_base_sha256": "a" * 64,
        "primary_extension_sha256": "b" * 64,
        "supplemental_source_sha256": "c" * 64,
        "target_fixture_id": "target",
        "target_kickoff_at": "2026-08-31T13:00:00+00:00",
        "prediction_at": "2026-08-30T08:00:00+00:00",
    }
    empty = build_causal_bundle(observations=[], **kwargs)
    with_future = build_causal_bundle(observations=[future], **kwargs)
    with_earlier = build_causal_bundle(observations=[earlier, future], **kwargs)
    assert empty["multi_source_continuity_sha256"] == with_future["multi_source_continuity_sha256"]
    assert empty["multi_source_continuity_sha256"] != with_earlier["multi_source_continuity_sha256"]


def test_observation_store_is_idempotent_and_requires_explicit_revision(tmp_path: Path) -> None:
    store = ImmutableObservationStore(tmp_path)
    value = observation()
    assert store.persist(value) == "CREATED"
    assert store.persist(value) == "ALREADY_PRESENT"
    changed = copy.deepcopy(value)
    changed["fields"]["home_corners"]["value"] = 8
    changed = seal_record(changed)
    with pytest.raises(Phase7EError, match="explicit revision"):
        store.persist(changed)


def test_append_only_revision_is_causal_as_of_cutoff(tmp_path: Path) -> None:
    store = ImmutableObservationStore(tmp_path)
    original = observation()
    store.persist(original)
    replacement = copy.deepcopy(original)
    replacement["fields"]["home_corners"]["value"] = 8
    replacement = seal_record(replacement)
    store.persist_revision(
        superseded_fingerprint=original["record_fingerprint"],
        replacement=replacement,
        changed_fields=["home_corners"],
        reason="provider correction",
        corrected_at="2026-08-30T10:00:00+00:00",
    )
    assert store.effective_records(as_of="2026-08-30T09:00:00+00:00")[0] == original
    assert store.effective_records(as_of="2026-08-30T11:00:00+00:00")[0] == replacement


def test_frozen_feature_contract_is_still_30_with_12_corner_features() -> None:
    spec = json.loads(
        (
            ROOT
            / "baselines/phase7/phase7-active-vs-candidate-shadow-79110f9a-20260829-v2/evaluation_specification.json"
        ).read_text(encoding="utf-8")
    )
    features = spec["feature_contract"]["ordered_feature_names"]
    assert len(features) == 30
    assert len([value for value in features if "corner" in value]) == 12
    assert sha256_json(features) == "18b0f442fd5457f8b35a6484f716d15cd250297749536ab23365aef13aae4ce7"


def test_qualified_source_reproduces_all_historical_features() -> None:
    from scripts.run_phase7e_multi_source_continuity import DEFAULT_RUN_DIR, _historical_reproduction

    result = _historical_reproduction(DEFAULT_RUN_DIR)
    assert result["reproduced"]
    assert result["matched_rows"] == 2460
    assert result["maximum_absolute_difference"] <= 1e-12


def test_partial_continuity_cannot_resume_phase7p() -> None:
    from scripts.run_phase7e_multi_source_continuity import DEFAULT_RUN_DIR

    readiness = json.loads((DEFAULT_RUN_DIR / "feature_readiness.json").read_text(encoding="utf-8"))
    resume = json.loads((DEFAULT_RUN_DIR / "phase7p_resume_result.json").read_text(encoding="utf-8"))
    assert readiness["classification"] == "SUPPLEMENTAL_SOURCE_PARTIAL"
    assert resume["phase7p_collection_resumed"] is False
    assert resume["created_count"] == 0


def test_phase0_to_phase7d_protected_snapshot_is_unchanged() -> None:
    from scripts.run_phase7e_multi_source_continuity import DEFAULT_RUN_DIR, protected_equal, protected_snapshot

    before = json.loads((DEFAULT_RUN_DIR / "protected_before.json").read_text(encoding="utf-8"))
    assert protected_equal(before, protected_snapshot())
