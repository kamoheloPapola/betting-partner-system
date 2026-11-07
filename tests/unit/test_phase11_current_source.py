"""Phase 11 source adapters, strict gates, and BL1 compatibility tests."""
from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import pandas as pd
import pytest
import requests

from src.evaluation import phase11_current_source as data


AT = "2026-09-06T10:00:00+00:00"


def archived_json(tmp_path, monkeypatch, payload, name="source"):
    monkeypatch.setattr(data, "ROOT", tmp_path)
    raw = tmp_path / f"{name}.raw"
    raw.write_bytes(json.dumps(payload).encode("utf-8"))
    return {"provider": "test", "status": "RETRIEVED", "http_status": 200,
            "retrieved_at": AT,
            "raw": {"path": raw.relative_to(tmp_path).as_posix(),
                    "sha256": data.sha256_file(raw), "size_bytes": raw.stat().st_size}}


def teams(league="SA", year=2026):
    return {"competition": {"code": league},
            "season": {"startDate": f"{year}-08-01"},
            "teams": [{"id": 1, "name": "Alpha", "shortName": "Alpha"},
                      {"id": 2, "name": "Beta", "shortName": "Beta"}]}


def match(league="SA", year=2026, fixture_id=10,
          kickoff="2026-08-01T00:00:00Z", status="FINISHED"):
    return {"id": fixture_id, "utcDate": kickoff, "status": status,
            "stage": "REGULAR_SEASON", "competition": {"code": league},
            "season": {"id": year, "startDate": f"{year}-08-02",
                       "endDate": f"{year + 1}-05-31"},
            "homeTeam": {"id": 1}, "awayTeam": {"id": 2},
            "lastUpdated": "2026-09-05T00:00:00Z",
            "score": {"fullTime": {"home": 1 if status == "FINISHED" else None,
                                    "away": 0 if status == "FINISHED" else None}}}


def match_payload(rows, league="SA", year=2026):
    dates = [row["utcDate"][:10] for row in rows]
    return {"competition": {"code": league}, "filters": {"season": year},
            "resultSet": {"count": len(rows), "first": min(dates), "last": max(dates)},
            "matches": rows}


def mapped():
    return {"1": {"canonical": "ALPHA"}, "2": {"canonical": "BETA"}}


def fotmob_payload(*, corners=True, fixture_name="Alpha", detail_name="Alpha"):
    stats = [{"stats": [{"title": "Corners", "stats": [4, 2]}]}] if corners else []
    league = {"details": {"id": 55, "name": "Serie A", "selectedSeason": "2026/2027"},
              "fixtures": {"allMatches": [{"id": 20,
                  "home": {"id": "1", "name": fixture_name},
                  "away": {"id": "2", "name": "Beta"},
                  "status": {"finished": True, "utcTime": "2026-08-03T12:00:00Z"}}]}}
    detail = {"general": {"matchId": 20, "leagueId": 55, "leagueName": "Serie A",
                           "finished": True, "matchTimeUTCDate": "2026-08-03T12:00:00Z",
                           "homeTeam": {"id": 1, "name": detail_name},
                           "awayTeam": {"id": 2, "name": "Beta"}},
              "header": {"status": {"scoreStr": "1 - 0"}},
              "content": {"stats": {"Periods": {"All": {"stats": stats}}}}}
    return league, detail


@pytest.mark.parametrize("status", [401, 403, 404, 429, 503])
def test_http_failures_are_archived_and_never_empty_success(tmp_path, monkeypatch, status):
    response = SimpleNamespace(status_code=status, content=b'{"error":"unavailable"}',
                               headers={"Content-Type": "application/json"})
    monkeypatch.setattr(requests, "get", lambda *args, **kwargs: response)
    record = data.fetch(tmp_path, f"status-{status}", "https://example.test/data",
                        provider="test", headers={"Authorization": "secret"})
    assert record["status"] == "HTTP_ERROR" and record["http_status"] == status
    assert (tmp_path / f"status-{status}.raw").read_bytes() == response.content
    stored = json.loads((tmp_path / f"status-{status}.json").read_text())
    assert "secret" not in json.dumps(stored)


def test_transport_failure_is_explicit(tmp_path, monkeypatch):
    def unavailable(*args, **kwargs):
        raise requests.Timeout("bounded timeout")
    monkeypatch.setattr(requests, "get", unavailable)
    record = data.fetch(tmp_path, "timeout", "https://example.test/data", provider="test")
    assert record["status"] == "SOURCE_UNAVAILABLE"
    assert record["error_type"] == "Timeout"


def test_reflected_credential_is_rejected_before_persistence(tmp_path, monkeypatch):
    response = SimpleNamespace(status_code=403, content=b"secret-value",
                               headers={"Content-Type": "text/plain"})
    monkeypatch.setattr(requests, "get", lambda *args, **kwargs: response)
    with pytest.raises(Exception, match="credential reflected"):
        data.fetch(tmp_path, "reflection", "https://example.test/data", provider="test",
                   headers={"X-Auth-Token": "secret-value"})
    assert not (tmp_path / "reflection.raw").exists()


def test_malformed_json_and_payload_revision_fail_closed(tmp_path, monkeypatch):
    source = archived_json(tmp_path, monkeypatch, {"ok": True})
    raw = tmp_path / "source.raw"
    raw.write_bytes(b"not json")
    source["raw"].update(sha256=data.sha256_file(raw), size_bytes=raw.stat().st_size)
    with pytest.raises(ValueError, match="MALFORMED_JSON"):
        data._json_payload(source)
    raw.write_bytes(b"revised")
    with pytest.raises(ValueError, match="SOURCE_BYTES_DRIFT"):
        data._raw_bytes(source)


@pytest.mark.parametrize("mutation,code", [
    ("wrong_competition", "WRONG_LEAGUE"), ("wrong_season", "STALE_SEASON"),
    ("boundary", "RESULT_SET_BOUNDARY_MISMATCH"), ("duplicate", "DUPLICATE_FIXTURE_ID"),
    ("changed_team", "INVALID_FIXTURE_TEAM"), ("timezone", "INVALID_TIMEZONE_TIMESTAMP"),
    ("stale_cache", "FUTURE_PROVIDER_UPDATE"), ("nonleague", "NON_LEAGUE_STAGE_INCLUDED"),
])
def test_primary_payload_failures(tmp_path, monkeypatch, mutation, code):
    rows = [match()]
    payload = match_payload(rows)
    if mutation == "wrong_competition": payload["competition"]["code"] = "PL"
    elif mutation == "wrong_season": payload["filters"]["season"] = 2025
    elif mutation == "boundary": payload["resultSet"]["first"] = "2026-08-02"
    elif mutation == "duplicate":
        payload["matches"].append(copy.deepcopy(rows[0])); payload["resultSet"]["count"] = 2
    elif mutation == "changed_team": payload["matches"][0]["homeTeam"]["id"] = 999
    elif mutation == "timezone": payload["matches"][0]["utcDate"] = "2026-08-01T00:00:00"
    elif mutation == "stale_cache": payload["matches"][0]["lastUpdated"] = "2026-09-07T00:00:00Z"
    elif mutation == "nonleague": payload["matches"][0]["stage"] = "PLAYOFFS"
    source = archived_json(tmp_path, monkeypatch, payload)
    with pytest.raises(ValueError, match=code):
        data._adapt_primary(source, "SA", 2026, mapped())


def test_reported_season_metadata_bug_does_not_relax_authoritative_bounds(tmp_path, monkeypatch):
    rows = [match(kickoff="2026-08-01T23:30:00Z")]
    payload = match_payload(rows)
    # The provider-reported start excludes its own regular-season fixture.  The
    # result-set bound is independently checked against the actual population.
    source = archived_json(tmp_path, monkeypatch, payload)
    normalized, evidence = data._adapt_primary(source, "SA", 2026, mapped())
    assert normalized[0]["fixture_date"] == "2026-08-01"
    assert evidence["result_set_first"] == "2026-08-01"
    assert evidence["season_metadata_discrepancy"]


def fotmob_index(tmp_path, monkeypatch, *, corners=True,
                 fixture_name="Alpha", detail_name="Alpha"):
    league, detail = fotmob_payload(corners=corners, fixture_name=fixture_name,
                                    detail_name=detail_name)
    league_source = archived_json(tmp_path, monkeypatch, league, "league")
    detail_source = archived_json(tmp_path, monkeypatch, detail, "detail")
    detail_source["request"] = {"params": {"matchId": "20"}}
    return {"league": "SA", "season": 2026, "selected_season": "2026/2027",
            "league_source": league_source, "finished_fixture_ids": ["20"],
            "detail_sources": [detail_source]}


def test_fotmob_requires_statistics_and_exact_team_identity(tmp_path, monkeypatch):
    index = fotmob_index(tmp_path, monkeypatch, corners=False)
    with pytest.raises(ValueError, match="FOTMOB_MISSING_CORNERS"):
        data._adapt_fotmob(index, "SA", 2026, {"ALPHA": {"ALPHA"}, "BETA": {"BETA"}})
    index = fotmob_index(tmp_path, monkeypatch, fixture_name="Alpha", detail_name="Other")
    with pytest.raises(ValueError, match="FOTMOB_TEAM_IDENTITY_CONFLICT"):
        data._adapt_fotmob(index, "SA", 2026, {"ALPHA": {"ALPHA"}, "BETA": {"BETA"}})


def test_fotmob_accepts_only_mechanical_accent_variant(tmp_path, monkeypatch):
    index = fotmob_index(tmp_path, monkeypatch, fixture_name="Atlético", detail_name="Atletico")
    rows, evidence = data._adapt_fotmob(
        index, "SA", 2026, {"ATLETICO": {"ATLETICO"}, "BETA": {"BETA"}})
    assert rows[0]["home_team"] == "ATLETICO"
    assert evidence["source_name_variants"][0]["team_id"] == "1"


def test_alternative_provider_disagreement_is_explicit():
    csv = [{"season": 2026, "fixture_date": "2026-08-03", "home_team": "ALPHA",
            "away_team": "BETA", "home_goals": 1, "away_goals": 0,
            "home_corners": 4, "away_corners": 2, "row_ordinal": 1}]
    alternative = [{**csv[0], "home_corners": 5, "provider_fixture_id": 20}]
    overlap, conflicts = data.provider_overlap(csv, alternative, 2026)
    assert conflicts == [["2026-08-03", "ALPHA", "BETA"]]
    assert overlap[0]["fields"]["home_corners"] is False


@pytest.mark.parametrize("league,team_id,expected", [
    ("FL1", 535, "LE MANS"), ("PD", 560, "LA CORUNA"),
    ("PD", 5335, "SANTANDER"), ("BL1", 29, "PADERBORN"),
    ("BL1", 719, "ELVERSBERG"),
])
def test_promoted_team_mappings_are_exact_and_league_owned(league, team_id, expected):
    rule = data.TEAM_MAPPING_OVERRIDES[league]["football-data.org:v4"][team_id]
    assert rule["canonical"] == expected
    assert all(team_id not in data.TEAM_MAPPING_OVERRIDES.get(other, {}).get(
        "football-data.org:v4", {}) for other in data.LEAGUES if other != league)


def test_provider_id_rename_fails_closed():
    names = data._phase11_names("FL1")
    payload = {"competition": {"code": "FL1"}, "season": {"startDate": "2026-08-01"},
               "teams": [{"id": 535, "name": "Different Club", "shortName": "Le Mans"}]}
    with pytest.raises(ValueError, match="SOURCE_TEAM_IDENTITY_CHANGED"):
        data._mapped_teams(payload, "FL1", 2026, names)


def test_bl1_apostrophe_adapter_is_reversible_and_collision_checked(monkeypatch):
    from src.evaluation import phase10_current_data as phase10
    frame = pd.DataFrame({"home_team": ["M'GLADBACH"], "away_team": ["DORTMUND"]})
    observed = {}
    def passthrough(value, league):
        observed["encoded"] = value.copy()
        return value.copy()
    monkeypatch.setattr(phase10, "pipeline", passthrough)
    result = data.pipeline(frame, "BL1")
    assert observed["encoded"].home_team.iloc[0] == "FFPS_MGLADBACH_PHASE11"
    assert result.home_team.iloc[0] == "M'GLADBACH"
    collision = pd.DataFrame({"home_team": ["FFPS_MGLADBACH_PHASE11"],
                              "away_team": ["DORTMUND"]})
    with pytest.raises(ValueError, match="SAFE_TEAM_SENTINEL_COLLISION"):
        data.pipeline(collision, "BL1")


def test_sanitizer_still_rejects_hostile_identifier():
    from src.features.pipeline import FeaturePipeline
    frame = pd.DataFrame({"home_team": ["../evil"], "away_team": ["DORTMUND"]})
    with pytest.raises(Exception, match="Invalid characters"):
        FeaturePipeline()._sanitize_dataframe(frame)


def test_cross_league_source_and_qualification_fail_closed():
    assert set(data.TEAM_TOKEN_OVERRIDES["PD"]).isdisjoint(data.TEAM_TOKEN_OVERRIDES["BL1"])
    with pytest.raises(ValueError, match="UNCONFIGURED_PHASE11_LEAGUE"):
        data.qualify_league("unused", "PL", "source", "fotmob")
