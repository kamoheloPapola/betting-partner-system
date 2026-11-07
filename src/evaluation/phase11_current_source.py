"""Phase 11 current-source diagnosis and qualification primitives.

This module is deliberately read-only with respect to production data and models.
Every network response, including an error response, is archived before it is
interpreted.  Provider adapters normalize source records; they never confer
qualification by themselves.
"""
from __future__ import annotations

import json
import re
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from dotenv import dotenv_values

from src.evaluation.multi_league_integrity import ROOT, identity, read, write
from src.ml.bundle_activation import require
from src.predictions.provenance import sha256_file, sha256_json


LEAGUES = ("SA", "FL1", "PD", "BL1")
PHASE9 = ROOT / "baselines/phase9/phase9-multi-league-20260905-v1"
PHASE10 = ROOT / "baselines/phase10/phase10-current-data-20260906-v1"
API_FOOTBALL_IDS = {"PL": 39, "BL1": 78, "FL1": 61, "SA": 135, "PD": 140}
SOURCE_SPECS = {
    "SA": {"division": "I1", "fotmob_id": 55, "fotmob_country": "ITA", "fotmob_name": "Serie A"},
    "FL1": {"division": "F1", "fotmob_id": 53, "fotmob_country": "FRA", "fotmob_name": "Ligue 1"},
    "PD": {"division": "SP1", "fotmob_id": 87, "fotmob_country": "ESP", "fotmob_name": "LaLiga"},
    "BL1": {"division": "D1", "fotmob_id": 54, "fotmob_country": "GER", "fotmob_name": "Bundesliga"},
}
TEAM_MAPPING_OVERRIDES = {
    "SA": {
        "football-data.org:v4": {
            98: {
                "canonical": "MILAN",
                "source_names": ["AC Milan", "Milan"],
                "reason": "frozen SA model-facing history uses MILAN; provider ID and season membership are stable",
            },
            100: {
                "canonical": "ROMA",
                "source_names": ["AS Roma", "Roma"],
                "reason": "frozen SA model-facing history uses ROMA; provider ID and season membership are stable",
            },
            487: {
                "canonical": "PISA",
                "source_names": ["AC Pisa 1909", "AC Pisa"],
                "reason": "exact football-data.co.uk Pisa identity plus frozen SA PISA membership",
            },
        },
    },
    "FL1": {
        "football-data.org:v4": {
            529: {
                "canonical": "RENNES",
                "source_names": ["Stade Rennais FC 1901", "Stade Rennais"],
                "reason": "stable provider ID and exact Rennes identity in both frozen history and current statistics",
            },
            535: {
                "canonical": "LE MANS",
                "source_names": ["Le Mans FC", "Le Mans"],
                "reason": "stable provider ID and exact Le Mans identity in both current statistics providers",
            },
        },
    },
    "PD": {
        "football-data.org:v4": {
            78: {
                "canonical": "ATH MADRID",
                "source_names": ["Club Atlético de Madrid", "Atleti"],
                "reason": "stable provider ID and exact Atletico identity; frozen PD history uses ATH MADRID",
            },
            560: {
                "canonical": "LA CORUNA",
                "source_names": ["RC Deportivo La Coruña", "Deportivo"],
                "reason": "stable provider ID and exact Deportivo La Coruna identity in both current statistics providers",
            },
            5335: {
                "canonical": "SANTANDER",
                "source_names": ["Real Racing Club de Santander", "Santander"],
                "reason": "stable provider ID and exact Racing Santander identity in both current statistics providers",
            },
        },
    },
    "BL1": {
        "football-data.org:v4": {
            29: {
                "canonical": "PADERBORN",
                "source_names": ["SC Paderborn 07", "SC Paderborn"],
                "reason": "stable provider ID and exact Paderborn identity in frozen history and current statistics",
            },
            719: {
                "canonical": "ELVERSBERG",
                "source_names": ["SV 07 Elversberg", "Elversberg"],
                "reason": "stable provider ID and exact Elversberg identity in both current statistics providers",
            },
        },
    },
}
MODEL_CANONICAL_OVERRIDES = {
    "SA": {"AC MILAN": "MILAN", "AS ROMA": "ROMA"},
    "FL1": {"PARIS SAINT GERMAIN": "PARIS SG", "AS MONACO": "MONACO"},
    "PD": {
        "ATHLETIC CLUB": "ATH BILBAO", "ATLETICO MADRID": "ATH MADRID",
        "ESPANYOL": "ESPANOL",
        "REAL BETIS": "BETIS", "REAL SOCIEDAD": "SOCIEDAD",
        "RAYO VALLECANO": "VALLECANO", "CELTA VIGO": "CELTA",
        "REAL OVIEDO": "OVIEDO",
    },
    "BL1": {
        "FC AUGSBURG": "AUGSBURG", "BORUSSIA DORTMUND": "DORTMUND",
        "TSG HOFFENHEIM": "HOFFENHEIM", "BAYER LEVERKUSEN": "LEVERKUSEN",
        "VFB STUTTGART": "STUTTGART", "VFL WOLFSBURG": "WOLFSBURG",
        "BORUSSIA MONCHENGLADBACH": "M'GLADBACH",
        "EINTRACHT FRANKFURT": "EIN FRANKFURT", "FC ST. PAULI": "ST PAULI",
        "FC HEIDENHEIM": "HEIDENHEIM", "FC SCHALKE 04": "SCHALKE 04",
        "HAMBURGER SV": "HAMBURG", "SC FREIBURG": "FREIBURG",
        "MAINZ 05": "MAINZ",
    },
}
TEAM_TOKEN_OVERRIDES = {
    "FL1": {"LE MANS": "LE MANS"},
    "PD": {"DEPORTIVO A CORUNA": "LA CORUNA", "LA CORUNA": "LA CORUNA",
           "RACING SANTANDER": "SANTANDER", "SANTANDER": "SANTANDER"},
    "BL1": {"ELVERSBERG": "ELVERSBERG", "PADERBORN": "PADERBORN",
            "SC PADERBORN": "PADERBORN"},
}
SAFE_TEAM_SENTINELS = {"BL1": {"M'GLADBACH": "FFPS_MGLADBACH_PHASE11"}}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def check(condition: bool, code: str) -> None:
    if not condition:
        raise ValueError(code)


def run_path(path: str | Path) -> Path:
    target = Path(path).resolve()
    root = (ROOT / "baselines/phase11").resolve()
    require(target.is_relative_to(root) and target != root,
            "Phase 11 evidence must be under its own run directory")
    return target


def protected_snapshot() -> dict[str, str]:
    """Protect all prior evidence, production data, and model artifacts."""
    result: dict[str, str] = {}
    phase11_root = (ROOT / "baselines/phase11").resolve()
    for base in ("baselines", "data", "src/ml/models"):
        for path in sorted((ROOT / base).rglob("*")):
            if path.is_file() and not path.resolve().is_relative_to(phase11_root):
                result[path.relative_to(ROOT).as_posix()] = sha256_file(path)
    return result


def _tree_snapshot(path: Path) -> dict[str, str]:
    return {
        item.relative_to(ROOT).as_posix(): sha256_file(item)
        for item in sorted(path.rglob("*")) if item.is_file()
    }


def prepare(path: str | Path) -> dict[str, Any]:
    target = run_path(path)
    require(not target.exists(), "immutable run already exists")
    before = protected_snapshot()
    write(target / "protected_before.json", before)
    candidates = {}
    for league in LEAGUES:
        deployment = read(PHASE9 / league / "deployment_specification.json")
        candidates[league] = {
            "bundle_id": deployment["candidate"]["bundle_id"],
            "bundle": deployment["candidate"],
        }
    phase7p = ROOT / "baselines/phase7p"
    spec = {
        "schema_version": "ffps.phase11_run.v1",
        "created_at": now(),
        "leagues": list(LEAGUES),
        "implementation_order": list(LEAGUES),
        "phase10_run": identity(PHASE10 / "artifact_index.json"),
        "phase9_candidates": candidates,
        "pl_phase7p_before": _tree_snapshot(phase7p) if phase7p.exists() else {},
        "model_training_authorized": False,
        "calibration_authorized": False,
        "promotion_authorized": False,
    }
    spec["specification_sha256"] = sha256_json(spec)
    write(target / "run_specification.json", spec)
    return spec


def open_run(path: str | Path) -> Path:
    target = run_path(path)
    check((target / "run_specification.json").exists(), "RUN_NOT_PREPARED")
    check(not (target / "artifact_index.json").exists(), "RUN_ALREADY_FINALIZED")
    return target


def credentials() -> dict[str, str | None]:
    values = dotenv_values(ROOT / ".env")
    return {
        "football_data_org": values.get("FOOTBALL_DATA_API_KEY") or
        os.getenv("FOOTBALL_DATA_API_KEY"),
        "api_football": values.get("API_FOOTBALL_KEY") or
        os.getenv("API_FOOTBALL_KEY"),
        "rapidapi": values.get("RAPIDAPI_KEY") or os.getenv("RAPIDAPI_KEY"),
    }


def fetch(
    directory: str | Path,
    name: str,
    url: str,
    *,
    provider: str,
    params: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    timeout: int = 30,
) -> dict[str, Any]:
    """Archive exact source bytes and safe request metadata, including errors."""
    check(name.replace("-", "").replace("_", "").isalnum(), "UNSAFE_SOURCE_NAME")
    target_dir = Path(directory)
    target_dir.mkdir(parents=True, exist_ok=True)
    raw_path = target_dir / f"{name}.raw"
    meta_path = target_dir / f"{name}.json"
    require(not raw_path.exists() and not meta_path.exists(), "source already archived")
    safe_params = params or {}
    record: dict[str, Any] = {
        "schema_version": "ffps.phase11_raw_source.v1",
        "provider": provider,
        "request": {"url": url, "params": safe_params},
        "started_at": now(),
        "credential_headers_present": sorted(
            key.lower() for key in (headers or {})
            if key.lower() in {"authorization", "x-auth-token", "x-apisports-key", "x-rapidapi-key"}
        ),
    }
    try:
        response = requests.get(
            url, params=safe_params, headers=headers, timeout=timeout,
            allow_redirects=False,
        )
        body = response.content
        for key, secret in (headers or {}).items():
            if key.lower() in {"authorization", "x-auth-token", "x-apisports-key", "x-rapidapi-key"}:
                require(not secret or str(secret).encode() not in body,
                        "credential reflected in provider response")
        with raw_path.open("xb") as stream:
            stream.write(body)
        allowed_headers = {
            key.lower(): value for key, value in response.headers.items()
            if key.lower() in {
                "content-type", "date", "last-modified", "retry-after",
                "x-ratelimit-limit", "x-ratelimit-remaining",
                "x-ratelimit-requests-limit", "x-ratelimit-requests-remaining",
            }
        }
        record.update({
            "status": "RETRIEVED" if response.status_code == 200 else "HTTP_ERROR",
            "http_status": response.status_code,
            "retrieved_at": now(),
            "response_headers": allowed_headers,
            "raw": identity(raw_path),
        })
    except requests.RequestException as exc:
        record.update({
            "status": "SOURCE_UNAVAILABLE",
            "error_type": type(exc).__name__,
            "retrieved_at": now(),
        })
    write(meta_path, record)
    return record


def acquire_league(path: str | Path, league: str, attempt: str = "qualified") -> dict[str, Any]:
    """Acquire both relevant seasons from the corrected primary/statistics routes."""
    target = open_run(path)
    check(league in SOURCE_SPECS, "UNCONFIGURED_PHASE11_LEAGUE")
    check(attempt.replace("-", "").isalnum(), "UNSAFE_ATTEMPT")
    from src.utils.naming import calculate_season
    season = calculate_season(datetime.now(timezone.utc), league)
    spec = SOURCE_SPECS[league]
    keys = credentials()
    primary_headers = ({"X-Auth-Token": keys["football_data_org"]}
                       if keys["football_data_org"] else {})
    csv_headers = {
        "User-Agent": "Mozilla/5.0 (compatible; FFPS-Phase11-Qualification/1.0)",
        "Accept": "text/csv,*/*;q=0.8",
    }
    directory = target / league / "raw_sources" / attempt
    records: list[dict[str, Any]] = []
    for year in (season - 1, season):
        for endpoint in ("matches", "teams"):
            records.append(fetch(
                directory, f"primary-{year}-{endpoint}",
                f"https://api.football-data.org/v4/competitions/{league}/{endpoint}",
                provider="football-data.org:v4", params={"season": year},
                headers=primary_headers,
            ))
        season_token = f"{year % 100:02d}{(year + 1) % 100:02d}"
        records.append(fetch(
            directory, f"statistics-{year}",
            f"https://football-data.co.uk/mmz4281/{season_token}/{spec['division']}.csv",
            provider="football-data.co.uk:csv:no-www", headers=csv_headers,
        ))
    index = {
        "schema_version": "ffps.phase11_source_index.v1", "league": league,
        "requested_season": season, "division": spec["division"],
        "attempt": attempt, "created_at": now(), "sources": records,
    }
    index["index_sha256"] = sha256_json(index)
    write(target / league / f"source_index_{attempt}.json", index)
    return {"league": league, "season": season, "attempt": attempt,
            "sources": [{"provider": item["provider"], "status": item["status"],
                         "http_status": item.get("http_status")} for item in records]}


def acquire_fotmob_current(path: str | Path, league: str, attempt: str = "fotmob-current") -> dict[str, Any]:
    """Archive a current league fixture set and details for every finished match."""
    target = open_run(path)
    check(league in SOURCE_SPECS, "UNCONFIGURED_PHASE11_LEAGUE")
    check(attempt.replace("-", "").isalnum(), "UNSAFE_ATTEMPT")
    from src.utils.naming import calculate_season
    year = calculate_season(datetime.now(timezone.utc), league)
    spec = SOURCE_SPECS[league]
    selected_season = f"{year}/{year + 1}"
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; FFPS-Phase11-Qualification/1.0)",
        "Accept": "application/json", "Referer": "https://www.fotmob.com/",
    }
    directory = target / league / "raw_sources" / attempt
    league_source = fetch(
        directory, f"league-{year}", "https://www.fotmob.com/api/data/leagues",
        provider="fotmob:web-api",
        params={"id": spec["fotmob_id"], "season": selected_season,
                "ccode3": spec["fotmob_country"]}, headers=headers,
    )
    records = [league_source]
    check(league_source.get("status") == "RETRIEVED", "FOTMOB_LEAGUE_UNAVAILABLE")
    payload = json.loads((ROOT / league_source["raw"]["path"]).read_text(encoding="utf-8"))
    details = payload.get("details", {})
    check(details.get("id") == spec["fotmob_id"] and details.get("selectedSeason") == selected_season,
          "FOTMOB_WRONG_LEAGUE_OR_SEASON")
    fixtures = payload.get("fixtures", {}).get("allMatches", [])
    check(isinstance(fixtures, list) and fixtures, "FOTMOB_EMPTY_FIXTURES")
    finished = [fixture for fixture in fixtures if fixture.get("status", {}).get("finished") is True]
    check(finished, "FOTMOB_NO_COMPLETED_FIXTURES")
    for fixture in finished:
        fixture_id = str(fixture.get("id", ""))
        check(fixture_id.isdigit(), "INVALID_FOTMOB_MATCH_ID")
        records.append(fetch(
            directory, f"match-{fixture_id}",
            "https://www.fotmob.com/api/data/matchDetails", provider="fotmob:web-api",
            params={"matchId": fixture_id}, headers=headers,
        ))
        time.sleep(0.15)
    index = {
        "schema_version": "ffps.phase11_fotmob_index.v1", "league": league,
        "season": year, "selected_season": selected_season, "attempt": attempt,
        "created_at": now(), "league_source": league_source,
        "finished_fixture_ids": [str(item["id"]) for item in finished],
        "detail_sources": records[1:],
    }
    index["index_sha256"] = sha256_json(index)
    write(target / league / f"fotmob_index_{attempt}.json", index)
    return {"league": league, "season": year, "fixtures": len(fixtures),
            "finished": len(finished),
            "details_retrieved": sum(item.get("status") == "RETRIEVED" for item in records[1:]),
            "details_failed": sum(item.get("status") != "RETRIEVED" for item in records[1:])}


def _raw_bytes(source: dict[str, Any]) -> bytes:
    check(source.get("status") == "RETRIEVED" and source.get("http_status") == 200,
          f"{source.get('provider', 'SOURCE')}_UNAVAILABLE")
    reference = source.get("raw", {})
    path = (ROOT / reference.get("path", "")).resolve()
    check(path.is_relative_to(ROOT), "SOURCE_PATH_ESCAPE")
    check(path.exists() and sha256_file(path) == reference.get("sha256"), "SOURCE_BYTES_DRIFT")
    check(path.stat().st_size == reference.get("size_bytes"), "SOURCE_SIZE_DRIFT")
    return path.read_bytes()


def _json_payload(source: dict[str, Any]) -> dict[str, Any]:
    try:
        value = json.loads(_raw_bytes(source))
    except (json.JSONDecodeError, UnicodeDecodeError):
        raise ValueError("MALFORMED_JSON") from None
    check(isinstance(value, dict), "MALFORMED_JSON")
    return value


def _mapped_teams(payload: dict[str, Any], league: str, season: int,
                  names: dict[str, set[str]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Apply existing exact aliases, then explicit league/provider-ID evidence."""
    from src.evaluation.phase10_current_data import map_teams
    mapped, failures = map_teams(payload, league, season, names)
    overrides = TEAM_MAPPING_OVERRIDES.get(league, {}).get("football-data.org:v4", {})
    evidence: list[dict[str, Any]] = []
    for provider_id, rule in overrides.items():
        key = str(provider_id)
        if key not in mapped:
            continue
        row = mapped[key]
        check([row["name"], row.get("short_name")] == rule["source_names"],
              "SOURCE_TEAM_IDENTITY_CHANGED")
        check(rule["canonical"] in {value for values in names.values() for value in values},
              "MAPPING_TARGET_OUTSIDE_MODEL_VOCABULARY")
        check(not any(other.get("canonical") == rule["canonical"] for k, other in mapped.items()
                      if k != key), "MANY_TO_ONE_TEAM_MAPPING")
        mapped[key] = {**row, "canonical": rule["canonical"],
                       "status": "EXACT_PHASE11_MAPPING"}
        failures = [failure for failure in failures if failure.get("team_id") != provider_id]
        evidence.append({"provider": "football-data.org:v4", "provider_team_id": provider_id,
                         "source_names": rule["source_names"], "canonical": rule["canonical"],
                         "league": league, "season": season, "reason": rule["reason"]})
    return mapped, failures + evidence


def _phase11_names(league: str) -> dict[str, set[str]]:
    """Return exact provider tokens bound to the frozen model-facing vocabulary."""
    from src.evaluation.phase10_current_data import exact_names
    names = exact_names(league)
    replacements = MODEL_CANONICAL_OVERRIDES.get(league, {})
    result = {key: {replacements.get(value, value) for value in values}
              for key, values in names.items()}
    for source_token, canonical in TEAM_TOKEN_OVERRIDES.get(league, {}).items():
        choices = result.setdefault(source_token, set())
        check(not choices or choices == {canonical}, "AMBIGUOUS_PHASE11_TEAM_TOKEN")
        choices.add(canonical)
    return result


def _adapt_primary(source: dict[str, Any], league: str, season: int,
                   teams: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Normalize football-data.org while preserving its contradictory season fields."""
    from src.evaluation.phase10_current_data import instant
    payload = _json_payload(source)
    check(payload.get("competition", {}).get("code") == league, "WRONG_LEAGUE")
    check(payload.get("filters", {}).get("season") == season, "STALE_SEASON")
    rows = payload.get("matches")
    check(isinstance(rows, list) and rows, "EMPTY_MATCHES")
    result_set = payload.get("resultSet", {})
    check(result_set.get("count") == len(rows), "FIXTURE_COUNT_MISMATCH")
    dates = [row.get("utcDate", "")[:10] for row in rows]
    check(all(re.fullmatch(r"\d{4}-\d{2}-\d{2}", value) for value in dates),
          "INVALID_FIXTURE_DATE")
    authoritative_start = result_set.get("first")
    authoritative_end = result_set.get("last")
    check(authoritative_start == min(dates) and authoritative_end == max(dates),
          "RESULT_SET_BOUNDARY_MISMATCH")
    season_metadata = sorted({
        (row.get("season", {}).get("id"), row.get("season", {}).get("startDate"),
         row.get("season", {}).get("endDate")) for row in rows
    })
    check(len(season_metadata) == 1, "AMBIGUOUS_SEASON_METADATA")
    season_id, reported_start, reported_end = season_metadata[0]
    check(isinstance(season_id, int) and reported_start[:4] == str(season), "STALE_SEASON")
    check(re.fullmatch(r"\d{4}-\d{2}-\d{2}", reported_end or "") is not None,
          "INVALID_SEASON_END")
    check(all(row.get("stage") == "REGULAR_SEASON" for row in rows),
          "NON_LEAGUE_STAGE_INCLUDED")
    events: list[dict[str, Any]] = []
    ids: set[int] = set()
    pair_keys: set[tuple[int, str, str]] = set()
    retrieval = instant(source["retrieved_at"])
    for raw in rows:
        kickoff = instant(raw["utcDate"])
        # KICKOFF_OUTSIDE_SEASON is preserved.  The source's resultSet.first/last
        # are used because they exactly define this league fixture population;
        # the conflicting season.startDate remains recorded as rejected metadata.
        check(authoritative_start <= kickoff.date().isoformat() <= authoritative_end,
              "KICKOFF_OUTSIDE_SEASON")
        fixture_id = raw.get("id")
        check(type(fixture_id) is int and fixture_id > 0 and fixture_id not in ids,
              "DUPLICATE_FIXTURE_ID")
        ids.add(fixture_id)
        home_id, away_id = str(raw.get("homeTeam", {}).get("id")), str(raw.get("awayTeam", {}).get("id"))
        check(home_id in teams and away_id in teams and home_id != away_id,
              "INVALID_FIXTURE_TEAM")
        check(teams[home_id].get("canonical") and teams[away_id].get("canonical"),
              "UNMAPPED_TEAM")
        pair = (season, home_id, away_id)
        check(pair not in pair_keys, "CHANGED_OR_DUPLICATE_FIXTURE_ID")
        pair_keys.add(pair)
        check(raw.get("competition", {}).get("code") == league, "WRONG_LEAGUE")
        updated = raw.get("lastUpdated")
        if updated:
            check(instant(updated) <= retrieval, "FUTURE_PROVIDER_UPDATE")
        score = raw.get("score", {}).get("fullTime", {})
        if raw.get("status") == "FINISHED":
            check(kickoff < retrieval, "FUTURE_RESULT")
            check(all(type(score.get(side)) is int and score[side] >= 0
                      for side in ("home", "away")), "INVALID_SCORE")
        events.append({
            "league": league, "season": season,
            "fixture_id": f"{league}_football_data_org_{fixture_id}",
            "provider_fixture_id": fixture_id, "kickoff_at": kickoff.isoformat(),
            "fixture_date": kickoff.date().isoformat(), "home_id": int(home_id),
            "away_id": int(away_id), "home_team": teams[home_id]["canonical"],
            "away_team": teams[away_id]["canonical"], "status": raw.get("status"),
            "home_goals": score.get("home"), "away_goals": score.get("away"),
            "provider_updated_at": updated, "retrieved_at": source["retrieved_at"],
        })
    discrepancy = [{"provider_fixture_id": row["id"], "kickoff_at": row["utcDate"]}
                   for row in rows if not reported_start <= row["utcDate"][:10] <= reported_end]
    evidence = {
        "requested_season": season, "competition": payload["competition"],
        "season_id": season_id, "reported_season_start": reported_start,
        "reported_season_end": reported_end, "result_set_first": authoritative_start,
        "result_set_last": authoritative_end,
        "authoritative_bounds_basis": "football-data.org resultSet.first/resultSet.last; exact min/max of the regular-season fixture population",
        "season_metadata_discrepancy": discrepancy,
        "normalized_fixture_count": len(events),
        "kickoff_outside_authoritative_bounds": [],
        "interpretation": "provider season start/end metadata rejected where internally inconsistent with regular-season fixtures; resultSet bounds remain source supplied and are enforced",
    }
    return events, evidence


def _adapt_fotmob(index: dict[str, Any], league: str, season: int,
                   names: dict[str, set[str]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from src.evaluation.phase10_current_data import instant, token
    spec = SOURCE_SPECS[league]
    league_payload = _json_payload(index["league_source"])
    details = league_payload.get("details", {})
    check(details.get("id") == spec["fotmob_id"] and details.get("name") == spec["fotmob_name"],
          "FOTMOB_WRONG_LEAGUE")
    check(details.get("selectedSeason") == index["selected_season"] and index.get("season") == season,
          "FOTMOB_WRONG_SEASON")
    fixtures = league_payload.get("fixtures", {}).get("allMatches", [])
    check(isinstance(fixtures, list) and fixtures, "FOTMOB_EMPTY_FIXTURES")
    fixture_by_id = {str(item.get("id")): item for item in fixtures}
    check(len(fixture_by_id) == len(fixtures), "FOTMOB_DUPLICATE_FIXTURE")
    finished_ids = {key for key, item in fixture_by_id.items()
                    if item.get("status", {}).get("finished") is True}
    check(finished_ids == set(index.get("finished_fixture_ids", [])),
          "FOTMOB_FINISHED_SET_DRIFT")
    source_by_id = {}
    for source in index.get("detail_sources", []):
        requested = str(source.get("request", {}).get("params", {}).get("matchId", ""))
        check(requested.isdigit() and requested not in source_by_id, "FOTMOB_DUPLICATE_DETAIL")
        source_by_id[requested] = source
    check(set(source_by_id) == finished_ids, "FOTMOB_DETAIL_COVERAGE_INCOMPLETE")
    rows: list[dict[str, Any]] = []
    source_name_variants: list[dict[str, Any]] = []
    for fixture_id in sorted(finished_ids, key=int):
        fixture, source = fixture_by_id[fixture_id], source_by_id[fixture_id]
        payload = _json_payload(source)
        general = payload.get("general", {})
        check(str(general.get("matchId")) == fixture_id and general.get("leagueId") == spec["fotmob_id"]
              and general.get("leagueName") == spec["fotmob_name"], "FOTMOB_DETAIL_IDENTITY_MISMATCH")
        check(general.get("finished") is True, "FOTMOB_NONFINAL_DETAIL")
        kickoff = instant(general.get("matchTimeUTCDate"))
        check(kickoff == instant(fixture.get("status", {}).get("utcTime")), "FOTMOB_KICKOFF_CONFLICT")
        mapped = []
        for side in ("home", "away"):
            fixture_team = fixture.get(side, {})
            detail_team = general.get(side + "Team", {})
            check(str(fixture_team.get("id")) == str(detail_team.get("id")),
                  "FOTMOB_TEAM_IDENTITY_CONFLICT")
            fixture_name, detail_name = fixture_team.get("name"), detail_team.get("name")
            check(token(fixture_name) == token(detail_name), "FOTMOB_TEAM_IDENTITY_CONFLICT")
            if fixture_name != detail_name:
                source_name_variants.append({
                    "provider_fixture_id": int(fixture_id), "side": side,
                    "team_id": str(fixture_team.get("id")),
                    "league_fixture_name": fixture_name, "match_detail_name": detail_name,
                    "comparison": "exact NFKD ASCII-folded uppercase token equality",
                })
            choices = names.get(token(fixture_team.get("name")), set())
            check(len(choices) == 1, "FOTMOB_UNMAPPED_OR_AMBIGUOUS_TEAM")
            mapped.append(next(iter(choices)))
        score_text = payload.get("header", {}).get("status", {}).get("scoreStr", "")
        score_match = re.fullmatch(r"\s*(\d+)\s*-\s*(\d+)\s*", str(score_text))
        check(score_match is not None, "FOTMOB_INVALID_SCORE")
        corners = []
        for group in payload.get("content", {}).get("stats", {}).get("Periods", {}).get("All", {}).get("stats", []):
            for statistic in group.get("stats", []):
                if statistic.get("title") == "Corners":
                    corners.append(statistic.get("stats"))
        check(len(corners) == 1 and isinstance(corners[0], list) and len(corners[0]) == 2,
              "FOTMOB_MISSING_CORNERS")
        check(all(type(value) is int and value >= 0 for value in corners[0]),
              "FOTMOB_INVALID_CORNERS")
        rows.append({
            "league": league, "season": season, "fixture_date": kickoff.date().isoformat(),
            "home_team": mapped[0], "away_team": mapped[1],
            "home_goals": int(score_match.group(1)), "away_goals": int(score_match.group(2)),
            "home_corners": corners[0][0], "away_corners": corners[0][1],
            "provider_fixture_id": int(fixture_id), "retrieved_at": source["retrieved_at"],
            "raw": source["raw"], "publication_timestamp": None,
            "publication_policy": "provider publication timestamp unavailable; FFPS retrieval is the availability upper bound",
        })
    return rows, {"league_id": details["id"], "league_name": details["name"],
                  "selected_season": details["selectedSeason"], "fixture_count": len(fixtures),
                  "finished_count": len(rows), "details_complete": len(rows) == len(finished_ids),
                  "source_name_variants": source_name_variants}


def _source_map(index: dict[str, Any]) -> dict[tuple[int, str], dict[str, Any]]:
    result = {}
    for source in index.get("sources", []):
        match = re.fullmatch(r"primary-(\d{4})-(matches|teams)|statistics-(\d{4})",
                             Path(source.get("raw", {}).get("path", "")).stem)
        if match:
            year = int(match.group(1) or match.group(3))
            kind = match.group(2) or "statistics"
            result[(year, kind)] = source
    return result


def provider_overlap(statistics: list[dict[str, Any]], alternative: list[dict[str, Any]],
                     season: int) -> tuple[list[dict[str, Any]], list[list[str]]]:
    """Compare current statistics authorities on exact deterministic fixture keys."""
    def key(row: dict[str, Any]) -> tuple[str, str, str]:
        return row["fixture_date"], row["home_team"], row["away_team"]
    other_by_key = {key(row): row for row in alternative}
    current = {key(row): row for row in statistics if row["season"] == season}
    overlap: list[dict[str, Any]] = []
    conflicts: list[list[str]] = []
    for join, row in sorted(current.items()):
        other = other_by_key.get(join)
        fields = {field: other is not None and row[field] == other[field]
                  for field in ("home_goals", "away_goals", "home_corners", "away_corners")}
        overlap.append({"key": list(join), "fields": fields,
                        "football_data_co_uk_row": row["row_ordinal"],
                        "fotmob_fixture_id": other.get("provider_fixture_id") if other else None})
        if not all(fields.values()):
            conflicts.append(list(join))
    return overlap, conflicts


def pipeline(frame: pd.DataFrame, league: str) -> pd.DataFrame:
    """Run the sealed feature logic with a reversible identifier-only adapter.

    The production sanitizer deliberately rejects apostrophes.  BL1's frozen
    canonical identity contains one, so Phase 11 substitutes a collision-checked
    safe token for grouping and restores the canonical label afterward.  Team
    identity equality, row ordering, and every numeric input remain unchanged.
    """
    from src.evaluation import phase10_current_data as phase10
    sentinels = SAFE_TEAM_SENTINELS.get(league, {})
    if not sentinels:
        return phase10.pipeline(frame, league)
    encoded = frame.copy()
    team_columns = ("home_team", "away_team")
    observed = {str(value) for column in team_columns for value in encoded[column].dropna()}
    for canonical, sentinel in sentinels.items():
        check(sentinel not in observed, "SAFE_TEAM_SENTINEL_COLLISION")
        check(canonical != sentinel and re.fullmatch(r"[\w\s\-\.]+", sentinel) is not None,
              "UNSAFE_TEAM_SENTINEL")
    for column in team_columns:
        encoded[column] = encoded[column].replace(sentinels)
    generated = phase10.pipeline(encoded, league)
    reverse = {sentinel: canonical for canonical, sentinel in sentinels.items()}
    check(len(reverse) == len(sentinels), "SAFE_TEAM_SENTINEL_COLLISION")
    for column in team_columns:
        generated[column] = generated[column].replace(reverse)
    generated.attrs["ffps_phase11_identifier_adapter"] = {
        "league": league, "mapping": sentinels,
        "scope": "team identity columns before sanitizer and grouping only",
    }
    return generated


def feature_reproduction(league: str, names: list[str]) -> dict[str, Any]:
    """Replay frozen features through the Phase 11 league adapter."""
    import numpy as np
    from src.evaluation import phase10_current_data as phase10
    generated = pipeline(phase10.base_frame(league), league)
    frozen = pd.read_csv(ROOT / "data/features/feature_matrix.csv", low_memory=False)
    frozen = frozen.loc[frozen.league.eq(league)]
    missing = sorted(set(names) - set(generated))
    if missing:
        return {"reproduced": False, "code": "FEATURE_SCHEMA_MISMATCH",
                "missing_features": missing}
    joined = generated.merge(frozen[["match_hash", *names]], left_on="match_id",
                             right_on="match_hash", suffixes=("_new", "_old"))
    differences: dict[str, int] = {}
    eligible_differences: dict[str, int] = {}
    eligible = np.isfinite(joined[[feature + "_old" for feature in names]].to_numpy(float)).all(axis=1)
    for feature in names:
        left, right = (joined[feature + suffix].to_numpy(float)
                       for suffix in ("_new", "_old"))
        same = np.isclose(left, right, rtol=0, atol=1e-12, equal_nan=True)
        if not same.all():
            differences[feature] = int((~same).sum())
        if not same[eligible].all():
            eligible_differences[feature] = int((~same[eligible]).sum())
    reproduced = (len(generated) == len(joined) == len(frozen) and bool(eligible.any())
                  and not eligible_differences)
    result = {
        "reproduced": reproduced,
        "code": "VERIFIED" if reproduced else "FEATURE_SEMANTICS_INCOMPATIBLE",
        "generated_rows": len(generated), "frozen_rows": len(frozen),
        "joined_rows": len(joined), "all_row_differing_features": differences,
        "model_eligible_differing_features": eligible_differences,
        "model_eligible_rows": int(eligible.sum()),
        "excluded_frozen_nonfinite_rows": int((~eligible).sum()),
        "eligibility_rule": "frozen ordered features all finite, identical to Phase 9 pre-partition cold-start exclusion; no imputation",
        "schema_sha256": sha256_json(names),
        "source": identity(ROOT / "data/features/feature_matrix.csv"),
    }
    if SAFE_TEAM_SENTINELS.get(league):
        result["identifier_adapter"] = SAFE_TEAM_SENTINELS[league]
    return result


def target_matrix(assessment: dict[str, Any], target: dict[str, Any], *,
                  as_of: str, source_index: dict[str, Any]):
    """Build Phase 10 features and attach replayable per-team causal cutoffs."""
    import numpy as np
    from src.evaluation import phase10_current_data as phase10
    from src.predictions.provenance import build_file_snapshot, source_event_index
    check(assessment["qualified"], "CURRENT_DATA_UNQUALIFIED")
    check(target in assessment["upcoming"], "UNKNOWN_TARGET_FIXTURE")
    check(phase10.instant(as_of) < phase10.instant(target["kickoff_at"]),
          "RETROSPECTIVE_RECONSTRUCTION")
    rows = []
    for event in assessment["composed_observations"]:
        check(phase10.instant(event["retrieved_at"]) <= phase10.instant(as_of), "LATE_RETRIEVAL")
        if phase10.instant(event["kickoff_at"]) >= phase10.instant(as_of):
            continue
        rows.append({
            "match_id": event["fixture_id"], "date": event["kickoff_at"],
            "league": event["league"], "season": event["season"],
            "home_team": event["home_team"], "away_team": event["away_team"],
            "home_score": event["home_goals"], "away_score": event["away_goals"],
            "home_corners": event["home_corners"], "away_corners": event["away_corners"],
            "status": "FINISHED",
        })
    rows.append({
        "match_id": target["fixture_id"], "date": target["kickoff_at"],
        "league": target["league"], "season": target["season"],
        "home_team": target["home_team"], "away_team": target["away_team"],
        "home_score": np.nan, "away_score": np.nan,
        "home_corners": np.nan, "away_corners": np.nan, "status": target["status"],
    })
    composite = pd.concat([phase10.base_frame(target["league"]), pd.DataFrame(rows)],
                          ignore_index=True)
    composite["date"] = pd.to_datetime(composite.date, utc=True)
    matrix = pipeline(composite, target["league"])
    matrix["match_hash"] = matrix.match_id
    artifacts = [dict(identity(ROOT / "data/processed/matches.csv"),
                      role="frozen_causal_history", load_order=0)]
    for source in source_index["sources"]:
        artifacts.append(dict(source["raw"], role="archived_current_provider_payload",
                              load_order=len(artifacts)))
    matrix.attrs["ffps_data_snapshot"] = build_file_snapshot(
        frame=matrix, artifacts=artifacts, source_type="phase11_causal_multi_source")
    matrix.attrs["ffps_source_events"] = source_event_index(composite)
    target_rows = matrix.loc[matrix.match_id.eq(target["fixture_id"])]
    check(len(target_rows) == 1, "NONUNIQUE_FEATURE_TARGET")
    fixture = target_rows.iloc[0]
    history = phase10.base_frame(target["league"])[["date", "home_team", "away_team"]].copy()
    observed = pd.DataFrame([{
        "date": row["kickoff_at"], "home_team": row["home_team"],
        "away_team": row["away_team"],
    } for row in assessment["composed_observations"]])
    if not observed.empty:
        observed["date"] = pd.to_datetime(observed["date"], utc=True)
        history = pd.concat([history, observed], ignore_index=True)
    history["date"] = pd.to_datetime(history["date"], utc=True)
    target_time = phase10.instant(target["kickoff_at"])
    as_of_time = phase10.instant(as_of)
    row_index = matrix.index[matrix.match_id.eq(target["fixture_id"])]
    check(len(row_index) == 1, "NONUNIQUE_FEATURE_TARGET")
    for side in ("home", "away"):
        team = target[side + "_team"]
        dates = history.loc[((history.home_team == team) | (history.away_team == team)) &
                            (history.date < target_time) & (history.date <= as_of_time), "date"]
        check(not dates.empty, "FEATURE_CAUSAL_CUTOFF_UNPROVEN")
        latest = dates.max()
        for suffix in ("_rolling_max_source_date_3", "_rolling_max_source_date_5",
                       "_rolling_max_source_date_10", "_std_max_source_date"):
            matrix.loc[row_index, side + "_" + suffix] = latest
    fixture = matrix.loc[row_index[0]]
    return matrix, fixture


def qualify_league(path: str | Path, league: str, source_attempt: str,
                   fotmob_attempt: str) -> dict[str, Any]:
    """Run strict league qualification and one current causal feature proof."""
    from src.evaluation import phase10_current_data as phase10
    from src.ml.league_bundle_activation import LeagueBundleController

    check(league in LEAGUES, "UNCONFIGURED_PHASE11_LEAGUE")
    target = open_run(path)
    assessment_root = target / league / "assessments"
    assessment_directory = assessment_root / f"assessment-{len(list(assessment_root.glob('assessment-*'))) + 1:03d}"
    source_path = target / league / f"source_index_{source_attempt}.json"
    fotmob_path = target / league / f"fotmob_index_{fotmob_attempt}.json"
    source_index, fotmob_index = read(source_path), read(fotmob_path)
    check(source_index.get("league") == league and fotmob_index.get("league") == league,
          "CROSS_LEAGUE_SOURCE")
    sources = _source_map(source_index)
    season = source_index["requested_season"]
    check(set(sources) == {(year, kind) for year in (season - 1, season)
                           for kind in ("matches", "teams", "statistics")},
          "SOURCE_INDEX_INCOMPLETE")
    names = _phase11_names(league)
    events: list[dict[str, Any]] = []
    statistics: list[dict[str, Any]] = []
    mappings: dict[str, Any] = {}
    mapping_failures: list[dict[str, Any]] = []
    mapping_evidence: list[dict[str, Any]] = []
    season_evidence: dict[str, Any] = {}
    for year in (season - 1, season):
        team_payload = _json_payload(sources[(year, "teams")])
        mapped, results = _mapped_teams(team_payload, league, year, names)
        mappings[str(year)] = mapped
        mapping_failures.extend(item for item in results if "code" in item)
        mapping_evidence.extend(item for item in results if "reason" in item)
        normalized, evidence = _adapt_primary(sources[(year, "matches")], league, year, mapped)
        events.extend(normalized)
        season_evidence[str(year)] = evidence
        statistics.extend(phase10.statistics_events(
            _raw_bytes(sources[(year, "statistics")]), league, year,
            source_index["division"], names))
    fotmob_statistics, fotmob_evidence = _adapt_fotmob(fotmob_index, league, season, names)

    def key(row: dict[str, Any]) -> tuple[str, str, str]:
        return row["fixture_date"], row["home_team"], row["away_team"]

    csv_by_season = {(row["season"], *key(row)): row for row in statistics}
    fotmob_by_key = {key(row): row for row in fotmob_statistics}
    overlap, provider_conflicts = provider_overlap(statistics, fotmob_statistics, season)
    check(overlap, "ALTERNATIVE_PROVIDER_OVERLAP_UNPROVEN")

    frozen = pd.read_csv(ROOT / "data/processed/matches.csv", low_memory=False)
    frozen = frozen.loc[frozen.league.eq(league)]
    frozen_index = {(str(row.match_date)[:10], row.home_team, row.away_team): row
                    for row in frozen.itertuples()}
    cutoff = str(frozen.match_date.max())[:10]
    cutoff_frontier_events = [event for event in events if event["fixture_date"] == cutoff
                              and key(event) in frozen_index and event["status"] == "FINISHED"]
    check(cutoff_frontier_events, "FROZEN_CAUSAL_FRONTIER_UNPROVEN")
    cutoff_frontier = max(phase10.instant(event["kickoff_at"])
                          for event in cutoff_frontier_events)
    historical_overlap, composed, missing, conflicts, unresolved = [], [], [], [], []
    frontier_extensions: list[dict[str, Any]] = []
    for event in events:
        if event["status"] != "FINISHED":
            continue
        join = key(event)
        statistic = csv_by_season.get((event["season"], *join))
        if event["season"] == season:
            fotmob_row = fotmob_by_key.get(join)
            if statistic is None:
                statistic = fotmob_row
            if fotmob_row is None:
                missing.append(event["fixture_id"])
        if statistic is None:
            missing.append(event["fixture_id"])
            continue
        if [event["home_goals"], event["away_goals"]] != [statistic["home_goals"], statistic["away_goals"]]:
            conflicts.append(event["fixture_id"])
        old = frozen_index.get(join)
        after_frontier = (event["fixture_date"] > cutoff or
                          (event["fixture_date"] == cutoff and old is None and
                           phase10.instant(event["kickoff_at"]) > cutoff_frontier))
        if not after_frontier:
            if old is None:
                unresolved.append(event["fixture_id"])
            else:
                fields = {
                    "home_goals": event["home_goals"] == old.home_goals,
                    "away_goals": event["away_goals"] == old.away_goals,
                    "home_corners": statistic["home_corners"] == old.home_corners,
                    "away_corners": statistic["away_corners"] == old.away_corners,
                }
                historical_overlap.append({"fixture_id": event["fixture_id"],
                                           "key": list(join), "fields": fields})
                if not all(fields.values()):
                    conflicts.append(event["fixture_id"])
        else:
            if event["fixture_date"] == cutoff:
                frontier_extensions.append({
                    "fixture_id": event["fixture_id"], "kickoff_at": event["kickoff_at"],
                    "frozen_same_day_latest_kickoff": cutoff_frontier.isoformat(),
                    "basis": "strictly later source fixture on the frozen maximum date; absent from frozen rows; two current authorities agree",
                })
            stat_source = (sources[(event["season"], "statistics")]["raw"]
                           if (event["season"], *join) in csv_by_season else statistic["raw"])
            composed.append({**event, "home_corners": statistic["home_corners"],
                             "away_corners": statistic["away_corners"],
                             "field_provenance": {
                                 "home_goals": {"source": sources[(event["season"], "matches")]["raw"],
                                                "provider_fixture_id": event["provider_fixture_id"]},
                                 "away_goals": {"source": sources[(event["season"], "matches")]["raw"],
                                                "provider_fixture_id": event["provider_fixture_id"]},
                                 "home_corners": {"source": stat_source,
                                                  "provider_fixture_id": statistic.get("provider_fixture_id"),
                                                  "row_ordinal": statistic.get("row_ordinal")},
                                 "away_corners": {"source": stat_source,
                                                  "provider_fixture_id": statistic.get("provider_fixture_id"),
                                                  "row_ordinal": statistic.get("row_ordinal")},
                             }})
    blockers: list[dict[str, Any]] = list(mapping_failures)
    if provider_conflicts:
        blockers.append({"code": "ALTERNATIVE_PROVIDER_DISAGREEMENT", "keys": provider_conflicts})
    if missing:
        blockers.append({"code": "MISSING_REQUIRED_CURRENT_STATISTICS", "fixtures": sorted(set(missing))})
    if conflicts:
        blockers.append({"code": "SOURCE_SEMANTICS_INCOMPATIBLE", "fixtures": sorted(set(conflicts))})
    if unresolved:
        blockers.append({"code": "HISTORICAL_PRIMARY_JOIN_UNPROVEN", "fixtures": sorted(set(unresolved))})
    if not historical_overlap:
        blockers.append({"code": "HISTORICAL_STATISTIC_OVERLAP_UNPROVEN"})
    as_of = now()
    upcoming = sorted([event for event in events if event["season"] == season and
                       event["status"] in ("TIMED", "SCHEDULED") and
                       phase10.instant(event["kickoff_at"]) > phase10.instant(as_of)],
                      key=lambda row: (row["kickoff_at"], row["fixture_id"]))
    if not upcoming:
        blockers.append({"code": "UPCOMING_CAUSAL_FIXTURES_UNPROVEN"})
    if not any(event["season"] == season and event["status"] == "FINISHED" for event in events):
        blockers.append({"code": "CURRENT_COMPLETED_COVERAGE_UNPROVEN"})
    assessment = {
        "league": league, "season": season, "as_of": as_of, "blockers": blockers,
        "team_mappings": mappings, "mapping_evidence": mapping_evidence,
        "events": events, "statistics": statistics, "fotmob_statistics": fotmob_statistics,
        "historical_overlap": historical_overlap, "cross_provider_overlap": overlap,
        "frozen_causal_frontier": {"date": cutoff, "latest_matched_kickoff": cutoff_frontier.isoformat(),
                                   "same_day_extensions": frontier_extensions},
        "season_evidence": season_evidence, "fotmob_evidence": fotmob_evidence,
        "composed_observations": sorted(composed, key=lambda row: row["kickoff_at"]),
        "upcoming": upcoming, "qualified": not blockers,
    }
    combined_sources = source_index["sources"] + [fotmob_index["league_source"]] + fotmob_index["detail_sources"]
    combined_index = {"schema_version": "ffps.phase11_qualified_source_index.v1",
                      "league": league, "requested_season": season,
                      "division": source_index["division"], "sources": combined_sources,
                      "primary_index": identity(source_path), "fotmob_index": identity(fotmob_path)}
    combined_index["index_sha256"] = sha256_json(combined_index)
    combined_path = assessment_directory / "qualified_source_index.json"
    write(combined_path, combined_index)
    controller = LeagueBundleController(PHASE9 / league / "deployment_specification.json", league=league)
    active, candidate = (controller.bundle(bundle_id)
                         for bundle_id in (controller.legacy_id, controller.candidate_id))
    check(active["features"] == candidate["features"], "ACTIVE_CANDIDATE_SCHEMA_MISMATCH")
    reproduction = feature_reproduction(league, active["features"])
    feature_proofs, feature_failures = [], []
    if not reproduction["reproduced"]:
        feature_failures.append({"code": reproduction["code"]})
    if assessment["qualified"] and reproduction["reproduced"] and upcoming:
        try:
            matrix, fixture = target_matrix(
                assessment, upcoming[0], as_of=as_of, source_index=combined_index)
            feature_proofs.append(phase10.validate_features(
                matrix, fixture, active["features"], as_of))
        except (ValueError, KeyError, TypeError, RuntimeError) as exc:
            feature_failures.append({"fixture_id": upcoming[0]["fixture_id"],
                                     "code": str(exc), "error_type": type(exc).__name__})
    else:
        feature_failures.append({"code": "CURRENT_FEATURE_CONTINUITY_BLOCKED_BY_SOURCE_QUALIFICATION"})
    ready = assessment["qualified"] and reproduction["reproduced"] and bool(feature_proofs) and not feature_failures
    qualification = {
        "schema_version": "ffps.phase11_qualification.v1", "league": league,
        "as_of": as_of, "assessment": assessment,
        "historical_feature_reproduction": reproduction,
        "feature_proofs": feature_proofs, "feature_failures": feature_failures,
        "feature_readiness": ready, "active_bundle_id": controller.legacy_id,
        "candidate_bundle_id": controller.candidate_id,
        "active_sha256": active["aggregate_sha256"],
        "candidate_sha256": candidate["aggregate_sha256"],
        "ordered_features": active["features"],
        "feature_schema_sha256": sha256_json(active["features"]),
        "source_index": identity(combined_path),
        "phase9_deployment": identity(PHASE9 / league / "deployment_specification.json"),
        "promotion_authorized": False,
    }
    qualification["qualification_sha256"] = sha256_json(qualification)
    qualification_path = assessment_directory / "qualification.json"
    write(qualification_path, qualification)
    write(assessment_directory / "season_boundary_evidence.json", season_evidence)
    write(assessment_directory / "team_mapping_evidence.json",
          {"league": league, "mappings": mappings, "new_mappings": mapping_evidence,
           "fuzzy_matching_used": False})
    write(assessment_directory / "provider_overlap_evidence.json",
          {"league": league, "football_data_co_uk_vs_fotmob": overlap,
           "rows": len(overlap), "conflicts": provider_conflicts,
           "required_fields": ["home_goals", "away_goals", "home_corners", "away_corners"]})
    if ready:
        contract = {
            "schema_version": "ffps.phase11_current_source_contract.v1", "league": league,
            "season": season, "qualification": identity(qualification_path),
            "frozen_at": now(), "feature_schema_sha256": qualification["feature_schema_sha256"],
            "source_index": qualification["source_index"],
            "authority": {"fixture_identity": "football-data.org:v4",
                          "kickoff": "football-data.org:v4", "goals": "football-data.org:v4",
                          "corners_primary": "football-data.co.uk:csv:no-www",
                          "corners_gap_fill": "fotmob:web-api"},
            "provider_overlap_rows": len(overlap),
            "cohort_policy": "future current-season league fixtures; capture requires a complete causal feature vector and retrieval before kickoff",
            "promotion_authorized": False,
        }
        contract["contract_sha256"] = sha256_json(contract)
        write(assessment_directory / "current_source_contract.json", contract)
    return {"league": league, "ready": ready,
            "blockers": sorted({item.get("code", "UNKNOWN") for item in blockers + feature_failures}),
            "historical_overlap": len(historical_overlap), "provider_overlap": len(overlap),
            "composed_observations": len(composed), "upcoming": len(upcoming),
            "first_fixture": upcoming[0] if upcoming else None,
            "qualification": identity(qualification_path)}


def qualify_sa(path: str | Path, source_attempt: str = "sa-qualified",
               fotmob_attempt: str = "sa-fotmob-qualified") -> dict[str, Any]:
    """Compatibility entry point for the completed first SA vertical slice."""
    return qualify_league(path, "SA", source_attempt, fotmob_attempt)


def diagnose_sa(path: str | Path, attempt: str = "initial") -> dict[str, Any]:
    """Acquire bounded SA diagnostics from all repository-configured sources."""
    target = open_run(path)
    check(attempt.replace("-", "").isalnum(), "UNSAFE_ATTEMPT")
    keys = credentials()
    directory = target / "SA" / "raw_sources" / attempt
    fd_headers = ({"X-Auth-Token": keys["football_data_org"]}
                  if keys["football_data_org"] else {})
    api_headers = ({"x-apisports-key": keys["api_football"]}
                   if keys["api_football"] else {})
    records = [
        fetch(directory, "football-data-org-competition",
              "https://api.football-data.org/v4/competitions/SA",
              provider="football-data.org:v4", headers=fd_headers),
        fetch(directory, "football-data-org-matches-2026",
              "https://api.football-data.org/v4/competitions/SA/matches",
              provider="football-data.org:v4", params={"season": 2026}, headers=fd_headers),
        fetch(directory, "football-data-org-standings-2026",
              "https://api.football-data.org/v4/competitions/SA/standings",
              provider="football-data.org:v4", params={"season": 2026}, headers=fd_headers),
        fetch(directory, "football-data-co-uk-data-page",
              "https://www.football-data.co.uk/data.php",
              provider="football-data.co.uk:web"),
        fetch(directory, "football-data-co-uk-statistics-2025",
              "https://www.football-data.co.uk/mmz4281/2526/I1.csv",
              provider="football-data.co.uk:csv"),
        fetch(directory, "football-data-co-uk-statistics-2026",
              "https://www.football-data.co.uk/mmz4281/2627/I1.csv",
              provider="football-data.co.uk:csv"),
        fetch(directory, "api-football-status",
              "https://v3.football.api-sports.io/status",
              provider="api-football:v3", headers=api_headers),
        fetch(directory, "api-football-league-2026",
              "https://v3.football.api-sports.io/leagues",
              provider="api-football:v3", params={"id": API_FOOTBALL_IDS["SA"], "season": 2026},
              headers=api_headers),
        fetch(directory, "api-football-fixtures-2026",
              "https://v3.football.api-sports.io/fixtures",
              provider="api-football:v3", params={"league": API_FOOTBALL_IDS["SA"], "season": 2026},
              headers=api_headers),
    ]
    index = {
        "schema_version": "ffps.phase11_diagnostic_index.v1",
        "league": "SA",
        "attempt": attempt,
        "created_at": now(),
        "sources": records,
    }
    index["index_sha256"] = sha256_json(index)
    write(target / "SA" / f"diagnostic_index_{attempt}.json", index)
    return {
        "league": "SA",
        "attempt": attempt,
        "sources": [
            {"provider": item["provider"], "status": item["status"],
             "http_status": item.get("http_status")}
            for item in records
        ],
    }


def diagnose_sa_recovery(path: str | Path, attempt: str = "recovery") -> dict[str, Any]:
    """Probe bounded alternate routes before accepting any new authority."""
    target = open_run(path)
    check(attempt.replace("-", "").isalnum(), "UNSAFE_ATTEMPT")
    keys = credentials()
    directory = target / "SA" / "raw_sources" / attempt
    rapid_headers = ({
        "x-rapidapi-key": keys["rapidapi"],
        "x-rapidapi-host": "api-football-v1.p.rapidapi.com",
    } if keys["rapidapi"] else {})
    browser_headers = {
        "User-Agent": "Mozilla/5.0 (compatible; FFPS-Phase11-Qualification/1.0)",
        "Accept": "application/json,text/csv;q=0.9,*/*;q=0.8",
    }
    records = [
        fetch(directory, "api-football-rapidapi-league-2026",
              "https://api-football-v1.p.rapidapi.com/v3/leagues",
              provider="api-football:v3:rapidapi",
              params={"id": API_FOOTBALL_IDS["SA"], "season": 2026},
              headers=rapid_headers),
        fetch(directory, "api-football-rapidapi-fixtures-2026",
              "https://api-football-v1.p.rapidapi.com/v3/fixtures",
              provider="api-football:v3:rapidapi",
              params={"league": API_FOOTBALL_IDS["SA"], "season": 2026},
              headers=rapid_headers),
        fetch(directory, "football-data-co-uk-no-www-2026",
              "https://football-data.co.uk/mmz4281/2627/I1.csv",
              provider="football-data.co.uk:csv:no-www", headers=browser_headers),
        fetch(directory, "football-data-co-uk-http-2026",
              "http://www.football-data.co.uk/mmz4281/2627/I1.csv",
              provider="football-data.co.uk:csv:http", headers=browser_headers),
        fetch(directory, "sofascore-tournament-catalog",
              "https://www.sofascore.com/api/v1/config/unique-tournaments/EN/football",
              provider="sofascore:web-api", headers=browser_headers),
        fetch(directory, "sofascore-serie-a-seasons",
              "https://www.sofascore.com/api/v1/unique-tournament/23/seasons",
              provider="sofascore:web-api", headers=browser_headers),
    ]
    index = {
        "schema_version": "ffps.phase11_diagnostic_index.v1",
        "league": "SA",
        "attempt": attempt,
        "created_at": now(),
        "sources": records,
    }
    index["index_sha256"] = sha256_json(index)
    write(target / "SA" / f"diagnostic_index_{attempt}.json", index)
    return {
        "league": "SA",
        "attempt": attempt,
        "sources": [{"provider": item["provider"], "status": item["status"],
                     "http_status": item.get("http_status")} for item in records],
    }


def diagnose_sa_sofascore(path: str | Path, attempt: str = "sofascore") -> dict[str, Any]:
    """Check SofaScore's API host without assuming its season identifier."""
    target = open_run(path)
    check(attempt.replace("-", "").isalnum(), "UNSAFE_ATTEMPT")
    directory = target / "SA" / "raw_sources" / attempt
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; FFPS-Phase11-Qualification/1.0)",
        "Accept": "application/json",
    }
    catalog = fetch(directory, "sofascore-api-tournament-catalog",
                    "https://api.sofascore.com/api/v1/config/unique-tournaments/EN/football",
                    provider="sofascore:web-api", headers=headers)
    seasons = fetch(directory, "sofascore-api-serie-a-seasons",
                    "https://api.sofascore.com/api/v1/unique-tournament/23/seasons",
                    provider="sofascore:web-api", headers=headers)
    records = [catalog, seasons]
    index = {
        "schema_version": "ffps.phase11_diagnostic_index.v1",
        "league": "SA", "attempt": attempt, "created_at": now(), "sources": records,
    }
    index["index_sha256"] = sha256_json(index)
    write(target / "SA" / f"diagnostic_index_{attempt}.json", index)
    return {"league": "SA", "attempt": attempt,
            "sources": [{"provider": item["provider"], "status": item["status"],
                         "http_status": item.get("http_status")} for item in records]}


def diagnose_sa_espn(path: str | Path, attempt: str = "espn") -> dict[str, Any]:
    """Probe ESPN scoreboards and summaries for overlap/current corner coverage."""
    target = open_run(path)
    check(attempt.replace("-", "").isalnum(), "UNSAFE_ATTEMPT")
    directory = target / "SA" / "raw_sources" / attempt
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; FFPS-Phase11-Qualification/1.0)",
        "Accept": "application/json",
    }
    records: list[dict[str, Any]] = []
    for day in ("20260822", "20260905"):
        source = fetch(
            directory, f"espn-scoreboard-{day}",
            "https://site.api.espn.com/apis/site/v2/sports/soccer/ita.1/scoreboard",
            provider="espn:site-api", params={"dates": day}, headers=headers,
        )
        records.append(source)
        if source.get("status") != "RETRIEVED":
            continue
        payload = json.loads((ROOT / source["raw"]["path"]).read_text(encoding="utf-8"))
        events = payload.get("events", [])
        if events:
            event_id = str(events[0].get("id", ""))
            check(event_id.isdigit(), "INVALID_ESPN_EVENT_ID")
            records.append(fetch(
                directory, f"espn-summary-{day}",
                "https://site.api.espn.com/apis/site/v2/sports/soccer/ita.1/summary",
                provider="espn:site-api", params={"event": event_id}, headers=headers,
            ))
    index = {
        "schema_version": "ffps.phase11_diagnostic_index.v1", "league": "SA",
        "attempt": attempt, "created_at": now(), "sources": records,
    }
    index["index_sha256"] = sha256_json(index)
    write(target / "SA" / f"diagnostic_index_{attempt}.json", index)
    return {"league": "SA", "attempt": attempt,
            "sources": [{"provider": item["provider"], "status": item["status"],
                         "http_status": item.get("http_status")} for item in records]}


def diagnose_sa_fotmob(path: str | Path, attempt: str = "fotmob") -> dict[str, Any]:
    """Probe FotMob league/date endpoints before considering it as authority."""
    target = open_run(path)
    check(attempt.replace("-", "").isalnum(), "UNSAFE_ATTEMPT")
    directory = target / "SA" / "raw_sources" / attempt
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; FFPS-Phase11-Qualification/1.0)",
        "Accept": "application/json",
        "Referer": "https://www.fotmob.com/",
    }
    records = [
        fetch(directory, "fotmob-league-2026",
              "https://www.fotmob.com/api/data/leagues", provider="fotmob:web-api",
              params={"id": 55, "season": "2026/2027", "ccode3": "ITA"}, headers=headers),
        fetch(directory, "fotmob-matches-20260822",
              "https://www.fotmob.com/api/data/matches", provider="fotmob:web-api",
              params={"date": "20260822", "timezone": "UTC", "ccode3": "ITA"}, headers=headers),
        fetch(directory, "fotmob-matches-20260905",
              "https://www.fotmob.com/api/data/matches", provider="fotmob:web-api",
              params={"date": "20260905", "timezone": "UTC", "ccode3": "ITA"}, headers=headers),
    ]
    for source, day in ((records[1], "20260822"), (records[2], "20260905")):
        if source.get("status") != "RETRIEVED":
            continue
        payload = json.loads((ROOT / source["raw"]["path"]).read_text(encoding="utf-8"))
        ids = [str(match.get("id", "")) for league in payload.get("leagues", [])
               if str(league.get("id")) == "55" for match in league.get("matches", [])]
        if ids:
            check(ids[0].isdigit(), "INVALID_FOTMOB_MATCH_ID")
            records.append(fetch(directory, f"fotmob-match-details-{day}",
                                 "https://www.fotmob.com/api/data/matchDetails",
                                 provider="fotmob:web-api", params={"matchId": ids[0]}, headers=headers))
    index = {"schema_version": "ffps.phase11_diagnostic_index.v1", "league": "SA",
             "attempt": attempt, "created_at": now(), "sources": records}
    index["index_sha256"] = sha256_json(index)
    write(target / "SA" / f"diagnostic_index_{attempt}.json", index)
    return {"league": "SA", "attempt": attempt,
            "sources": [{"provider": item["provider"], "status": item["status"],
                         "http_status": item.get("http_status")} for item in records]}
