"""Immutable Phase 11 status reporting and offline verification."""
from __future__ import annotations

import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from src.evaluation import phase10_current_data as phase10
from src.evaluation import phase11_current_source as data
from src.evaluation import phase11_shadow as shadow
from src.ml.league_bundle_activation import LeagueBundleController


SOURCES = (
    "src/evaluation/phase11_current_source.py",
    "src/evaluation/phase11_shadow.py",
    "src/evaluation/phase11_evidence.py",
    "scripts/run_phase11_current_source.py",
    "tests/unit/test_phase11_current_source.py",
    "docs/phase11-current-source-operations.md",
)


def write_once_or_verify(path: Path, value: Any) -> None:
    """Resume a partial seal only when the existing immutable value is exact."""
    if path.exists():
        data.check(data.read(path) == value, "CONFLICTING_PARTIAL_FINALIZATION")
    else:
        data.write(path, value)


def copy_once_or_verify(source: Path, destination: Path) -> None:
    if destination.exists():
        data.check(data.sha256_file(source) == data.sha256_file(destination),
                   "CONFLICTING_PARTIAL_FINALIZATION")
    else:
        shutil.copyfile(source, destination)


def test_summary(path: str | Path) -> dict[str, Any]:
    root = ET.parse(path)
    cases = list(root.iter("testcase"))
    failures = [
        case.attrib["classname"] + "::" + case.attrib["name"]
        for case in cases if case.find("failure") is not None or case.find("error") is not None
    ]
    skipped = sum(case.find("skipped") is not None for case in cases)
    return {"cases": len(cases), "passed": len(cases) - len(failures) - skipped,
            "skipped": skipped, "failures": failures}


def latest_qualification(path: Path, league: str) -> Path:
    items = sorted((path / league / "assessments").glob("assessment-*/qualification.json"))
    data.check(bool(items), "LEAGUE_NOT_ASSESSED")
    return items[-1]


def verify_phase10(path: str | Path) -> dict[str, Any]:
    """Verify the sealed Phase 10 index without misclassifying Phase 11 as drift.

    Phase 10's historical snapshot scans every baseline except its own, so it
    necessarily treats a later Phase 11 directory as a mutation.  Its immutable
    artifact index plus Phase 11's pre-run snapshot provide the intended check.
    """
    target = data.open_run(path) / "prior_phase10_verification.json"
    data.check(not target.exists(), "PRIOR_PHASE10_VERIFICATION_ALREADY_RECORDED")
    run = data.read(Path(path) / "run_specification.json")
    data.check(run["phase10_run"] == data.identity(data.PHASE10 / "artifact_index.json"),
               "PHASE10_INDEX_DRIFT")
    index = data.read(data.PHASE10 / "artifact_index.json")
    for ref in index["artifacts"] + index["implementation_sources"]:
        shadow.reference(ref)
    archived = data.read(data.PHASE10 / "offline_verification.json")
    data.check(archived["verified"] and not archived["new_unexplained_failures"],
               "PRIOR_PHASE10_VERIFICATION_FAILED")
    result = {"verified": True, "sealed_artifact_index_verified": True,
              "artifact_count": len(index["artifacts"]),
              "implementation_source_count": len(index["implementation_sources"]),
              "archived_offline_verification": data.identity(
                  data.PHASE10 / "offline_verification.json"),
              "later_baseline_excluded_from_historical_snapshot": True,
              "new_unexplained_failures": archived["new_unexplained_failures"]}
    data.write(target, result)
    return result


def replay_source_evidence(q: dict[str, Any]) -> dict[str, Any]:
    """Reparse every raw provider payload bound by a qualification."""
    league = q["league"]
    assessment = q["assessment"]
    combined = data.read(shadow.reference(q["source_index"]))
    shadow.sealed(combined, "index_sha256")
    primary = data.read(shadow.reference(combined["primary_index"]))
    fotmob = data.read(shadow.reference(combined["fotmob_index"]))
    shadow.sealed(primary, "index_sha256")
    shadow.sealed(fotmob, "index_sha256")
    data.check(primary["league"] == fotmob["league"] == combined["league"] == league,
               "CROSS_LEAGUE_SOURCE")
    source_map = data._source_map(primary)
    season = primary["requested_season"]
    names = data._phase11_names(league)
    mappings: dict[str, Any] = {}
    mapping_evidence: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    statistics: list[dict[str, Any]] = []
    seasons: dict[str, Any] = {}
    for year in (season - 1, season):
        mapped, results = data._mapped_teams(
            data._json_payload(source_map[(year, "teams")]), league, year, names)
        data.check(not [item for item in results if "code" in item], "TEAM_MAPPING_REPLAY_FAILED")
        mappings[str(year)] = mapped
        mapping_evidence.extend(item for item in results if "reason" in item)
        rows, evidence = data._adapt_primary(
            source_map[(year, "matches")], league, year, mapped)
        events.extend(rows)
        seasons[str(year)] = evidence
        statistics.extend(phase10.statistics_events(
            data._raw_bytes(source_map[(year, "statistics")]), league, year,
            primary["division"], names))
    fotmob_rows, fotmob_evidence = data._adapt_fotmob(fotmob, league, season, names)
    data.check(events == assessment["events"], "PRIMARY_NORMALIZATION_REPLAY_MISMATCH")
    data.check(statistics == assessment["statistics"], "STATISTICS_NORMALIZATION_REPLAY_MISMATCH")
    data.check(fotmob_rows == assessment["fotmob_statistics"],
               "ALTERNATIVE_NORMALIZATION_REPLAY_MISMATCH")
    data.check(mappings == assessment["team_mappings"] and
               mapping_evidence == assessment["mapping_evidence"],
               "TEAM_MAPPING_REPLAY_MISMATCH")
    stored_seasons = assessment["season_evidence"]
    data.check(set(seasons) == set(stored_seasons), "SOURCE_METADATA_REPLAY_MISMATCH")
    for year in seasons:
        data.check({k: v for k, v in seasons[year].items() if k != "interpretation"} ==
                   {k: v for k, v in stored_seasons[year].items() if k != "interpretation"},
                   "SOURCE_METADATA_REPLAY_MISMATCH")
    stored_fotmob = assessment["fotmob_evidence"]
    if "source_name_variants" not in stored_fotmob and not fotmob_evidence["source_name_variants"]:
        fotmob_evidence = {k: v for k, v in fotmob_evidence.items()
                           if k != "source_name_variants"}
    data.check(fotmob_evidence == stored_fotmob, "SOURCE_METADATA_REPLAY_MISMATCH")
    data.check(assessment["qualified"] and not assessment["blockers"],
               "CURRENT_DATA_UNQUALIFIED")
    data.check(assessment["historical_overlap"] and all(
        all(row["fields"].values()) for row in assessment["historical_overlap"]),
        "HISTORICAL_STATISTIC_OVERLAP_UNPROVEN")
    replayed_overlap, provider_conflicts = data.provider_overlap(statistics, fotmob_rows, season)
    data.check(not provider_conflicts and replayed_overlap == assessment["cross_provider_overlap"],
               "ALTERNATIVE_PROVIDER_DISAGREEMENT")
    data.check(assessment["cross_provider_overlap"] and all(
        all(row["fields"].values()) for row in assessment["cross_provider_overlap"]),
        "ALTERNATIVE_PROVIDER_DISAGREEMENT")
    event_ids = {row["fixture_id"] for row in events if row["status"] == "FINISHED"}
    for row in assessment["composed_observations"]:
        data.check(row["fixture_id"] in event_ids, "COMPOSED_EVENT_NOT_IN_PRIMARY")
        for provenance in row["field_provenance"].values():
            shadow.reference(provenance["source"])
    for row in assessment["upcoming"]:
        data.check(row["season"] == season and row["status"] in ("TIMED", "SCHEDULED") and
                   phase10.instant(row["kickoff_at"]) > phase10.instant(q["as_of"]),
                   "INVALID_UPCOMING_FIXTURE")
    return {"league": league, "raw_sources_replayed": len(combined["sources"]),
            "primary_events": len(events), "statistics_rows": len(statistics),
            "alternative_statistics_rows": len(fotmob_rows),
            "historical_overlap_rows": len(assessment["historical_overlap"]),
            "current_provider_overlap_rows": len(assessment["cross_provider_overlap"]),
            "composed_observations": len(assessment["composed_observations"]),
            "upcoming_fixtures": len(assessment["upcoming"])}


def matrix_row(path: Path, league: str) -> dict[str, Any]:
    path = Path(path).resolve()
    qualification_path = latest_qualification(path, league)
    q, _, _ = shadow.load_qualification(qualification_path)
    source_replay = replay_source_evidence(q)
    protocol_path = path / league / "prospective_protocol.json"
    pairs = []
    if protocol_path.exists():
        _, pairs, _ = shadow.verify_cohort(path, league)
    assessment = q["assessment"]
    current_mappings = assessment["team_mappings"][str(assessment["season"])]
    source = "football-data.org:v4 + football-data.co.uk:csv:no-www + fotmob:web-api"
    return {
        "league": league, "source": source,
        "source_identity": q["source_index"],
        "season_valid": all(not item["kickoff_outside_authoritative_bounds"]
                            for item in assessment["season_evidence"].values()),
        "teams_mapped": all(item.get("canonical") for item in current_mappings.values()),
        "mapped_current_teams": len(current_mappings),
        "historical_features": q["historical_feature_reproduction"]["code"],
        "historical_statistics": "VERIFIED" if assessment["historical_overlap"] else "BLOCKED",
        "historical_statistics_rows": len(assessment["historical_overlap"]),
        "current_statistics": "VERIFIED" if assessment["cross_provider_overlap"] else "BLOCKED",
        "current_statistics_overlap_rows": len(assessment["cross_provider_overlap"]),
        "current_causal_features": "VERIFIED" if q["feature_readiness"] else "BLOCKED",
        "protocol": "FROZEN" if protocol_path.exists() else "NOT_FROZEN",
        "collection": "PROSPECTIVE_COLLECTION_ACTIVE" if pairs else "DATA_BLOCKED",
        "admitted_pairs": len(pairs),
        "first_pair": ({"fixture_id": pairs[0]["fixture_id"],
                        "created_at": pairs[0]["created_at"],
                        "kickoff_at": pairs[0]["kickoff_at"],
                        "pair_sha256": pairs[0]["pair_sha256"]} if pairs else None),
        "promotion": "NOT_AUTHORIZED_PHASE11",
        "active_bundle_id": q["active_bundle_id"], "active_sha256": q["active_sha256"],
        "candidate_bundle_id": q["candidate_bundle_id"],
        "candidate_sha256": q["candidate_sha256"],
        "qualification": data.identity(qualification_path),
        "blocker_codes": [], "source_replay": source_replay,
    }


def provider_failure_classification(path: Path) -> dict[str, Any]:
    sa = path / "SA"
    files = {item.stem.removeprefix("diagnostic_index_"): data.identity(item)
             for item in sorted(sa.glob("diagnostic_index_*.json"))}
    return {
        "schema_version": "ffps.phase11_provider_failure_classification.v1",
        "diagnostic_indexes": files,
        "classifications": [
            {"source": "initial sandbox-only diagnostics", "classification": "EXECUTION_ENVIRONMENT_NETWORK_ISOLATION",
             "disposition": "not treated as provider evidence"},
            {"source": "www.football-data.co.uk current and prior-season CSV routes",
             "classification": "ROUTE_OR_HOST_SPECIFIC_HTTP_503",
             "disposition": "retained as failed responses; never converted to empty data"},
            {"source": "football-data.co.uk no-www HTTPS CSV route",
             "classification": "CORRECTED_ENDPOINT_RETRIEVED",
             "disposition": "qualified as primary corners source after historical and current overlap"},
            {"source": "API-Football direct v3", "classification": "PLAN_CAPABILITY_GAP",
             "disposition": "HTTP 200 carried an application error denying season 2026; rejected"},
            {"source": "API-Football RapidAPI", "classification": "SUBSCRIPTION_OR_AUTHORIZATION_HTTP_403",
             "disposition": "rejected"},
            {"source": "SofaScore and ESPN probes", "classification": "HTTP_403_OR_UNAVAILABLE",
             "disposition": "rejected"},
            {"source": "FotMob web API", "classification": "CURRENT_STATISTICS_CAPABILITY_VERIFIED",
             "disposition": "qualified only as gap-fill after exact current overlap against football-data.co.uk"},
        ],
        "season_boundary_classification": {
            "classification": "PROVIDER_SEASON_METADATA_INCONSISTENT_WITH_OWN_REGULAR_SEASON_FIXTURE_POPULATION",
            "accepted_authority": "resultSet.first/resultSet.last, each required to equal the exact fixture min/max",
            "gate_preserved": "every normalized kickoff is still checked against the accepted source-supplied bounds",
        },
        "credentials_recorded": False,
    }


def verify(path: str | Path, *, indexed: bool = True) -> dict[str, Any]:
    path = data.run_path(path)
    from src.config.leagues import ACTIVE_LEAGUES
    data.check(set(ACTIVE_LEAGUES) - {"PL"} == set(data.LEAGUES),
               "PRODUCTION_LEAGUE_INVENTORY_CHANGED")
    before = data.read(path / "protected_before.json")
    data.check(before == data.protected_snapshot(), "PROTECTED_FILE_MUTATION")
    run = data.read(path / "run_specification.json")
    shadow.sealed(run, "specification_sha256")
    phase7p = data.ROOT / "baselines/phase7p"
    data.check(run["pl_phase7p_before"] == data._tree_snapshot(phase7p), "PL_PHASE7P_MUTATION")
    prior = data.read(path / "prior_phase10_verification.json")
    data.check(prior["verified"] and not prior["new_unexplained_failures"],
               "PRIOR_PHASE10_VERIFICATION_FAILED")
    for league in data.LEAGUES:
        controller = LeagueBundleController(data.PHASE9 / league / "deployment_specification.json",
                                            league=league)
        expected = run["phase9_candidates"][league]
        candidate = controller.bundle(controller.candidate_id)
        frozen_candidate_path = shadow.reference(expected["bundle"])
        frozen_candidate = data.read(frozen_candidate_path)
        data.check(candidate["bundle_id"] == frozen_candidate["bundle_id"] ==
                   controller.candidate_id == expected["bundle_id"],
                   "PHASE9_CANDIDATE_MUTATION")
    if indexed:
        index = data.read(path / "artifact_index.json")
        for ref in index["artifacts"] + index["implementation_sources"]:
            shadow.reference(ref)
        expected = {ref["path"] for ref in index["artifacts"]}
        actual = {item.relative_to(data.ROOT).as_posix() for item in path.rglob("*")
                  if item.is_file() and item.name not in ("artifact_index.json", "offline_verification.json")}
        data.check(expected == actual, "UNINDEXED_OR_MISSING_EVIDENCE")
    matrix = data.read(path / "multi_league_phase11_status_matrix.json")
    data.check([row["league"] for row in matrix["leagues"]] == list(data.LEAGUES),
               "OMITTED_OR_DUPLICATED_LEAGUE")
    rebuilt = [matrix_row(path, league) for league in data.LEAGUES]
    data.check(rebuilt == matrix["leagues"], "MATRIX_STATE_MISMATCH")
    data.check(all(row["collection"] in ("PROSPECTIVE_COLLECTION_ACTIVE", "DATA_BLOCKED")
                   for row in rebuilt), "INVALID_PHASE11_STATE")
    tests = data.read(path / "tests/classification.json")
    for label in ("focused", "protected", "full"):
        data.check(test_summary(path / f"tests/{label}.xml") == tests[label],
                   "TEST_REPORT_MISMATCH")
    inherited = set(test_summary(path / "tests/phase10_full_reference.xml")["failures"])
    new = set(tests["full"]["failures"]) - inherited
    data.check(not tests["focused"]["failures"] and not tests["protected"]["failures"] and not new,
               "NEW_REGRESSION_FAILURE")
    return {
        "schema_version": "ffps.phase11_offline_verification.v1", "verified": True,
        "protected_file_count": len(before), "protected_files_unchanged": True,
        "pl_phase7p_unchanged": True, "phase9_candidates_unchanged": True,
        "production_models_and_pointers_unchanged": True,
        "model_training_performed": False, "calibration_performed": False,
        "production_activation_performed": False,
        "leagues": [{"league": row["league"], "state": row["collection"],
                     "source_replay": row["source_replay"],
                     "pair_verified": row["admitted_pairs"] > 0} for row in rebuilt],
        "tests_verified": True, "new_unexplained_failures": sorted(new),
    }


def finalize(path: str | Path) -> dict[str, Any]:
    path = data.run_path(path)
    data.check(not (path / "artifact_index.json").exists(), "RUN_ALREADY_FINALIZED")
    rows = [matrix_row(path, league) for league in data.LEAGUES]
    matrix = {
        "schema_version": "ffps.phase11_status_matrix.v1", "leagues": rows,
        "pl_policy": "READ_ONLY_EXISTING_PHASE7P_UNCHANGED",
        "production_activation_performed": False,
    }
    write_once_or_verify(path / "multi_league_phase11_status_matrix.json", matrix)
    write_once_or_verify(path / "provider_failure_classification.json",
                         provider_failure_classification(path))
    raw = [data.identity(item) for item in sorted(path.glob("*/raw_sources/**/*")) if item.is_file()]
    write_once_or_verify(path / "raw_source_identity_index.json", {
        "schema_version": "ffps.phase11_raw_source_identity_index.v1", "sources": raw,
        "credentials_recorded": False,
        "publication_policy": "source update fields are retained where present; FFPS retrieval is the availability upper bound",
    })
    test_dir = path / "tests"
    test_dir.mkdir(exist_ok=True)
    for label in ("focused", "protected", "full"):
        copy_once_or_verify(data.ROOT / f"tmp/phase11-{label}.xml", test_dir / f"{label}.xml")
    copy_once_or_verify(data.PHASE10 / "tests/full.xml", test_dir / "phase10_full_reference.xml")
    tests = {label: test_summary(test_dir / f"{label}.xml")
             for label in ("focused", "protected", "full")}
    inherited = set(test_summary(test_dir / "phase10_full_reference.xml")["failures"])
    tests["inherited_failures"] = sorted(set(tests["full"]["failures"]) & inherited)
    tests["new_unexplained_failures"] = sorted(set(tests["full"]["failures"]) - inherited)
    write_once_or_verify(test_dir / "classification.json", tests)
    after = data.protected_snapshot()
    before = data.read(path / "protected_before.json")
    data.check(after == before, "PROTECTED_FILE_MUTATION")
    write_once_or_verify(path / "protected_after.json", after)
    write_once_or_verify(path / "preservation_verification.json", {
        "protected_count": len(before), "unchanged": True, "pl_phase7p_preserved": True,
        "phase9_candidates_preserved": True, "production_models_preserved": True,
        "production_pointers_preserved": True, "retraining_performed": False,
        "calibration_performed": False, "promotion_performed": False,
    })
    lines = [
        "# Phase 11 current-source recovery and qualification", "",
        "Shared implementation passed provider failure handling, strict source parsing, exact mapping, BL1 compatibility, pair isolation, and offline replay. "
        f"League-specific current-data evidence passed for {sum(row['collection'] == 'PROSPECTIVE_COLLECTION_ACTIVE' for row in rows)}/{len(rows)} leagues.", "",
        "| League | Source | Season valid | Teams mapped | Historical features | Historical statistics | Current statistics | Current causal features | Protocol | Collection | Promotion |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['league']} | {row['source']} | {'Yes' if row['season_valid'] else 'No'} | "
            f"{row['mapped_current_teams']}/{row['mapped_current_teams'] if row['teams_mapped'] else '?'} | "
            f"{row['historical_features']} | {row['historical_statistics']} ({row['historical_statistics_rows']}) | "
            f"{row['current_statistics']} ({row['current_statistics_overlap_rows']}) | {row['current_causal_features']} | "
            f"{row['protocol']} | {row['collection']} ({row['admitted_pairs']}) | Not authorized |")
    lines += ["", "## Provider diagnosis", "",
        "The `www.football-data.co.uk` data and CSV routes repeatedly returned HTTP 503. The equivalent HTTPS CSV path on the no-www host returned valid current and prior-season bytes for all four leagues. API-Football transport succeeded, but its payload denied 2026 data under the configured plan; its RapidAPI route returned 403. SofaScore and ESPN probes were also rejected. All failures remain archived.", "",
        "Football-data.org supplied league fixture identity, kickoff, status, and goals. Its `season.startDate` and, for one FL1 season, `season.endDate` conflict with regular-season fixtures returned in the same payload. Phase 11 did not relax the kickoff gate: it accepts `resultSet.first` and `resultSet.last` only after proving they equal the exact minimum and maximum dates of the returned regular-season fixture population, then checks every kickoff against those bounds.", "",
        "FotMob supplied current corner statistics missing from the lagging CSV snapshots. It was accepted only after exact score, orientation, team, and corner equality on overlapping current fixtures. The source identity remains explicit as `fotmob:web-api`, retrieval timestamps bound availability, and each raw response is hashed.", "",
        "## League-specific evidence", ""]
    for row in rows:
        first = row["first_pair"]
        lines += [f"### {row['league']}", "",
            f"Historical statistic overlaps: {row['historical_statistics_rows']}; current cross-provider overlaps: {row['current_statistics_overlap_rows']}. "
            f"Exact current mappings: {row['mapped_current_teams']}. Qualification: `{row['qualification']['path']}`.", "",
            f"The first pair is `{first['fixture_id']}`, captured at `{first['created_at']}` for kickoff `{first['kickoff_at']}`. "
            "The active and candidate records share the same feature snapshot and common RNG and both pass Phase 4 replay.", ""]
    lines += ["## Mapping and BL1 compatibility", "",
        "New mappings bind stable football-data.org team IDs and exact provider spellings to each league's model-facing identity. Current football-data.co.uk and FotMob spellings corroborate promoted teams. Qualification uses no fuzzy matching and rejects changed IDs, changed names, ambiguity, and collisions.", "",
        "BL1's legitimate frozen identity `M'GLADBACH` conflicts with the production sanitizer's identifier whitelist. Phase 11 leaves that global sanitizer unchanged. A BL1-only adapter replaces this exact canonical identifier with a collision-checked alphanumeric sentinel immediately before sanitization and grouping, then restores it afterward. Historical model-eligible feature replay is exact. Hostile identifiers remain rejected.", "",
        "## Preservation and limits", "",
        f"All {len(before)} protected pre-Phase 11 files are unchanged. This includes sealed Phase 10 evidence, every Phase 9 candidate and active bundle, production model bytes and pointers, and PL Phase 7P. No model was trained, calibrated, activated, promoted, or rolled back.", "",
        "Each protocol and cohort is league-bound. Phase 11 contains no promotion command. Future settlement and any eventual decision remain subject to the existing league-specific Phase 8 controls.", "",
        "## Validation", ""]
    for label in ("focused", "protected", "full"):
        summary = tests[label]
        lines.append(f"- {label}: {summary['passed']} passed, {len(summary['failures'])} failed, {summary['skipped']} skipped.")
    lines += ["", f"New unexplained failures: {len(tests['new_unexplained_failures'])}.", "",
              "Inherited full-suite failures:", ""]
    lines.extend(f"- `{failure}`" for failure in tests["inherited_failures"])
    lines += ["", "Offline verification reparses all archived source bytes, recomputes mappings and feature vectors, replays all prediction pairs, checks league ownership and artifact hashes, validates test reports, and compares every protected file against the pre-run snapshot.", ""]
    report_path = path / "PHASE11_CURRENT_SOURCE_IMPLEMENTATION_REPORT.md"
    report_text = "\n".join(lines) + "\n"
    if report_path.exists():
        data.check(report_path.read_text(encoding="utf-8") == report_text,
                   "CONFLICTING_PARTIAL_FINALIZATION")
    else:
        report_path.write_text(report_text, encoding="utf-8")
    preliminary = verify(path, indexed=False)
    artifacts = [data.identity(item) for item in sorted(path.rglob("*"))
                 if item.is_file() and item.name not in ("artifact_index.json", "offline_verification.json")]
    data.write(path / "artifact_index.json", {
        "schema_version": "ffps.phase11_artifact_index.v1", "artifacts": artifacts,
        "implementation_sources": [data.identity(data.ROOT / source) for source in SOURCES],
    })
    data.write(path / "offline_verification.json", preliminary)
    return preliminary
