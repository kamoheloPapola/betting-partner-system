"""Root-scoped, read-only source composition using unchanged Phase 11 science.

The caller supplies exact archived indexes, frozen history and vocabulary.
No source acquisition, source acceptance or collection permission is issued.
"""
from __future__ import annotations

from pathlib import Path
import re

from src.evaluation.future_lineage_context import (
    fingerprint, instant, read, reference, require, seal, sealed,
)


def observation(metadata, repository, *, cutoff, kickoff=None):
    """Conservative retained observation bounds, never first-publication time."""
    require(metadata.get("status") == "RETRIEVED" and metadata.get("http_status") == 200,
            "FUTURE_SOURCE_UNAVAILABLE")
    raw = metadata.get("raw")
    reference(repository, raw)
    bounds = metadata.get("archived_observation")
    require(isinstance(bounds, dict), "FUTURE_OBSERVATION_MISSING")
    require(bounds.get("basis") == "archived_response_observation" and
            bounds.get("raw") == raw and bounds.get("provider") == metadata.get("provider")
            and bounds.get("request_sha256") == fingerprint(metadata.get("request")),
            "FUTURE_OBSERVATION_INVALID")
    sealed(bounds)
    started = instant(metadata["started_at"])
    retrieved = instant(metadata["retrieved_at"])
    lower, upper = instant(bounds["not_before"]), instant(bounds["observed_by"])
    require(started <= lower <= upper <= retrieved <= instant(cutoff),
            "FUTURE_OBSERVATION_LATE_OR_CONTRADICTORY")
    if kickoff is not None:
        require(upper < instant(kickoff) and retrieved < instant(kickoff),
                "FUTURE_OBSERVATION_AFTER_KICKOFF")
    # lastUpdated is checked by the unchanged normalizer for internal time
    # consistency. It supplies no first-publication evidence.
    return bounds


def validate_indexes(store, primary, alternative, *, as_of):
    from src.evaluation import phase11_current_source as legacy
    require(primary.get("league") == alternative.get("league") == "BL1" and
            primary.get("requested_season") == alternative.get("season") == 2026 and
            primary.get("division") == legacy.SOURCE_SPECS["BL1"]["division"] and
            alternative.get("selected_season") == "2026/2027", "FUTURE_SOURCE_INDEX_SCOPE")
    roles = {}
    for metadata in primary.get("sources", []):
        name = Path(metadata.get("raw", {}).get("path", "")).stem
        match = re.fullmatch(r"primary-(2025|2026)-(matches|teams)|statistics-(2025|2026)", name)
        require(match is not None, "FUTURE_UNDECLARED_SOURCE_ROLE")
        role = (int(match.group(1) or match.group(3)), match.group(2) or "statistics")
        require(role not in roles, "FUTURE_DUPLICATE_SOURCE_ROLE")
        roles[role] = metadata
        year, kind = role
        if kind in ("matches", "teams"):
            require(metadata.get("provider") == "football-data.org:v4" and
                    metadata.get("request") == {
                        "url": f"https://api.football-data.org/v4/competitions/BL1/{kind}",
                        "params": {"season": year}} and
                    "x-auth-token" in metadata.get("credential_headers_present", []),
                    "FUTURE_PRIMARY_SOURCE_SCOPE")
        else:
            code = "2526" if year == 2025 else "2627"
            require(metadata.get("provider") == "football-data.co.uk:csv:no-www" and
                    metadata.get("request") == {
                        "url": f"https://football-data.co.uk/mmz4281/{code}/D1.csv",
                        "params": {}}, "FUTURE_STATISTICS_SOURCE_SCOPE")
    require(set(roles) == {(year, kind) for year in (2025, 2026)
                          for kind in ("matches", "teams", "statistics")},
            "SOURCE_INDEX_INCOMPLETE")
    league_source = alternative.get("league_source", {})
    require(league_source.get("provider") == "fotmob:web-api" and
            league_source.get("request") == {
                "url": "https://www.fotmob.com/api/data/leagues",
                "params": {"ccode3": "GER", "id": 54, "season": "2026/2027"}},
            "FUTURE_ALTERNATIVE_SOURCE_SCOPE")
    combined = list(primary["sources"]) + [league_source] + list(alternative.get("detail_sources", []))
    paths = []
    for metadata in combined:
        raw_path = metadata.get("raw", {}).get("path", "")
        require(raw_path.startswith(store.relative_root + "/BL1/qualification/"),
                "FUTURE_SOURCE_OUTSIDE_LINEAGE")
        relative = raw_path[len(store.relative_root) + 1:]
        require(store.path(relative).suffix == ".raw", "FUTURE_SOURCE_SLOT")
        observation(metadata, store.repository, cutoff=as_of)
        paths.append(raw_path)
    require(len(paths) == len(set(paths)), "FUTURE_DUPLICATE_SOURCE_IDENTITY")
    for metadata in alternative.get("detail_sources", []):
        params = metadata.get("request", {}).get("params", {})
        require(metadata.get("provider") == "fotmob:web-api" and
                metadata.get("request", {}).get("url") ==
                "https://www.fotmob.com/api/data/matchDetails" and
                set(params) == {"matchId"} and str(params["matchId"]).isdigit(),
                "FUTURE_ALTERNATIVE_DETAIL_SCOPE")
    return roles, combined


def compose_assessment(store, primary, alternative, *, frozen, names, as_of):
    """Translated composition; normalization/statistics/mappings remain unchanged.

    No feature or model work occurs here. Complete composition, not merely
    source membership, is rederived for subsequent qualification verification.
    """
    from src.evaluation import phase11_current_source as source
    from src.evaluation import phase10_current_data as phase10

    require(store.repository.resolve() == source.ROOT.resolve(),
            "FUTURE_SHARED_ADAPTER_REPOSITORY_MISMATCH")
    sources, combined = validate_indexes(store, primary, alternative, as_of=as_of)
    events, statistics, mappings = [], [], {}
    mapping_failures, mapping_evidence, season_evidence = [], [], {}
    for year in (2025, 2026):
        mapped, results = source._mapped_teams(
            source._json_payload(sources[(year, "teams")]), "BL1", year, names)
        mappings[str(year)] = mapped
        mapping_failures.extend(row for row in results if "code" in row)
        mapping_evidence.extend(row for row in results if "reason" in row)
        normalized, evidence = source._adapt_primary(sources[(year, "matches")], "BL1", year, mapped)
        events.extend(normalized)
        season_evidence[str(year)] = evidence
        statistics.extend(phase10.statistics_events(
            source._raw_bytes(sources[(year, "statistics")]), "BL1", year, "D1", names))
    require(len({e["fixture_id"] for e in events}) == len(events), "DUPLICATE_FIXTURE_ID")
    fotmob, fotmob_evidence = source._adapt_fotmob(alternative, "BL1", 2026, names)
    key = lambda row: (row["fixture_date"], row["home_team"], row["away_team"])
    csv = {(row["season"], *key(row)): row for row in statistics}
    other = {key(row): row for row in fotmob}
    require(len(csv) == len(statistics) and len(other) == len(fotmob), "FUTURE_AMBIGUOUS_JOIN")
    overlap, provider_conflicts = source.provider_overlap(statistics, fotmob, 2026)
    require(overlap, "ALTERNATIVE_PROVIDER_OVERLAP_UNPROVEN")
    frozen = frozen.loc[frozen.league.eq("BL1")]
    frozen_index = {(str(row.match_date)[:10], row.home_team, row.away_team): row
                    for row in frozen.itertuples()}
    require(len(frozen_index) == len(frozen), "FUTURE_AMBIGUOUS_FROZEN_JOIN")
    cutoff = str(frozen.match_date.max())[:10]
    frontier = [e for e in events if e["fixture_date"] == cutoff and
                key(e) in frozen_index and e["status"] == "FINISHED"]
    require(frontier, "FROZEN_CAUSAL_FRONTIER_UNPROVEN")
    frontier_time = max(instant(e["kickoff_at"]) for e in frontier)
    historical, composed, missing, conflicts, unresolved, extensions = [], [], [], [], [], []
    for event in events:
        if event["status"] != "FINISHED":
            continue
        join = key(event)
        statistic = csv.get((event["season"], *join))
        if event["season"] == 2026:
            fotmob_row = other.get(join)
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
        after = event["fixture_date"] > cutoff or (
            event["fixture_date"] == cutoff and old is None and
            instant(event["kickoff_at"]) > frontier_time)
        if not after:
            if old is None:
                unresolved.append(event["fixture_id"])
            else:
                fields = {field: (event if field.endswith("goals") else statistic)[field] ==
                          getattr(old, field) for field in
                          ("home_goals", "away_goals", "home_corners", "away_corners")}
                historical.append({"fixture_id": event["fixture_id"], "key": list(join), "fields": fields})
                if not all(fields.values()):
                    conflicts.append(event["fixture_id"])
        else:
            if event["fixture_date"] == cutoff:
                extensions.append({"fixture_id": event["fixture_id"],
                    "kickoff_at": event["kickoff_at"],
                    "frozen_same_day_latest_kickoff": frontier_time.isoformat(),
                    "basis": "strictly later source fixture on the frozen maximum date; absent from frozen rows; two current authorities agree"})
            stat_ref = (sources[(event["season"], "statistics")]["raw"]
                        if (event["season"], *join) in csv else statistic["raw"])
            composed.append({**event, "home_corners": statistic["home_corners"],
                "away_corners": statistic["away_corners"], "field_provenance": {
                    **{f: {"source": sources[(event["season"], "matches")]["raw"],
                            "provider_fixture_id": event["provider_fixture_id"]}
                       for f in ("home_goals", "away_goals")},
                    **{f: {"source": stat_ref,
                            "provider_fixture_id": statistic.get("provider_fixture_id"),
                            "row_ordinal": statistic.get("row_ordinal")}
                       for f in ("home_corners", "away_corners")}}})
    blockers = list(mapping_failures)
    for values, code in (
        (provider_conflicts, "ALTERNATIVE_PROVIDER_DISAGREEMENT"),
        (missing, "MISSING_REQUIRED_CURRENT_STATISTICS"),
        (conflicts, "SOURCE_SEMANTICS_INCOMPATIBLE"),
        (unresolved, "HISTORICAL_PRIMARY_JOIN_UNPROVEN"),
    ):
        if values:
            blockers.append({"code": code, "keys" if values is provider_conflicts else "fixtures":
                             values if values is provider_conflicts else sorted(set(values))})
    if not historical:
        blockers.append({"code": "HISTORICAL_STATISTIC_OVERLAP_UNPROVEN"})
    upcoming = sorted([e for e in events if e["season"] == 2026 and
                       e["status"] in ("TIMED", "SCHEDULED") and
                       instant(e["kickoff_at"]) > instant(as_of)],
                      key=lambda e: (e["kickoff_at"], e["fixture_id"]))
    if not upcoming:
        blockers.append({"code": "UPCOMING_CAUSAL_FIXTURES_UNPROVEN"})
    if not any(e["season"] == 2026 and e["status"] == "FINISHED" for e in events):
        blockers.append({"code": "CURRENT_COMPLETED_COVERAGE_UNPROVEN"})
    assessment = {"league": "BL1", "season": 2026, "as_of": as_of, "blockers": blockers,
        "team_mappings": mappings, "mapping_evidence": mapping_evidence,
        "events": events, "statistics": statistics, "fotmob_statistics": fotmob,
        "historical_overlap": historical, "cross_provider_overlap": overlap,
        "frozen_causal_frontier": {"date": cutoff, "latest_matched_kickoff": frontier_time.isoformat(),
                                 "same_day_extensions": extensions},
        "season_evidence": season_evidence, "fotmob_evidence": fotmob_evidence,
        "composed_observations": sorted(composed, key=lambda e: e["kickoff_at"]),
        "upcoming": upcoming, "qualified": not blockers}
    return assessment, {"schema_version": "ffps.future_lineage.source_index.v1",
        "league": "BL1", "requested_season": 2026, "division": "D1", "sources": combined}


def verify_composition(qualification, store, primary, alternative, *, frozen, names):
    sealed(qualification, "qualification_sha256")
    assessment, index = compose_assessment(
        store, primary, alternative, frozen=frozen, names=names, as_of=qualification["as_of"])
    require(qualification.get("assessment") == assessment,
            "FUTURE_COMPLETE_COMPOSITION_REPLAY_MISMATCH")
    require(assessment["qualified"], "CURRENT_DATA_UNQUALIFIED")
    return assessment, index


def candidate_availability(index, repository, *, at, kickoff):
    for metadata in index["sources"]:
        observation(metadata, repository, cutoff=at, kickoff=kickoff)


def _feature_continuity(assessment, ordered, features):
    """Shared reproduction/first-target calculation; no publication or grant.

    Keeping this calculation separate permits an explicitly authorized offline
    comparison using pinned archived features without labeling those inputs as
    synthetic or enabling qualification of live sources. The caller supplies
    the feature implementation and must establish its identity and authority.
    """
    from src.evaluation.phase10_current_data import validate_features
    require(isinstance(ordered, list) and ordered and len(set(ordered)) == len(ordered),
            "FEATURE_SCHEMA_MISMATCH")
    reproduction = features.reproduce()
    proofs, failures = [], []
    if not reproduction.get("reproduced"):
        failures.append({"code": reproduction.get("code", "FEATURE_REPRODUCTION_FAILED")})
    if assessment["qualified"] and reproduction.get("reproduced") and assessment["upcoming"]:
        target = assessment["upcoming"][0]
        try:
            matrix, fixture = features.build(assessment, target, assessment["as_of"])
            require(fixture.match_id == target["fixture_id"], "FUTURE_FEATURE_TARGET_MISMATCH")
            proofs.append(validate_features(matrix, fixture, ordered, assessment["as_of"]))
        except (ValueError, KeyError, TypeError, RuntimeError) as exc:
            failures.append({"fixture_id": target["fixture_id"], "code": str(exc),
                             "error_type": type(exc).__name__})
    else:
        failures.append({"code": "CURRENT_FEATURE_CONTINUITY_BLOCKED_BY_SOURCE_QUALIFICATION"})
    ready = bool(assessment["qualified"] and reproduction.get("reproduced") and proofs and not failures)
    return reproduction, proofs, failures, ready


def qualification_body(assessment, index_ref, *, science, features):
    """Rebuild qualification readiness with synthetic feature science only.

    This exercises the same reproduction/first-target/finite/causal decisions
    as the historical qualifier. It cannot qualify live sources or load a real
    model. Full live feature equivalence requires separate review/permission.
    """
    require(science.get("execution_mode") == getattr(features, "execution_mode", None)
            == "synthetic", "FUTURE_SYNTHETIC_SCIENCE_REQUIRED")
    ordered = science["ordered_features"]
    reproduction, proofs, failures, ready = _feature_continuity(assessment, ordered, features)
    return seal({"schema_version": "ffps.future_lineage.synthetic_qualification.v1",
        "execution_mode": "synthetic", "league": "BL1", "as_of": assessment["as_of"],
        "assessment": assessment, "historical_feature_reproduction": reproduction,
        "feature_proofs": proofs, "feature_failures": failures, "feature_readiness": ready,
        "active_bundle_id": science["active_bundle_id"],
        "candidate_bundle_id": science["candidate_bundle_id"],
        "active_sha256": science["active_sha256"], "candidate_sha256": science["candidate_sha256"],
        "ordered_features": ordered, "feature_schema_sha256": fingerprint(ordered),
        "source_index": index_ref, "scientific_contract_sha256": fingerprint(science),
        "promotion_authorized": False, "collection_authorization": False}, "qualification_sha256")


def qualify_fixture(store, primary, alternative, *, frozen, names, as_of,
                    science, features, attempt, assessment_number):
    """Publish only a contained synthetic assessment, never a live acceptance.

    Source/index inputs have already been archived by the test. No acquisition,
    credential lookup, source-contract approval or live permission is issued.
    """
    require(store.mode == "synthetic" and store.access is not None,
            "FUTURE_SYNTHETIC_SESSION_REQUIRED")
    store.access.check(as_of)
    require(re.fullmatch(r"[0-9]{6}", attempt or "") is not None and
            re.fullmatch(r"[0-9]{3}", assessment_number or "") is not None,
            "FUTURE_ASSESSMENT_SLOT_INVALID")
    assessment, index = compose_assessment(store, primary, alternative,
        frozen=frozen, names=names, as_of=as_of)
    prefix = f"BL1/qualification/attempt-{attempt}/assessments/assessment-{assessment_number}/"
    index = seal({**index, "execution_mode": "synthetic"}, "index_sha256")
    index_ref = store.publish(prefix+"qualified_source_index.json", index, at=as_of)
    body = qualification_body(assessment, index_ref, science=science, features=features)
    qualification_ref = store.publish(prefix+"qualification.json", body, at=as_of)
    store.publish(prefix+"season_boundary_evidence.json",
        {"execution_mode": "synthetic", "seasons": assessment["season_evidence"]}, at=as_of)
    store.publish(prefix+"team_mapping_evidence.json",
        {"execution_mode": "synthetic", "league": "BL1", "mappings": assessment["team_mappings"],
         "new_mappings": assessment["mapping_evidence"], "fuzzy_matching_used": False}, at=as_of)
    store.publish(prefix+"provider_overlap_evidence.json",
        {"execution_mode": "synthetic", "league": "BL1",
         "football_data_co_uk_vs_fotmob": assessment["cross_provider_overlap"],
         "required_fields": ["home_goals", "away_goals", "home_corners", "away_corners"]}, at=as_of)
    store.publish(prefix+"feature_continuity_evidence.json",
        {"execution_mode": "synthetic", "qualification": qualification_ref,
         "reproduction": body["historical_feature_reproduction"],
         "proofs": body["feature_proofs"], "failures": body["feature_failures"]}, at=as_of)
    if body["feature_readiness"]:
        store.publish(prefix+"current_source_contract.json", seal({
            "execution_mode": "synthetic", "qualification": qualification_ref,
            "source_index": index_ref, "league": "BL1", "season": 2026,
            "authority": {"fixture_identity": "football-data.org:v4",
                "kickoff": "football-data.org:v4", "goals": "football-data.org:v4",
                "corners_primary": "football-data.co.uk:csv:no-www", "corners_gap_fill": "fotmob:web-api"},
            "provider_overlap_rows": len(assessment["cross_provider_overlap"]),
            "collection_authorization": False, "promotion_authorized": False}), at=as_of)
    return {"execution_mode": "synthetic", "qualification": qualification_ref,
            "source_index": index_ref, "ready": body["feature_readiness"],
            "collection_authorization": False}


def verify_fixture_qualification(qualification, store, primary, alternative, *,
                                frozen, names, science, features):
    assessment, rederived = verify_composition(qualification, store, primary, alternative,
                                               frozen=frozen, names=names)
    index = read(reference(store.repository, qualification["source_index"]))
    sealed(index, "index_sha256")
    require(index == seal({**rederived, "execution_mode": "synthetic"}, "index_sha256"),
            "FUTURE_SOURCE_INDEX_REPLAY_MISMATCH")
    require(qualification == qualification_body(assessment, qualification["source_index"],
            science=science, features=features), "FUTURE_FEATURE_QUALIFICATION_REPLAY_MISMATCH")
    require(qualification["feature_readiness"], "CURRENT_FEATURE_CONTINUITY_BLOCKED")
    return index
