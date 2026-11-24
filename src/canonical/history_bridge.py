"""Offline, one-opening E17 state bridge across reviewed source boundaries."""

from collections import Counter, defaultdict
from copy import deepcopy
import csv
from datetime import date, timedelta
from hashlib import sha256
import io
import json
import math
import os
from pathlib import Path, PureWindowsPath

from src.data_foundation.approved_matcher import MatchError, read_metadata
from src.data_foundation.audit import MetadataError, project_metadata
from src.data_foundation.identity import normalize_name
from src.data_foundation.snapshot_metadata import project_snapshot
from src.data_foundation.source_manifest import SourceManifestError, _checked_stat, _manifest_entries, _metadata

from .clock import CanonicalError, canonical, digest, now_utc, utc
from .features import FEATURE_VERSION, _advance_day, _fresh
from .model import Model
from . import provider as live_provider
from .provider import LEAGUES, STATE_UPDATING_STATUS, STATUSES
from .provider_team_mapping import APPROVED_INPUTS, ESPANYOL, ROOT as MAPPING_ROOT
from .team_mapping_consumer import MAPPING_PATH, PROVIDER_SEED_IDENTITY, _has_sha256, _object, _provider_id, _read_pinned


SEED_DAY = "2024-06-02"
CSV_START = "2024-07-01"
PROVIDER_START = "2025-07-01"
CONFIRMATORY_END = "2026-05-01"
SEALED_WINDOWS = (
    ("final_test", CSV_START, PROVIDER_START),
    ("confirmatory", PROVIDER_START, CONFIRMATORY_END),
)
PACKET = Path("tmp/canonical-history-bridge")
FEATURES = Path("src/canonical/features.py")
READER = Path("src/canonical/history_bridge.py")
PROVIDER_IMPLEMENTATION = Path("src/canonical/provider.py")
FROZEN_RECIPE = (
    Path("tmp/BL1-T3-offline-dependency-manifest-binding-20261027-01/NEXT-32-trackB-stageC-01/FREEZE_C.md"),
    "12c3e64c7bdd76c05bbf1ef28fc846cef3a18bbfd34d437fb093c0162dbcbebe",
)
ESPANYOL_PUBLICATION_SHA256 = "1b5efbd10f6c09599eb822da15bdaaac01653a3e16045652fc5941a54fa8d5a7"
BRIDGE_CSV_SEASONS = (2023, 2024)
IDENTITY_REVIEW_SCHEMA = 4
VOID_DISPOSITIONS = {"AWARDED": "excluded_awarded", "CANCELLED": "excluded_cancelled"}
SEASON_CLUBS = {"BL1": 18, "FL1": 18, "PD": 20, "PL": 20, "SA": 20}
GRANT_SCOPE = (
    "Read outcomes only to advance feature and rating state for the history bridge. "
    "Exclude model selection, recipe changes, performance claims and confirmatory evaluation. "
    "Verify the frozen recipe and model hashes before and after."
)


def _require(condition, code, detail=""):
    if not condition:
        raise CanonicalError(code, detail)


class SeasonCoverageError(CanonicalError):
    def __init__(self, gaps):
        self.gaps = gaps
        super().__init__("BRIDGE_COMPLETED_SEASON_GAPS", "; ".join(gaps))


def _day(value):
    _require(isinstance(value, str) and date.fromisoformat(value).isoformat() == value,
             "BRIDGE_DATE_INVALID")
    return value


def _input(root, reference, kind):
    path = Path(reference["path"])
    _require(not path.is_absolute() and not path.anchor and ".." not in path.parts
             and _has_sha256(reference["sha256"]), "BRIDGE_INPUT_REFERENCE_INVALID", kind)
    return _read_pinned(root / path, reference["sha256"], kind)


def _document(root, reference, kind):
    return _object(_input(root, reference, kind), kind + "_INVALID")


def _validate_state(state):
    matches = state["matches"]
    _require(type(matches) is int and matches >= 0, "BRIDGE_SEED_STATE_INVALID")
    for key in ("for", "against"):
        goals = state[key]
        _require(isinstance(goals, list) and len(goals) == min(matches, 10)
                 and all(type(value) in (int, float) and math.isfinite(value) and value >= 0
                         for value in goals), "BRIDGE_SEED_STATE_INVALID")
    elo, form = state["elo"], state["form"]
    _require(type(elo) in (int, float) and math.isfinite(elo)
             and (form is None if not matches else type(form) in (int, float) and 0 <= form <= 3),
             "BRIDGE_SEED_STATE_INVALID")


def approved_stale_cutoff(register, proposals):
    rule = register.get("binding_rule", {})
    _require(rule.get("status") == "APPROVED" and isinstance(proposals.get("binding_rule"), str)
             and rule.get("architect_proposal") == proposals["binding_rule"]
             and proposals.get("seed_end_inclusive") == SEED_DAY, "BRIDGE_STALE_RULE_NOT_APPROVED")
    cutoff = _day(proposals.get("recent_cutoff_inclusive"))
    _require(cutoff < SEED_DAY, "BRIDGE_STALE_RULE_CUTOFF_INVALID")
    return cutoff


def processed_seed_sources(root, inventory):
    seen = set()
    for source in inventory["canonical_inputs"]:
        name = PureWindowsPath(source["path"]).name
        league, _, season = Path(name).stem.partition("_")
        _require(league in LEAGUES and season.isascii() and season.isdecimal()
                 and name == f"{league}_{season}.csv" and name not in seen
                 and _has_sha256(source["sha256"]), "BRIDGE_SEED_APPEARANCE_SOURCE_INVALID", name)
        seen.add(name)
        yield root / "data/processed/matches" / name, league, int(season), source["sha256"]


def record_seed_appearances(spans, league, records, fixture_chains=None):
    for row in records:
        day = row["day"]
        if day <= SEED_DAY:
            home, away = row["home"], row["away"]
            for label in (home, away):
                first, last = spans.get((league, label), (day, day))
                spans[league, label] = min(first, day), max(last, day)
                if fixture_chains is not None and league == ESPANYOL[0]:
                    fixture_chains[league, label].add((day, home, away))


def latest_seed_chain(league, labels, spans):
    """Return one latest chain only if the evidenced chains occupy distinct seasons."""
    if not labels or any((league, label) not in spans for label in labels):
        return None
    chains = sorted((*spans[league, label], label) for label in labels)

    def season(day):
        return int(day[:4]) - (day[5:7] < "07")

    if any(season(first) <= season(prior_end)
           for (_, prior_end, _), (first, _, _) in zip(chains, chains[1:])):
        return None
    return chains[-1][-1]


def espanyol_decision_reference():
    path, pin = APPROVED_INPUTS["espanyol_debt"]
    return {"path": path.relative_to(MAPPING_ROOT).as_posix(), "sha256": pin}


def espanyol_seed_exception(root, key, carried, candidates, spans, fixture_chains, decision_ref):
    """Carry the published Espanyol state only for the recorded one-fixture debt."""
    if key != ESPANYOL or decision_ref != espanyol_decision_reference():
        return False
    decision = _document(root, decision_ref, "BRIDGE_ESPANYOL_DEBT")["decision_supplement"]
    excluded = decision["legacy_label"]
    if ((decision["league"], decision["provider_id"]) != key
            or decision["status"] != "APPROVED" or decision["provenance"] != "RECORDED_OWNER_DECISION"
            or decision["decision_type"] != "FIXTURE_EXCLUDED_AND_COUNTED"
            or type(decision["excluded_fixture_count"]) is not int or decision["excluded_fixture_count"] != 1
            or carried is None or carried == excluded or not {carried, excluded} <= candidates
            or latest_seed_chain(key[0], candidates - {excluded}, spans) != carried):
        return False
    league, _ = key
    carried_chain = fixture_chains.get((league, carried))
    excluded_chain = fixture_chains.get((league, excluded))
    if not carried_chain or not excluded_chain:
        return False
    difference = excluded_chain - carried_chain
    if (len(difference) != decision["excluded_fixture_count"]
            or {digest([league, *fixture]) for fixture in difference} != {decision["track_b_fixture_id"]}
            or spans[league, carried][1] > spans[league, excluded][1]):
        return False
    publication = _document(root, {"path": MAPPING_PATH.as_posix(), "sha256": ESPANYOL_PUBLICATION_SHA256},
                            "BRIDGE_ESPANYOL_PUBLICATION")
    bindings = [row for row in publication["entries"] if (row["league"], row["provider_id"]) == key]
    return (len(bindings) == 1 and bindings[0]["kind"] == "BINDING"
            and bindings[0]["legacy_label"] == carried)


def proposal_seed_tokens(proposal):
    candidates = proposal.get("candidates", [])
    supported = {normalize_name(row["legacy_label"]) for row in candidates if row["binding_supported"]}
    possible = {normalize_name(row["legacy_label"])
                for group in (candidates, proposal.get("tail_state_conflicts", [])) for row in group}
    return supported, possible


def _seed_review_evidence(root, review, leagues):
    sources = review["seed_appearance_sources"]
    inventory = _document(root, sources, "BRIDGE_SEED_APPEARANCE_SOURCES")
    rule = review["stale_rule"]
    proposals = _document(root, rule["proposals"], "BRIDGE_STALE_RULE_PROPOSALS")
    cutoff = approved_stale_cutoff(_document(root, rule["register"], "BRIDGE_STALE_RULE_REGISTER"), proposals)
    spans, fixture_chains = {}, defaultdict(set)
    covered = set()
    for path, league, season, pin in processed_seed_sources(root, inventory):
        if league in leagues and season < int(CSV_START[:4]):
            try:
                records, _ = read_metadata(path, league, season, expected_sha256=pin)
            except (MatchError, MetadataError, OSError) as exc:
                raise CanonicalError("BRIDGE_SEED_APPEARANCE_SOURCE_INVALID", path.name) from exc
            if records:
                covered.add(league)
            record_seed_appearances(spans, league, records, fixture_chains)
    _require(set(leagues) <= covered, "BRIDGE_SEED_APPEARANCE_SOURCES_INCOMPLETE")
    by_club = {(row["league"], row["id"]): row for row in proposals["rows"]}
    _require(len(by_club) == len(proposals["rows"]), "BRIDGE_STALE_RULE_PROPOSALS_INVALID")
    return spans, cutoff, sources["sha256"], by_club, fixture_chains


def _seed_states(root, seed, bindings, leagues, prior_clubs, replayed_clubs, identities, evidence):
    _require(seed["feature_version"] == FEATURE_VERSION and seed["last_day"] == SEED_DAY,
             "BRIDGE_SEED_VERSION_OR_CUTOFF_INVALID")
    lookup = {(row["league"], row["team_id"]): row["state"] for row in seed["states"]}
    _require(len(lookup) == len(seed["states"]), "BRIDGE_DUPLICATE_SEED_LABEL")
    by_token = defaultdict(set)
    for league, label in lookup:
        by_token[league, normalize_name(label)].add(label)
    spans, cutoff, source_sha256, proposals, fixture_chains = evidence
    prior_tokens = {(league, normalize_name(label)) for league, label in spans}
    states, labels = {}, set()
    for row in bindings:
        league, provider_id, label = row["league"], row["provider_id"], row["seed_label"]
        key = league, provider_id
        _require(league in leagues and _provider_id(provider_id) and key not in states,
                 "BRIDGE_SEED_BINDING_INVALID")
        detail = f"{league}/{provider_id}"
        _require(label is None or (isinstance(label, str) and bool(label.strip())
                 and (league, label) in lookup and "finding" not in row), "BRIDGE_SEED_BINDING_INVALID", detail)
        if key in replayed_clubs:
            supported, possible = proposal_seed_tokens(proposals.get(key, {}))
            tokens = identities[key] | supported
            candidates = set().union(*(by_token[league, token] for token in tokens | possible))
            prior = key in prior_clubs or any((league, token) in prior_tokens for token in tokens)
            if label is not None:
                candidates.add(label)
            latest = latest_seed_chain(league, candidates, spans)
            if latest is None and len(candidates) > 1 and espanyol_seed_exception(
                    root, key, label, candidates, spans, fixture_chains, row.get("proposal_evidence")):
                latest = label
            _require(len(candidates) <= 1 or latest is not None, "BRIDGE_SEED_BINDING_AMBIGUOUS",
                     f"{detail}: {', '.join(sorted(candidates))}")
            last = spans[league, latest][1] if latest else None
            if label is None:
                code = ("BRIDGE_FROZEN_SEED_BINDING_REQUIRED" if candidates or prior
                        else "BRIDGE_NO_USABLE_STATE_FINDING_REQUIRED")
                _require(row.get("finding") == "NO_USABLE_STATE", code, detail)
                reason = row.get("reason")
                if reason == "NO_SEED_LABEL":
                    _require(not candidates and not prior, "BRIDGE_FINDING_CONTRADICTS_EVIDENCE", detail)
                elif reason == "STALE_BEYOND_THRESHOLD":
                    _require(last is not None and normalize_name(latest) in tokens
                             and last < cutoff and row.get("evidence") == {
                        "last_seed_appearance": last, "threshold": cutoff, "source_sha256": source_sha256,
                    }, "BRIDGE_FINDING_CONTRADICTS_EVIDENCE", detail)
                else:
                    raise CanonicalError("BRIDGE_FINDING_REASON_REQUIRED", detail)
            else:
                _require(latest == label and last >= cutoff, "BRIDGE_SEED_BINDING_RULE_MISMATCH", detail)
        if label is None:
            states[key] = _fresh()
        else:
            legacy = league, label
            _require(legacy in lookup and legacy not in labels, "BRIDGE_SEED_BINDING_INVALID")
            labels.add(legacy)
            _validate_state(lookup[legacy])
            states[key] = deepcopy(lookup[legacy])
    missing = replayed_clubs - states.keys()
    _require(not missing, "BRIDGE_FIXTURE_TEAM_UNBOUND",
             ", ".join(f"{league}/{team}" for league, team in sorted(missing)))
    return {(league, str(team)): state for (league, team), state in states.items()}


def _completed_rosters(rows, required):
    rosters = {}
    for row in rows:
        key = row["league"], row["season"]
        _require(key in required and key not in rosters, "BRIDGE_COMPLETED_SEASON_ROSTER_INVALID", str(key))
        clubs = {}
        for club in row["clubs"]:
            team, name = club["provider_id"], club["name"]
            _require(_provider_id(team) and team not in clubs and isinstance(name, str) and bool(name.strip()),
                     "BRIDGE_COMPLETED_SEASON_ROSTER_INVALID", str(key))
            clubs[team] = name
        _require(len(clubs) == SEASON_CLUBS[key[0]], "BRIDGE_COMPLETED_SEASON_ROSTER_INVALID", str(key))
        rosters[key] = clubs
    missing = required - rosters.keys()
    _require(not missing, "BRIDGE_COMPLETED_SEASON_ROSTER_MISSING",
             ", ".join(f"{league}/{season}" for league, season in sorted(missing)))
    return rosters


def _verify_completed_seasons(rosters, schedule, exclusions=()):
    grouped = defaultdict(Counter)
    for league, season, _, home, away in schedule:
        grouped[league, season][home, away] += 1

    void_counts = Counter((row["league"], row["season"]) for row in exclusions)
    gaps, summaries = [], []
    for (league, season), clubs in sorted(rosters.items()):
        counts = grouped[league, season]
        expected = {(home, away) for home in clubs for away in clubs if home != away}
        for home, away in sorted(expected | counts.keys()):
            observed = counts[home, away]
            required = int((home, away) in expected)
            if observed != required:
                gaps.append(f"{league}/{season} {clubs.get(home, 'Unlisted club')} ({home}) home vs "
                            f"{clubs.get(away, 'Unlisted club')} ({away}) away: expected {required}, found {observed}")
        summaries.append({"league": league, "season": season, "clubs": len(clubs), "fixtures": len(expected),
                          "accounted_void_fixtures": void_counts[league, season]})
    if gaps:
        raise SeasonCoverageError(gaps)
    return summaries


def _csv_label_coverage(league, rows):
    clubs = {label for _, home, away in rows for label in (home, away)}
    pairs = Counter((home, away) for _, home, away in rows)
    expected = {(home, away) for home in clubs for away in clubs if home != away}
    missing = len(expected - pairs.keys())
    repeated = sum(count - 1 for count in pairs.values())
    self_fixtures = sum(count for (home, away), count in pairs.items() if home == away)
    passed = len(clubs) == SEASON_CLUBS[league] and not (missing or repeated or self_fixtures)
    return {"league": league, "season": 2023, "status": "PASS" if passed else "FAIL",
            "clubs": len(clubs), "expected_clubs": SEASON_CLUBS[league], "fixtures": len(rows),
            "missing_pairs": missing, "repeated_pairs": repeated, "self_fixtures": self_fixtures}


def _add_identity(identities, league, team, *labels):
    tokens = identities[league, team]
    tokens.add(str(team))
    tokens.update(normalize_name(label) for label in labels if isinstance(label, str) and label.strip())


def _check_unresolved_pre_seed(rows, identities):
    lookup = defaultdict(set)
    for (league, team), tokens in identities.items():
        for token in tokens:
            lookup[league, token].add(team)
    for row in rows:
        clashes = lookup[row["league"], normalize_name(row["label"])]
        _require(not clashes, "BRIDGE_PRE_SEED_IDENTITY_OVERLAPS_POST_SEED",
                 f"{row['path']}:{row['line']} {row['side']} {row['label']}: {', '.join(map(str, sorted(clashes)))}")


def _csv_labels(rows):
    labels = {}
    for row in rows:
        key = row["league"], row["season"], row["label"]
        _require(key[1] in BRIDGE_CSV_SEASONS and key not in labels and _provider_id(row["provider_id"]),
                 "BRIDGE_CSV_LABEL_BINDING_INVALID")
        labels[key] = row["provider_id"]
    return labels


def csv_fixture_key(league, season, day, home, away, labels):
    return league, season, day, labels.get((league, season, home)), labels.get((league, season, away))


def fixture_exclusions(schedule, matched_csv):
    exclusions = []
    for key, match in sorted(schedule.items()):
        status = match["status"]
        if status in VOID_DISPOSITIONS:
            league, season, day, home, away = key
            exclusions.append({"league": league, "season": season, "day": day, "fixture_id": match["id"],
                               "home_id": home, "away_id": away, "status": status,
                               "disposition": VOID_DISPOSITIONS[status], "matched_csv": key in matched_csv})
    return exclusions


def _snapshot_reference(root, manifests, league, season, source, observed_as_of):
    ref, expected_count = source["manifest"], source["expected_fixtures"]
    _require(_provider_id(expected_count), "BRIDGE_PROVIDER_EXPECTED_FIXTURES_INVALID")
    ref_key = ref["path"], ref["sha256"]
    if ref_key not in manifests:
        manifests[ref_key] = _document(root, ref, "BRIDGE_PROVIDER_MANIFEST")
    manifest = manifests[ref_key]
    _require(manifest["schema"] == "football-data-org.snapshots.v1"
             and manifest["host"] == "api.football-data.org", "BRIDGE_PROVIDER_MANIFEST_INVALID")
    entries = [row for row in manifest["entries"] if (row["league"], row["season"]) == (league, season)]
    _require(len(entries) == 1 and entries[0]["http_status"] == 200, "BRIDGE_PROVIDER_MATCH_BODY_UNAVAILABLE")
    entry, = entries
    received = utc(entry["received_utc"])
    _require(utc(entry["requested_utc"]) <= received <= observed_as_of,
             "BRIDGE_PROVIDER_OBSERVATION_INVALID", f"{league}/{season}")
    path = Path(ref["path"]).parent / f"{league}_{season}.json"
    _require(PureWindowsPath(entry["path"]).name == path.name, "BRIDGE_PROVIDER_BODY_NAME_MISMATCH")
    return {"league": league, "season": season, "path": path.as_posix(), "sha256": entry["sha256"],
            "size_bytes": entry["size_bytes"], "received_utc": received, "expected_fixtures": expected_count}


def _provider_sources(root, sources, leagues, through, observed_as_of):
    last_season = int(through[:4])
    if through[5:] <= "07-01":
        last_season -= 1
    required = {(league, season) for league in leagues for season in range(2024, last_season + 1)}
    supplied = {(entry["league"], entry["season"]): entry for entry in sources}
    _require(len(supplied) == len(sources) and supplied.keys() == required, "BRIDGE_PROVIDER_INVENTORY_INCOMPLETE")
    manifests = {}
    bodies, stale = {}, []
    for (league, season), source in sorted(supplied.items()):
        body = _snapshot_reference(root, manifests, league, season, source, observed_as_of)
        required_day = min(through, f"{season + 1:04}-07-01")
        if body["received_utc"][:10] < required_day:
            stale.append(f"{league}/{season}")
        bodies[league, season] = body
    _require(not stale, "BRIDGE_PROVIDER_CAPTURE_BEFORE_CUTOFF", ", ".join(stale))
    for league in leagues:
        key = league, last_season
        baseline = supplied[key].get("gap_baseline")
        _require(isinstance(baseline, dict), "BRIDGE_GAP_BASELINE_REQUIRED", f"{league}/{last_season}")
        body = bodies[key]
        baseline_ref = _snapshot_reference(root, manifests, *key, baseline, observed_as_of)
        _require(baseline_ref["received_utc"][:10] < through and baseline_ref["received_utc"] < body["received_utc"],
                 "BRIDGE_GAP_BASELINE_NOT_OLDER", f"{league}/{last_season}")
        body["gap_baseline"] = baseline_ref
    return bodies


def _snapshot_metadata(root, ref):
    raw = _input(root, ref, "BRIDGE_PROVIDER_BODY")
    payload = _object(raw, "BRIDGE_PROVIDER_BODY_INVALID")
    projected = project_snapshot(payload)
    count = projected["resultSet"].get("count")
    rows = projected["matches"]
    _require(len(raw) == ref["size_bytes"] and payload.get("competition", {}).get("code") == ref["league"]
             and type(count) is int and count == len(rows) == ref["expected_fixtures"]
             and not projected["metadata_errors"], "BRIDGE_PROVIDER_BODY_INVALID",
             f"{ref['league']}/{ref['season']}")
    return payload, rows


def _reconcile_gap(root, refreshed, through, baseline, identities):
    _, previous = _snapshot_metadata(root, baseline)
    seen = set()
    for row in previous:
        fixture_id = row["id"]
        teams = row["homeTeam"]["id"], row["awayTeam"]["id"]
        _require(all(_provider_id(value) for value in (fixture_id, *teams)) and fixture_id not in seen,
                 "BRIDGE_GAP_BASELINE_INVALID")
        for side in ("homeTeam", "awayTeam"):
            team = row[side]
            _add_identity(identities, baseline["league"], team["id"], team.get("name"),
                          team.get("shortName"), team.get("tla"))
        seen.add(fixture_id)
        if utc(row["utcDate"])[:10] < through:
            _require(refreshed.get(fixture_id) == teams, "BRIDGE_GAP_FIXTURE_NOT_RECONCILED", str(fixture_id))


def _provider_metadata(root, sources, through):
    fixtures = {}
    seen_ids = set()
    pending = []
    payloads = {}
    identities, max_dates = defaultdict(set), {}
    for (league, season), ref in sources.items():
        payload, rows = _snapshot_metadata(root, ref)
        provider_ids = {}
        dates = []
        pair_has_result = {}
        for index, row in enumerate(rows):
            fixture_id = row["id"]
            home, away = row["homeTeam"]["id"], row["awayTeam"]["id"]
            day = utc(row["utcDate"])[:10]
            status = row["status"]
            _require(all(_provider_id(value) for value in (fixture_id, home, away)) and home != away
                     and status in STATUSES and _day(row["season"]["startDate"])[:4] == str(season)
                     and f"{season:04}-07-01" <= day < f"{season + 1:04}-07-01",
                     "BRIDGE_PROVIDER_FIXTURE_INVALID", f"{league}/{season}")
            _require(fixture_id not in seen_ids, "BRIDGE_DUPLICATE_FIXTURE_ID", str(fixture_id))
            seen_ids.add(fixture_id)
            provider_ids[fixture_id] = (home, away)
            dates.append(day)
            for side in ("homeTeam", "awayTeam"):
                team = row[side]
                _add_identity(identities, league, team["id"], team.get("name"), team.get("shortName"), team.get("tla"))
            if day >= through:
                continue
            updates_state = status == STATE_UPDATING_STATUS
            if not updates_state and status not in VOID_DISPOSITIONS:
                pending.append(f"{league}/{season} {day} fixture {fixture_id} home {home} away {away} {status}")
                continue
            if updates_state:
                _require(utc(row["utcDate"]) < ref["received_utc"], "BRIDGE_FINISHED_FIXTURE_IN_FUTURE")
            match = {"id": fixture_id, "league": league, "day": day, "home_id": home, "away_id": away,
                     "status": status}
            key = league, season, day, home, away
            _require(key not in fixtures, "BRIDGE_DUPLICATE_FIXTURE_METADATA")
            _require(pair_has_result.setdefault((home, away), updates_state) == updates_state,
                     "BRIDGE_PROVIDER_PAIR_RESULT_CONFLICT", f"{league}/{season} {home} home vs {away} away")
            fixtures[key] = (match, ref["path"], index)
        if "gap_baseline" in ref:
            _reconcile_gap(root, provider_ids, through, ref["gap_baseline"], identities)
        payloads[ref["path"]] = payload
        max_dates[league, season] = max(dates)
    _require(not pending, "BRIDGE_PROVIDER_RESULTS_UNRESOLVED",
             ", ".join(sorted(pending)))
    return fixtures, payloads, identities, max_dates


def _required_completed_seasons(manifest, csv_max_dates, provider_max_dates, through):
    completed_provider = {key for key in provider_max_dates if key[1] >= 2025 and f"{key[1] + 1:04}-07-01" <= through}
    endings = {}
    for row in manifest.get("provider_season_max_dates", []):
        key = row["league"], row["season"]
        if key not in completed_provider:
            continue
        expected = _day(row["max_date"])
        _require(key not in endings and expected == provider_max_dates[key],
                 "BRIDGE_PROVIDER_MANIFEST_MAX_DATE_MISMATCH", f"{key[0]}/{key[1]}")
        endings[key] = expected
    missing = completed_provider - endings.keys()
    _require(not missing, "BRIDGE_PROVIDER_MANIFEST_MAX_DATE_MISSING",
             ", ".join(f"{league}/{season}" for league, season in sorted(missing)))
    return {(league, season) for (league, season), last in {**csv_max_dates, **endings}.items()
            if f"{season + 1:04}-07-01" <= through and last > SEED_DAY}


def _csv_matches(root, authority, season, through, labels, schedule, used):
    raw = _input(root, authority, "BRIDGE_AUTHORITY_CSV")
    path, league = authority["path"], authority["league"]
    try:
        metadata = _metadata(path, authority, raw)
    except SourceManifestError as error:
        raise CanonicalError("BRIDGE_CSV_MANIFEST_METADATA_MISMATCH", str(error)) from None
    stream = io.BytesIO(raw)
    header = next(csv.reader([stream.readline().decode("utf-8-sig")], strict=True))
    _require(all(header.count(name) == 1 for name in ("FTHG", "FTAG")), "BRIDGE_CSV_GOAL_COLUMNS_INVALID", path)
    goal_columns = {header.index("FTHG"): "home_goals", header.index("FTAG"): "away_goals"}
    rows, dated_pairs, unresolved = [], [], []
    prior_clubs = set()
    records = ((line, record) for line, record in enumerate(stream, 2) if record.strip())
    for (day, home, away), (line, record) in zip(metadata, records, strict=True):
        _require(f"{season:04}-07-01" <= day < f"{season + 1:04}-07-01",
                 "BRIDGE_CSV_SEASON_MISMATCH", f"{path}:{line}")
        if day >= through:
            continue
        key = csv_fixture_key(league, season, day, home, away, labels)
        if day <= SEED_DAY:
            for side, label, team in (("home", home, key[3]), ("away", away, key[4])):
                if team is None:
                    unresolved.append({"league": league, "path": path, "line": line, "day": day,
                                       "side": side, "label": label})
                else:
                    prior_clubs.add((league, team))
            if None in key[3:]:
                continue
        _require(key[3] is not None and key[4] is not None and key[3] != key[4],
                 "BRIDGE_CSV_LABEL_UNBOUND", f"{path}:{line} {day} {home} vs {away}")
        dated_pairs.append(key)
        if day <= SEED_DAY:
            continue
        match = schedule.get(key)
        _require(match is not None and key not in used, "BRIDGE_CSV_FIXTURE_NOT_UNIQUELY_RECONCILED",
                 f"{path}:{line} {day} {home} ({key[3]}) vs {away} ({key[4]})")
        used.add(key)
        if match["status"] == STATE_UPDATING_STATUS:
            rows.append((match, path, (record, goal_columns)))
    return rows, dated_pairs, metadata, prior_clubs, unresolved


def _reviewed_identities(root, plan, review_sha256, leagues, through):
    ref = plan.get("identity_review")
    _require(isinstance(ref, dict) and _has_sha256(review_sha256) and ref.get("sha256") == review_sha256,
             "BRIDGE_INDEPENDENT_IDENTITY_REVIEW_REQUIRED")
    review = _document(root, ref, "BRIDGE_IDENTITY_REVIEW")
    _require(type(review.get("schema_version")) is int and review["schema_version"] == IDENTITY_REVIEW_SCHEMA
             and review.get("status") == "APPROVED_FOR_BRIDGE"
             and isinstance(review.get("reviewed_by"), str) and bool(review["reviewed_by"].strip())
             and review.get("leagues") == list(leagues) and review.get("through_exclusive") == through
             and review.get("model_sha256") == plan["model"]["sha256"]
             and review.get("source_manifest_sha256") == plan["source_manifest"]["sha256"]
             and isinstance(review.get("seed_appearance_sources"), dict)
             and isinstance(review.get("stale_rule"), dict)
             and isinstance(review.get("tail_expected_counts"), dict),
             "BRIDGE_IDENTITY_REVIEW_NOT_APPLICABLE")
    _require(review.get("blockers") == [], "BRIDGE_IDENTITY_REVIEW_BLOCKED")
    return review


def _plan(root, reference, review_sha256):
    plan = _document(root, reference, "BRIDGE_PLAN")
    through = _day(plan["through_exclusive"])
    leagues = tuple(sorted(plan["leagues"]))
    _require(type(plan["schema_version"]) is int and plan["schema_version"] == 1
             and leagues and len(set(leagues)) == len(leagues) and set(leagues) <= set(LEAGUES)
             and through >= CONFIRMATORY_END, "BRIDGE_PLAN_INVALID")
    observed_as_of = utc(plan["observed_as_of"])
    _require(through <= observed_as_of[:10], "BRIDGE_OBSERVATION_BEFORE_CUTOFF")
    review = _reviewed_identities(root, plan, review_sha256, leagues, through)
    sources = _provider_sources(root, plan["provider_sources"], leagues, through, observed_as_of)
    model_ref, recipe = plan["model"], plan["recipe"]
    model = Model(_document(root, model_ref, "BRIDGE_FROZEN_MODEL")["model"])
    _require((Path(recipe["path"]), recipe["sha256"]) == FROZEN_RECIPE, "BRIDGE_FROZEN_RECIPE_REFERENCE_INVALID")
    _input(root, recipe, "BRIDGE_FROZEN_RECIPE")
    implementation = plan["feature_implementation"]
    _require(Path(implementation["path"]) == FEATURES, "BRIDGE_FEATURE_REFERENCE_INVALID")
    _input(root, implementation, "BRIDGE_FROZEN_FEATURES")
    _require(sha256(Path(__file__).with_name("features.py").read_bytes()).hexdigest() == implementation["sha256"],
             "BRIDGE_RUNNING_FEATURES_MISMATCH")
    reader = plan["bridge_reader"]
    _require(Path(reader["path"]) == READER, "BRIDGE_READER_REFERENCE_INVALID")
    _input(root, reader, "BRIDGE_FROZEN_READER")
    _require(sha256(Path(__file__).read_bytes()).hexdigest() == reader["sha256"],
             "BRIDGE_RUNNING_READER_MISMATCH")
    verification = _document(root, plan["offline_verification"], "BRIDGE_OFFLINE_VERIFICATION")
    provider_sha256 = sha256(Path(live_provider.__file__).read_bytes()).hexdigest()
    _require(verification.get("status") == "PASS"
             and verification.get("bridge_reader_sha256") == reader["sha256"]
             and verification.get("feature_implementation_sha256") == implementation["sha256"]
             and verification.get("provider_implementation_sha256") == provider_sha256,
             "BRIDGE_OFFLINE_VERIFICATION_NOT_APPLICABLE")
    labels = _csv_labels(review["csv_labels"])
    manifest = _document(root, plan["source_manifest"], "BRIDGE_SOURCE_MANIFEST")
    _, authorities, _ = _manifest_entries(manifest)
    provider, payloads, identities, provider_max_dates = _provider_metadata(root, sources, through)
    provider_fixtures = {key: match for key, (match, _, _) in provider.items()}
    csv_schedule = {key: match for key, match in provider_fixtures.items() if key[1] == 2024}
    fixture_ids = {row["id"] for row in provider_fixtures.values()}
    for row in review["tail_fixtures"]:
        league, day, home, away = row["league"], _day(row["day"]), row["home_id"], row["away_id"]
        key = (league, 2023, day, home, away)
        _require(league in leagues and SEED_DAY < day < CSV_START
                 and all(_provider_id(value) for value in (row["id"], home, away))
                 and home != away and key not in csv_schedule and row["id"] not in fixture_ids
                 and (row.get("status") == STATE_UPDATING_STATUS or row.get("status") in VOID_DISPOSITIONS),
                 "BRIDGE_TAIL_FIXTURE_INVALID", f"{league} {day} fixture {row['id']} home {home} away {away}")
        fixture_ids.add(row["id"])
        csv_schedule[key] = {name: row[name] for name in ("id", "league", "day", "home_id", "away_id", "status")}
        _add_identity(identities, league, home, row.get("home_name"))
        _add_identity(identities, league, away, row.get("away_name"))
    used = set()
    csv_rows, csv_sources, pre_snapshot_schedule = [], [], []
    csv_max_dates, label_coverage = {}, []
    tail_counts = dict.fromkeys(leagues, 0)
    prior_clubs, unresolved = set(), []
    for league in leagues:
        for season in BRIDGE_CSV_SEASONS:
            authority = authorities.get((league, season))
            _require(authority is not None, "BRIDGE_CSV_AUTHORITY_MISSING", f"{league}/{season}")
            matches, pairs, metadata, prior, unresolved_labels = _csv_matches(
                root, authority, season, through, labels, csv_schedule, used)
            csv_max_dates[league, season] = authority["max_date"]
            if season == 2023:
                pre_snapshot_schedule.extend(pairs)
                label_coverage.append(_csv_label_coverage(league, metadata))
                tail_counts[league] = len(matches)
            prior_clubs.update(prior)
            unresolved.extend(unresolved_labels)
            csv_rows.extend(matches)
            csv_sources.append({"league": league, "season": season, "path": authority["path"],
                                "sha256": authority["sha256"], "max_date": authority["max_date"],
                                "admitted_fixtures": len(matches)})
    expected = review["tail_expected_counts"]
    _require(expected.keys() == tail_counts.keys()
             and all(type(value) is int for value in expected.values())
             and expected == tail_counts, "BRIDGE_TAIL_EXPECTATION_MISMATCH")
    required = _required_completed_seasons(manifest, csv_max_dates, provider_max_dates, through)
    rosters = _completed_rosters(review.get("completed_season_rosters", []), required)
    unreconciled = {key for key, match in csv_schedule.items()
                    if match["status"] == STATE_UPDATING_STATUS and key not in used}
    _require(not unreconciled, "BRIDGE_CSV_PROVIDER_COVERAGE_MISMATCH", "; ".join(
        f"{league}/{season} {day} {rosters.get((league, season), {}).get(home, home)} ({home}) vs "
        f"{rosters.get((league, season), {}).get(away, away)} ({away})"
        for league, season, day, home, away in sorted(unreconciled)))
    provider_rows = [row for key, row in provider.items()
                     if key[1] >= 2025 and row[0]["status"] == STATE_UPDATING_STATUS]
    replayed_clubs = {(row["league"], row[side]) for row, _, _ in (*csv_rows, *provider_rows)
                      for side in ("home_id", "away_id")}
    for (league, _), clubs in rosters.items():
        for team, name in clubs.items():
            _add_identity(identities, league, team, name)
    for (league, _, label), team in labels.items():
        if (league, team) in identities:
            _add_identity(identities, league, team, label)
    for binding in review["seed_bindings"]:
        league, team = binding["league"], binding["provider_id"]
        aliases = binding.get("seed_aliases", [])
        _require(isinstance(aliases, list) and all(isinstance(alias, str) and alias.strip() for alias in aliases),
                 "BRIDGE_SEED_BINDING_INVALID", f"{league}/{team}")
        if (league, team) in identities:
            _add_identity(identities, league, team, binding["seed_label"], *aliases)
    _check_unresolved_pre_seed(unresolved, identities)
    seed_evidence = _seed_review_evidence(root, review, leagues)
    states = _seed_states(root, model.seed, review["seed_bindings"], leagues, prior_clubs, replayed_clubs, identities,
                          seed_evidence)
    ids, occupied = set(), set()
    for row, _, _ in (*csv_rows, *provider_rows):
        _require(row["id"] not in ids, "BRIDGE_DUPLICATE_FIXTURE_ID", str(row["id"]))
        ids.add(row["id"])
        for side in ("home_id", "away_id"):
            pair = row["league"], str(row[side])
            assignment = row["day"], pair
            _require(assignment not in occupied, "BRIDGE_DUPLICATE_TEAM_DATE")
            occupied.add(assignment)
    reviewed_tail = {key: match for key, match in csv_schedule.items() if key[1] == 2023}
    exclusions = fixture_exclusions({**provider_fixtures, **reviewed_tail}, used)
    completed_schedule = (*pre_snapshot_schedule, *(reviewed_tail.keys() - used), *provider)
    completed_seasons = _verify_completed_seasons(rosters, completed_schedule, exclusions)
    summary = {
        "leagues": list(leagues), "seed_day": SEED_DAY, "through_exclusive": through,
        "observed_as_of": observed_as_of, "csv_fixtures": len(csv_rows), "provider_fixtures": len(provider_rows),
        "void_provider_fixtures": sum(match["status"] in VOID_DISPOSITIONS for match in provider_fixtures.values()),
        "excluded_fixture_count": len(exclusions), "excluded_fixtures": exclusions,
        "completed_seasons": completed_seasons,
        "pre_seed_csv_label_coverage": label_coverage,
        "unresolved_pre_seed_labels": unresolved,
        "tail_fixtures_by_league": tail_counts,
        "csv_sources": csv_sources, "provider_sources": list(sources.values()), "model_id": model.model_id,
        "identity_review_sha256": review_sha256,
        "seed_appearance_sources_sha256": review["seed_appearance_sources"]["sha256"],
        "stale_rule_sha256": {name: review["stale_rule"][name]["sha256"] for name in ("register", "proposals")},
    }
    return plan, summary, states, csv_rows, provider_rows, payloads


def _goals(value, *, csv_value=False):
    if csv_value and isinstance(value, str) and value.isascii() and value.isdecimal():
        value = int(value)
    _require(type(value) is int and value >= 0, "BRIDGE_GOALS_INVALID")
    return value


def _project_scores(csv_rows, provider_rows, payloads):
    for row, path, locator in csv_rows:
        raw, columns = locator
        try:
            values = project_metadata(raw, columns)
            yield {**row, **{name: _goals(values[name], csv_value=True) for name in columns.values()}}
        except (CanonicalError, KeyError, ValueError, MetadataError):
            raise CanonicalError("BRIDGE_CSV_GOALS_INVALID", path) from None
    for row, path, index in provider_rows:
        try:
            score = payloads[path]["matches"][index]["score"]["fullTime"]
            yield {**row, "home_goals": _goals(score["home"]), "away_goals": _goals(score["away"])}
        except (CanonicalError, KeyError, TypeError):
            raise CanonicalError("BRIDGE_PROVIDER_GOALS_INVALID", str(row["id"])) from None


def _replay(states, rows, through):
    states = deepcopy(states)
    days = defaultdict(list)
    for row in rows:
        days[row["day"]].append(row)
    for day in sorted(days):
        _advance_day(states, days[day])
    return {"feature_version": FEATURE_VERSION, "identity_key": PROVIDER_SEED_IDENTITY,
            "last_day": (date.fromisoformat(through) - timedelta(days=1)).isoformat(),
            "states": [
            {"league": league, "team_id": team, "state": state}
            for (league, team), state in sorted(states.items())
    ]}


def _replay_by_window(states, rows, through):
    start = SEED_DAY
    for end in (PROVIDER_START, CONFIRMATORY_END, through):
        checkpoint = _replay(states, (row for row in rows if start <= row["day"] < end), end)
        states = {(row["league"], row["team_id"]): row["state"] for row in checkpoint["states"]}
        start = end
    return checkpoint


def _write_once(path, document):
    raw = (canonical(document) + "\n").encode("utf-8")
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    return sha256(raw).hexdigest()


def run_bridge(root, reference, *, identity_review_sha256=None, operator=None):
    """Preflight without score projection, or open once for state replay."""
    root = Path(root).absolute()
    packet = root / PACKET
    ledger_path = packet / "access-ledger.json"
    if operator is not None and os.path.lexists(ledger_path):
        raise CanonicalError("BRIDGE_OPENING_ALREADY_RECORDED")
    try:
        plan, summary, states, csv_rows, provider_rows, payloads = _plan(root, reference, identity_review_sha256)
    except (KeyError, TypeError, ValueError, AttributeError, OverflowError, csv.Error):
        raise CanonicalError("BRIDGE_INPUT_INVALID") from None
    if operator is None:
        return {**summary, "status": "PREFLIGHT_PASSED", "scores_projected": False}

    _require(isinstance(operator, str) and bool(operator.strip()), "BRIDGE_OPERATOR_REQUIRED")
    grant_ref = plan.get("owner_grant", {})
    _require(isinstance(grant_ref, dict) and {"path", "sha256"} <= grant_ref.keys(),
             "BRIDGE_OWNER_GRANT_REQUIRED")
    grant = _document(root, grant_ref, "BRIDGE_OWNER_GRANT")
    sealed = grant.get("sealed_windows", {})
    _require(grant.get("record_type") == "owner_decision_addendum" and isinstance(sealed, dict)
             and sealed.get("authority") == "GRANTED"
             and sealed.get("architect_proposed_scope") == GRANT_SCOPE, "BRIDGE_OWNER_GRANT_INVALID")

    for folder in (packet.parent, packet):
        if os.path.lexists(folder):
            _checked_stat(folder)
            _require(folder.is_dir(), "BRIDGE_PACKET_PATH_INVALID")
        else:
            _checked_stat(folder.parent)
            folder.mkdir(exist_ok=True)
            _checked_stat(folder)
    ledger = {"status": "OPENING_RESERVED", "operator": operator.strip(), "observed_utc": now_utc(),
              "windows": SEALED_WINDOWS, "plan": reference, "owner_grant": grant_ref,
              "identity_review": plan["identity_review"],
              "model": plan["model"], "feature_implementation": plan["feature_implementation"],
              "recipe": plan["recipe"], "reader_sha256": plan["bridge_reader"]["sha256"],
              "offline_verification": plan["offline_verification"],
              "through_exclusive": summary["through_exclusive"],
              "excluded_fixture_count": summary["excluded_fixture_count"],
              "excluded_fixtures": summary["excluded_fixtures"],
              "tail_fixtures_by_league": summary["tail_fixtures_by_league"]}
    try:
        ledger_pin = _write_once(ledger_path, ledger)
    except FileExistsError:
        raise CanonicalError("BRIDGE_OPENING_ALREADY_RECORDED") from None

    try:
        scores = list(_project_scores(csv_rows, provider_rows, payloads))
        seed = _replay(states, scores, summary["through_exclusive"])
        _require(digest(seed) == digest(_replay(states, reversed(scores), summary["through_exclusive"])),
                 "BRIDGE_STATE_REPLAY_MISMATCH")
        _require(digest(seed) == digest(_replay_by_window(states, scores, summary["through_exclusive"])),
                 "BRIDGE_CHECKPOINT_REPLAY_MISMATCH")
        for key in ("model", "recipe", "feature_implementation", "bridge_reader"):
            _input(root, plan[key], "BRIDGE_FROZEN_INPUT_AFTER_REPLAY")
        seed_pin = _write_once(packet / "candidate-seed.json", seed)
        result = {**summary, "status": "CANDIDATE_COMPLETE", "scores_projected": True,
                  "purpose": "FEATURE_AND_RATING_STATE_ONLY", "plan_sha256": reference["sha256"],
                  "ledger_sha256": ledger_pin, "candidate_seed_sha256": seed_pin}
        _write_once(packet / "history-bridge.json", result)
        return result
    except BaseException as error:
        _write_once(packet / "opening-failure.json", {
            "status": "OPENING_INTERRUPTED", "ledger_sha256": ledger_pin,
            "error_code": error.code if isinstance(error, CanonicalError) else type(error).__name__,
        })
        raise
