"""Generate an unapproved, score-blind mapping review from pinned local sources."""

import argparse
from collections import Counter, defaultdict
import csv
from datetime import date
import io
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.canonical.history_bridge import (
    BRIDGE_CSV_SEASONS, CSV_START, FEATURES, FROZEN_RECIPE, IDENTITY_REVIEW_SCHEMA, PROVIDER_IMPLEMENTATION, READER,
    SEASON_CLUBS, SEED_DAY, VOID_DISPOSITIONS, _csv_labels as _reviewed_csv_labels, approved_stale_cutoff,
    csv_fixture_key, espanyol_decision_reference, espanyol_seed_exception, fixture_exclusions, latest_seed_chain,
    processed_seed_sources, proposal_seed_tokens, record_seed_appearances,
)
from src.canonical.clock import utc
from src.canonical.provider import STATE_UPDATING_STATUS
from src.canonical import provider_team_mapping as mapping
from src.data_foundation.approved_matcher import QUARANTINE_SHA256, load_decisions, read_metadata
from src.data_foundation.identity import normalize_name
from src.data_foundation.snapshot_metadata import project_snapshot
from src.data_foundation.source_manifest import _held_bytes, _manifest_entries, _metadata


CAPTURES = {
    "20261108T193117.443560Z": "23dd6a705b7656d6ed3de53900421800e84e11cc22598037f1f55f8bda5495ff",
    "20261119T121847.246815Z": "426fe5a138cb1311add38252d13be1953db1f5da77b6f97f9c0161b9018089cc",
}
AUTHORITY = ("NEXT-38d-source-authority-01/source-authority-manifest.json",
             "73dd69514b7501a49a144bbf51e7bb42470aaa6886ec50c4532c438d7f481cc1")
CROSSWALK = ("NEXT-38-system-implementation-01/historical-identity-review.csv",
             "247a550475a405c8763c4c2527296f2a0adc566745e591f303572c420d571195")
PROCESSED_PINS = ("NEXT-30-trackB-stageA-02/freeze-A-input-references.json",
                  "7407c46daeb17c84e485eaebc4829e955bfef1357085d70cafdaf1fbc826b6dc")
FROZEN_SEED = ("NEXT-38-system-implementation-01/replay/portable-unbound-model.json",
               "577bacf580c78c6d1fc215a55b0961ee7357028c6550190c7b975bd22a2b33b1")
STALE_RULE_REGISTER = ("NEXT-38k2-catalog-hardening-01/owner-decision-addendum.json",
                       "034b9f685e759537e24df133d341e0acfd07de2cca6e84848a510f8771bd9cf5")
REPLAY_THROUGH_EXCLUSIVE = "2026-11-15"
PROPOSED_ALIASES = {
    ("BL1", 29): ("Paderborn", "PADERBORN"),
    ("BL1", 36): ("Bochum", "VFL BOCHUM"),
    ("BL1", 44): ("Heidenheim", "FC HEIDENHEIM"),
    ("PD", 560): ("La Coruna", "La Coruna"),
    ("PL", 322): ("Hull", "Hull"),
}
SEED_WINDOW_CARRY = "Carry state (existing approved seed-window binding)"
OWNER_DECISIONS = {
    "stated_by": "Kamzo",
    "stated_on": "2026-11-21",
    "status": "STATED_PENDING_FORMAL_APPROVAL",
    "verbatim": (
        "five aliases approved (Paderborn, Bochum, Heidenheim, Deportivo La Coruña, Hull); "
        "carry ESPANYOL under the recorded decision; AWARDED fixtures never update state and are counted and listed "
        "in the receipt. Stale rule unchanged; Barcelona carries under the existing seed-window binding."
    ),
}


def _record_pin(pins, path, pin):
    pins[path.relative_to(ROOT).as_posix()] = pin


def _read_pinned(pins, path, pin):
    raw = _held_bytes(path, pin)
    _record_pin(pins, path, pin)
    return raw


def _check_mapping_counts(entries):
    counts = Counter(row["league"] for row in entries)
    keys = {(row["league"], row["provider_id"]) for row in entries}
    if counts != SEASON_CLUBS or len(entries) != 96 or len(keys) != 96:
        raise ValueError(f"Published mapping club counts do not match 96 unique entries: {dict(counts)}")
    return counts


def _capture_metadata(pins):
    matches, names, exact = {}, {}, defaultdict(set)
    replay, void_by_season = {}, {}
    inventory = {}
    for capture, pin in sorted(CAPTURES.items()):
        folder = ROOT / "data/raw/football_data_org" / capture
        manifest = json.loads(_read_pinned(pins, folder / "manifest.json", pin))
        for entry in manifest["entries"]:
            league, season = entry["league"], entry["season"]
            if season not in (2024, 2025, 2026):
                continue
            raw = _read_pinned(pins, folder / f"{league}_{season}.json", entry["sha256"])
            payload = json.loads(raw)
            projected = project_snapshot(payload)
            rows = projected["matches"]
            if (entry["http_status"] != 200 or len(raw) != entry["size_bytes"]
                    or payload["competition"]["code"] != league or projected["metadata_errors"]
                    or len(rows) != projected["resultSet"]["count"]):
                raise ValueError(f"Unverified capture metadata for {league}/{season}")
            by_id = {}
            replay_rows, void_rows = {}, {}
            for row in rows:
                teams = row["homeTeam"], row["awayTeam"]
                day, status = utc(row["utcDate"])[:10], row["status"]
                fixture = (day, *(team["id"] for team in teams))
                by_id[row["id"]] = fixture
                if SEED_DAY < day < REPLAY_THROUGH_EXCLUSIVE:
                    if status == STATE_UPDATING_STATUS:
                        replay_rows[row["id"]] = fixture
                    elif status in VOID_DISPOSITIONS:
                        void_rows[league, season, *fixture] = {"id": row["id"], "status": status}
                for team in teams:
                    key = league, team["id"]
                    names[key] = team["name"]
                    for field in ("name", "shortName", "tla"):
                        if team.get(field):
                            exact[league, normalize_name(team[field])].add(team["id"])
            if len(by_id) != len(rows):
                raise ValueError(f"Repeated fixture ID for {league}/{season}")
            inventory[capture, league, season] = (len(rows), utc(entry["received_utc"]))
            matches[league, season] = by_id
            replay[league, season] = replay_rows
            void_by_season[league, season] = void_rows
    void_schedule = {key: row for rows in void_by_season.values() for key, row in rows.items()}
    return matches, names, exact, replay, inventory, void_schedule


def _csv_labels(evidence, pins):
    authority = json.loads(_read_pinned(pins, evidence / AUTHORITY[0], AUTHORITY[1]))
    _, authorities, _ = _manifest_entries(authority)
    labels = {}
    inventory_rows = {}
    alias_leagues = {league for league, _ in PROPOSED_ALIASES}
    for key, entry in authorities.items():
        if key[1] not in BRIDGE_CSV_SEASONS and key[0] not in alias_leagues:
            continue
        path = entry["path"]
        rows = _metadata(path, entry, _read_pinned(pins, ROOT / path, entry["sha256"]))
        inventory_rows[key[0], key[1], "authority"] = rows
        if key[1] in BRIDGE_CSV_SEASONS:
            labels[key] = (rows, "AUTHORITATIVE_CSV")

    # 2025 processed files are label evidence for review only. Provider remains the match authority.
    inventory = json.loads(_read_pinned(pins, evidence / PROCESSED_PINS[0], PROCESSED_PINS[1]))
    spans, fixture_chains = {}, defaultdict(set)
    for path, league, season, pin in processed_seed_sources(ROOT, inventory):
        records, identity = read_metadata(path, league, season, expected_sha256=pin)
        _record_pin(pins, path, identity["sha256"])
        metadata = [(r["day"], r["home"], r["away"]) for r in records]
        inventory_rows[league, season, "processed"] = metadata
        if season == 2025:
            labels[league, season] = (metadata, "PROCESSED_LABELS_ONLY")
        if season < int(CSV_START[:4]):
            record_seed_appearances(spans, league, records, fixture_chains)
    return labels, spans, fixture_chains, inventory_rows, authority


def _date_crosswalk(rows, fixtures):
    """Resolve a label only when every dated home/away appearance allows the same single ID."""
    by_date_side, occurrences = defaultdict(set), defaultdict(list)
    for day, home, away in fixtures:
        by_date_side[day, "home"].add(home)
        by_date_side[day, "away"].add(away)
    for day, home, away in rows:
        occurrences[home].append(by_date_side[day, "home"])
        occurrences[away].append(by_date_side[day, "away"])
    return {label: ids for label, options in occurrences.items() if len(ids := set.intersection(*options)) == 1}


def _review_rosters(matches, names, exact, labels, decisions, historical):
    recorded_labels, rosters = defaultdict(list), []
    dated = {league: _date_crosswalk(rows, matches[league, season].values())
             for (league, season), (rows, _) in labels.items() if season == 2024}
    for (league, season), (rows, source) in sorted(labels.items()):
        games = matches.get((league, season), {}).values()
        official = {team for _, home, away in games for team in (home, away)}
        assigned, unresolved = set(), []
        source_labels = {label for _, home, away in rows for label in (home, away)}
        for label in sorted(source_labels):
            key = league, normalize_name(label)
            if season == 2025 and key in decisions.splits:
                unresolved.append(label)
                continue
            ids = ({decisions.aliases[key]} if key in decisions.aliases
                   else exact.get(key, set()) | historical.get(key, set()))
            ids = ids & official if official else ids
            dated_ids = dated[league].get(label)
            if dated_ids and season in (2023, 2024):
                ids = ids & dated_ids if ids else dated_ids
            if len(ids) != 1:
                unresolved.append(label)
                continue
            team, = ids
            assigned.add(team)
            recorded_labels[league, team, season].append(label)
        roster = official or assigned
        for team in roster:
            names.setdefault((league, team), "")
        counts = Counter(team for _, home, away in games for team in (home, away))
        rosters.append({"league": league, "season": season, "expected_clubs": SEASON_CLUBS[league],
                        "source": "PROVIDER_CAPTURE" if official else "CSV_CROSSWALK_CANDIDATE",
                        "label_source": source, "observed_clubs": len(roster),
                        "count_matches": len(roster) == SEASON_CLUBS[league], "unresolved_csv_labels": unresolved,
                        "clubs": [{"provider_id": team, "name": names[league, team] or None,
                                   "csv_labels": recorded_labels[league, team, season],
                                   "provider_fixture_count": counts.get(team)} for team in sorted(roster)]})
    return rosters, recorded_labels


def _bridge_binding_rows(replay, names, exact, labels, historical, proposals, published, seed, spans, cutoff,
                         pre_seed_labels):
    first = {}
    for (league, season), games in replay.items():
        for fixture_id, (day, home, away) in games.items():
            for team in (home, away):
                key = league, team
                appearance = (day, fixture_id, season)
                first[key] = min(first.get(key, appearance), appearance)

    tokens = defaultdict(set)
    for league, token in exact.keys() | historical.keys():
        ids = exact.get((league, token), set()) | historical.get((league, token), set())
        if len(ids) == 1:
            team, = ids
            tokens[league, team].add(token)
    for (league, team, _), values in labels.items():
        tokens[league, team].update(map(normalize_name, values))
    suggested = {(row["league"], row["id"]): row for row in proposals["rows"]}
    frozen = defaultdict(lambda: defaultdict(set))
    for row in seed["states"]:
        label = row["team_id"]
        frozen[row["league"]][normalize_name(label)].add(label)
    prior_tokens = {(league, normalize_name(label)) for league, label in spans.keys() | pre_seed_labels}

    rows = []
    for (league, team), (day, fixture_id, season) in sorted(first.items()):
        key = league, team
        proposal = suggested.get(key, {})
        supported, possible = proposal_seed_tokens(proposal)
        supported.update(tokens[key] | {str(team)})
        seed_labels = set().union(*(frozen[league].get(token, set()) for token in supported | possible))
        last = {label: spans.get((league, label), (None, None))[1] for label in sorted(seed_labels)}
        latest = latest_seed_chain(league, seed_labels, spans)
        live = published.get(key, {})
        disposition = live.get("disposition", "OUTSIDE_CURRENT_MAPPING")
        live_label = live.get("legacy_label")
        selected, finding, hold, evidence = None, None, None, None
        seed_window_carry = (disposition == "OWNER_SEED_WINDOW_ONLY" and live_label is not None
                             and latest == live_label and last[latest] >= cutoff
                             and live.get("valid_through") == SEED_DAY)
        if seed_window_carry:
            selected = live_label
        elif seed_labels and not latest:
            hold = ("SEED_APPEARANCE_UNVERIFIED" if None in last.values()
                    else "OVERLAPPING_OR_SAME_SEASON_CHAINS")
        elif latest and last[latest] < cutoff and (
                normalize_name(latest) in supported or disposition == "COLD_START_STALE"):
            finding = "STALE_BEYOND_THRESHOLD"
            evidence = {"last_seed_appearance": last[latest], "threshold": cutoff, "source_sha256": PROCESSED_PINS[1]}
        elif latest and normalize_name(latest) not in supported:
            hold = "LATEST_CHAIN_NAME_UNSUPPORTED"
        elif latest:
            selected = latest
        elif any((league, token) in prior_tokens for token in supported):
            hold = "INCUMBENT_WITHOUT_SEED_LABEL"
        else:
            finding = "NO_SEED_LABEL"

        flags = [] if live else ["OUTSIDE_LIVE_MAPPING"]
        if finding == "STALE_BEYOND_THRESHOLD" and normalize_name(latest) not in supported:
            flags.append("REVIEWED_SEED_ALIAS_REQUIRED")
        cold_disposition = "COLD_START_STALE" if finding == "STALE_BEYOND_THRESHOLD" else "COLD_START_NO_SUPPORTED_STATE"
        if seed_window_carry or (hold and disposition in mapping.BIND_ACTIONS):
            flags.append("BRIDGE_LIVE_DISAGREEMENT")
        elif live and not hold and ((disposition, live_label) != ("BIND_LATEST_STATE", selected) if selected
                                    else disposition not in (cold_disposition, "OWNER_WITHHELD_COLD_START")):
            flags.append("BRIDGE_LIVE_DISAGREEMENT")
            hold = "RULE_CONFLICTS_WITH_PUBLICATION"
        if hold:
            flags.append("BRIDGE_BINDING_UNRESOLVED")
        treatment = ("Hold for review" if hold else SEED_WINDOW_CARRY if seed_window_carry
                     else "Carry state" if selected else "Fresh (proposed)")
        older = [label for label in last if label != latest] if latest and not hold and selected else []
        last_day = max((date.fromisoformat(value) for value in last.values() if value), default=None)
        missed_seasons = season - 2024
        if last_day is not None:
            last_season = last_day.year - (last_day.month < 7)
            missed_seasons = max(missed_seasons, season - last_season - 1)
        rows.append({"league": league, "id": team, "name": names[key], "first_day": day, "fixture_id": fixture_id,
                     "seed_label": selected, "seed_labels_found": last, "treatment": treatment, "live": disposition,
                     "older_chains": older, "finding": finding, "evidence": evidence, "hold": hold,
                     "live_label": live_label, "flag": ", ".join(flags) or "—", "absent_seasons": missed_seasons,
                     "absence_days": (date.fromisoformat(day) - last_day).days if last_day else None})
    return rows


def _seed_review_sections(rows):
    counts = Counter(row["treatment"] for row in rows)
    lines = ["", "## Bridge seed bindings", "",
             f"First finished provider fixture strictly before `{REPLAY_THROUGH_EXCLUSIVE}`. Treatments are proposals for explicit review.",
             "", f"{len(rows)} replaying clubs. {counts['Carry state'] + counts[SEED_WINDOW_CARRY]} carry bindings, "
             f"{counts['Fresh (proposed)']} cold-start proposals, {counts['Hold for review']} held.",
             "", "| League | Official ID | Official name | Rule result | Exists in frozen seed | Last seed appearance | First replay (UTC day / ID) | Treatment | Published live disposition / label | Flag |",
             "|---|---:|---|---|---|---|---|---|---|---|"]
    for row in rows:
        last = row["seed_labels_found"]
        binding = row["seed_label"] or (f"NO_USABLE_STATE / {row['finding']}" if row["finding"] else "Review required")
        dates = ", ".join(f"{label}: {day or 'unverified'}"
                          + (" (older chain, not carried)" if label in row["older_chains"] else "")
                          for label, day in last.items()) or "—"
        live = row["live"] + (f" / {row['live_label']}" if row["live_label"] else "")
        lines.append(f"| {row['league']} | {row['id']} | {row['name']} | {binding} | {'Yes' if last else 'No match found'} | "
                     f"{dates} | {row['first_day']} / {row['fixture_id']} | {row['treatment']} | {live} | {row['flag']} |")
        if row["evidence"]:
            evidence = row["evidence"]
            lines.append(f"| | | | Latest {evidence['last_seed_appearance']}; cutoff {evidence['threshold']}; "
                         f"appearance source `{evidence['source_sha256']}` | | | | | | |")

    lines.extend(["", "### All clubs held for review", "", "| Club (league / ID) | Reason |", "|---|---|"])
    lines.extend(f"| {row['name']} ({row['league']} / {row['id']}) | {row['hold']} |"
                 for row in rows if row["hold"])

    lines.extend(["", "## Returning clubs and cold starts", "",
                  "A returning club has at least one complete season without an evidenced appearance before its first replay fixture. "
                  "Days run from the latest evidenced pre-seed appearance to that fixture. When no seed date is evidenced, "
                  "the full-season count covers only the post-seed captures.", ""])
    for heading, group in (
        ("Returning clubs carrying state", [row for row in rows if row["absent_seasons"] and row["seed_label"] and not row["hold"]]),
        ("All proposed cold starts", [row for row in rows if row["treatment"] == "Fresh (proposed)"]),
        ("Returning clubs held for review", [row for row in rows if row["absent_seasons"] and row["treatment"] == "Hold for review"]),
    ):
        lines.extend([f"### {heading}", "", "| Club (league / ID) | Full seasons missed | Days since last seed appearance |",
                      "|---|---:|---:|"])
        lines.extend(f"| {row['name']} ({row['league']} / {row['id']}) | {row['absent_seasons']} | {row['absence_days'] or '—'} |"
                     for row in group)
        lines.append("")
    return lines


def _stale_rule_summary(proposals, cutoff):
    return (f"Published stale threshold: `{cutoff}` inclusive. "
            f"Rule: {proposals['binding_rule']}. Register entry: `{STALE_RULE_REGISTER[0]}#binding_rule`, "
            f"SHA-256 `{STALE_RULE_REGISTER[1]}`. Threshold date: pinned binding proposals `recent_cutoff_inclusive`. "
            "Older appearances qualify for a verified stale finding.")


def _alias_evidence(inventory, spans, names, exact, historical, labels, published):
    owners = defaultdict(set)
    for source in (exact, historical):
        for key, ids in source.items():
            owners[key].update(ids)
    for (league, team, _), values in labels.items():
        for label in values:
            owners[league, normalize_name(label)].add(team)
    for (league, team), row in published.items():
        if row.get("legacy_label"):
            owners[league, normalize_name(row["legacy_label"])].add(team)
    for (league, team), aliases in PROPOSED_ALIASES.items():
        for label in aliases:
            owners[league, normalize_name(label)].add(team)

    result = []
    for (league, team), (csv_label, seed_label) in sorted(PROPOSED_ALIASES.items()):
        known = {normalize_name(csv_label), normalize_name(seed_label)}
        by_season = []
        for (code, season, source), rows in sorted(inventory.items()):
            if code == league:
                found = sorted({label for _, home, away in rows for label in (home, away)
                                if normalize_name(label) in known})
                if found:
                    by_season.append({"season": season, "source": source, "labels": found})
        result.append({"league": league, "provider_id": team, "provider_name": names[league, team],
                       "seed_label": seed_label, "last_seed_appearance": spans[league, seed_label][1],
                       "csv_by_season": by_season,
                       "other_provider_ids": sorted(set().union(*(owners[league, label] for label in known)) - {team}),
                       "review_status": OWNER_DECISIONS["status"]})
    return result


def _espanyol_report(binding, fixture_chains, recorded, decision_ref, matched):
    league = binding["league"]
    excluded, carried = recorded["legacy_label"], binding["live_label"]
    excluded_chain = fixture_chains.get((league, excluded), set())
    carried_chain = fixture_chains.get((league, carried), set())
    dates = binding["seed_labels_found"]
    report = ["# Espanyol frozen seed chain comparison", "",
              f"Exact CSV chains through {SEED_DAY}. The reader's exception checks {'pass' if matched else 'block'}.",
              f"Published binding proposed for the bridge: `{carried}`. Decision SHA-256: `{decision_ref['sha256']}`.",
              f"Recorded excluded fixture: `{recorded['track_b_fixture_id']}` (count {recorded['excluded_fixture_count']}).",
              f"Last seed appearances: `{carried}` {dates.get(carried)}, `{excluded}` {dates.get(excluded)}.",
              recorded["one_sided_update"]["reason"], f"Revisit when: {recorded['one_sided_update']['revisit_when']}", ""]
    for label, rows in ((excluded, excluded_chain - carried_chain), (carried, carried_chain - excluded_chain)):
        report.extend([f"## {label} only ({len(rows)})", "", "| Date | Venue | Opponent |", "|---|---|---|"])
        for day, home, away in sorted(rows):
            side, opponent = ("home", away) if home == label else ("away", home)
            report.append(f"| {day} | {side} | {opponent} |")
        report.append("")
    return "\n".join(report)


def _file_ref(pins, path, expected=None):
    if expected is None:
        expected = mapping.digest(path.read_bytes())
        _record_pin(pins, path, expected)
    else:
        _read_pinned(pins, path, expected)
    return {"path": path.relative_to(ROOT).as_posix(), "sha256": expected}


def _artifact_ref(folder, name, raw):
    return {"path": (folder / name).relative_to(ROOT).as_posix(), "sha256": mapping.digest(raw)}


def _bridge_csv_labels(labels, alias_by_club):
    csv_bindings = {(league, season, label): team for (league, team, season), values in labels.items()
                    if season in BRIDGE_CSV_SEASONS for label in values}
    alias_keys = set()
    for (league, team), row in alias_by_club.items():
        if row["other_provider_ids"]:
            raise ValueError(f"Draft alias collides with another official ID: {league}/{team}")
        for season in row["csv_by_season"]:
            if season["season"] not in BRIDGE_CSV_SEASONS:
                continue
            for label in season["labels"]:
                key = league, season["season"], label
                if csv_bindings.setdefault(key, team) != team:
                    raise ValueError(f"Draft CSV alias conflicts with review crosswalk: {league}/{team}")
                alias_keys.add(key)
    return [{"league": league, "season": season, "label": label, "provider_id": team,
             **({"review_status": OWNER_DECISIONS["status"]} if key in alias_keys else {})}
            for key, team in sorted(csv_bindings.items()) for league, season, label in (key,)]


def _bridge_draft(folder, artifacts, pins, verification_path, *, authority, sources, matches, captures,
                  rosters, csv_labels, bindings, alias_by_club, release, evidence):
    # Draft copies add verified provider season endings beside the original authority pin.
    completed = [row for row in rosters if row["season"] in (2024, 2025)]
    if len(completed) != 2 * len(SEASON_CLUBS) or not all(row["count_matches"] for row in completed):
        raise ValueError("Draft bridge requires complete 2024/25 and 2025/26 roster counts")
    endings = [{"league": league, "season": season, "max_date": max(day for day, *_ in games.values())}
               for (league, season), games in sorted(matches.items()) if season == 2025]
    manifest_name = "source-authority.UNAPPROVED.json"
    authority_ref = _file_ref(pins, evidence / AUTHORITY[0], AUTHORITY[1])
    artifacts[manifest_name] = mapping.encode({**authority, "draft_base_authority": authority_ref,
                                               "provider_season_max_dates": endings})
    manifest_ref = _artifact_ref(folder, manifest_name, artifacts[manifest_name])

    seed_bindings, blockers = [], []
    for row in bindings:
        key = row["league"], row["id"]
        binding = {"league": key[0], "provider_id": key[1], "seed_label": row["seed_label"]}
        alias = alias_by_club.get(key)
        if alias:
            binding.update(seed_aliases=[alias["seed_label"]], alias_review_status=alias["review_status"])
            if row["hold"] == "INCUMBENT_WITHOUT_SEED_LABEL":
                binding["seed_label"] = alias["seed_label"]
            elif row["hold"]:
                blockers.append({"league": key[0], "provider_id": key[1], "reason": row["hold"]})
        elif row.get("proposal_evidence"):
            binding["proposal_evidence"] = row["proposal_evidence"]
        elif row["hold"] or key == mapping.ESPANYOL:
            blockers.append({"league": key[0], "provider_id": key[1],
                             "reason": row["hold"] or "ESPANYOL_DEBT_EXCEPTION_UNVERIFIED"})
        if key == release:
            binding["binding_basis"] = "PUBLISHED_SEED_WINDOW_BINDING"
        if row["finding"]:
            binding.update(finding="NO_USABLE_STATE", reason=row["finding"])
            if row["evidence"]:
                binding["evidence"] = row["evidence"]
        seed_bindings.append(binding)

    tail = {league: sum(SEED_DAY < day < CSV_START for day, *_ in sources[league, 2023][0])
            for league in sorted(SEASON_CLUBS)}
    if any(tail.values()):
        raise ValueError("June tail requires reviewed official fixture IDs")
    model = _file_ref(pins, evidence / FROZEN_SEED[0], FROZEN_SEED[1])
    register = _file_ref(pins, evidence / STALE_RULE_REGISTER[0], STALE_RULE_REGISTER[1])
    proposal_path, proposal_pin = mapping.APPROVED_INPUTS["proposals"]
    identity = {"schema_version": IDENTITY_REVIEW_SCHEMA, "status": "UNAPPROVED", "leagues": sorted(SEASON_CLUBS),
                "through_exclusive": REPLAY_THROUGH_EXCLUSIVE, "model_sha256": model["sha256"],
                "source_manifest_sha256": manifest_ref["sha256"], "seed_bindings": seed_bindings,
                "csv_labels": csv_labels, "tail_fixtures": [], "tail_expected_counts": tail, "blockers": blockers,
                "owner_decisions": OWNER_DECISIONS,
                "seed_appearance_sources": _file_ref(pins, evidence / PROCESSED_PINS[0], PROCESSED_PINS[1]),
                "stale_rule": {"register": register, "proposals": _file_ref(pins, proposal_path, proposal_pin)},
                "completed_season_rosters": [
                    {"league": row["league"], "season": row["season"], "clubs": [
                        {"provider_id": club["provider_id"], "name": club["name"]} for club in row["clubs"]]}
                    for row in completed]}
    identity_name = "identity-review.UNAPPROVED.json"
    reader = _file_ref(pins, ROOT / READER)
    features = _file_ref(pins, ROOT / FEATURES)
    provider = _file_ref(pins, ROOT / PROVIDER_IMPLEMENTATION)
    verification = _file_ref(pins, verification_path)
    verification_doc = json.loads(_held_bytes(verification_path, verification["sha256"]))
    if (verification_doc.get("status") != "PASS" or verification_doc.get("bridge_reader_sha256") != reader["sha256"]
            or verification_doc.get("feature_implementation_sha256") != features["sha256"]
            or verification_doc.get("provider_implementation_sha256") != provider["sha256"]):
        raise ValueError("Offline verification must pass and match the current reader, features and provider rule")
    old, refreshed = sorted(CAPTURES)
    def provider_manifest(capture, league, season):
        path = f"data/raw/football_data_org/{capture}/manifest.json"
        return {"manifest": {"path": path, "sha256": CAPTURES[capture]},
                "expected_fixtures": captures[capture, league, season][0]}

    provider_sources = []
    for league, season in sorted(matches):
        capture = refreshed if (refreshed, league, season) in captures else old
        provider_sources.append({"league": league, "season": season, **provider_manifest(capture, league, season),
                                 **({"gap_baseline": provider_manifest(old, league, season)} if capture == refreshed else {})})
    artifacts[identity_name] = mapping.encode(identity)
    plan = {"schema_version": 1, "status": "UNAPPROVED", "leagues": identity["leagues"],
            "through_exclusive": REPLAY_THROUGH_EXCLUSIVE, "observed_as_of": max(when for _, when in captures.values()),
            "source_manifest": manifest_ref, "provider_sources": provider_sources,
            "model": model, "bridge_reader": reader, "feature_implementation": features,
            "recipe": _file_ref(pins, ROOT / FROZEN_RECIPE[0], FROZEN_RECIPE[1]),
            "owner_grant": register, "offline_verification": verification,
            "identity_review": _artifact_ref(folder, identity_name, artifacts[identity_name])}
    artifacts["bridge-plan.UNAPPROVED.json"] = mapping.encode(plan)


def build_review(folder, release, *, offline_verification=None):
    folder = (ROOT / folder).resolve()
    scratch = (ROOT / "tmp").resolve()
    if folder == scratch or not folder.is_relative_to(scratch):
        raise ValueError("Review packet must be a new folder under tmp/")
    receipt = {"input_sha256": {}}
    inputs = mapping.read_inputs(mapping.APPROVED_INPUTS, receipt)
    hashes = receipt["input_sha256"]
    pins = {}
    for name, (path, _) in mapping.APPROVED_INPUTS.items():
        _record_pin(pins, path, hashes[name])
    candidate = mapping.derive_mapping(inputs, hashes)
    baseline = mapping.DESTINATION.read_bytes()
    published_entries = json.loads(baseline)["entries"]
    mapping_counts = _check_mapping_counts(published_entries)
    if mapping.encode(candidate) != baseline:
        raise ValueError("Current approved inputs do not reproduce the publication")
    _record_pin(pins, mapping.DESTINATION, mapping.digest(baseline))
    published = {(row["league"], row["provider_id"]): row for row in published_entries}
    entries = {(row["league"], row["provider_id"]): row for row in candidate["entries"]}
    entry = entries[release]
    if entry["disposition"] != "OWNER_SEED_WINDOW_ONLY":
        raise ValueError("Selected review binding is already unrestricted")
    candidate.update(review_status="UNAPPROVED_OWNER_REVIEW", publication_allowed=False,
                     baseline_sha256=mapping.digest(baseline), live_release={
                         "league": release[0], "provider_id": release[1], "status": "PENDING_BRIDGE_RECEIPT",
                         "bridge_receipt_sha256": None,
                     })

    evidence = mapping.REVISION.parents[1]
    decision_path = evidence / "NEXT-37d-owner-decisions-scheduler-hold-01/owner-alias-decisions.json"
    alias_path = evidence / "NEXT-37b-normaliser-probe-02/alias-review-sheet-v3.csv"
    quarantine_path = alias_path.with_name("quarantine-dry-run.json")
    decisions = load_decisions(decision_path, alias_path, quarantine_path=quarantine_path)
    for identity in decisions.identities:
        _record_pin(pins, Path(identity["path"]), identity["sha256"])
    _record_pin(pins, quarantine_path, QUARANTINE_SHA256)
    historical = defaultdict(set)
    raw = _read_pinned(pins, evidence / CROSSWALK[0], CROSSWALK[1])
    for row in csv.DictReader(io.StringIO(raw.decode("utf-8-sig"))):
        ids = row["candidate_provider_ids"]
        if row["status"] == "EXACT_CANDIDATE_NOT_APPLIED" and ids.isascii() and ids.isdecimal():
            historical[row["league"], normalize_name(row["source_name"])].add(int(ids))
    matches, names, exact, replay, capture_inventory, void_schedule = _capture_metadata(pins)
    sources, spans, fixture_chains, inventory, authority = _csv_labels(evidence, pins)
    seed = json.loads(_read_pinned(pins, evidence / FROZEN_SEED[0], FROZEN_SEED[1]))["model"]["feature_seed"]
    if seed["last_day"] != SEED_DAY:
        raise ValueError("Frozen seed cutoff does not match bridge cutoff")
    register = json.loads(_read_pinned(pins, evidence / STALE_RULE_REGISTER[0], STALE_RULE_REGISTER[1]))
    cutoff = approved_stale_cutoff(register, inputs["proposals"])
    rosters, labels = _review_rosters(matches, names, exact, sources, decisions, historical)
    expected = {(league, season) for league in SEASON_CLUBS for season in (2023, 2024, 2025)}
    if {(row["league"], row["season"]) for row in rosters} != expected:
        raise ValueError("Missing league-season label source")
    aliases = _alias_evidence(inventory, spans, names, exact, historical, labels, published)
    alias_by_club = {(row["league"], row["provider_id"]): row for row in aliases}
    csv_labels = _bridge_csv_labels(labels, alias_by_club)
    label_index = _reviewed_csv_labels(csv_labels)
    csv_keys = {csv_fixture_key(league, season, day, home, away, label_index)
                for (league, season), (rows, _) in sources.items() if season in BRIDGE_CSV_SEASONS
                for day, home, away in rows}
    exclusions = fixture_exclusions(void_schedule, csv_keys)
    debt_ref = espanyol_decision_reference()
    pre_seed_labels = {(league, label) for (league, season), (rows, _) in sources.items() if season == 2023
                       for day, home, away in rows if day <= SEED_DAY for label in (home, away)}
    binding_rows = _bridge_binding_rows(replay, names, exact, labels, historical, inputs["proposals"],
                                        published, seed, spans, cutoff, pre_seed_labels)
    espanyol, = [row for row in binding_rows if (row["league"], row["id"]) == mapping.ESPANYOL]
    carried = espanyol["live_label"]
    debt = inputs["espanyol_debt"]["decision_supplement"]
    chain_labels = set(espanyol["seed_labels_found"])
    debt_matches = (espanyol["hold"] == "OVERLAPPING_OR_SAME_SEASON_CHAINS"
                    and espanyol_seed_exception(ROOT, mapping.ESPANYOL, carried, chain_labels,
                                                spans, fixture_chains, debt_ref))
    if debt_matches:
        espanyol.update(seed_label=carried, hold=None, treatment="Carry state", proposal_evidence=debt_ref,
                        older_chains=sorted(chain_labels - {carried, debt["legacy_label"]}),
                        flag="ACCEPTED_ESPANYOL_FIXTURE_DEBT")
    espanyol_report = _espanyol_report(espanyol, fixture_chains, debt, debt_ref, debt_matches)
    artifacts = {"espanyol-fixture-diff.md": espanyol_report.encode("utf-8")}
    if offline_verification is not None:
        _bridge_draft(folder, artifacts, pins, (ROOT / offline_verification).resolve(), authority=authority,
                      sources=sources, matches=matches, captures=capture_inventory, rosters=rosters,
                      csv_labels=csv_labels, bindings=binding_rows, alias_by_club=alias_by_club,
                      release=release, evidence=evidence)

    summary = ["# Unapproved club identity review v6", "",
               "The SHA-256 of this exact sheet is in [owner-review.receipt.json](owner-review.receipt.json).",
               "", "Published mapping counts: " + ", ".join(f"{league} {mapping_counts[league]}" for league in sorted(SEASON_CLUBS)) + " (96 total).",
               "", "<details>", "<summary>Pinned inputs read by the generator</summary>", "",
               "| Input | SHA-256 |", "|---|---|"]
    summary.extend(f"| `{path}` | `{pin}` |" for path, pin in sorted(pins.items()))
    summary.extend(["", "</details>", "", _stale_rule_summary(inputs["proposals"], cutoff),
               "", "| League / season | Clubs | Unresolved CSV labels |",
               "|---|---:|---|"])
    for row in rosters:
        summary.append(f"| {row['league']}/{row['season']} | {row['observed_clubs']}/{row['expected_clubs']} | "
                       f"{', '.join(row['unresolved_csv_labels']) or '—'} |")
    name = names.get(release) or entry["legacy_label"]
    summary.extend(["", f"**{name} fixture evidence**", "", "| Season | Provider fixtures | CSV rows under linked labels |",
                    "|---|---:|---:|"])
    for season in (2023, 2024, 2025, 2026):
        games = matches.get((release[0], season))
        count = sum(release[1] in (home, away) for _, home, away in games.values()) if games else "No capture"
        club_labels = set(labels[release[0], release[1], season])
        csv_rows = sources.get((release[0], season), ([], None))[0]
        csv_count = sum(home in club_labels or away in club_labels for _, home, away in csv_rows) if club_labels else "—"
        summary.append(f"| {season}/{str(season + 1)[-2:]} | {count} | {csv_count} |")
    summary.extend(["", "### Live release", "",
                    f"{name} — pending, decided after the bridge receipt. The live release decision must cite the bridge receipt SHA-256.",
                    "", "| League | Official ID | Official name | Published live disposition | 2023/24 CSV | 2024/25 CSV | 2025/26 CSV |",
                    "|---|---:|---|---|---|---|---|"])
    for (league, team), name in sorted(names.items()):
        status = entries.get((league, team), {}).get("disposition", "OUTSIDE_CURRENT_MAPPING")
        values = [", ".join(labels[league, team, season]) or "—" for season in (2023, 2024, 2025)]
        summary.append(f"| {league} | {team} | {name or 'Name unavailable'} | {status} | {' | '.join(values)} |")
    summary.extend(["", "A dash indicates no unique crosswalk label for that season. The 2025/26 labels are review evidence only.",
                    "The 2024/25 date and home/away crosswalk is proposed evidence for exact labels also found in 2023/24.",
                    "A missing 2023/24 ID needs independent evidence. No 2026/27 CSV is in scope."])
    summary.extend(_seed_review_sections(binding_rows))
    summary.extend(["", "## Aliases stated by the owner, pending formal approval", "",
                    "The collision check covers the pinned league catalog, crosswalk, and published seed bindings.",
                    "", "| League / ID | Provider name | CSV labels by season and source | Frozen seed label | Last seed appearance | Other known club IDs sharing a label |",
                    "|---|---|---|---|---|---|"])
    for row in aliases:
        seasons = "; ".join(f"{item['season']}/{str(item['season'] + 1)[-2:]} ({item['source']}): {', '.join(item['labels'])}"
                            for item in row["csv_by_season"])
        summary.append(f"| {row['league']} / {row['provider_id']} | {row['provider_name']} | {seasons or 'None'} | "
                       f"{row['seed_label']} | {row['last_seed_appearance']} | "
                       f"{', '.join(map(str, row['other_provider_ids'])) or 'None found'} |")
    summary.extend(["", "## Espanyol seed binding", "",
                    "[Dates and opponents in both directions](espanyol-fixture-diff.md). "
                    f"The reader's exception checks {'pass' if debt_matches else 'block'} against the decision "
                    f"pinned at `{debt_ref['sha256']}`. "
                    "The recorded one-sided fixture debt remains in the frozen seed."])
    summary.extend(["", "## Excluded fixtures", "",
                    "The round-robin counts these fixtures as accounted. They contribute no goals or state updates.",
                    "", "| League / season | UTC date | Fixture ID | Home ID | Away ID | Provider status | Disposition | CSV match |",
                    "|---|---|---:|---:|---:|---|---|---|"])
    for row in exclusions:
        csv_match = "Yes" if row["matched_csv"] else "No"
        summary.append(f"| {row['league']}/{row['season']} | {row['day']} | {row['fixture_id']} | {row['home_id']} | "
                       f"{row['away_id']} | {row['status']} | {row['disposition']} | {csv_match} |")
    summary.extend(["", "The frozen seed's handling of awarded matches before 2024-06-03 is unevidenced.",
                    "", "## Owner decisions pending formal approval", "",
                    f"{OWNER_DECISIONS['stated_by']}, {OWNER_DECISIONS['stated_on']}", "",
                    f"> {OWNER_DECISIONS['verbatim']}"])

    folder.mkdir(exist_ok=False)
    artifacts.update({
        "completed-season-rosters.json": mapping.encode({"review_status": "UNAPPROVED_OWNER_REVIEW",
            "source_pins": {"captures": CAPTURES, "authority": AUTHORITY[1], "crosswalk": CROSSWALK[1],
                            "processed_label_sources": PROCESSED_PINS[1]}, "rosters": rosters}),
        "provider-team-mapping.UNAPPROVED.json": mapping.encode(candidate),
        "owner-review.md": ("\n".join(summary) + "\n").encode("utf-8"),
    })
    artifacts["owner-review.receipt.json"] = mapping.encode({"schema_version": 6, "sheet": "owner-review.md",
        "sha256": mapping.digest(artifacts["owner-review.md"]), "input_sha256": pins,
        "artifact_sha256": {name: mapping.digest(raw) for name, raw in artifacts.items()}})
    for name, raw in artifacts.items():
        (folder / name).write_bytes(raw)
        print(mapping.digest(raw), name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("packet", type=Path)
    parser.add_argument("league", choices=SEASON_CLUBS)
    parser.add_argument("provider_id", type=int)
    parser.add_argument("--offline-verification", type=Path, help="Pin an offline PASS receipt and include the draft plan")
    args = parser.parse_args()
    build_review(args.packet, (args.league, args.provider_id), offline_verification=args.offline_verification)


if __name__ == "__main__":
    main()
