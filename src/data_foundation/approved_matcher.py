"""Owner-approved, score-blind identity matching. No fuzzy matching or import-time IO."""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
from dataclasses import dataclass
from datetime import date
import hashlib
import io
import json
import os
from pathlib import Path
from types import MappingProxyType

from .identity import normalize_name, provider_id
from .audit import project_metadata
from .repair_plan import GROUPS

DECISIONS_SHA256 = "faf9c7177266ff4f6ce134049cbb45a13d5085667e59101b79318411fa907d19"
V3_SHA256 = "1ec07fdca3f32678d354f0a6d8e7c784bdc68ef43b58a937e812cf4241dc1d22"
QUARANTINE_SHA256 = "1cd0c45168aa0154b2662011a20ff12e84ce1ec7a837f3a7efb67b5147dd9ab5"


class MatchError(ValueError):
    """A required identity cannot be established without guessing."""


def held_bytes(path):
    path = Path(path)
    if path.name.startswith(".env"):
        raise MatchError("Secret configuration is not a matcher input")
    with path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        raw = stream.read()
        after = os.fstat(stream.fileno())
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns) or len(raw) != before.st_size:
        raise MatchError(f"Input changed during held read: {path}")
    return raw, {"path": str(path.resolve()), "size_bytes": len(raw),
                 "sha256": hashlib.sha256(raw).hexdigest()}


@dataclass(frozen=True)
class Decisions:
    aliases: object
    splits: object
    identities: tuple
    quarantine: object
    quarantine_lines: object


def load_decisions(decision_path, v3_path, *, expected_decision=DECISIONS_SHA256,
                   expected_v3=V3_SHA256, quarantine_path=None, expected_quarantine=QUARANTINE_SHA256):
    # Verify all pins before parsing the same held bytes.
    decision_raw, decision_pin = held_bytes(decision_path)
    v3_raw, v3_pin = held_bytes(v3_path)
    if decision_pin["sha256"] != expected_decision or v3_pin["sha256"] != expected_v3:
        raise MatchError("Owner decision or accepted v3 hash mismatch")
    quarantine_raw, quarantine_pin = held_bytes(quarantine_path or Path(v3_path).with_name("quarantine-dry-run.json"))
    if quarantine_pin["sha256"] != expected_quarantine:
        raise MatchError("Quarantine evidence hash mismatch")
    quarantine_doc = json.loads(quarantine_raw)
    if (quarantine_doc.get("rule") != "QUARANTINE_RULE_V1" or quarantine_doc.get("status") != "PASS"
            or quarantine_doc.get("applied") is not False or not isinstance(quarantine_doc.get("rows"), list)):
        raise MatchError("Invalid quarantine evidence")
    quarantine_lines = {}
    for row in quarantine_doc["rows"]:
        day = date.fromisoformat(row["date"])
        season = day.year if day.month >= 7 else day.year - 1
        location = row["source_sha256"], row["line"]
        if (location in quarantine_lines or type(row["line"]) is not int or row["line"] <= 1
                or row.get("season", season) != season or len(set(row["candidate_fixture_ids"])) < 2):
            raise MatchError("Invalid quarantine row")
        quarantine_lines[location] = (row["league"], season, day.isoformat(), row["original_home"], row["original_away"])
    document = json.loads(decision_raw)
    sheet = list(csv.DictReader(io.StringIO(v3_raw.decode("utf-8-sig"))))
    records = document.get("rows", [])
    if not records or len(records) != len(sheet) or document.get("source_v3", {}).get("sha256") != expected_v3:
        raise MatchError("Decision/v3 row count or provenance mismatch")
    by_key = {}
    for row in sheet:
        key = (row["league"], row["canonical_name"], row["candidate_id"], row["operation"])
        if key in by_key:
            raise MatchError("Duplicate v3 identity")
        by_key[key] = row
    aliases, splits, seen = {}, {}, set()
    for record in records:
        key = tuple(record.get(k) for k in ("league", "canonical_name", "candidate_id", "operation"))
        if key not in by_key or key in seen:
            raise MatchError("Unexpected or duplicate owner decision")
        seen.add(key)
        if record.get("source_v3_sha256") != expected_v3 or not record.get("reviewer") or not record.get("date_utc"):
            raise MatchError("Decision approval provenance missing")
        lookup = (record["league"], normalize_name(record["canonical_name"]))
        tid = provider_id(record["candidate_id"])
        if key[3] == "ALIAS_1_TO_1" and record.get("decision") == "APPROVED":
            target = aliases
        elif (key[3] == "LABEL_SPLIT" and record.get("decision") == "CONFIRMED_LABEL_SPLIT"
              and record.get("assignment_decision") == "UNRESOLVED_NO_ASSIGNMENT_APPROVED"
              and record.get("unresolved_disposition") == "EXCLUDE_AND_COUNT_FAIL_CLOSED"):
            recorded = record.get("unresolved_csv_lines")
            matching = [line for (_, line), fixture in quarantine_lines.items()
                        if fixture[0] == lookup[0] and lookup[1] in map(normalize_name, fixture[3:])]
            if (not isinstance(recorded, list) or not all(type(line) is int for line in recorded)
                    or not matching or sorted(matching) != sorted(recorded)
                    or record.get("unresolved_rows") != len(matching)):
                raise MatchError("Unresolved split does not match pinned quarantine rows")
            target = splits
        else:
            raise MatchError("Decision is not the accepted approval/disposition")
        if lookup in aliases or lookup in splits:
            raise MatchError("Conflicting normalized approval key")
        target[lookup] = tid
    return Decisions(MappingProxyType(aliases), MappingProxyType(splits), (decision_pin, v3_pin),
                     MappingProxyType(dict(Counter(quarantine_lines.values()))), MappingProxyType(quarantine_lines))


def read_metadata(path, league, season, *, expected_sha256=None):
    raw, pin = held_bytes(path)
    if expected_sha256 is not None and pin["sha256"] != expected_sha256:
        raise MatchError(f"Processed source hash mismatch: {path}")
    lines = raw.splitlines(keepends=True)
    if not lines:
        raise MatchError(f"Empty metadata source: {path}")
    header = next(csv.reader([lines[0].decode("utf-8-sig")]))
    wanted = {"date": "day", "home_team": "home", "away_team": "away"}
    if any(header.count(name) != 1 for name in wanted):
        raise MatchError(f"Missing/duplicate metadata columns: {path}")
    indices = {header.index(name): field for name, field in wanted.items()}
    result = []
    for line, record in enumerate(lines[1:], 2):
        fields = project_metadata(record, indices)
        try:
            day = date.fromisoformat(fields["day"]).isoformat()
            if not fields["home"].strip() or not fields["away"].strip():
                raise ValueError("Empty team")
        except (KeyError, ValueError) as exc:
            raise MatchError(f"Invalid metadata {path}:{line}") from exc
        result.append({"league": league, "season": int(season), "day": day,
                       "home": fields["home"], "away": fields["away"], "line": line,
                       "source_sha256": pin["sha256"]})
    return result, pin


def row_key(row):
    return (row["league"], int(row["season"]), row["day"], row["home"], row["away"])


def _catalog(fixtures):
    names, ids = defaultdict(set), set()
    for fixture in fixtures:
        for side in ("home", "away"):
            tid = provider_id(fixture[side + "_id"])
            ids.add(tid)
            meta = fixture[side + "_meta"]
            for field in ("name", "shortName", "tla"):
                value = meta.get(field)
                if isinstance(value, str) and value.strip():
                    names[normalize_name(value)].add(tid)
    return names, ids


def match_row(row, fixtures, decisions):
    names, valid = _catalog(fixtures)
    def lookup(label):
        key = (row["league"], normalize_name(label))
        approved = decisions.aliases.get(key)
        split = decisions.splits.get(key)
        if approved is not None or split is not None:
            tid = approved if approved is not None else split
            if tid not in valid:
                raise MatchError(f"Approved target absent in {row['league']}: {tid}")
            return {tid}
        return names.get(key[1], set())
    h, a = lookup(row["home"]), lookup(row["away"])
    if len(h) != 1 or len(a) != 1:
        raise MatchError(f"Unmatched/ambiguous label at {row['league']}:{row.get('line')}")
    groups = GROUPS.get(row["league"], {})
    home = set().union(*(groups.get(t, {t}) for t in h))
    away = set().union(*(groups.get(t, {t}) for t in a))
    candidates = [f for f in fixtures if f["date"] == row["day"] and
                  f["home_id"] in home and f["away_id"] in away]
    ids = [provider_id(f["id"]) for f in candidates]
    if len(ids) != len(set(ids)):
        raise MatchError("Duplicate provider fixture ID in candidate input")
    result = {"key": row_key(row), "line": row.get("line"), "source_sha256": row.get("source_sha256"),
              "candidate_fixture_ids": sorted(ids)}
    if row_key(row) in decisions.quarantine:
        if len(candidates) < 2:
            raise MatchError("Quarantine key no longer satisfies the ambiguous-fixture rule")
        return dict(result, status="EXCLUDED", reason="QUARANTINE_RULE_V1")
    if len(candidates) != 1:
        raise MatchError(f"Unexpected unresolved fixture at {row['league']}:{row.get('line')}")
    chosen = candidates[0]
    if chosen["home_id"] == chosen["away_id"]:
        raise MatchError("Self fixture resolved from provider metadata")
    return dict(result, status="MATCHED", fixture_id=chosen["id"],
                home_id=chosen["home_id"], away_id=chosen["away_id"])


def match_rows(rows, fixtures_by_league, decisions):
    rows = list(rows)
    locations = {(row.get("source_sha256"), row.get("line")): row_key(row) for row in rows}
    sources = {source for source, _ in locations}
    if any(source in sources and locations.get((source, line)) != key
           for (source, line), key in decisions.quarantine_lines.items()):
        raise MatchError("Pinned quarantine row-location/key disagreement")
    results = [match_row(r, fixtures_by_league[r["league"]], decisions) for r in rows]
    excluded = Counter(tuple(r["key"]) for r in results if r["status"] == "EXCLUDED")
    if excluded != Counter(decisions.quarantine):
        raise MatchError(f"Physical exclusions differ from pinned quarantine; got {sum(excluded.values())}")
    return results

