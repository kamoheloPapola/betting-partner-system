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
PD_SHA256 = "12dc208fbc4e586fb243255721bdca8a330c6f71ce33c637e5d3658c8360e2cc"
QUARANTINE = MappingProxyType({
    ("PD", 2025, "2026-01-04", "REAL MADRID", "REAL MADRID"): 2,
    ("PD", 2025, "2026-04-04", "REAL MADRID", "BARCELONA"): 2,
})


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


def load_decisions(decision_path, v3_path, *, expected_decision=DECISIONS_SHA256,
                   expected_v3=V3_SHA256):
    # Both files are verified BEFORE either document is parsed; parse these same bytes.
    decision_raw, decision_pin = held_bytes(decision_path)
    v3_raw, v3_pin = held_bytes(v3_path)
    if decision_pin["sha256"] != expected_decision or v3_pin["sha256"] != expected_v3:
        raise MatchError("Owner decision or accepted v3 hash mismatch")
    document = json.loads(decision_raw)
    sheet = list(csv.DictReader(io.StringIO(v3_raw.decode("utf-8-sig"))))
    records = document.get("rows", [])
    if len(records) != 22 or len(sheet) != 22 or document.get("source_v3", {}).get("sha256") != expected_v3:
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
              and lookup == ("PD", normalize_name("BARCELONA")) and tid == 81
              and record.get("assignment_decision") == "UNRESOLVED_NO_ASSIGNMENT_APPROVED"
              and record.get("unresolved_disposition") == "EXCLUDE_AND_COUNT_FAIL_CLOSED"
              and record.get("unresolved_rows") == 2):
            target = splits
        else:
            raise MatchError("Decision is not the accepted approval/disposition")
        if lookup in aliases or lookup in splits:
            raise MatchError("Conflicting normalized approval key")
        target[lookup] = tid
    if len(aliases) != 21 or len(splits) != 1:
        raise MatchError("Expected 21 aliases and one label split")
    return Decisions(MappingProxyType(aliases), MappingProxyType(splits), (decision_pin, v3_pin))


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
    if expected_sha256 == PD_SHA256:
        by_line = {r["line"]: row_key(r) for r in result}
        expected = {179: next(iter(QUARANTINE)), 182: next(iter(QUARANTINE)),
                    295: list(QUARANTINE)[1], 296: list(QUARANTINE)[1]}
        if any(by_line.get(n) != key for n, key in expected.items()):
            raise MatchError("Pinned PD row-location/key disagreement")
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
    if row_key(row) in QUARANTINE:
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
    results = [match_row(r, fixtures_by_league[r["league"]], decisions) for r in rows]
    excluded = Counter(tuple(r["key"]) for r in results if r["status"] == "EXCLUDED")
    if excluded != Counter(QUARANTINE) or sum(excluded.values()) != 4:
        raise MatchError(f"Expected exactly four physical exclusions at the two approved keys; got {sum(excluded.values())}")
    return results

