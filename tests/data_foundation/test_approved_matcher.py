"""Synthetic owner-decision gates and score-blind identity regression tests."""
from collections import Counter
import csv
import hashlib
import io
import json
import sys
import types
import pytest

from src.data_foundation import approved_matcher as m


@pytest.fixture
def documents(tmp_path):
    sheet = [{"league": "PL", "canonical_name": f"TEAM {i}", "candidate_id": str(100+i),
              "operation": "ALIAS_1_TO_1"} for i in range(21)]
    sheet.append({"league": "PD", "canonical_name": "BARCELONA", "candidate_id": "81",
                  "operation": "LABEL_SPLIT"})
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=list(sheet[0]))
    writer.writeheader()
    writer.writerows(sheet)
    raw = output.getvalue().encode()
    v3hash = hashlib.sha256(raw).hexdigest()
    records = []
    for row in sheet:
        record = dict(row, reviewer="SYNTHETIC_OWNER", date_utc="2026-01-01",
                      source_v3_sha256=v3hash)
        if row["operation"] == "ALIAS_1_TO_1":
            record["decision"] = "APPROVED"
        else:
            record.update(decision="CONFIRMED_LABEL_SPLIT", assignment_decision="UNRESOLVED_NO_ASSIGNMENT_APPROVED",
                          unresolved_rows=2, unresolved_disposition="EXCLUDE_AND_COUNT_FAIL_CLOSED")
        records.append(record)
    doc = {"source_v3": {"sha256": v3hash}, "rows": records}
    decision = tmp_path/"decisions.json"
    v3 = tmp_path/"v3.csv"
    v3.write_bytes(raw)
    def load(changed=None):
        decision.write_text(json.dumps(changed or doc), encoding="utf-8")
        return m.load_decisions(decision, v3,
                               expected_decision=hashlib.sha256(decision.read_bytes()).hexdigest(),
                               expected_v3=v3hash)
    return doc, decision, v3, v3hash, load


def fixture(mid, day, home, away):
    names = {86: "Real Madrid", 90: "Real Betis", 92: "Real Sociedad",
             78: "Atletico Madrid", 81: "FC Barcelona", 80: "Espanyol"}
    return {"id": mid, "date": day, "home_id": home, "away_id": away,
            "home_meta": {"name": names.get(home, f"Official {home}")},
            "away_meta": {"name": names.get(away, f"Official {away}")}}


@pytest.fixture
def pd_inputs(documents):
    decisions = documents[-1]()
    fixtures = [fixture(544385, "2026-01-04", 86, 90), fixture(544387, "2026-01-04", 92, 78),
                fixture(544502, "2026-04-04", 86, 81), fixture(544503, "2026-04-04", 90, 80)]
    rows = [{"league": k[0], "season": k[1], "day": k[2], "home": k[3], "away": k[4], "line": i}
            for i, k in enumerate([*list(m.QUARANTINE)[:1]*2, *list(m.QUARANTINE)[1:]*2], 1)]
    return decisions, fixtures, rows


def test_both_hashes_checked_before_parsing(documents, monkeypatch):
    _, decision, v3, _, load = documents
    load()
    monkeypatch.setattr(m.json, "loads", lambda _: pytest.fail("Parsed before both hash gates"))
    with pytest.raises(m.MatchError, match="hash mismatch"):
        m.load_decisions(decision, v3, expected_decision="0"*64, expected_v3="0"*64)


def test_approved_count_and_split_not_alias(documents):
    decisions = documents[-1]()
    assert len(decisions.aliases) == 21
    assert ("PD", "barcelona") not in decisions.aliases
    assert decisions.splits[("PD", "barcelona")] == 81
    with pytest.raises(TypeError):
        decisions.aliases[("PL", "new")] = 100


@pytest.mark.parametrize("change", ["duplicate", "missing", "reviewer", "hash", "rejected", "split_assignment"])
def test_malformed_decisions_fail_closed(documents, change):
    doc, _, _, _, load = documents
    if change == "duplicate": doc["rows"][1] = doc["rows"][0]
    elif change == "missing": doc["rows"].pop()
    elif change == "reviewer": doc["rows"][0]["reviewer"] = ""
    elif change == "hash": doc["rows"][0]["source_v3_sha256"] = "0"*64
    elif change == "rejected": doc["rows"][0]["decision"] = "REJECTED"
    else: doc["rows"][-1]["assignment_decision"] = "APPROVED"
    with pytest.raises(m.MatchError):
        load(doc)


def test_exact_approved_alias_precedes_other_name_evidence(documents):
    decisions = documents[-1]()
    f = fixture(1, "2026-08-01", 100, 101)
    other = fixture(2, "2026-08-02", 999, 101)
    other["home_meta"]["name"] = "TEAM 0"
    row = {"league": "PL", "season": 2026, "day": "2026-08-01", "home": " Team 0 ", "away": "TEAM 1"}
    assert m.match_row(row, [f, other], decisions)["home_id"] == 100


def test_unknown_no_fuzzy_fallback(documents, monkeypatch):
    trap = types.ModuleType("rapidfuzz")
    trap.process = types.SimpleNamespace(extractOne=lambda *a, **k: pytest.fail("Fuzzy called"))
    monkeypatch.setitem(sys.modules, "rapidfuzz", trap)
    row = {"league": "PL", "season": 2026, "day": "2026-08-01", "home": "TEAM 0 NEAR", "away": "TEAM 1"}
    with pytest.raises(m.MatchError, match="Unmatched"):
        m.match_row(row, [fixture(1, row["day"], 100, 101)], documents[-1]())


def test_target_missing_is_failure(documents):
    row = {"league": "PL", "season": 2026, "day": "2026-08-01", "home": "TEAM 0", "away": "TEAM 1"}
    with pytest.raises(m.MatchError, match="target absent"):
        m.match_row(row, [fixture(1, row["day"], 102, 103)], documents[-1]())


def test_four_physical_exclusions_not_two_keys(pd_inputs):
    decisions, fixtures, rows = pd_inputs
    results = m.match_rows(rows, {"PD": fixtures}, decisions)
    assert len(results) == 4 and all(x["status"] == "EXCLUDED" for x in results)
    assert Counter(tuple(x["key"]) for x in results) == Counter(m.QUARANTINE)


@pytest.mark.parametrize("count", [2, 3, 5])
def test_wrong_physical_count_is_hard_failure(pd_inputs, count):
    decisions, fixtures, rows = pd_inputs
    sample = rows[:count] if count < 4 else rows+[rows[-1]]
    with pytest.raises(m.MatchError, match="four physical"):
        m.match_rows(sample, {"PD": fixtures}, decisions)


def test_row_order_and_line_numbers_do_not_select(pd_inputs):
    decisions, fixtures, rows = pd_inputs
    for i, row in enumerate(rows):
        row["line"] = 9000+i
    result = m.match_rows(list(reversed(rows)), {"PD": list(reversed(fixtures))}, decisions)
    assert len(result) == 4


def test_unexpected_ambiguity_fails_not_quiet_exclusion(pd_inputs):
    decisions, fixtures, rows = pd_inputs
    bad = dict(rows[0], day="2026-01-05")
    fs = [dict(f, date="2026-01-05") for f in fixtures[:2]]
    with pytest.raises(m.MatchError, match="Unexpected unresolved"):
        m.match_row(bad, fs, decisions)


def test_quarantine_still_requires_ambiguous_anchor(pd_inputs):
    decisions, fixtures, rows = pd_inputs
    with pytest.raises(m.MatchError, match="no longer satisfies"):
        m.match_row(rows[0], fixtures[:1], decisions)


def test_score_poison_bytes_never_decoded(tmp_path):
    p = tmp_path/"PD_2025.csv"
    p.write_bytes(b"date,home_team,away_team,home_score,odds\n2026-01-01,A,B,\xff,\xfe\n")
    rows, _ = m.read_metadata(p, "PD", 2025)
    assert rows[0]["home"] == "A"


def test_csv_pin_before_decode(tmp_path):
    p = tmp_path/"PD_2025.csv"
    p.write_bytes(b"\xff")
    with pytest.raises(m.MatchError, match="hash mismatch"):
        m.read_metadata(p, "PD", 2025, expected_sha256="0"*64)


def test_no_score_status_or_time_reads(pd_inputs):
    class Guard(dict):
        def __getitem__(self, key):
            if any(t in key.lower() for t in ("score", "status", "time", "odds")):
                pytest.fail("Forbidden provider access")
            return super().__getitem__(key)
    decisions, fixtures, rows = pd_inputs
    assert len(m.match_rows(rows, {"PD": [Guard(f) for f in fixtures]}, decisions)) == 4

