from contextlib import contextmanager
from hashlib import sha256
import json
from pathlib import Path

import pytest

from scripts import build_canonical_mapping_review as review
from src.canonical import provider_team_mapping as mapping
from src.canonical.clock import digest


@pytest.fixture
def destination(tmp_path):
    return tmp_path / "canonical" / "provider-team-mapping.json"


def receipts(destination):
    folder = destination.parent / "mapping-publication-receipts"
    return [json.loads(path.read_bytes()) for path in sorted(folder.glob("*.json"))]


def last_failure(destination):
    failure, = [doc for doc in receipts(destination) if doc["result"] == "FAILED"]
    return failure


def plan_publish(destination):
    assert mapping.main(["dry-run"], destination=destination) == 0
    dry_run, = receipts(destination)
    assert dry_run["result"] == "DRY_RUN"
    return [
        "publish", "--baseline-sha256", dry_run["previous_sha256"] or "absent",
        "--candidate-sha256", dry_run["output_sha256"],
    ]


def seed_baseline(destination):
    destination.parent.mkdir()
    baseline = b'{"test":"previous baseline"}\r\n'
    destination.write_bytes(baseline)
    return baseline


def test_current_decisions_reproduce_publication_byte_for_byte():
    receipt = {"input_sha256": {}}
    inputs = mapping.read_inputs(mapping.APPROVED_INPUTS, receipt)
    candidate = mapping.encode(mapping.derive_mapping(inputs, receipt["input_sha256"]))
    assert sha256(candidate).hexdigest() == "1b5efbd10f6c09599eb822da15bdaaac01653a3e16045652fc5941a54fa8d5a7"


@pytest.mark.parametrize("fault", [None, "missing-diff", "wrong-pin", "extra-decision-field"])
def test_release_publisher_requires_pinned_derivation_of_signed_pending_bytes(destination, tmp_path, monkeypatch, fault):
    monkeypatch.setattr(mapping, "ROOT", tmp_path)
    source_pins = {"input_sha256": {}}
    inputs = mapping.read_inputs(mapping.APPROVED_INPUTS, source_pins)
    baseline = mapping.derive_mapping(inputs, source_pins["input_sha256"])
    baseline_raw = mapping.encode(baseline)
    candidate_raw = mapping.encode(mapping.barcelona_release_mapping(baseline))
    candidate_pin = sha256(candidate_raw).hexdigest()
    assert candidate_pin == "6db652d621c614d4338994655c1672bb7d86214e0ac90037e7964af2a85b8346"
    destination.parent.mkdir()
    destination.write_bytes(baseline_raw)

    def write(name, value):
        raw = mapping.encode(value) if isinstance(value, dict) else value
        (tmp_path / name).write_bytes(raw)
        return {"path": name, "sha256": sha256(raw).hexdigest()}

    bridge = write("history-bridge.json", {"status": "CANDIDATE_COMPLETE", "purpose": "FEATURE_AND_RATING_STATE_ONLY"})
    pending = {"status": "PENDING_FORMAL_APPROVAL", "schema_version": 1, "league": "PD", "provider_id": 81,
               "legacy_label": "BARCELONA", "binding_disposition": "BIND_LATEST_STATE",
               "baseline_mapping_sha256": sha256(baseline_raw).hexdigest(), "released_mapping_sha256": candidate_pin,
               "history_bridge_sha256": bridge["sha256"]}
    signoff = write("formal-approval.json", {"status": "APPROVED", "approved_by": "Synthetic owner",
                    "approved_on": "2026-11-22", "sheet": write("decision-sheet.txt", b"Reviewed release"),
                    "pending_decision": write("pending.json", pending), "history_bridge": bridge})
    approved = {**pending, "status": "APPROVED", "approved_by": "Synthetic owner", "approved_on": "2026-11-22"}
    if fault == "extra-decision-field":
        approved["new_scope"] = "another row"
    approved_ref = write("barcelona-release.json", approved)
    diff = write("approval-diff.json", {"schema_version": 1, "formal_approval": signoff,
                 "approved_decision": approved_ref, "changes": mapping.release_decision_changes(pending, approved)})
    args = ["publish", "--baseline-sha256", pending["baseline_mapping_sha256"], "--candidate-sha256", candidate_pin]
    if fault != "missing-diff":
        args.extend(["--release-diff", diff["path"], "--release-diff-sha256",
                     "0" * 64 if fault == "wrong-pin" else diff["sha256"]])
    assert mapping.main(args, destination=destination) == (0 if fault is None else 2)
    if fault is None:
        assert destination.read_bytes() == candidate_raw
        receipt, = receipts(destination)
        assert receipt["verified_counts"] == {"bindings": 82, "cold_starts": 14}
        assert receipt["barcelona_release"] == approved_ref and receipt["release_approval_diff"] == diff
        assert receipt["history_bridge_sha256"] == bridge["sha256"]
    else:
        assert destination.read_bytes() == baseline_raw


def test_candidate_crosswalk_requires_consistent_dated_appearances():
    rows = [("2024-08-01", "Club A", "Club B"), ("2024-08-08", "Club C", "Club A"),
            ("2024-08-15", "Club B", "Club C")]
    official = [("2024-08-01", 11, 12), ("2024-08-01", 99, 98), ("2024-08-08", 13, 11)]
    assert review._date_crosswalk(rows, official) == {"Club A": {11}}


def test_candidate_generator_refuses_runtime_directory():
    with pytest.raises(ValueError, match="new folder under tmp/"):
        review.build_review(Path("data/canonical/review"), ("PD", 81))


def test_bridge_csv_aliases_include_only_the_two_authoritative_seasons():
    club = ("BL1", 36)
    labels = {(*club, 2025): ["Processed evidence"]}
    aliases = {club: {"other_provider_ids": [], "csv_by_season": [
        {"season": season, "labels": [f"Bochum {season}"]} for season in (2022, 2023, 2024, 2025, 2026)
    ]}}
    assert [(row["season"], row["label"]) for row in review._bridge_csv_labels(labels, aliases)] == [
        (2023, "Bochum 2023"), (2024, "Bochum 2024"),
    ]


@pytest.mark.parametrize("fault", ["misallocated-league", "duplicate-id"])
def test_review_mapping_counts_require_exact_leagues_and_unique_ids(fault):
    entries = [{"league": league, "provider_id": team}
               for league, count in review.SEASON_CLUBS.items() for team in range(1, count + 1)]
    assert review._check_mapping_counts(entries) == review.SEASON_CLUBS
    entries[0] = ({**entries[0], "league": "PL", "provider_id": 999} if fault == "misallocated-league" else entries[1])
    with pytest.raises(ValueError, match="96 unique entries"):
        review._check_mapping_counts(entries)


def test_bridge_review_applies_latest_chain_rule_and_flags_publication_disagreement():
    replay = {("PD", 2025): {11: ("2025-08-19", 1, 2), 10: ("2025-08-17", 1, 2)},
              ("PD", 2026): {12: ("2026-08-19", 1, 3), 13: ("2026-08-20", 4, 5),
                             14: ("2026-08-23", 5, 6), 15: ("2026-09-01", 7, 5)}}
    names = {("PD", team): name for team, name in enumerate(
        ["Carry Club", "Fresh Club", "Split Club", "Stale Club", "Incumbent", "Conflict Club", "Old Live Label"], 1)}
    exact = {("PD", name.lower()): {team} for (_, team), name in names.items()}
    spans = {("PD", label): dates for label, dates in {
        "Carry Club": ("2016-07-01", "2017-05-22"), "CARRY CLUB": ("2023-09-01", "2024-06-02"),
        "SPLIT CLUB": ("2023-09-01", "2024-04-10"), "Split Club": ("2024-04-20", "2024-05-20"),
        "STALE CLUB": ("2019-07-01", "2020-05-22"), "CONFLICT CLUB": ("2020-07-01", "2021-06-02"),
        "Old Live Label": ("2015-09-01", "2016-04-10"), "OLD LIVE LABEL": ("2021-09-01", "2022-04-10"),
    }.items()}
    seed = {"states": [{"league": league, "team_id": label} for league, label in spans]}
    published = {
        ("PD", 1): {"disposition": "BIND_LATEST_STATE", "legacy_label": "CARRY CLUB"},
        ("PD", 3): {"disposition": "BIND_LATEST_STATE", "legacy_label": "SPLIT CLUB"},
        ("PD", 6): {"disposition": "COLD_START_STALE"},
        ("PD", 7): {"disposition": "BIND_LATEST_STATE", "legacy_label": "Old Live Label"},
    }
    rows = review._bridge_binding_rows(replay, names, exact, {}, {}, {"rows": []}, published, seed, spans,
                                       "2021-06-02", {("PD", "Incumbent")})
    carry, fresh, split, stale, incumbent, conflict, old_live = rows
    assert (carry["seed_label"], carry["fixture_id"], carry["absent_seasons"]) == ("CARRY CLUB", 10, 1)
    assert carry["older_chains"] == ["Carry Club"]
    assert (fresh["treatment"], fresh["flag"], fresh["seed_labels_found"]) == ("Fresh (proposed)", "OUTSIDE_LIVE_MAPPING", {})
    assert split["hold"] == "OVERLAPPING_OR_SAME_SEASON_CHAINS"
    assert stale["finding"] == "STALE_BEYOND_THRESHOLD" and stale["treatment"] == "Fresh (proposed)"
    assert stale["evidence"] == {"last_seed_appearance": "2020-05-22", "threshold": "2021-06-02",
                                 "source_sha256": review.PROCESSED_PINS[1]}
    assert incumbent["hold"] == "INCUMBENT_WITHOUT_SEED_LABEL" and incumbent["finding"] is None
    for row, label in ((conflict, "CONFLICT CLUB"), (old_live, "OLD LIVE LABEL")):
        assert row["seed_label"] == label and row["hold"] == "RULE_CONFLICTS_WITH_PUBLICATION"
        assert "BRIDGE_LIVE_DISAGREEMENT" in row["flag"]
    sheet = "\n".join(review._seed_review_sections(rows))
    assert "Returning clubs carrying state" in sheet and "All proposed cold starts" in sheet
    assert "Carry Club: 2017-05-22 (older chain, not carried)" in sheet


def test_review_identifies_the_pinned_stale_rule_and_refuses_without_approval():
    from src.canonical.clock import CanonicalError

    proposals = {"binding_rule": "Synthetic rule", "seed_end_inclusive": review.SEED_DAY,
                 "recent_cutoff_inclusive": "2021-06-02"}
    with pytest.raises(CanonicalError, match="BRIDGE_STALE_RULE_NOT_APPROVED"):
        review.approved_stale_cutoff({}, proposals)
    rule = {"binding_rule": {"status": "APPROVED", "architect_proposal": "Synthetic rule"}}
    block = review._stale_rule_summary(proposals, review.approved_stale_cutoff(rule, proposals))
    assert "2021-06-02" in block and "#binding_rule" in block and review.STALE_RULE_REGISTER[1] in block


def test_seed_review_sheet_bytes_are_independent_of_source_order():
    games = {13: ("2025-08-23", 3, 2), 11: ("2025-08-17", 1, 2)}
    names = {("PD", team): f"Club {team}" for team in range(1, 4)}
    exact = {(league, name.lower()): {team} for (league, team), name in names.items()}
    spans = {("PD", "Club 1"): ("2017-01-01", "2017-06-01"),
             ("PD", "CLUB 1"): ("2023-08-01", "2024-05-01")}
    states = [{"league": league, "team_id": label} for league, label in spans]

    def rendered(fixtures, frozen_states):
        rows = review._bridge_binding_rows({("PD", 2025): fixtures}, names, exact, {}, {}, {"rows": []}, {},
                                           {"states": frozen_states}, spans, "2021-06-02", set())
        return "\n".join(review._seed_review_sections(rows)).encode()

    assert rendered(games, states) == rendered(dict(reversed(games.items())), list(reversed(states)))


def test_seed_window_binding_carries_for_bridge_with_informational_live_flag():
    replay = {("PD", 2024): {11: ("2024-08-17", 81, 2)}}
    names = {("PD", 81): "Barcelona", ("PD", 2): "Other"}
    exact = {("PD", "barcelona"): {81}}
    published = {("PD", 81): {"disposition": "OWNER_SEED_WINDOW_ONLY", "legacy_label": "BARCELONA",
                               "valid_through": review.SEED_DAY}}
    spans = {("PD", "BARCELONA"): ("2019-08-16", "2024-05-26"),
             ("PD", "Barcelona"): ("2012-08-19", "2017-05-21")}
    seed = {"states": [{"league": league, "team_id": label} for league, label in spans]}
    rows = review._bridge_binding_rows(replay, names, exact, {}, {}, {"rows": []},
                                       published, seed, spans, "2021-06-02", set())
    barcelona = next(row for row in rows if row["id"] == 81)
    assert barcelona["seed_label"] == "BARCELONA" and barcelona["hold"] is None
    assert barcelona["treatment"] == "Carry state (existing approved seed-window binding)"
    assert barcelona["seed_labels_found"]["BARCELONA"] == "2024-05-26"
    assert barcelona["flag"] == "BRIDGE_LIVE_DISAGREEMENT"
    assert published["PD", 81]["disposition"] == "OWNER_SEED_WINDOW_ONLY"


def test_alias_evidence_reports_another_club_using_a_label(monkeypatch):
    monkeypatch.setattr(review, "PROPOSED_ALIASES", {("BL1", 36): ("Bochum", "VFL BOCHUM")})
    evidence, = review._alias_evidence(
        {("BL1", 2023, "authority"): [("2024-05-18", "Bochum", "Other")]},
        {("BL1", "VFL BOCHUM"): ("2021-08-14", "2024-05-18")},
        {("BL1", 36): "VfL Bochum 1848"}, {("BL1", "bochum"): {36, 99}},
        {}, {}, {},
    )
    assert evidence["review_status"] == "STATED_PENDING_FORMAL_APPROVAL"
    assert evidence["other_provider_ids"] == [99]
    assert evidence["csv_by_season"] == [{"season": 2023, "source": "authority", "labels": ["Bochum"]}]


def test_espanyol_report_lists_both_chains_and_the_reader_decision():
    excluded = ("2023-06-04", "ESPANOL", "ALMERIA")
    carried = ("2023-05-28", "VALENCIA", "ESPANYOL")
    fixture_chains = {("PD", "ESPANOL"): {excluded}, ("PD", "ESPANYOL"): {carried}}
    binding = {"league": "PD", "live_label": "ESPANYOL",
               "seed_labels_found": {"ESPANOL": excluded[0], "ESPANYOL": carried[0]}}
    recorded = {"legacy_label": "ESPANOL", "excluded_fixture_count": 1,
                "track_b_fixture_id": digest(["PD", *excluded]),
                "one_sided_update": {"reason": "Frozen asymmetry", "revisit_when": "Regeneration"}}
    ref = {"sha256": "a" * 64}
    report = review._espanyol_report(binding, fixture_chains, recorded, ref, True)
    assert "2023-06-04 | home | ALMERIA" in report and "2023-05-28 | away | VALENCIA" in report
    assert "exception checks pass" in report and recorded["track_b_fixture_id"] in report
    assert "exception checks block" in review._espanyol_report(binding, fixture_chains, recorded, ref, False)


@pytest.mark.parametrize("approval", ["APPROVED", "PROPOSED"])
def test_binding_release_requires_an_explicit_approved_register_entry(approval):
    receipt = {"input_sha256": {}}
    inputs = mapping.read_inputs(mapping.APPROVED_INPUTS, receipt)
    hashes = receipt["input_sha256"]
    original = mapping.derive_mapping(inputs, hashes)
    restricted, = [entry for entry in original["entries"] if entry["disposition"] == "OWNER_SEED_WINDOW_ONLY"]
    inputs["decision_register"]["recorded_owner_decisions"].append({
        "league": restricted["league"], "provider_id": restricted["provider_id"],
        "legacy_label": restricted["legacy_label"], "decision_type": "APPROVED_ALIAS",
        "binding_disposition": "BIND_LATEST_STATE", "status": approval,
    })
    reference = hashes["decision_register"] = mapping.digest(mapping.encode(inputs["decision_register"]))
    inputs["proposals"]["revision"]["decision_register_sha256"] = reference
    inputs["espanyol_debt"]["accepted_decision_register_sha256"] = reference
    if approval == "PROPOSED":
        with pytest.raises(mapping.Refusal, match="BINDING_RELEASE_NOT_APPROVED"):
            mapping.derive_mapping(inputs, hashes)
        return
    released = mapping.derive_mapping(inputs, hashes)
    entry, = [row for row in released["entries"] if (row["league"], row["provider_id"]) ==
              (restricted["league"], restricted["provider_id"])]
    assert entry == {key: "BIND_LATEST_STATE" if key == "disposition" else value
                     for key, value in restricted.items() if key != "valid_through"}


@pytest.mark.parametrize("input_name", tuple(mapping.APPROVED_INPUTS))
def test_each_input_pin_mismatch_refuses_before_parsing(destination, tmp_path, input_name):
    original, expected = mapping.APPROVED_INPUTS[input_name]
    changed = tmp_path / f"{input_name}.json"
    changed.write_bytes(original.read_bytes() + b"\xff")
    inputs = {**mapping.APPROVED_INPUTS, input_name: (changed, expected)}

    assert mapping.main(["dry-run"], destination=destination, inputs=inputs) != 0

    failure = last_failure(destination)
    assert failure["error_code"] == "INPUT_PIN_MISMATCH"
    assert failure["disposition"] == "NOT_PUBLISHED"
    assert failure["input_pin_mismatches"] == [input_name]
    assert all(failure["input_sha256"].values())
    assert not destination.exists()


@pytest.mark.parametrize("input_name, change, error_code", [
    ("proposals", "declared_hold", "PROPOSAL_HOLD"),
    ("proposals", "row_hold", "PROPOSAL_HOLD"),
    ("proposals", "status", "PROPOSAL_STATUS_INVALID"),
    ("proposals", "bad_seed_date", "SEED_WINDOW_CONSTRAINTS_INVALID"),
    ("espanyol_debt", "missing_revisit", "ESPANYOL_DEBT_MISMATCH"),
    ("proposals", "missing_catalog_id", "CATALOG_COVERAGE_MISMATCH"),
])
def test_mapping_invariants_can_refuse_a_hash_matched_file(destination, tmp_path, input_name, change, error_code):
    document = json.loads(mapping.APPROVED_INPUTS[input_name][0].read_bytes())
    if change == "declared_hold":
        document["statistics"]["proposal_actions"]["HOLD_ALIAS_REVIEW"] = 1
    elif change == "row_hold":
        document["rows"][0]["proposal_action"] = "HOLD_ALIAS_REVIEW"
    elif change == "status":
        document["rows"][0]["row_status"] = "ALREADY_APPLIED"
    elif change == "bad_seed_date":
        document["revision"]["later_application_checks"]["barcelona"]["seed_end_inclusive"] = "2024-02-30"
    elif change == "missing_revisit":
        del document["decision_supplement"]["one_sided_update"]["revisit_when"]
    elif change == "missing_catalog_id":
        document["rows"][0]["id"] = 999999
    path = tmp_path / f"synthetic-{input_name}.json"
    raw = json.dumps(document).encode("utf-8")
    path.write_bytes(raw)
    inputs = {**mapping.APPROVED_INPUTS, input_name: (path, sha256(raw).hexdigest())}

    assert mapping.main(["dry-run"], destination=destination, inputs=inputs) == 2
    failure = last_failure(destination)
    assert failure["error_code"] == error_code
    assert failure["input_sha256"][input_name] == inputs[input_name][1]
    assert failure["disposition"] == "NOT_PUBLISHED"
    assert not destination.exists()


@pytest.mark.parametrize("replacement, error_code", [
    ("0" * 64, "DRY_RUN_CANDIDATE_MISMATCH"),
    ("not-a-sha256", "INVALID_DRY_RUN_HASH"),
])
def test_publish_refuses_mismatched_or_malformed_dry_run_candidate(destination, replacement, error_code):
    argv = plan_publish(destination)
    approved_hash = argv[-1]
    assert replacement != approved_hash
    argv[-1] = replacement

    assert mapping.main(argv, destination=destination) == 2

    failure = last_failure(destination)
    assert failure["error_code"] == error_code
    assert failure["disposition"] == "NOT_PUBLISHED"
    if error_code == "DRY_RUN_CANDIDATE_MISMATCH":
        assert failure["output_sha256"] == approved_hash
    else:
        assert all(value is None for value in failure["input_sha256"].values())
    assert all(receipt["result"] != "SUCCESS" for receipt in receipts(destination))
    assert not destination.exists()


def test_baseline_changed_between_dry_run_and_publish_is_preserved(destination):
    seed_baseline(destination)
    argv = plan_publish(destination)
    changed = b"changed between invocations"
    destination.write_bytes(changed)

    assert mapping.main(argv, destination=destination) != 0

    assert destination.read_bytes() == changed
    failure = last_failure(destination)
    assert failure["error_code"] == "BASELINE_CHANGED"
    assert failure["disposition"] == "NOT_PUBLISHED"


def test_baseline_changed_after_temp_fsync_is_refused_before_replace(destination, monkeypatch):
    seed_baseline(destination)
    argv = plan_publish(destination)
    original_temp = mapping.synced_temp
    changed = b"baseline changed just before replace"

    @contextmanager
    def move_baseline_after_fsync(path, raw):
        with original_temp(path, raw) as temp:
            if path == destination:
                destination.write_bytes(changed)
            yield temp

    monkeypatch.setattr(mapping, "synced_temp", move_baseline_after_fsync)
    assert mapping.main(argv, destination=destination) != 0
    assert last_failure(destination)["error_code"] == "BASELINE_CHANGED"
    assert destination.read_bytes() == changed


def test_replace_failure_leaves_baseline_intact(destination, monkeypatch):
    baseline = seed_baseline(destination)
    argv = plan_publish(destination)
    replace = mapping.os.replace

    def fail_mapping_replace(source, target):
        if target == destination:
            raise PermissionError("synthetic replace error")
        return replace(source, target)

    monkeypatch.setattr(mapping.os, "replace", fail_mapping_replace)
    assert mapping.main(argv, destination=destination) != 0
    failure = last_failure(destination)
    assert failure["error_code"] == "REPLACE_FAILED"
    assert failure["disposition"] == "NOT_PUBLISHED"
    assert destination.read_bytes() == baseline
    assert Path(failure["retained_previous"]).read_bytes() == baseline
    assert not list(destination.parent.glob("*.tmp"))


def verification_failure(*args):
    raise mapping.Refusal("INJECTED_VERIFICATION_FAILURE")


def test_failed_first_verification_quarantines_and_writes_receipt(destination, monkeypatch):
    argv = plan_publish(destination)
    monkeypatch.setattr(mapping, "verify_published", verification_failure)

    assert mapping.main(argv, destination=destination) != 0

    assert not destination.exists()
    failed, = destination.parent.glob("*.failed")
    failure = last_failure(destination)
    assert failure["error_code"] == "INJECTED_VERIFICATION_FAILURE"
    assert failure["disposition"] == "FIRST_PUBLICATION_QUARANTINED"
    assert failure["previous_sha256"] is None
    assert Path(failure["recovery_path"]) == failed
    assert sha256(failed.read_bytes()).hexdigest() == failure["output_sha256"]
    assert all(doc["result"] != "SUCCESS" for doc in receipts(destination))


def test_failed_verification_restores_identical_previous_bytes(destination, monkeypatch):
    baseline = seed_baseline(destination)
    argv = plan_publish(destination)
    monkeypatch.setattr(mapping, "verify_published", verification_failure)

    assert mapping.main(argv, destination=destination) != 0

    failure = last_failure(destination)
    assert failure["disposition"] == "BASELINE_RESTORED"
    assert failure["error_code"] == "INJECTED_VERIFICATION_FAILURE"
    assert destination.read_bytes() == Path(failure["retained_previous"]).read_bytes() == baseline
    assert failure["previous_sha256"] == sha256(baseline).hexdigest()
    assert all(doc["result"] != "SUCCESS" for doc in receipts(destination))


def test_unwritable_primary_receipt_recovers_and_leaves_durable_fallback(destination, monkeypatch):
    baseline = seed_baseline(destination)
    argv = plan_publish(destination)

    def unavailable(*args):
        raise PermissionError("synthetic receipt directory error")

    monkeypatch.setattr(mapping, "write_receipt", unavailable)
    assert mapping.main(argv, destination=destination) != 0

    fallback, = destination.parent.glob("*.receipt.json")
    failure = json.loads(fallback.read_bytes())
    assert destination.read_bytes() == baseline
    assert failure["result"] == "FAILED"
    assert failure["error_code"] == "RECEIPT_FAILED"
    assert failure["disposition"] == "BASELINE_RESTORED"
    assert failure["receipt_storage"] == "DESTINATION_DIRECTORY_FALLBACK"
    assert all(doc["result"] != "SUCCESS" for doc in receipts(destination))


def test_unwritable_receipt_paths_emit_failure_and_quarantine(destination, monkeypatch, capsys):
    argv = plan_publish(destination)
    write_atomic = mapping.write_atomic

    def unavailable(path, raw):
        if path.name.endswith(".receipt.json") or path.parent.name == mapping.RECEIPTS_DIR:
            raise PermissionError("synthetic failure of both receipt paths")
        return write_atomic(path, raw)

    monkeypatch.setattr(mapping, "write_atomic", unavailable)
    assert mapping.main(argv, destination=destination) != 0

    failure = json.loads(capsys.readouterr().err)
    assert failure["receipt_storage"] == "STDERR_ONLY_NO_DURABLE_LOCATION"
    assert failure["result"] == "FAILED"
    assert failure["receipt_path"] is None
    assert failure["disposition"] == "FIRST_PUBLICATION_QUARANTINED"
    assert not destination.exists()
    assert Path(failure["recovery_path"]).is_file()


def test_rollback_failure_is_nonzero_and_records_recovery_failure(destination, monkeypatch):
    seed_baseline(destination)
    argv = plan_publish(destination)
    monkeypatch.setattr(mapping, "verify_published", verification_failure)

    def unavailable(*args):
        raise OSError("synthetic rollback failure")

    monkeypatch.setattr(mapping, "recover", unavailable)
    assert mapping.main(argv, destination=destination) != 0
    failure = last_failure(destination)
    assert failure["disposition"] == "RECOVERY_FAILED"
    assert failure["error_code"] == "INJECTED_VERIFICATION_FAILURE"
    assert all(doc["result"] != "SUCCESS" for doc in receipts(destination))


def test_reopened_mapping_has_exact_binding_set_and_counts(destination, tmp_path):
    assert not destination.exists()
    argv = plan_publish(destination)
    assert not destination.exists()
    assert mapping.main(argv, destination=destination) == 0
    published = destination.read_bytes()
    observed = json.loads(published)

    # Independently read the result and split its entries by kind.
    by_kind = {"BINDING": [], "COLD_START": []}
    for entry in observed["entries"]:
        by_kind[entry["kind"]].append(entry)
    assert [len(entries) for entries in by_kind.values()] == [82, 14]
    assert all("legacy_label" not in entry for entry in by_kind["COLD_START"])
    bound = {(entry["league"], entry["provider_id"]):entry for entry in by_kind["BINDING"]}
    assert bound["PD", 81]["valid_through"] == "2024-06-02"
    assert bound["SA", 454]["legacy_label"] == "VENEZIA"
    assert bound["PD", 80]["legacy_label"] == "ESPANYOL"
    assert {item["source_file"] for item in observed["unassigned"]} == {"data/processed/matches/PD_2025.csv"}
    assert {item["line"] for item in observed["unassigned"]} == {179, 182, 295, 296}
    debt = observed["espanyol_debt"]
    assert debt["one_sided_update"]["revisit_when"] == "If Track B is regenerated or the frozen PD artifacts are reopened."
    assert debt["one_sided_update"]["status"] == "ACCEPTED_DEBT"
    assert "ESPANOL" not in {entry["legacy_label"] for entry in bound.values()}
    assert "UNLISTED LABEL" not in {entry["legacy_label"] for entry in bound.values()}
    assert observed["lookup_mode"] == "CLOSED_WORLD_EXACT"
    assert observed["consumer_approved"] is False
    assert "timestamp_utc" not in observed and "attempt_id" not in observed
    success, = [doc for doc in receipts(destination) if doc["result"] == "SUCCESS"]
    assert success["output_sha256"] == sha256(published).hexdigest()
    assert success["verified_counts"] == {"bindings":82, "cold_starts":14}
    assert success["previous_sha256"] is None and len(success["input_sha256"]) == 4

    second = tmp_path / "second" / destination.name
    assert mapping.main(plan_publish(second), destination=second) == 0
    assert second.read_bytes() == published


def test_reopened_verifier_refuses_a_bad_recount_even_with_matching_candidate_bytes(destination):
    assert mapping.main(plan_publish(destination), destination=destination) == 0
    expected = json.loads(destination.read_bytes())
    modified = json.loads(destination.read_bytes())
    entry = next(entry for entry in modified["entries"] if entry["kind"] == "BINDING")
    entry["kind"] = "COLD_START"
    tampered = json.dumps(modified).encode("utf-8")
    destination.write_bytes(tampered)

    with pytest.raises(mapping.Refusal, match="PUBLISHED_RECOUNT_FAILED"):
        mapping.verify_published(destination, tampered, expected)


def test_success_with_baseline_retains_previous_bytes(destination):
    baseline = seed_baseline(destination)
    argv = plan_publish(destination)
    assert mapping.main(argv, destination=destination) == 0

    success, = [doc for doc in receipts(destination) if doc["result"] == "SUCCESS"]
    assert Path(success["retained_previous"]).read_bytes() == baseline
    assert success["previous_sha256"] == sha256(baseline).hexdigest()
    assert success["output_sha256"] == sha256(destination.read_bytes()).hexdigest()
