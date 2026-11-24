from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path

import pytest

from scripts import run_history_bridge
from src.canonical import history_bridge
from src.canonical.clock import CanonicalError, digest
from src.canonical.features import FEATURE_VERSION
from src.canonical import provider
from src.canonical.provider import LEAGUES
from test_model import payload


@pytest.fixture
def bridge_plan(tmp_path, monkeypatch):
    monkeypatch.setitem(history_bridge.SEASON_CLUBS, "PD", 2)
    def write(path, value):
        raw = value if isinstance(value, bytes) else json.dumps(value).encode("utf-8")
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
        return {"path": path, "sha256": sha256(raw).hexdigest()}

    def make_fixture(fixture_id, season, day, status, score, *, home=81, away=2):
        return {"id": fixture_id, "utcDate": day + "T18:00:00Z", "status": status,
                "season": {"startDate": f"{season}-08-01"},
                "homeTeam": {"id": home}, "awayTeam": {"id": away}, "score": score}

    csv_refs = []
    for season, first, last, rows in (
        (2023, "2024-06-01", "2024-06-10", "01/06/2024,Other,FC Barcelona,ignore,ignore,ignored\n"
         "10/06/2024,FC Barcelona,Other,1,1,ignored\n"),
        (2024, "2024-08-17", "2024-08-24", "17/08/2024,FC Barcelona,Other,2,0,ignored\n"
         "24/08/2024,Other,FC Barcelona,0,3,ignored\n"),
    ):
        reference = write(f"data/historical/PD/PD_{season}.csv",
                          ("Date,HomeTeam,AwayTeam,FTHG,FTAG,Unused\n" + rows).encode())
        csv_refs.append({**reference, "league": "PD", "declared_season_start": season,
                         "filename_season": season, "role": "AUTHORITATIVE", "row_count": 2,
                         "min_date": first, "max_date": last})
    authority = write("fixture/source-manifest.json", {
        "schema_version": 1, "files": csv_refs,
        "season_scope": {league: {"first": 2023, "last": 2024} for league in LEAGUES},
        "known_gaps": [
            {"league": league, "declared_season_start": season, "status": "MISSING"}
            for league in LEAGUES if league != "PD" for season in (2023, 2024)
        ], "unverified_seasons": [],
        "provider_season_max_dates": [{"league": "PD", "season": 2025, "max_date": "2025-08-23"}],
    })
    snapshots = {
        # This score cannot be used. The archived provider fixture identifies the CSV row.
        2024: [make_fixture(24, 2024, "2024-08-17", "FINISHED", "ignored"),
               make_fixture(34, 2024, "2024-08-24", "FINISHED", "ignored", home=2, away=81)],
        2025: [make_fixture(25, 2025, "2025-08-16", "FINISHED", {"fullTime": {"home": 4, "away": 1}}),
               make_fixture(35, 2025, "2025-08-23", "FINISHED", {"fullTime": {"home": 2, "away": 0}}, home=2, away=81)],
        2026: [make_fixture(26, 2026, "2026-11-14", "FINISHED", {"fullTime": {"home": 1, "away": 0}}),
               make_fixture(27, 2026, "2026-11-15", "SCHEDULED", "ignored"),
               make_fixture(28, 2026, "2026-11-01", "CANCELLED", "ignored", home=2, away=81)],
    }
    received = {2024: "2026-11-08T19:33:00Z", 2025: "2026-11-08T19:32:00Z", 2026: "2026-11-18T18:00:00Z"}
    feature_file = Path(history_bridge.__file__).with_name("features.py")
    implementation = write("src/canonical/features.py", feature_file.read_bytes())
    reader = write("src/canonical/history_bridge.py", Path(history_bridge.__file__).read_bytes())
    verification = write("fixture/offline-verification.json", {
        "status": "PASS", "bridge_reader_sha256": reader["sha256"],
        "feature_implementation_sha256": implementation["sha256"],
        "provider_implementation_sha256": sha256(Path(provider.__file__).read_bytes()).hexdigest(),
    })
    seed = {"elo": 1500.0, "for": [1.0], "against": [0.0], "form": 3.0, "matches": 1}
    model = {**payload(), "feature_seed": {"feature_version": FEATURE_VERSION, "last_day": "2024-06-02",
             "states": [{"league": "PD", "team_id": label, "state": seed}
                        for label in ("BARCELONA", "OTHER LEGACY")]}}
    recipe = write("fixture/frozen-recipe.md", b"# Synthetic frozen recipe\n")
    monkeypatch.setattr(history_bridge, "FROZEN_RECIPE", (Path(recipe["path"]), recipe["sha256"]))
    plan = {"schema_version": 1, "leagues": ["PD"], "observed_as_of": "2026-11-19T00:00:00Z",
            "through_exclusive": "2026-11-15", "source_manifest": authority,
            "model": write("fixture/seed.json", {"model": model}),
            "recipe": recipe,
            "feature_implementation": implementation, "bridge_reader": reader,
            "offline_verification": verification,
            "seed_appearance_sources": write("fixture/seed-appearance-sources.json", {"canonical_inputs": [
                write("data/processed/matches/PD_2020.csv", b"date,home_team,away_team\n2021-05-22,OTHER LEGACY,Retired\n"),
                write("data/processed/matches/PD_2023.csv", b"date,home_team,away_team\n2024-06-01,BARCELONA,OTHER LEGACY\n"),
            ]}),
            "stale_rule": {
                "register": write("fixture/stale-register.json", {
                    "binding_rule": {"status": "APPROVED", "architect_proposal": "Synthetic rule"}}),
                "proposals": write("fixture/binding-proposals.json", {
                    "binding_rule": "Synthetic rule", "seed_end_inclusive": history_bridge.SEED_DAY,
                    "recent_cutoff_inclusive": "2021-06-02", "rows": []}),
            },
            "seed_bindings": [{"league": "PD", "provider_id": 81, "seed_label": "BARCELONA"},
                              {"league": "PD", "provider_id": 2, "seed_label": "OTHER LEGACY"}],
            "csv_labels": [{"league": "PD", "season": season, "label": label, "provider_id": team}
                           for season in (2023, 2024) for label, team in (("FC Barcelona", 81), ("Other", 2))],
            "tail_fixtures": [{"league": "PD", "day": "2024-06-10", "id": 23,
                               "home_id": 81, "away_id": 2, "status": "FINISHED"}],
            "completed_season_rosters": [
                {"league": "PD", "season": season, "clubs": [
                    {"provider_id": 81, "name": "Barcelona"}, {"provider_id": 2, "name": "Other"},
                ]} for season in (2023, 2024, 2025)],
            "owner_grant": write("fixture/grant.json", {
                "record_type": "owner_decision_addendum", "sealed_windows": {"authority": "GRANTED",
                "architect_proposed_scope": history_bridge.GRANT_SCOPE},
            })}

    def save():
        review = {
            "schema_version": 4, "status": "APPROVED_FOR_BRIDGE", "reviewed_by": "synthetic reviewer",
            "leagues": plan["leagues"], "through_exclusive": plan["through_exclusive"],
            "model_sha256": plan["model"]["sha256"], "source_manifest_sha256": plan["source_manifest"]["sha256"],
            "blockers": [], "tail_expected_counts": {"PD": len(plan["tail_fixtures"])},
            **{key: plan[key] for key in ("seed_bindings", "csv_labels", "tail_fixtures", "completed_season_rosters",
                                         "seed_appearance_sources", "stale_rule")},
            **plan.get("review_override", {}),
        }
        plan["identity_review"] = write("fixture/identity-review.json", review)
        entries = []
        for season, matches in snapshots.items():
            body = write(f"fixture/provider/PD_{season}.json", {
                "competition": {"code": "PD"}, "resultSet": {"count": len(matches)}, "matches": matches,
            })
            entries.append({"league": "PD", "season": season, "http_status": 200, **body,
                            "size_bytes": (tmp_path / body["path"]).stat().st_size,
                            "requested_utc": received[season], "received_utc": received[season]})
        manifest = write("fixture/provider/manifest.json", {
            "schema": "football-data-org.snapshots.v1", "host": "api.football-data.org", "entries": entries,
        })
        plan["provider_sources"] = [{"league": "PD", "season": season, "manifest": manifest,
                                     "expected_fixtures": len(matches)} for season, matches in snapshots.items()]
        baseline = write("fixture/baseline/PD_2026.json", {
            "competition": {"code": "PD"}, "resultSet": {"count": 1},
            "matches": [make_fixture(26, 2026, "2026-11-14", "TIMED", "unused")],
        })
        baseline_manifest = write("fixture/baseline/manifest.json", {
            "schema": "football-data-org.snapshots.v1", "host": "api.football-data.org",
            "entries": [{"league": "PD", "season": 2026, "http_status": 200, **baseline,
                        "size_bytes": (tmp_path / baseline["path"]).stat().st_size,
                        "requested_utc": "2026-11-08T18:00:00Z", "received_utc": "2026-11-08T18:00:00Z"}],
        })
        plan["provider_sources"][-1]["gap_baseline"] = {"manifest": baseline_manifest, "expected_fixtures": 1}
        return write("fixture/plan.json", plan)

    return plan, snapshots, received, save


def _replace_authoritative_csv(tmp_path, plan, season, rows, *, max_date=None):
    manifest_path = tmp_path / plan["source_manifest"]["path"]
    manifest = json.loads(manifest_path.read_bytes())
    entry = next(row for row in manifest["files"] if row["declared_season_start"] == season)
    csv_path = tmp_path / entry["path"]
    raw = b"Date,HomeTeam,AwayTeam,FTHG,FTAG,Unused\n" + b"".join(rows)
    csv_path.write_bytes(raw)
    entry.update(sha256=sha256(raw).hexdigest(), row_count=len(rows))
    if max_date is not None:
        entry["max_date"] = max_date
    manifest_path.write_bytes(json.dumps(manifest).encode("utf-8"))
    plan["source_manifest"]["sha256"] = sha256(manifest_path.read_bytes()).hexdigest()


def _cli_approval(tmp_path, ref, *, plan_changes=None, review_changes=None):
    def write(name, document):
        raw = document if isinstance(document, bytes) else json.dumps(document).encode()
        path = tmp_path / "fixture" / name
        path.write_bytes(raw)
        return {"path": path.relative_to(tmp_path).as_posix(), "sha256": sha256(raw).hexdigest()}

    plan = json.loads((tmp_path / ref["path"]).read_bytes())
    review = json.loads((tmp_path / plan["identity_review"]["path"]).read_bytes())
    draft_review = {**review, "status": "UNAPPROVED"}
    draft_review.pop("reviewed_by")
    draft_review_ref = write("draft-review.json", draft_review)
    draft = {**plan, "status": "UNAPPROVED", "identity_review": draft_review_ref}
    draft_ref = write("draft-plan.json", draft)
    signoff = write("formal-approval.json", {
        "status": "APPROVED", "approved_by": review["reviewed_by"], "approved_on": "2026-11-22",
        "draft_plan": draft_ref, "identity_review": draft_review_ref, "sheet": write("sheet.md", b"Synthetic approval"),
    })
    approved_review = write("approved-review.json", {**review, "reviewed_on": "2026-11-22",
                                                     "formal_approval": signoff, **(review_changes or {})})
    approved = {**plan, "status": "APPROVED_FOR_BRIDGE", "identity_review": approved_review, **(plan_changes or {})}
    approved_ref = write("approved-plan.json", approved)
    evidence = write("approval-plan-diff.json", {
        "schema_version": 1, "formal_approval": signoff, "draft_plan": draft_ref, "approved_plan": approved_ref,
        "changes": run_history_bridge.approval_plan_changes(draft, approved),
    })
    return [approved_ref["path"], "--sha256", approved_ref["sha256"],
            "--identity-review-sha256", approved_review["sha256"], "--approval-diff", evidence["path"],
            "--approval-diff-sha256", evidence["sha256"]]


def _pre_seed_season_with_unresolved_label(tmp_path, plan, label, *, reverse=True):
    rows = [f"01/06/2024,Other,{label},ignored,ignored,unused\n".encode()]
    if reverse:
        rows.append(f"02/06/2024,{label},Other,ignored,ignored,unused\n".encode())
    _replace_authoritative_csv(tmp_path, plan, 2023, rows, max_date="2024-06-02" if reverse else "2024-06-01")
    plan["tail_fixtures"].clear()
    plan["completed_season_rosters"] = [row for row in plan["completed_season_rosters"] if row["season"] != 2023]


@pytest.mark.parametrize("reverse", [True, False], ids=["label-round-robin-passes", "label-audit-failure-is-nonblocking"])
def test_relegated_pre_seed_labels_pass_and_receipt_records_coverage_and_empty_tail(tmp_path, bridge_plan, reverse):
    plan, _, _, save = bridge_plan
    _pre_seed_season_with_unresolved_label(tmp_path, plan, "Relegated", reverse=reverse)
    ref = save()
    report = history_bridge.run_bridge(tmp_path, ref, identity_review_sha256=plan["identity_review"]["sha256"],
                                      operator="synthetic")
    assert report["status"] == "CANDIDATE_COMPLETE"
    receipt = json.loads((tmp_path / history_bridge.PACKET / "history-bridge.json").read_bytes())
    assert [row["season"] for row in receipt["completed_seasons"]] == [2024, 2025]
    assert receipt["tail_fixtures_by_league"] == {"PD": 0}
    ledger = json.loads((tmp_path / history_bridge.PACKET / "access-ledger.json").read_bytes())
    assert ledger["tail_fixtures_by_league"] == {"PD": 0}
    assert receipt["pre_seed_csv_label_coverage"] == [{
        "league": "PD", "season": 2023, "status": "PASS" if reverse else "FAIL", "clubs": 2,
        "expected_clubs": 2, "fixtures": 2 if reverse else 1, "missing_pairs": 0 if reverse else 1,
        "repeated_pairs": 0, "self_fixtures": 0,
    }]
    unresolved = receipt["unresolved_pre_seed_labels"]
    assert [(row["line"], row["label"], row["side"]) for row in unresolved] == (
        [(2, "Relegated", "away"), (3, "Relegated", "home")] if reverse else [(2, "Relegated", "away")])


@pytest.mark.parametrize("label, team", [
    ("FC Barcelona", 81), ("81", 81), ("Future Club", 99), ("Roster Alias", 81),
])
def test_unresolved_pre_seed_label_matching_post_seed_identity_blocks(tmp_path, bridge_plan, monkeypatch, label, team):
    plan, snapshots, _, save = bridge_plan
    _pre_seed_season_with_unresolved_label(tmp_path, plan, label)
    plan["csv_labels"] = [row for row in plan["csv_labels"] if (row["season"], row["label"]) != (2023, label)]
    snapshots[2026][1]["homeTeam"] = {"id": 99, "shortName": "Future Club"}
    plan["completed_season_rosters"][0]["clubs"][0]["name"] = "Roster Alias"
    ref = save()
    monkeypatch.setattr(history_bridge, "_project_scores", lambda *a: pytest.fail("score projection"))
    with pytest.raises(CanonicalError, match=rf"BRIDGE_PRE_SEED_IDENTITY_OVERLAPS_POST_SEED: .*: {team}$"):
        history_bridge.run_bridge(tmp_path, ref, identity_review_sha256=plan["identity_review"]["sha256"])
    assert not (tmp_path / history_bridge.PACKET).exists()


def test_half_resolved_pre_seed_row_still_requires_incumbent_review(tmp_path, bridge_plan):
    plan, _, _, save = bridge_plan
    _pre_seed_season_with_unresolved_label(tmp_path, plan, "Relegated")
    plan["seed_bindings"][1]["seed_label"] = None
    ref = save()
    with pytest.raises(CanonicalError, match="BRIDGE_FROZEN_SEED_BINDING_REQUIRED: PD/2"):
        history_bridge.run_bridge(tmp_path, ref, identity_review_sha256=plan["identity_review"]["sha256"])
    assert not (tmp_path / history_bridge.PACKET).exists()


def test_returning_club_cannot_cold_start_with_an_existing_frozen_state(tmp_path, bridge_plan):
    plan, _, _, save = bridge_plan
    _pre_seed_season_with_unresolved_label(tmp_path, plan, "Relegated")
    plan["seed_bindings"][0].update(seed_label=None, finding="NO_USABLE_STATE", reason="NO_SEED_LABEL")
    ref = save()
    with pytest.raises(CanonicalError, match="BRIDGE_FINDING_CONTRADICTS_EVIDENCE: PD/81"):
        history_bridge.run_bridge(tmp_path, ref, identity_review_sha256=plan["identity_review"]["sha256"])


@pytest.mark.parametrize("kind, code", [
    ("incumbent-without-seed-label", "BRIDGE_FINDING_CONTRADICTS_EVIDENCE: PD/2"),
    ("ambiguous-seed", "BRIDGE_SEED_BINDING_AMBIGUOUS: PD/81"),
])
def test_seed_review_refuses_incumbent_cold_start_and_ambiguous_seed(tmp_path, bridge_plan, kind, code):
    plan, _, _, save = bridge_plan
    _pre_seed_season_with_unresolved_label(tmp_path, plan, "Relegated")
    model_path = tmp_path / plan["model"]["path"]
    model = json.loads(model_path.read_bytes())
    states = model["model"]["feature_seed"]["states"]
    if kind == "incumbent-without-seed-label":
        states.pop(1)
        plan["seed_bindings"][1].update(seed_label=None, finding="NO_USABLE_STATE", reason="NO_SEED_LABEL")
    else:
        states.append({**states[0], "team_id": "Barcelona"})
    raw = json.dumps(model).encode("utf-8")
    model_path.write_bytes(raw)
    plan["model"]["sha256"] = sha256(raw).hexdigest()
    ref = save()
    options = {"identity_review_sha256": plan["identity_review"]["sha256"]}
    with pytest.raises(CanonicalError, match=code):
        history_bridge.run_bridge(tmp_path, ref, **options)
    assert not (tmp_path / history_bridge.PACKET).exists()


def _prepare_stale_binding(tmp_path, plan, *, inclusive_cutoff=False):
    source = plan["seed_appearance_sources"]
    inventory_path = tmp_path / source["path"]
    inventory = json.loads(inventory_path.read_bytes())
    recent = inventory["canonical_inputs"][1]
    recent_path = tmp_path / recent["path"]
    rows = b"date,home_team,away_team\n2024-06-01,BARCELONA,Retired\n"
    if inclusive_cutoff:
        rows += b"2021-06-02,OTHER LEGACY,Retired\n"
    recent_path.write_bytes(rows)
    recent["sha256"] = sha256(rows).hexdigest()
    raw = json.dumps(inventory).encode()
    inventory_path.write_bytes(raw)
    source["sha256"] = sha256(raw).hexdigest()

    evidence = {"last_seed_appearance": "2021-05-22", "threshold": "2021-06-02",
                "source_sha256": source["sha256"]}
    binding = plan["seed_bindings"][1]
    binding.update(seed_label=None, seed_aliases=["OTHER LEGACY"], finding="NO_USABLE_STATE",
                   reason="STALE_BEYOND_THRESHOLD", evidence=evidence)
    return binding, evidence


@pytest.mark.parametrize("finding", ["verified", "wrong-date", "wrong-cutoff", "wrong-source",
                                    "missing-reason", "inclusive-cutoff"])
def test_stale_finding_is_verified_against_pinned_seed_history(tmp_path, bridge_plan, finding, monkeypatch):
    plan, _, _, save = bridge_plan
    binding, evidence = _prepare_stale_binding(tmp_path, plan, inclusive_cutoff=finding == "inclusive-cutoff")
    if finding == "wrong-date":
        evidence["last_seed_appearance"] = "2020-05-22"
    elif finding == "wrong-cutoff":
        evidence["threshold"] = "2022-06-02"
    elif finding == "wrong-source":
        evidence["source_sha256"] = "0" * 64
    elif finding == "inclusive-cutoff":
        evidence["last_seed_appearance"] = "2021-06-02"
    if finding == "missing-reason":
        del binding["reason"]
    ref = save()
    review_pin = plan["identity_review"]["sha256"]
    monkeypatch.setattr(history_bridge, "_project_scores", lambda *args: pytest.fail("score projection"))
    if finding == "verified":
        _, _, states, *_ = history_bridge._plan(tmp_path, ref, review_pin)
        assert states["PD", "2"] == history_bridge._fresh()
    else:
        expected = ("BRIDGE_FINDING_REASON_REQUIRED" if finding == "missing-reason"
                    else "BRIDGE_FINDING_CONTRADICTS_EVIDENCE")
        with pytest.raises(CanonicalError, match=f"{expected}: PD/2"):
            history_bridge.run_bridge(tmp_path, ref, identity_review_sha256=review_pin)
    assert not (tmp_path / history_bridge.PACKET).exists()


def test_cold_start_seed_contains_only_replayed_matches(tmp_path, bridge_plan):
    plan, _, _, save = bridge_plan
    _prepare_stale_binding(tmp_path, plan)
    ref = save()
    frozen = json.loads((tmp_path / plan["model"]["path"]).read_bytes())["model"]["feature_seed"]
    assert next(row["state"] for row in frozen["states"] if row["team_id"] == "OTHER LEGACY")["matches"] == 1

    receipt = history_bridge.run_bridge(tmp_path, ref, identity_review_sha256=plan["identity_review"]["sha256"],
                                        operator="synthetic")
    assert receipt["status"] == "CANDIDATE_COMPLETE"
    seed = json.loads((tmp_path / history_bridge.PACKET / "candidate-seed.json").read_bytes())
    cold = next(row["state"] for row in seed["states"] if row["team_id"] == "2")
    assert cold["matches"] == receipt["csv_fixtures"] + receipt["provider_fixtures"] == 6
    assert cold["for"] == [1, 0, 0, 1, 2, 0]
    assert cold["against"] == [1, 2, 3, 4, 0, 1]


def test_truncated_csv_cannot_exempt_season_with_post_seed_manifest_max_date(tmp_path, bridge_plan, monkeypatch):
    plan, _, _, save = bridge_plan
    _replace_authoritative_csv(tmp_path, plan, 2023, [b"01/06/2024,Other,FC Barcelona,ignored,ignored,unused\n"])
    ref = save()
    monkeypatch.setattr(history_bridge, "_project_scores", lambda *a: pytest.fail("score projection"))
    with pytest.raises(CanonicalError, match="BRIDGE_CSV_MANIFEST_METADATA_MISMATCH"):
        history_bridge.run_bridge(tmp_path, ref, identity_review_sha256=plan["identity_review"]["sha256"],
                                  operator="synthetic")
    assert not (tmp_path / history_bridge.PACKET).exists()


def test_bridge_replays_barcelona_by_id_from_unchanged_frozen_seed(tmp_path, bridge_plan, monkeypatch, capsys):
    plan, _, _, save = bridge_plan
    ref = save()
    packet = tmp_path / history_bridge.PACKET
    monkeypatch.setattr(run_history_bridge, "ROOT", tmp_path)
    args = _cli_approval(tmp_path, ref)
    assert run_history_bridge.main(["preflight", *args]) == 0
    preflight = json.loads(capsys.readouterr().out)
    assert preflight["status"] == "PREFLIGHT_PASSED" and not preflight["scores_projected"]
    assert not packet.exists()
    assert preflight["completed_seasons"] == [
        {"league": "PD", "season": season, "clubs": 2, "fixtures": 2, "accounted_void_fixtures": 0}
        for season in (2023, 2024, 2025)]

    project = history_bridge._project_scores
    replay = history_bridge._replay
    frozen = json.loads((tmp_path / plan["model"]["path"]).read_bytes())["model"]["feature_seed"]["states"][0]["state"]

    def anchored_replay(states, rows, through):
        if through == plan["through_exclusive"] and isinstance(rows, list):
            assert states["PD", "81"] == frozen
            assert all(type(row["home_id"]) is type(row["away_id"]) is int for row in rows)
            assert [row["id"] for row in rows] == [23, 24, 34, 25, 35, 26]
        return replay(states, rows, through)

    def anchored_projection(*args):
        ledger = json.loads((packet / "access-ledger.json").read_bytes())
        assert ledger["status"] == "OPENING_RESERVED"
        assert len(ledger["windows"]) == 2
        return project(*args)

    monkeypatch.setattr(history_bridge, "_project_scores", anchored_projection)
    monkeypatch.setattr(history_bridge, "_replay", anchored_replay)
    assert run_history_bridge.main(["build", *args, "--operator", "synthetic"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "CANDIDATE_COMPLETE"
    assert result["csv_fixtures"] == result["provider_fixtures"] == 3
    assert result["void_provider_fixtures"] == 1
    seed_bytes = (packet / "candidate-seed.json").read_bytes()
    assert sha256(seed_bytes).hexdigest() == result["candidate_seed_sha256"]
    seed = json.loads(seed_bytes)
    assert seed["last_day"] == "2026-11-14" and seed["identity_key"] == "provider_id"
    barcelona = next(row["state"] for row in seed["states"] if row["team_id"] == "81")
    assert barcelona["matches"] == 7 and barcelona["for"] == [1., 1., 2., 3., 4., 0., 1.]
    assert run_history_bridge.main(["build", *args, "--operator", "synthetic"]) == 2
    assert json.loads(capsys.readouterr().out)["error_code"] == "BRIDGE_OPENING_ALREADY_RECORDED"


@pytest.mark.parametrize("kickoff", ["2026-11-15", "2026-11-16"], ids=["cutoff-day", "after-cutoff"])
def test_bridge_build_never_projects_poisoned_post_cutoff_score(tmp_path, bridge_plan, kickoff):
    plan, snapshots, _, save = bridge_plan
    snapshots[2026][1].update(utcDate=kickoff + "T18:00:00Z", status="FINISHED", score="poisoned")
    ref = save()
    result = history_bridge.run_bridge(tmp_path, ref, identity_review_sha256=plan["identity_review"]["sha256"],
                                       operator="synthetic")
    assert result["status"] == "CANDIDATE_COMPLETE" and result["provider_fixtures"] == 3
    seed = json.loads((tmp_path / history_bridge.PACKET / "candidate-seed.json").read_bytes())
    assert seed["last_day"] == "2026-11-14" and {row["state"]["matches"] for row in seed["states"]} == {7}


@pytest.mark.parametrize("changes", [
    {"plan_changes": {"through_exclusive": "2026-11-16"}},
    {"plan_changes": {"provider_sources": []}},
    {"plan_changes": {"schema_version": True}},
    {"review_changes": {"seed_bindings": []}},
], ids=["cutoff", "captures", "schema-type", "bindings"])
def test_runner_refuses_changes_outside_the_hashed_approval_diff(
        tmp_path, bridge_plan, monkeypatch, capsys, changes):
    _, _, _, save = bridge_plan
    args = _cli_approval(tmp_path, save(), **changes)
    monkeypatch.setattr(run_history_bridge, "ROOT", tmp_path)
    monkeypatch.setattr(run_history_bridge, "run_bridge", lambda *args, **kwargs: pytest.fail("opened the bridge"))
    assert run_history_bridge.main(["build", *args, "--operator", "synthetic"]) == 2
    assert json.loads(capsys.readouterr().out)["error_code"] == "BRIDGE_APPROVAL_DERIVATION_INVALID"
    assert not (tmp_path / history_bridge.PACKET).exists()


@pytest.mark.parametrize("status", ["AWARDED", "CANCELLED"])
def test_bridge_accounts_for_csv_void_without_projecting_goals_or_updating_either_club(
        tmp_path, bridge_plan, monkeypatch, status):
    plan, snapshots, _, save = bridge_plan
    snapshots[2024][0].update(status=status, score="unreadable provider goals")
    _replace_authoritative_csv(tmp_path, plan, 2024, [
        b"17/08/2024,FC Barcelona,Other,must-not-read,must-not-read,unused\n",
        b"24/08/2024,Other,FC Barcelona,0,3,unused\n",
    ])
    ref = save()
    options = {"identity_review_sha256": plan["identity_review"]["sha256"]}
    packet = tmp_path / history_bridge.PACKET
    with monkeypatch.context() as preflight_only:
        preflight_only.setattr(history_bridge, "_project_scores", lambda *args: pytest.fail("score projection"))
        preflight = history_bridge.run_bridge(tmp_path, ref, **options)
    assert preflight["status"] == "PREFLIGHT_PASSED" and not preflight["scores_projected"]
    assert not packet.exists()
    assert preflight["completed_seasons"][1]["accounted_void_fixtures"] == 1
    excluded = preflight["excluded_fixtures"]
    assert excluded[0] == {
        "league": "PD", "season": 2024, "day": "2024-08-17", "fixture_id": 24,
        "home_id": 81, "away_id": 2, "status": status, "disposition": f"excluded_{status.lower()}",
        "matched_csv": True,
    }
    project_scores = history_bridge._project_scores

    def project_admitted_only(csv_rows, provider_rows, payloads):
        assert [row["id"] for row, _, _ in csv_rows] == [23, 34]
        ledger = json.loads((packet / "access-ledger.json").read_bytes())
        assert ledger["excluded_fixture_count"] == 2
        assert ledger["excluded_fixtures"] == excluded
        return project_scores(csv_rows, provider_rows, payloads)

    monkeypatch.setattr(history_bridge, "_project_scores", project_admitted_only)
    candidate = history_bridge.run_bridge(tmp_path, ref, operator="synthetic", **options)
    assert candidate["excluded_fixtures"] == excluded
    seed = json.loads((packet / "candidate-seed.json").read_bytes())
    states = {row["team_id"]: row["state"] for row in seed["states"]}
    assert states["81"]["for"] == [1., 1., 3., 4., 0., 1.]
    assert states["2"]["for"] == [1., 1., 0., 1., 2., 0.]
    assert states["81"]["matches"] == states["2"]["matches"] == 6


@pytest.mark.parametrize("missing_pair", [False, True], ids=["accounted", "missing-pair"])
def test_provider_authoritative_award_is_accounted_without_a_csv_or_score_read(
        tmp_path, bridge_plan, monkeypatch, missing_pair):
    plan, snapshots, _, save = bridge_plan
    snapshots[2025][1]["status"] = "AWARDED"
    del snapshots[2025][1]["score"]
    if missing_pair:
        snapshots[2025].pop(0)
    ref = save()
    options = {"identity_review_sha256": plan["identity_review"]["sha256"]}
    if missing_pair:
        monkeypatch.setattr(history_bridge, "_project_scores", lambda *args: pytest.fail("score projection"))
        with pytest.raises(history_bridge.SeasonCoverageError, match=r"Barcelona \(81\) home.*expected 1, found 0"):
            history_bridge.run_bridge(tmp_path, ref, **options)
        return
    result = history_bridge.run_bridge(tmp_path, ref, operator="synthetic", **options)
    assert result["status"] == "CANDIDATE_COMPLETE" and result["provider_fixtures"] == 2
    assert result["completed_seasons"][-1]["accounted_void_fixtures"] == 1
    assert result["excluded_fixtures"][0] == {
        "league": "PD", "season": 2025, "day": "2025-08-23", "fixture_id": 35,
        "home_id": 2, "away_id": 81, "status": "AWARDED", "disposition": "excluded_awarded",
        "matched_csv": False,
    }


def test_bridge_exclusion_listing_is_independent_of_capture_row_order(tmp_path, bridge_plan):
    plan, snapshots, _, save = bridge_plan
    snapshots[2024][0]["status"] = "AWARDED"
    snapshots[2025][1]["status"] = "AWARDED"

    def preflight():
        ref = save()
        return history_bridge.run_bridge(tmp_path, ref, identity_review_sha256=plan["identity_review"]["sha256"])

    result = preflight()
    for rows in snapshots.values():
        rows.reverse()
    reversed_result = preflight()
    assert [row["fixture_id"] for row in result["excluded_fixtures"]] == [24, 35, 28]
    assert reversed_result["excluded_fixtures"] == result["excluded_fixtures"]


def test_reviewed_june_void_counts_as_accounted_without_a_tail_state_update(tmp_path, bridge_plan):
    plan, _, _, save = bridge_plan
    plan["tail_fixtures"][0]["status"] = "AWARDED"
    plan["review_override"] = {"tail_expected_counts": {"PD": 0}}
    ref = save()
    result = history_bridge.run_bridge(tmp_path, ref, identity_review_sha256=plan["identity_review"]["sha256"])
    assert result["status"] == "PREFLIGHT_PASSED"
    assert result["csv_fixtures"] == 2 and result["tail_fixtures_by_league"] == {"PD": 0}
    assert result["completed_seasons"][0]["accounted_void_fixtures"] == 1
    assert result["excluded_fixtures"][0]["fixture_id"] == 23
    assert result["excluded_fixtures"][0]["matched_csv"] is True


@pytest.mark.parametrize("day, code", [
    ("2024-08-17", "BRIDGE_DUPLICATE_FIXTURE_METADATA"),
    ("2024-08-18", "BRIDGE_PROVIDER_PAIR_RESULT_CONFLICT"),
], ids=["same-date", "different-date"])
def test_bridge_blocks_awarded_and_finished_for_the_same_ordered_pair(
        tmp_path, bridge_plan, monkeypatch, day, code):
    plan, snapshots, _, save = bridge_plan
    snapshots[2024].append({**snapshots[2024][0], "id": 124, "status": "AWARDED", "utcDate": day + "T18:00:00Z"})
    ref = save()
    monkeypatch.setattr(history_bridge, "_project_scores", lambda *args: pytest.fail("score projection"))
    with pytest.raises(CanonicalError, match=code):
        history_bridge.run_bridge(tmp_path, ref, identity_review_sha256=plan["identity_review"]["sha256"])


@pytest.mark.parametrize("status", ["POSTPONED", "SUSPENDED"])
def test_bridge_blocks_unresolved_selected_status_before_cutoff(tmp_path, bridge_plan, monkeypatch, status):
    plan, snapshots, _, save = bridge_plan
    snapshots[2024][0]["status"] = status
    ref = save()
    monkeypatch.setattr(history_bridge, "_project_scores", lambda *args: pytest.fail("score projection"))
    with pytest.raises(CanonicalError, match=f"BRIDGE_PROVIDER_RESULTS_UNRESOLVED: .*{status}"):
        history_bridge.run_bridge(tmp_path, ref, identity_review_sha256=plan["identity_review"]["sha256"])


@pytest.mark.parametrize("fault, code", [
    ("stale", "BRIDGE_PROVIDER_CAPTURE_BEFORE_CUTOFF"),
    ("pending", "BRIDGE_PROVIDER_RESULTS_UNRESOLVED"),
    ("wrong_side", "BRIDGE_CSV_FIXTURE_NOT_UNIQUELY_RECONCILED"),
    ("unmatched_csv_date", "BRIDGE_CSV_FIXTURE_NOT_UNIQUELY_RECONCILED"),
    ("duplicate_csv_void_row", "BRIDGE_CSV_FIXTURE_NOT_UNIQUELY_RECONCILED"),
    ("unmatched_provider", "BRIDGE_CSV_PROVIDER_COVERAGE_MISMATCH"),
    ("unbound", "BRIDGE_FIXTURE_TEAM_UNBOUND"),
    ("duplicate", "BRIDGE_DUPLICATE_FIXTURE_ID"),
    ("pin_mismatch", "BRIDGE_PLAN_PIN_OR_READ_FAILED"),
    ("no_owner_grant", "BRIDGE_OWNER_GRANT_REQUIRED"),
    ("unapproved_crosswalk", "BRIDGE_IDENTITY_REVIEW_NOT_APPLICABLE"),
    ("no_independent_crosswalk_pin", "BRIDGE_INDEPENDENT_IDENTITY_REVIEW_REQUIRED"),
    ("missing_gap_fixture", "BRIDGE_GAP_FIXTURE_NOT_RECONCILED"),
    ("invalid_starting_state", "BRIDGE_SEED_BINDING_INVALID"),
    ("incumbent_cold_start", "BRIDGE_FROZEN_SEED_BINDING_REQUIRED"),
    ("unresolved_pre_seed_label", "BRIDGE_CSV_LABEL_UNBOUND"),
    ("out_of_scope_csv_alias", "BRIDGE_CSV_LABEL_BINDING_INVALID"),
    ("unresolved_tail_fixture", "BRIDGE_TAIL_FIXTURE_INVALID"),
    ("missing_roster", "BRIDGE_COMPLETED_SEASON_ROSTER_MISSING"),
    ("reader_hash_mismatch", "BRIDGE_FROZEN_READER_PIN_OR_READ_FAILED"),
    ("provider_verification_hash_mismatch", "BRIDGE_OFFLINE_VERIFICATION_NOT_APPLICABLE"),
    ("different_recipe_with_matching_bytes", "BRIDGE_FROZEN_RECIPE_REFERENCE_INVALID"),
])
def test_bridge_refuses_incomplete_or_unreviewed_inputs_without_opening(
        tmp_path, bridge_plan, monkeypatch, fault, code):
    plan, snapshots, received, save = bridge_plan
    if fault == "stale":
        received[2026] = "2026-11-08T18:00:00Z"
    elif fault == "pending":
        snapshots[2026][0]["status"] = "TIMED"
    elif fault == "wrong_side":
        snapshots[2024][0]["homeTeam"], snapshots[2024][0]["awayTeam"] = (
            snapshots[2024][0]["awayTeam"], snapshots[2024][0]["homeTeam"])
    elif fault == "unmatched_csv_date":
        snapshots[2024][0]["utcDate"] = "2024-08-18T18:00:00Z"
    elif fault == "duplicate_csv_void_row":
        snapshots[2024][0]["status"] = "AWARDED"
        _replace_authoritative_csv(tmp_path, plan, 2024, [
            b"17/08/2024,FC Barcelona,Other,2,0,unused\n",
            b"17/08/2024,FC Barcelona,Other,2,0,unused\n",
            b"24/08/2024,Other,FC Barcelona,0,3,unused\n",
        ])
    elif fault == "unmatched_provider":
        snapshots[2024].append({**snapshots[2024][0], "id": 31, "utcDate": "2024-09-01T18:00:00Z"})
    elif fault == "unbound":
        snapshots[2025][0]["awayTeam"]["id"] = 999
    elif fault == "duplicate":
        snapshots[2026][0]["id"] = snapshots[2025][0]["id"]
    elif fault == "no_owner_grant":
        del plan["owner_grant"]
    elif fault == "unapproved_crosswalk":
        plan["review_override"] = {"status": "PROPOSED"}
    elif fault == "missing_gap_fixture":
        snapshots[2026][0]["id"] = 40
    elif fault == "invalid_starting_state":
        plan["seed_bindings"][0]["seed_label"] = "MISSING"
    elif fault == "incumbent_cold_start":
        plan["seed_bindings"][0]["seed_label"] = None
    elif fault == "unresolved_pre_seed_label":
        plan["csv_labels"] = [row for row in plan["csv_labels"]
                              if (row["season"], row["provider_id"]) != (2023, 81)]
    elif fault == "out_of_scope_csv_alias":
        plan["csv_labels"].append({"league": "PD", "season": 2025, "label": "Other", "provider_id": 2})
    elif fault == "unresolved_tail_fixture":
        plan["tail_fixtures"][0]["status"] = "TIMED"
    elif fault == "missing_roster":
        plan["completed_season_rosters"].pop()
    elif fault == "reader_hash_mismatch":
        plan["bridge_reader"]["sha256"] = "0" * 64
    elif fault == "provider_verification_hash_mismatch":
        verification_path = tmp_path / plan["offline_verification"]["path"]
        verification = json.loads(verification_path.read_bytes())
        verification["provider_implementation_sha256"] = "0" * 64
        raw = json.dumps(verification).encode()
        verification_path.write_bytes(raw)
        plan["offline_verification"]["sha256"] = sha256(raw).hexdigest()
    elif fault == "different_recipe_with_matching_bytes":
        path = "fixture/protocol-c.json"
        (tmp_path / path).write_bytes(b"{}")
        plan["recipe"] = {"path": path, "sha256": sha256(b"{}").hexdigest()}
    ref = save()
    if fault == "pin_mismatch":
        ref["sha256"] = "0" * 64
    monkeypatch.setattr(history_bridge, "_project_scores", lambda *args: pytest.fail("score projection"))
    review_pin = None if fault == "no_independent_crosswalk_pin" else plan["identity_review"]["sha256"]
    with pytest.raises(CanonicalError, match=code):
        history_bridge.run_bridge(tmp_path, ref, identity_review_sha256=review_pin, operator="synthetic")
    assert not (tmp_path / history_bridge.PACKET).exists()


def test_score_failure_records_opening_without_publishing_a_checkpoint(tmp_path, bridge_plan):
    plan, snapshots, _, save = bridge_plan
    snapshots[2025][0]["score"]["fullTime"]["home"] = True
    ref = save()
    options = {"identity_review_sha256": plan["identity_review"]["sha256"]}
    assert history_bridge.run_bridge(tmp_path, ref, **options)["status"] == "PREFLIGHT_PASSED"
    with pytest.raises(CanonicalError, match="BRIDGE_PROVIDER_GOALS_INVALID"):
        history_bridge.run_bridge(tmp_path, ref, operator="synthetic", **options)
    packet = tmp_path / history_bridge.PACKET
    failure = json.loads((packet / "opening-failure.json").read_bytes())
    assert failure["error_code"] == "BRIDGE_PROVIDER_GOALS_INVALID"
    assert {path.name for path in packet.iterdir()} == {"access-ledger.json", "opening-failure.json"}
    with pytest.raises(CanonicalError, match="BRIDGE_OPENING_ALREADY_RECORDED"):
        history_bridge.run_bridge(tmp_path, ref, operator="synthetic", **options)


@pytest.mark.parametrize("override, error_code", [
    ({"status": "UNAPPROVED"}, "BRIDGE_IDENTITY_REVIEW_NOT_APPLICABLE"),
    ({"schema_version": 2}, "BRIDGE_IDENTITY_REVIEW_NOT_APPLICABLE"),
    ({"schema_version": 3}, "BRIDGE_IDENTITY_REVIEW_NOT_APPLICABLE"),
    ({"blockers": [{"league": "PD", "provider_id": 80}]}, "BRIDGE_IDENTITY_REVIEW_BLOCKED"),
    ({"blockers": None}, "BRIDGE_IDENTITY_REVIEW_BLOCKED"),
    ({"tail_expected_counts": {"PD": 0}}, "BRIDGE_TAIL_EXPECTATION_MISMATCH"),
    ({}, None),
], ids=["unapproved", "v2", "v3", "blockers", "invalid-blockers", "tail-mismatch", "v4"])
def test_v4_review_requires_approval_empty_blockers_and_matching_tail_counts(
        tmp_path, bridge_plan, monkeypatch, override, error_code):
    plan, _, _, save = bridge_plan
    plan["review_override"] = override
    ref = save()
    monkeypatch.setattr(history_bridge, "_project_scores", lambda *args: pytest.fail("score projection"))
    if error_code in {"BRIDGE_IDENTITY_REVIEW_NOT_APPLICABLE", "BRIDGE_IDENTITY_REVIEW_BLOCKED"}:
        monkeypatch.setattr(history_bridge, "_provider_sources", lambda *args: pytest.fail("provider inspection"))
    options = {"identity_review_sha256": plan["identity_review"]["sha256"]}
    if error_code:
        with pytest.raises(CanonicalError, match=error_code):
            history_bridge.run_bridge(tmp_path, ref, operator="synthetic", **options)
    else:
        result = history_bridge.run_bridge(tmp_path, ref, **options)
        assert result["status"] == "PREFLIGHT_PASSED" and result["tail_fixtures_by_league"] == {"PD": 1}
    assert not (tmp_path / history_bridge.PACKET).exists()


@pytest.fixture
def espanyol_chain_check(tmp_path, monkeypatch):
    def select(fault=None, *, reverse=False):
        league, club = history_bridge.ESPANYOL
        carried, excluded, older = "ESPANYOL", "ESPANOL", "Espanol"
        excluded_fixture = ("2024-05-26", excluded, "Opponent")
        older_day = "2024-05-05" if fault == "third-overlapping-chain" else "2017-05-19"
        fixtures = [
            (older_day, older, "Opponent C"),
            ("2024-05-12", "Opponent A", carried),
            ("2024-05-19", carried, "Opponent B"),
            excluded_fixture,
        ]
        decision = {"league": league, "provider_id": club, "legacy_label": excluded, "status": "APPROVED",
                    "provenance": "RECORDED_OWNER_DECISION", "decision_type": "FIXTURE_EXCLUDED_AND_COUNTED",
                    "excluded_fixture_count": 1, "track_b_fixture_id": digest([league, *excluded_fixture])}
        if fault == "wrong-decision-club":
            decision["provider_id"] += 1
        elif fault == "wrong-decision-count":
            decision["excluded_fixture_count"] = 2
        elif fault == "wrong-fixture":
            fixtures[-1] = (excluded_fixture[0], excluded, "Different Opponent")
        elif fault == "extra-excluded-fixture":
            fixtures.append(("2024-05-25", excluded, "Extra Opponent"))
        elif fault == "carried-ends-later":
            fixtures.append(("2024-06-01", carried, "Later Opponent"))

        def write(path, document):
            target = tmp_path / path
            target.parent.mkdir(parents=True, exist_ok=True)
            raw = json.dumps(document).encode()
            target.write_bytes(raw)
            return sha256(raw).hexdigest()

        decision_path = Path("fixture/espanyol-debt-addendum.json")
        decision_pin = write(decision_path, {"decision_supplement": decision})
        monkeypatch.setitem(history_bridge.APPROVED_INPUTS, "espanyol_debt",
                            (history_bridge.MAPPING_ROOT / decision_path, decision_pin))
        ref = history_bridge.espanyol_decision_reference()
        if fault == "decision-ref-hash-mismatch":
            ref["sha256"] = "0" * 64
        elif fault == "decision-file-hash-mismatch":
            (tmp_path / decision_path).write_bytes((tmp_path / decision_path).read_bytes() + b"\n")
        published_label = excluded if fault == "carried-not-published" else carried
        publication_pin = write(history_bridge.MAPPING_PATH, {"entries": [
            {"league": league, "provider_id": club, "kind": "BINDING", "legacy_label": published_label},
        ]})
        monkeypatch.setattr(history_bridge, "ESPANYOL_PUBLICATION_SHA256", publication_pin)
        if fault == "different-club":
            club += 1
            ref = None
        elif fault == "missing-decision":
            ref = None
        if reverse:
            fixtures.reverse()
        records = [{"day": day, "home": home, "away": away} for day, home, away in fixtures]
        spans, chains = {}, defaultdict(set)
        history_bridge.record_seed_appearances(spans, league, records, chains)
        labels = (older, excluded, carried) if reverse else (carried, excluded, older)
        state = {"elo": 1501.0, "for": [1.0], "against": [0.0], "form": 3.0, "matches": 1}
        seed = {"feature_version": FEATURE_VERSION, "last_day": history_bridge.SEED_DAY, "states": [
            {"league": league, "team_id": label, "state": state if label == carried else {**state, "elo": 1499.0}}
            for label in labels
        ]}
        binding = {"league": league, "provider_id": club, "seed_label": carried}
        if ref is not None:
            binding["proposal_evidence"] = ref
        key = league, club
        evidence = (spans, "2021-06-02", "unused", {}, chains)
        return history_bridge._seed_states(tmp_path, seed, [binding], [league], set(), {key},
                                           {key: {label.lower() for label in labels}}, evidence)

    return select


def test_espanyol_exception_carries_published_state_independent_of_input_order(espanyol_chain_check):
    forward = espanyol_chain_check()
    assert forward == espanyol_chain_check(reverse=True)
    assert forward["PD", "80"]["elo"] == 1501.0


@pytest.mark.parametrize("fault", [
    "missing-decision", "wrong-decision-club", "wrong-decision-count", "wrong-fixture", "extra-excluded-fixture",
    "carried-not-published", "carried-ends-later", "different-club", "decision-ref-hash-mismatch",
    "decision-file-hash-mismatch", "third-overlapping-chain",
])
def test_espanyol_exception_refuses_each_failed_condition(espanyol_chain_check, fault):
    code = ("BRIDGE_ESPANYOL_DEBT_PIN_OR_READ_FAILED" if fault == "decision-file-hash-mismatch"
            else "BRIDGE_SEED_BINDING_AMBIGUOUS")
    with pytest.raises(CanonicalError, match=code):
        espanyol_chain_check(fault)


@pytest.mark.parametrize("header", [b"FTHG,UnusedGoal", b"FTHG,FTAG,FTAG"], ids=["missing", "duplicated"])
def test_preflight_rejects_invalid_goal_headers_before_opening(tmp_path, bridge_plan, monkeypatch, header):
    plan, _, _, save = bridge_plan
    manifest_path = tmp_path / plan["source_manifest"]["path"]
    manifest = json.loads(manifest_path.read_bytes())
    entry = manifest["files"][1]
    csv_path = tmp_path / entry["path"]
    csv_path.write_bytes(csv_path.read_bytes().replace(b"FTHG,FTAG", header, 1))
    entry["sha256"] = sha256(csv_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    plan["source_manifest"]["sha256"] = sha256(manifest_path.read_bytes()).hexdigest()
    ref = save()
    monkeypatch.setattr(history_bridge, "_project_scores", lambda *a: pytest.fail("score projection"))
    for operator in (None, "synthetic"):
        with pytest.raises(CanonicalError, match="BRIDGE_CSV_GOAL_COLUMNS_INVALID"):
            history_bridge.run_bridge(tmp_path, ref, identity_review_sha256=plan["identity_review"]["sha256"],
                                      operator=operator)
    assert not (tmp_path / history_bridge.PACKET).exists()


@pytest.mark.parametrize("league, clubs, matches_per_club", [
    ("PD", 20, 38), ("PL", 20, 38), ("SA", 20, 38), ("BL1", 18, 34), ("FL1", 18, 34),
])
def test_completed_season_requires_both_directions_of_every_pair(league, clubs, matches_per_club):
    roster = [{"league": league, "season": 2023,
               "clubs": [{"provider_id": team, "name": f"Club {team}"} for team in range(1, clubs + 1)]}]
    rosters = history_bridge._completed_rosters(roster, {(league, 2023)})
    schedule = [(league, 2023, "2024-05-01", home, away)
                for home in range(1, clubs + 1) for away in range(1, clubs + 1) if home != away]
    assert history_bridge._verify_completed_seasons(rosters, schedule) == [
        {"league": league, "season": 2023, "clubs": clubs, "fixtures": clubs * matches_per_club // 2,
         "accounted_void_fixtures": 0}]

    # Preserve the total fixture count while replacing one away leg with a duplicate home leg.
    schedule.remove((league, 2023, "2024-05-01", 1, 2))
    schedule.append((league, 2023, "2024-05-02", 2, 1))
    with pytest.raises(history_bridge.SeasonCoverageError) as caught:
        history_bridge._verify_completed_seasons(rosters, schedule)
    assert caught.value.gaps == [
        f"{league}/2023 Club 1 (1) home vs Club 2 (2) away: expected 1, found 0",
        f"{league}/2023 Club 2 (2) home vs Club 1 (1) away: expected 1, found 2",
    ]


def test_bridge_reports_named_barcelona_gap_before_goals_or_opening(tmp_path, bridge_plan, monkeypatch, capsys):
    plan, snapshots, _, save = bridge_plan
    snapshots[2025][0]["utcDate"] = snapshots[2025].pop()["utcDate"]
    ref = save()
    monkeypatch.setattr(run_history_bridge, "ROOT", tmp_path)
    monkeypatch.setattr(history_bridge, "_project_scores", lambda *args: pytest.fail("score projection"))
    assert run_history_bridge.main(["build", *_cli_approval(tmp_path, ref), "--operator", "synthetic"]) == 2
    assert json.loads(capsys.readouterr().out) == {
        "status": "BLOCKED", "error_code": "BRIDGE_COMPLETED_SEASON_GAPS",
        "gaps": ["PD/2025 Other (2) home vs Barcelona (81) away: expected 1, found 0"],
    }
    assert not (tmp_path / history_bridge.PACKET).exists()
