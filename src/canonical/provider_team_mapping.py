"""Generate and publish the reviewed provider team mapping. No consumer is wired."""

import argparse
from collections import Counter
from contextlib import contextmanager
from datetime import date, datetime, timezone
from difflib import unified_diff
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import sys
from tempfile import NamedTemporaryFile
from uuid import uuid4


ROOT = Path(__file__).resolve().parents[2]
REVISION = ROOT / (
    "tmp/BL1-T3-offline-dependency-manifest-binding-20261027-01/"
    "NEXT-38l-binding-review-01/revision-02"
)
# Debt: these accepted inputs live under untracked tmp/, so a fresh clone cannot
# run this publisher. Reconcile the pinned input delivery when this module is committed.
APPROVED_INPUTS = {
    "proposals": (REVISION / "binding-proposals.json",
                  "bd7e2b1f189a20c94483b5c42431ee587be0e803da195aa5d4ac6ee45ba51c43"),
    "catalog": (REVISION / "provider-2026-team-catalog.json",
                "9b46dc2379d884a1f1da4ad9b43fec4ad052501b2465c86f2805b7cd7cb4b082"),
    "decision_register": (REVISION / "revision-02-decisions.json",
                          "709246734bb47d7616651e0f5e5951f72d361c144b0b78357b94548100e1d163"),
    "espanyol_debt": (REVISION / "review-addendum/espanyol-debt-addendum.json",
                      "90605615cfc30a410a26e6b7b718a677b27c48f75698d1bc4558edd7a964b793"),
}
DESTINATION = ROOT / "data/canonical/provider-team-mapping.json"
STATUS = "PROPOSED_NOT_APPLIED"
BIND_ACTIONS = {"BIND_LATEST_STATE", "OWNER_SEED_WINDOW_ONLY"}
COLD_ACTIONS = {"OWNER_WITHHELD_COLD_START", "COLD_START_STALE", "COLD_START_NO_SUPPORTED_STATE"}
ESPANYOL = ("PD", 80)
RECEIPTS_DIR = "mapping-publication-receipts"


class Refusal(ValueError):
    """A fixed error code suitable for the operator receipt."""


class AttemptParser(argparse.ArgumentParser):
    def error(self, message):
        raise Refusal("INVALID_ARGUMENTS")


def require(condition, code):
    if not condition:
        raise Refusal(code)


def digest(raw):
    return sha256(raw).hexdigest() if raw is not None else None


def encode(document):
    return (json.dumps(document, ensure_ascii=True, allow_nan=False, sort_keys=True, indent=2) + "\n").encode("ascii")


def read_inputs(inputs, receipt):
    held = {}
    mismatches = []
    unavailable = []
    for name in APPROVED_INPUTS:
        path, expected = inputs[name]
        try:
            raw = path.read_bytes()
        except OSError:
            unavailable.append(name)
            continue
        held[name] = raw
        receipt["input_sha256"][name] = digest(raw)
        if receipt["input_sha256"][name] != expected:
            mismatches.append(name)
    if unavailable or mismatches:
        receipt["inputs_unavailable"] = unavailable
        receipt["input_pin_mismatches"] = mismatches
        raise Refusal("INPUT_UNAVAILABLE" if unavailable else "INPUT_PIN_MISMATCH")
    return {name: json.loads(raw) for name, raw in held.items()}


def debt_record(inputs, hashes, rows):
    supplement = inputs["espanyol_debt"]
    expected = rows[ESPANYOL]["tail_state_resolution"]
    decision, = [d for d in inputs["decision_register"]["recorded_owner_decisions"]
                 if (d["league"], d["provider_id"]) == ESPANYOL and d["decision_type"] == "FIXTURE_EXCLUDED_AND_COUNTED"]
    debt = supplement["decision_supplement"]
    update = debt["one_sided_update"]
    revisit = update.get("revisit_when")
    prior = {**debt, "one_sided_update": {k: v for k, v in update.items() if k != "revisit_when"}}
    require(
        supplement["accepted_proposals_sha256"] == hashes["proposals"]
        and supplement["accepted_decision_register_sha256"] == hashes["decision_register"]
        and prior == decision == expected
        and decision["track_b_fixture_id"] == inputs["proposals"]["revision"]["fixture_metadata"]["track_b_fixture_id"]
        and "TAIL_STATE_CONFLICT" in rows[ESPANYOL]["flags"]
        and isinstance(revisit, str) and bool(revisit.strip()),
        "ESPANYOL_DEBT_MISMATCH",
    )
    return debt


def derive_mapping(inputs, hashes):
    proposals = inputs["proposals"]
    require(proposals["catalog_pin"]["sha256"] == hashes["catalog"]
            and proposals["revision"]["decision_register_sha256"] == hashes["decision_register"],
            "PROPOSAL_PROVENANCE_MISMATCH")
    rows = proposals["rows"]
    require(all(row["row_status"] == STATUS for row in rows), "PROPOSAL_STATUS_INVALID")
    declared = proposals["statistics"]["proposal_actions"]
    observed = Counter(row["proposal_action"] for row in rows)
    require(not any(action.startswith("HOLD_") and count > 0
                    for counts in (declared, observed) for action, count in counts.items()), "PROPOSAL_HOLD")
    require(all(type(count) is int and count >= 0 for count in declared.values())
            and {action: count for action, count in declared.items() if count} == observed
            and observed.keys() <= BIND_ACTIONS | COLD_ACTIONS,
            "PROPOSAL_ACTION_COUNTS_INVALID")

    catalog = {(group["league"], team["id"]): team["name"]
               for group in inputs["catalog"]["catalogs"] for team in group["teams"]}
    by_club = {(row["league"], row["id"]): row for row in rows}
    require(len(rows) == len(by_club) == len(catalog) == 96
            and sum(len(group["teams"]) for group in inputs["catalog"]["catalogs"]) == 96
            and {key: row["name"] for key, row in by_club.items()} == catalog, "CATALOG_COVERAGE_MISMATCH")
    require(sum(observed[action] for action in BIND_ACTIONS) == 82
            and sum(observed[action] for action in COLD_ACTIONS) == 14, "PROPOSAL_TOTALS_INVALID")

    checks = proposals["revision"]["later_application_checks"]
    excluded = checks["leave_unassigned"]
    lines = excluded["lines"]
    require(isinstance(excluded["source_file"], str) and bool(excluded["source_file"])
            and isinstance(lines, list) and all(type(line) is int and line > 1 for line in lines)
            and lines == sorted(set(lines)), "UNASSIGNED_ROWS_INVALID")
    restricted = {key for key, row in by_club.items() if row["proposal_action"] == "OWNER_SEED_WINDOW_ONLY"}
    cutoffs = {}
    for name, check in checks.items():
        if name == "leave_unassigned":
            continue
        keys = [key for key in restricted if key[1] == check.get("provider_id")
                and key[0] == check.get("league", key[0])]
        through = check.get("seed_end_inclusive")
        try:
            valid_date = date.fromisoformat(through).isoformat() == through
        except (TypeError, ValueError):
            valid_date = False
        require(len(keys) == 1 and keys[0] not in cutoffs and check.get("reject_later_dates") is True
                and valid_date, "SEED_WINDOW_CONSTRAINTS_INVALID")
        cutoffs[keys[0]] = through
    require(cutoffs.keys() == restricted, "SEED_WINDOW_CONSTRAINTS_INVALID")

    released = set()
    for decision in inputs["decision_register"]["recorded_owner_decisions"]:
        if "binding_disposition" not in decision:
            continue
        key = decision["league"], decision["provider_id"]
        require(key in restricted and key not in released and decision.get("status") == "APPROVED"
                and decision.get("decision_type") == "APPROVED_ALIAS"
                and decision.get("binding_disposition") == "BIND_LATEST_STATE"
                and decision.get("legacy_label") == by_club[key]["proposed_legacy_state"],
                "BINDING_RELEASE_NOT_APPROVED")
        released.add(key)

    entries = []
    bound_states = set()
    for (league, provider_id), row in sorted(by_club.items()):
        action = row["proposal_action"]
        state = row["proposed_legacy_state"]
        key = league, provider_id
        entry = {"league": league, "provider_id": provider_id,
                 "disposition": "BIND_LATEST_STATE" if key in released else action}
        if action in BIND_ACTIONS:
            require(isinstance(state, str) and bool(state.strip()) and (league, state) not in bound_states,
                    "LEGACY_STATE_INVALID")
            require(any(c["legacy_label"] == state and c["proposed"] and c["binding_supported"]
                        for c in row["candidates"]), "PROPOSED_STATE_UNSUPPORTED")
            bound_states.add((league, state))
            entry.update(kind="BINDING", legacy_label=state)
            if key in cutoffs and key not in released:
                entry["valid_through"] = cutoffs[key]
        else:
            require(state is None, "COLD_START_WITH_LEGACY_STATE")
            entry["kind"] = "COLD_START"
        entries.append(entry)
    return {
        "schema_version": 1, "input_sha256": dict(hashes), "lookup_mode": "CLOSED_WORLD_EXACT",
        "consumer_approved": False, "entries": entries,
        "unassigned": [{"source_file": excluded["source_file"], "line": line} for line in excluded["lines"]],
        "espanyol_debt": debt_record(inputs, hashes, by_club),
    }


def barcelona_release_mapping(baseline):
    rows = baseline.get("entries")
    require(isinstance(rows, list) and all(isinstance(row, dict) for row in rows), "BARCELONA_RELEASE_BASELINE_INVALID")
    targets = [row for row in rows if (row.get("league"), row.get("provider_id")) == ("PD", 81)]
    require(len(targets) == 1 and targets[0].get("kind") == "BINDING"
            and targets[0].get("legacy_label") == "BARCELONA"
            and targets[0].get("disposition") == "OWNER_SEED_WINDOW_ONLY"
            and "valid_through" in targets[0], "BARCELONA_RELEASE_BASELINE_INVALID")
    approved = {key: value for key, value in targets[0].items() if key != "valid_through"}
    approved["disposition"] = "BIND_LATEST_STATE"
    return {**baseline, "entries": [approved if row is targets[0] else row for row in rows]}


def release_decision_changes(pending, approved):
    return [{"field": "status", "before": pending["status"], "after": approved["status"]},
            *({"field": field, "before": None, "after": approved[field]}
              for field in ("approved_by", "approved_on"))]


def _approved_barcelona_release(root, diff_ref):
    from .clock import CanonicalError
    from .team_mapping_consumer import _has_sha256, _object, _read_pinned

    code = "BARCELONA_RELEASE_APPROVAL_INVALID"

    def held(ref):
        path = Path(ref["path"])
        require(not path.anchor and ".." not in path.parts and _has_sha256(ref["sha256"]), code)
        return _read_pinned(root / path, ref["sha256"], "BARCELONA_RELEASE_PROOF")

    def document(ref):
        return _object(held(ref), code)

    try:
        diff = document(diff_ref)
        signoff = document(diff["formal_approval"])
        pending = document(signoff["pending_decision"])
        approved_ref = diff["approved_decision"]
        approved = document(approved_ref)
        bridge_ref = signoff["history_bridge"]
        bridge = document(bridge_ref)
        held(signoff["sheet"])
        reviewed = {**pending, "status": "APPROVED", "approved_by": signoff["approved_by"],
                    "approved_on": signoff["approved_on"]}
        require(signoff["status"] == "APPROVED"
                and isinstance(signoff["approved_by"], str) and bool(signoff["approved_by"].strip())
                and date.fromisoformat(signoff["approved_on"]).isoformat() == signoff["approved_on"]
                and pending["status"] == "PENDING_FORMAL_APPROVAL"
                and {"approved_by", "approved_on"}.isdisjoint(pending)
                and encode(approved) == encode(reviewed)
                and encode(diff) == encode({"schema_version": 1, "formal_approval": diff["formal_approval"],
                                             "approved_decision": approved_ref,
                                             "changes": release_decision_changes(pending, approved)})
                and (approved["league"], approved["provider_id"], approved["legacy_label"],
                     approved["binding_disposition"], approved["schema_version"]) ==
                     ("PD", 81, "BARCELONA", "BIND_LATEST_STATE", 1)
                and approved["history_bridge_sha256"] == bridge_ref["sha256"]
                and bridge["status"] == "CANDIDATE_COMPLETE"
                and bridge["purpose"] == "FEATURE_AND_RATING_STATE_ONLY", code)
    except (CanonicalError, KeyError, TypeError, ValueError) as exc:
        raise Refusal(code) from exc
    return approved, approved_ref


def verify_published(destination, candidate, expected):
    raw = destination.read_bytes()
    require(raw == candidate, "PUBLISHED_BYTES_MISMATCH")
    mapping = json.loads(raw)
    entries = mapping["entries"]
    binding_count = sum(entry["kind"] == "BINDING" for entry in entries)
    cold_count = sum(entry["kind"] == "COLD_START" for entry in entries)
    require((len(entries), binding_count, cold_count) == (96, 82, 14), "PUBLISHED_RECOUNT_FAILED")
    require(mapping["input_sha256"] == expected["input_sha256"]
            and mapping["espanyol_debt"] == expected["espanyol_debt"]
            and mapping["espanyol_debt"]["one_sided_update"]["revisit_when"]
            and mapping["lookup_mode"] == "CLOSED_WORLD_EXACT" and mapping["consumer_approved"] is False,
            "PUBLISHED_PROVENANCE_FAILED")
    pairs = {(entry["league"], entry["provider_id"]) for entry in entries}
    require(pairs == {(entry["league"], entry["provider_id"]) for entry in expected["entries"]}
            and len(pairs) == 96, "PUBLISHED_CATALOG_COVERAGE_FAILED")
    require(entries == expected["entries"] and mapping["unassigned"] == expected["unassigned"],
            "PUBLISHED_DECISIONS_MISMATCH")
    return {"bindings": binding_count, "cold_starts": cold_count}


def read_baseline(destination):
    try:
        return destination.read_bytes()
    except FileNotFoundError:
        return None


@contextmanager
def synced_temp(destination, raw):
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = None
    try:
        with NamedTemporaryFile(dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp",
                                delete=False) as stream:
            temp = Path(stream.name)
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        yield temp
    finally:
        if temp is not None:
            temp.unlink(missing_ok=True)


def write_atomic(destination, raw):
    with synced_temp(destination, raw) as temp:
        os.replace(temp, destination)


def retain_baseline(destination, baseline):
    retained = destination.with_name(f"{destination.stem}.{digest(baseline)}.previous.json")
    if not retained.exists():
        write_atomic(retained, baseline)
    require(retained.read_bytes() == baseline, "RETAINED_BASELINE_MISMATCH")
    return retained


def recover(destination, retained, previous, attempt_id):
    if retained is not None:
        raw = retained.read_bytes()
        require(digest(raw) == previous, "RETAINED_BASELINE_MISMATCH")
        write_atomic(destination, raw)
        require(destination.read_bytes() == raw, "RESTORE_VERIFICATION_FAILED")
        return "BASELINE_RESTORED", str(retained)
    failed = destination.with_name(f"{destination.stem}.{attempt_id}.failed")
    os.replace(destination, failed)
    require(not destination.exists() and failed.is_file(), "QUARANTINE_VERIFICATION_FAILED")
    return "FIRST_PUBLICATION_QUARANTINED", str(failed)


def write_receipt(destination, receipt):
    path = destination.parent / RECEIPTS_DIR / f"{receipt['attempt_id']}.json"
    write_atomic(path, encode(receipt))
    return path


def persist_failure(destination, receipt):
    try:
        path = write_receipt(destination, receipt)
    except Exception:
        # The same failure may make the primary directory unusable. Keep a local
        # fallback next to the mapping, or emit the receipt if neither path works.
        receipt["receipt_storage"] = "DESTINATION_DIRECTORY_FALLBACK"
        path = destination.with_name(f"{destination.stem}.{receipt['attempt_id']}.receipt.json")
        try:
            write_atomic(path, encode(receipt))
        except Exception:
            receipt["receipt_storage"] = "STDERR_ONLY_NO_DURABLE_LOCATION"
            path = None
    print(encode({**receipt, "receipt_path": str(path) if path else None}).decode("ascii"), end="", file=sys.stderr)


def parse_args(argv):
    parser = AttemptParser(description=__doc__)
    commands = parser.add_subparsers(dest="operation", required=True)
    commands.add_parser("dry-run", help="show a diff and the hashes required for publication")
    publish = commands.add_parser("publish", help="publish using the reviewed dry-run hashes")
    publish.add_argument("--baseline-sha256", required=True, help="dry-run previous hash, or 'absent'")
    publish.add_argument("--candidate-sha256", required=True, help="dry-run output hash")
    publish.add_argument("--release-diff", help="checkout-relative Barcelona approval diff")
    publish.add_argument("--release-diff-sha256", help="reviewed approval diff digest")
    args = parser.parse_args(argv)
    if args.operation == "publish":
        require(re.fullmatch(r"[a-f0-9]{64}", args.candidate_sha256)
                and (args.baseline_sha256 == "absent" or re.fullmatch(r"[a-f0-9]{64}", args.baseline_sha256)),
                "INVALID_DRY_RUN_HASH")
        require((args.release_diff is None) == (args.release_diff_sha256 is None), "BARCELONA_RELEASE_PIN_REQUIRED")
    return args


def main(argv=None, *, destination=DESTINATION, inputs=APPROVED_INPUTS):
    receipt = {
        "attempt_id": uuid4().hex, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "operation": None, "destination": str(destination),
        "input_sha256": dict.fromkeys(APPROVED_INPUTS), "output_sha256": None, "previous_sha256": None,
        "retained_previous": None, "result": None,
    }
    retained = None
    published = False
    stage = "ARGUMENTS"
    try:
        args = parse_args(argv)
        receipt["operation"] = args.operation
        stage = "INPUTS"
        documents = read_inputs(inputs, receipt)
        stage = "DERIVATION"
        expected = derive_mapping(documents, receipt["input_sha256"])
        if args.operation == "publish" and args.release_diff is not None:
            stage = "RELEASE_APPROVAL"
            diff_ref = {"path": args.release_diff, "sha256": args.release_diff_sha256}
            decision, decision_ref = _approved_barcelona_release(ROOT, diff_ref)
            require(digest(encode(expected)) == decision["baseline_mapping_sha256"] == args.baseline_sha256,
                    "BARCELONA_RELEASE_BASELINE_MISMATCH")
            expected = barcelona_release_mapping(expected)
            require(digest(encode(expected)) == decision["released_mapping_sha256"] == args.candidate_sha256,
                    "BARCELONA_RELEASE_CANDIDATE_MISMATCH")
            receipt.update(barcelona_release=decision_ref, release_approval_diff=diff_ref,
                           history_bridge_sha256=decision["history_bridge_sha256"])
        candidate = encode(expected)
        receipt["output_sha256"] = digest(candidate)
        stage = "BASELINE"
        baseline = read_baseline(destination)
        receipt["previous_sha256"] = digest(baseline)
        if args.operation == "dry-run":
            stage = "RECEIPT"
            receipt.update(result="DRY_RUN", disposition="NOT_PUBLISHED")
            receipt_path = write_receipt(destination, receipt)
            before = (baseline or b"").decode("utf-8", errors="backslashreplace").splitlines()
            after = candidate.decode("ascii").splitlines()
            print("\n".join(unified_diff(before, after, fromfile="baseline", tofile="candidate", lineterm="")))
        else:
            require(args.candidate_sha256 == receipt["output_sha256"], "DRY_RUN_CANDIDATE_MISMATCH")
            require(args.baseline_sha256 == (receipt["previous_sha256"] or "absent"), "BASELINE_CHANGED")
            if baseline is not None:
                stage = "RETAIN_BASELINE"
                retained = retain_baseline(destination, baseline)
                receipt["retained_previous"] = str(retained)
            stage = "REPLACE"
            with synced_temp(destination, candidate) as temp:
                require(read_baseline(destination) == baseline, "BASELINE_CHANGED")
                os.replace(temp, destination)
                published = True
            stage = "VERIFY"
            receipt["verified_counts"] = verify_published(destination, candidate, expected)
            stage = "RECEIPT"
            receipt.update(result="SUCCESS", disposition="PUBLISHED")
            receipt_path = write_receipt(destination, receipt)
    except (Exception, KeyboardInterrupt) as exc:
        receipt.update(result="FAILED", disposition="NOT_PUBLISHED", stage=stage,
                       error_code=str(exc) if isinstance(exc, Refusal) else f"{stage}_FAILED")
        if published:
            try:
                disposition, recovery_path = recover(destination, retained, receipt["previous_sha256"], receipt["attempt_id"])
                receipt.update(disposition=disposition, recovery_path=recovery_path)
            except Exception:
                receipt.update(disposition="RECOVERY_FAILED")
        persist_failure(destination, receipt)
        return 2
    print(encode({**receipt, "receipt_path": str(receipt_path)}).decode("ascii"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
