"""Run the offline canonical history bridge with an independently reviewed plan."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.canonical.clock import CanonicalError, canonical
from src.canonical.history_bridge import SeasonCoverageError, _document, _input, run_bridge


def approval_plan_changes(draft, approved):
    return [
        {"field": "status", "before": draft["status"], "after": approved["status"]},
        *({"field": f"identity_review.{field}", "before": draft["identity_review"][field],
           "after": approved["identity_review"][field]} for field in ("path", "sha256")),
    ]


def verify_approval_derivation(root, plan_ref, identity_sha256, diff_ref):
    code = "BRIDGE_APPROVAL_DERIVATION_INVALID"
    evidence = _document(root, diff_ref, "BRIDGE_APPROVAL_DERIVATION")
    try:
        approved = _document(root, plan_ref, "BRIDGE_PLAN")
        identity_ref = approved["identity_review"]
        identity = _document(root, identity_ref, "BRIDGE_IDENTITY_REVIEW")
        signoff_ref = identity["formal_approval"]
        signoff = _document(root, signoff_ref, "BRIDGE_FORMAL_APPROVAL")
        draft_ref = signoff["draft_plan"]
        draft = _document(root, draft_ref, "BRIDGE_DRAFT_PLAN")
        original_identity = _document(root, signoff["identity_review"], "BRIDGE_DRAFT_IDENTITY_REVIEW")
        _input(root, signoff["sheet"], "BRIDGE_APPROVED_SHEET")
        review_fields = {"status": "APPROVED_FOR_BRIDGE", "reviewed_by": signoff["approved_by"],
                         "reviewed_on": signoff["approved_on"], "formal_approval": signoff_ref}
        expected_plan = {**draft, "status": "APPROVED_FOR_BRIDGE", "identity_review": identity_ref}
        expected_evidence = {"schema_version": 1, "formal_approval": signoff_ref, "draft_plan": draft_ref,
                             "approved_plan": plan_ref, "changes": approval_plan_changes(draft, approved)}
        if (signoff["status"] != "APPROVED" or draft["status"] != "UNAPPROVED"
                or original_identity["status"] != "UNAPPROVED" or draft["identity_review"] != signoff["identity_review"]
                or identity_ref["sha256"] != identity_sha256
                or canonical(identity) != canonical({**original_identity, **review_fields})
                or canonical(approved) != canonical(expected_plan)
                or canonical(evidence) != canonical(expected_evidence)):
            raise CanonicalError(code)
    except (KeyError, TypeError) as error:
        raise CanonicalError(code) from error


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("preflight", "build"):
        target = commands.add_parser(command)
        target.add_argument("plan", help="Review plan path relative to the checkout root")
        target.add_argument("--sha256", required=True, help="Independently reviewed plan digest")
        target.add_argument("--identity-review-sha256", required=True,
                            help="Separately approved historical identity-review digest")
        target.add_argument("--approval-diff", required=True, help="Checkout-relative field-level approval diff")
        target.add_argument("--approval-diff-sha256", required=True, help="Pinned digest of the approval diff")
        if command == "build":
            target.add_argument("--operator", required=True)
    args = parser.parse_args(argv)
    try:
        plan_ref = {"path": args.plan, "sha256": args.sha256}
        verify_approval_derivation(ROOT, plan_ref, args.identity_review_sha256,
                                   {"path": args.approval_diff, "sha256": args.approval_diff_sha256})
        result = run_bridge(ROOT, plan_ref,
                            identity_review_sha256=args.identity_review_sha256,
                            operator=getattr(args, "operator", None))
    except (CanonicalError, OSError, ValueError) as error:
        report = {"status": "BLOCKED", "error_code": getattr(error, "code", type(error).__name__)}
        if isinstance(error, SeasonCoverageError):
            report["gaps"] = error.gaps
        elif report["error_code"] in {"BRIDGE_PROVIDER_CAPTURE_BEFORE_CUTOFF", "BRIDGE_PROVIDER_RESULTS_UNRESOLVED"}:
            report["affected_sources"] = str(error).partition(": ")[2].split(", ")
        elif report["error_code"] in {"BRIDGE_CSV_LABEL_UNBOUND", "BRIDGE_CSV_FIXTURE_NOT_UNIQUELY_RECONCILED",
                                     "BRIDGE_CSV_PROVIDER_COVERAGE_MISMATCH", "BRIDGE_TAIL_FIXTURE_INVALID",
                                     "BRIDGE_CSV_MANIFEST_METADATA_MISMATCH", "BRIDGE_PRE_SEED_IDENTITY_OVERLAPS_POST_SEED",
                                     "BRIDGE_PROVIDER_MANIFEST_MAX_DATE_MISSING", "BRIDGE_PROVIDER_MANIFEST_MAX_DATE_MISMATCH",
                                     "BRIDGE_FROZEN_SEED_BINDING_REQUIRED", "BRIDGE_NO_USABLE_STATE_FINDING_REQUIRED",
                                     "BRIDGE_SEED_BINDING_AMBIGUOUS"}:
            report["detail"] = str(error).partition(": ")[2]
        print(canonical(report))
        return 2
    print(canonical(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
