"""Read-only inspection of an exact future-lineage implementation grant.

No preparation, acquisition, capture, collection or issuer command is exposed.
The application fixes the ProgramData trust source. CLI identities are comparison
inputs only and cannot provision or override that source.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation import future_lineage_context as context


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("inspect-implementation",))
    parser.add_argument("--decision", required=True)
    parser.add_argument("--decision-sha256", required=True)
    parser.add_argument("--implementation-grant", required=True)
    parser.add_argument("--implementation-grant-sha256", required=True)
    parser.add_argument("--candidate-sha256", required=True)
    parser.add_argument("--procedure-sha256", required=True)
    args = parser.parse_args(argv)
    try:
        anchor, _ = context.trusted_approval(ROOT)
        context.require(args.decision == anchor["decision"]["path"] and
                        args.implementation_grant == anchor["grant"]["path"],
                        "FUTURE_TRUSTED_APPROVAL_MISMATCH")
        decision, grant = anchor["decision"], anchor["grant"]
        access = context.verify_implementation(
            ROOT, decision, grant, decision_pin=args.decision_sha256,
            grant_pin=args.implementation_grant_sha256, candidate_pin=args.candidate_sha256,
            procedure_pin=args.procedure_sha256, at=context.now())
        print(json.dumps({"verified": True, "implementation_only": True,
                          "dependency_manifest": access.anchor["manifest"],
                          "runtime_closure": "UNQUALIFIED",
                          "valid_until_utc_exclusive": access.grant["valid_until_utc_exclusive"],
                          "preparation_permission": False, "collection_authorization": False}))
        return 0
    except (ValueError, KeyError, TypeError, OSError) as exc:
        print(json.dumps({"verified": False, "error": str(exc)}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
