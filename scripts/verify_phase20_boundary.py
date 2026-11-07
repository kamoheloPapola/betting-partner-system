"""Inspect a separate Phase 20 boundary contract without changing settlement."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation import phase20_boundary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorization", type=Path,
                        default=ROOT / "docs/phase20-boundary-authorization-proposal-v6.json")
    parser.add_argument("--approved-sha256", help="hash supplied by an independent approval decision")
    args = parser.parse_args()
    result = phase20_boundary.verify(ROOT, phase20_boundary.RUN, args.authorization,
                                     approved_sha256=args.approved_sha256)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["verified"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
