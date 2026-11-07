"""Read-only Phase 18 five-league readiness and offline verification."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation import phase18_evidence, phase18_verifier


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('status','verify'))
    args = parser.parse_args()
    try:
        value = phase18_verifier.all_leagues() if args.command == 'status' else phase18_evidence.verify_package()
        print(json.dumps(value, sort_keys=True, indent=2))
        return 0
    except (ValueError, KeyError, TypeError, RuntimeError) as error:
        print(json.dumps({'verified': False, 'error': str(error)}, sort_keys=True))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
