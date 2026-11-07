"""Read-only Phase 17 readiness/status commands; no capture or activation."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation import phase17_readiness as readiness
from src.evaluation import phase17_evidence as evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('status', 'verify'))
    args = parser.parse_args()
    try:
        result = readiness.status() if args.command == 'status' else evidence.verify_package()
        print(json.dumps(result, sort_keys=True, indent=2))
        return 0
    except (ValueError, KeyError, TypeError, RuntimeError) as error:
        print(json.dumps({'verified': False, 'error': str(error)}, sort_keys=True))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
