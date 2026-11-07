"""Dedicated prospective operation; deliberately separate from nightly training."""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.evaluation import phase12_operations as operations
from src.evaluation import phase12_operations_evidence as evidence


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('run', 'verify', 'certify-tests'))
    parser.add_argument('--run', type=Path, default=ROOT / 'baselines/phase12/phase12-prospective-20260908-v1')
    parser.add_argument('--key', help='Retry key; defaults to current UTC hour. No scientific settings are configurable.')
    parser.add_argument('--focused', type=Path)
    parser.add_argument('--protected', type=Path)
    parser.add_argument('--full', type=Path)
    args = parser.parse_args()
    if args.command == 'run':
        result = operations.run(args.run, args.key)
    elif args.command == 'verify':
        result = evidence.verify(args.run)
    else:
        if not all((args.focused, args.protected, args.full)):
            parser.error('certify-tests requires --focused, --protected and --full')
        result = evidence.certify_tests(args.run, args.focused, args.protected, args.full)
    print(json.dumps(result, indent=2, sort_keys=True))
    if result.get('degraded'):
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
