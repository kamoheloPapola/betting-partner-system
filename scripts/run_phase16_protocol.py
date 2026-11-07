"""Phase16 protocol and clean-cohort controls. No activation or promotion."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluation import phase16_cohort as cohort
from src.evaluation import phase16_regression as regression
from src.evaluation import phase16_verifier as verifier


def main():
    parser = argparse.ArgumentParser(description='Versioned prospective fixture-revision protocol')
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('freeze-rule')
    start = commands.add_parser('initialize')
    start.add_argument('cohort_id')
    start.add_argument('league')
    inspect = commands.add_parser('status')
    inspect.add_argument('cohort_id')
    commands.add_parser('historical-regression')
    commands.add_parser('verify-regression')
    args = parser.parse_args()
    if args.command == 'freeze-rule':
        result = cohort.freeze_rule()
    elif args.command == 'initialize':
        result = cohort.initialize(args.cohort_id, args.league)
    elif args.command == 'status':
        result = verifier.verify_cohort(args.cohort_id)
    elif args.command == 'historical-regression':
        result = regression.publish()
    else:
        result = verifier.verify_regression()
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
