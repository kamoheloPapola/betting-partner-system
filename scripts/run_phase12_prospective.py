"""Run, report, and independently verify frozen multi-league continuation."""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.evaluation import phase12_prospective as phase12


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('command', choices=['prepare', 'run', 'status', 'checkpoint', 'finalize', 'verify'])
    args = parser.parse_args()
    if args.command == 'prepare':
        result = phase12.prepare(args.run)
    elif args.command == 'run':
        result = phase12.operate(args.run)
    else:
        from src.evaluation import phase12_evidence
        result = getattr(phase12_evidence, args.command)(args.run)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
