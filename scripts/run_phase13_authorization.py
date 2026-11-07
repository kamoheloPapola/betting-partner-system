"""Authorization prerequisites only. No activation or decision-production command."""
from pathlib import Path
import argparse
import json
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluation import phase13_authorization as a
from src.evaluation import phase13_evidence as e


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('bind', 'status', 'certify-tests', 'seal', 'verify'))
    parser.add_argument('--focused', type=Path)
    parser.add_argument('--protected', type=Path)
    parser.add_argument('--full', type=Path)
    args = parser.parse_args()
    if args.command == 'bind':
        result = [a.install(league) for league in a.p.source.LEAGUES]
    elif args.command == 'status':
        result = e.status()
    elif args.command == 'certify-tests':
        if not all((args.focused, args.protected, args.full)):
            parser.error('three test report paths are required')
        result = e.certify(args.focused, args.protected, args.full)
    elif args.command == 'seal':
        e.seal_report()
        result = e.verify()
        a.put(a.RUN / 'offline_verification.json', result)
    else:
        result = e.verify()
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
