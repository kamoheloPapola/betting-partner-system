"""Inventory, assess and offline-verify the production league integrity matrix."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main():
    from src.evaluation.multi_league_integrity import inventory, assess, verify
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['inventory', 'assess', 'qualify', 'prepare-matrix', 'finalize', 'verify'])
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--league')
    parser.add_argument('--all-production-leagues', action='store_true')
    for name in ('focused','protected','full'):
        parser.add_argument('--'+name,type=Path)
    args = parser.parse_args()
    if args.command == 'inventory':
        result = inventory(args.run_dir.resolve())
    elif args.command in ('assess', 'qualify'):
        if not (bool(args.league) ^ args.all_production_leagues):
            parser.error('select --league or --all-production-leagues')
        if args.command == 'assess':
            result = assess(args.run_dir.resolve(), args.league)
        else:
            from src.evaluation.multi_league_integrity import qualify
            result = qualify(args.run_dir.resolve(), args.league)
    elif args.command == 'prepare-matrix':
        from src.evaluation.multi_league_integrity import prepare_matrix
        result = prepare_matrix(args.run_dir.resolve())
    elif args.command == 'finalize':
        if not all((args.focused,args.protected,args.full)):
            parser.error('finalize requires --focused, --protected and --full JUnit files')
        from src.evaluation.phase9_report import finalize
        result = finalize(args.run_dir.resolve(),focused=args.focused,protected=args.protected,full=args.full)
    else:
        result = verify(args.run_dir.resolve())
    print(json.dumps(result, indent=2))
    return 1 if result.get('verified') is False else 0


if __name__ == '__main__':
    raise SystemExit(main())
