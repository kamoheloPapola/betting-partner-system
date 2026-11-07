"""Readiness, immutable scientific finalization and offline verification only."""
from pathlib import Path
import argparse
import json
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluation import phase14_decision as d
from src.evaluation import phase14_verifier as v


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    status = sub.add_parser('status')
    status.add_argument('league', choices=(*d.a.p.source.LEAGUES, 'all'))
    final = sub.add_parser('finalize')
    final.add_argument('league', choices=d.a.p.source.LEAGUES)
    final.add_argument('--publication-id', default='final')
    final.add_argument('--finalized-at')
    verify = sub.add_parser('verify')
    verify.add_argument('decision', type=Path)
    args = parser.parse_args()
    if args.command == 'status':
        result = [d.status(league) for league in d.a.p.source.LEAGUES] if args.league == 'all' else d.status(args.league)
    elif args.command == 'finalize':
        result = d.finalize(args.league, publication_id=args.publication_id, finalized_at=args.finalized_at)
    else:
        result = v.verify(args.decision)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 2 if isinstance(result, dict) and result.get('blockers') else 0


if __name__ == '__main__':
    raise SystemExit(main())
