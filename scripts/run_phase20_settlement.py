"""Manually authorize, archive, settle, and audit Phase 19 prospective pairs."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation import phase20_settlement as phase20


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('authorize', 'acquire', 'settle', 'verify', 'integrity'))
    parser.add_argument('--league', choices=phase20.LEAGUES)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--authorize-settlement', action='store_true')
    args = parser.parse_args()
    try:
        if args.command == 'authorize':
            phase20.p.check(args.league is None and args.source is None and
                            args.authorize_settlement, 'PHASE20_EXPLICIT_AUTHORIZATION_REQUIRED')
            result = phase20.authorize()
        elif args.command == 'acquire':
            phase20.p.check(args.league is not None and args.source is None,
                            'PHASE20_ACQUIRE_REQUIRES_LEAGUE')
            result = phase20.acquire(args.league, authorized=args.authorize_settlement)
        elif args.command == 'settle':
            phase20.p.check(args.league is not None and args.source is not None,
                            'PHASE20_SETTLE_REQUIRES_LEAGUE_AND_SOURCE')
            result = phase20.settle(args.league, args.source,
                                    authorized=args.authorize_settlement)
        elif args.command == 'verify':
            phase20.p.check(args.source is None and not args.authorize_settlement,
                            'PHASE20_VERIFY_ARGUMENT_INVALID')
            result = (phase20.verify(args.league) if args.league else
                      {'leagues': [phase20.verify(league) for league in phase20.LEAGUES]})
        else:
            phase20.p.check(args.league is None and args.source is None and
                            not args.authorize_settlement, 'PHASE20_INTEGRITY_ARGUMENT_INVALID')
            result = phase20.integrity()
        print(json.dumps(result, sort_keys=True))
        return 0
    except (ValueError, KeyError, TypeError, RuntimeError, OSError) as exc:
        print(json.dumps({'verified': False, 'error': str(exc)}, sort_keys=True))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
