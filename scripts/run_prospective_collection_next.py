"""Manual acquisition and prospective capture in the separate new lineage."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation import prospective_collection_next as collection


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('authorize', 'acquire', 'collect', 'verify', 'status'))
    parser.add_argument('--league', choices=collection.LEAGUES)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--date-from')
    parser.add_argument('--date-to')
    args = parser.parse_args(argv)
    if args.command in ('acquire', 'collect', 'verify') and not args.league:
        parser.error(f'{args.command} requires --league')
    if args.command in ('authorize', 'status') and args.league:
        parser.error(f'{args.command} does not accept --league')
    if args.command == 'collect' and not (args.source and args.date_from and args.date_to):
        parser.error('collect requires --source, --date-from, and --date-to')
    if args.command != 'collect' and (args.source or args.date_from or args.date_to):
        parser.error('source and date filters apply only to collect')
    try:
        if args.command == 'authorize':
            result = collection.authorize()
        elif args.command == 'acquire':
            result = collection.acquire(args.league)
        elif args.command == 'collect':
            result = collection.collect(args.league, args.source,
                date_from=args.date_from, date_to=args.date_to)
        elif args.command == 'verify':
            result = collection.verify(args.league)
        else:
            collection.authorization()
            result = {'lineage': collection.RUN.relative_to(ROOT).as_posix(),
                'leagues': [{'league': league, 'recorded': len(collection.journal(league))}
                            for league in collection.LEAGUES],
                'settlement_authorized': False, 'promotion_authorized': False}
        print(json.dumps(result, sort_keys=True))
        return 0
    except (ValueError, KeyError, TypeError, RuntimeError, OSError) as exc:
        print(json.dumps({'verified': False, 'error': str(exc)}, sort_keys=True))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
