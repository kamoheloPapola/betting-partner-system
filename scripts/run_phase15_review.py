"""Review status/evaluate/verify and the strict additive Phase14 status consumer."""
from pathlib import Path
import argparse
import json
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluation import phase15_review as review
from src.evaluation import phase15_review_verifier as verifier
from src.evaluation import phase15_phase14 as integration


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('status', 'evaluate', 'phase14-status'):
        cmd = commands.add_parser(name)
        cmd.add_argument('league', choices=review.c.a.p.source.LEAGUES)
        cmd.add_argument('fixture', type=int)
    commands.add_parser('verify').add_argument('artifact', type=Path)
    args = parser.parse_args()
    if args.command == 'verify':
        result = verifier.verify(args.artifact)
    elif args.command == 'phase14-status':
        result = integration.status(args.league, args.fixture)
    else:
        result = getattr(review, args.command)(args.league, args.fixture)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
