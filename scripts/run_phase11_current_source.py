"""CLI for immutable Phase 11 current-source evidence."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.evaluation import phase11_current_source as phase11
from src.evaluation import phase11_shadow


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("prepare")
    diagnose = commands.add_parser("diagnose-sa")
    diagnose.add_argument("--attempt", default="initial")
    recovery = commands.add_parser("diagnose-sa-recovery")
    recovery.add_argument("--attempt", default="recovery")
    sofascore = commands.add_parser("diagnose-sa-sofascore")
    sofascore.add_argument("--attempt", default="sofascore")
    espn = commands.add_parser("diagnose-sa-espn")
    espn.add_argument("--attempt", default="espn")
    fotmob = commands.add_parser("diagnose-sa-fotmob")
    fotmob.add_argument("--attempt", default="fotmob")
    acquire = commands.add_parser("acquire")
    acquire.add_argument("--league", required=True)
    acquire.add_argument("--attempt", default="qualified")
    current = commands.add_parser("acquire-fotmob-current")
    current.add_argument("--league", required=True)
    current.add_argument("--attempt", default="fotmob-current")
    qualify_sa = commands.add_parser("qualify-sa")
    qualify_sa.add_argument("--source-attempt", default="sa-qualified")
    qualify_sa.add_argument("--fotmob-attempt", default="sa-fotmob-qualified")
    qualify = commands.add_parser("qualify")
    qualify.add_argument("--league", required=True)
    qualify.add_argument("--source-attempt", required=True)
    qualify.add_argument("--fotmob-attempt", required=True)
    freeze = commands.add_parser("freeze")
    freeze.add_argument("--league", required=True)
    freeze.add_argument("--qualification", type=Path, required=True)
    collect = commands.add_parser("collect-one")
    collect.add_argument("--league", required=True)
    commands.add_parser("verify-phase10")
    commands.add_parser("finalize")
    commands.add_parser("verify")
    args = parser.parse_args()
    if args.command == "prepare":
        result = phase11.prepare(args.run)
    elif args.command == "diagnose-sa":
        result = phase11.diagnose_sa(args.run, args.attempt)
    elif args.command == "diagnose-sa-recovery":
        result = phase11.diagnose_sa_recovery(args.run, args.attempt)
    elif args.command == "diagnose-sa-sofascore":
        result = phase11.diagnose_sa_sofascore(args.run, args.attempt)
    elif args.command == "diagnose-sa-espn":
        result = phase11.diagnose_sa_espn(args.run, args.attempt)
    elif args.command == "diagnose-sa-fotmob":
        result = phase11.diagnose_sa_fotmob(args.run, args.attempt)
    elif args.command == "acquire":
        result = phase11.acquire_league(args.run, args.league, args.attempt)
    elif args.command == "acquire-fotmob-current":
        result = phase11.acquire_fotmob_current(args.run, args.league, args.attempt)
    elif args.command == "qualify-sa":
        result = phase11.qualify_sa(args.run, args.source_attempt, args.fotmob_attempt)
    elif args.command == "qualify":
        result = phase11.qualify_league(
            args.run, args.league, args.source_attempt, args.fotmob_attempt)
    elif args.command == "freeze":
        result = phase11_shadow.freeze(args.run, args.league, args.qualification)
    elif args.command == "collect-one":
        result = phase11_shadow.collect_one(args.run, args.league)
    else:
        from src.evaluation import phase11_evidence
        result = getattr(phase11_evidence, args.command.replace("-", "_"))(args.run)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
