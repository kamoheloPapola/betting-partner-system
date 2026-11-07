"""Explicit Phase 8 administrative CLI; no override or automatic promotion."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ml.bundle_activation import (
    ActivationError, BundleController, LEGACY_ID, PHASE7, persist_new,
    read_json, verify_activation_record, verify_seal,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--league', default='PL')
    parser.add_argument('--league-specification', type=Path)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('dry-run', 'activate'):
        cmd = sub.add_parser(name)
        cmd.add_argument('--evidence', type=Path, default=ROOT / PHASE7)
        cmd.add_argument('--expected-active', default=LEGACY_ID, required=name == 'activate')
        if name == 'activate':
            cmd.add_argument('--operator', required=True)
        else:
            cmd.add_argument('--output', type=Path, required=True)
    rollback = sub.add_parser('rollback')
    rollback.add_argument('--expected-active', required=True)
    rollback.add_argument('--operator', required=True)
    rollback.add_argument('--reason', required=True)
    recover = sub.add_parser('recover')
    recover.add_argument('--operator', required=True)
    verify = sub.add_parser('verify')
    verify.add_argument('record', type=Path)
    infrastructure = sub.add_parser('verify-infrastructure')
    infrastructure.add_argument('run_dir', type=Path)
    finalize = sub.add_parser('finalize')
    finalize.add_argument('run_dir', type=Path)
    for name in ('focused', 'protected', 'full'):
        finalize.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    if args.league == 'PL':
        if args.league_specification:
            parser.error('PL retains its frozen Phase 8 specification')
        controller = BundleController()
    else:
        from src.ml.league_bundle_activation import LeagueBundleController
        if not args.league_specification:
            parser.error('non-PL operations require an explicit --league-specification')
        controller = LeagueBundleController(args.league_specification, league=args.league)
        if getattr(args, 'expected_active', None) == LEGACY_ID and args.command == 'dry-run':
            args.expected_active = controller.legacy_id
    try:
        if args.command == 'dry-run':
            result = controller.dry_run(args.evidence, args.expected_active)
            persist_new(args.output, result)
            verify_seal(read_json(args.output))
            result = {'classification': result['classification'], 'eligibility': result['eligibility'],
                      'preflight': result['preflight']['status'], 'record': str(args.output)}
        elif args.command == 'activate':
            result = controller.activate(args.evidence, expected_active=args.expected_active, operator=args.operator)
        elif args.command == 'rollback':
            result = controller.rollback(expected_active=args.expected_active, operator=args.operator, reason=args.reason)
        elif args.command == 'recover':
            result = controller.recover(operator=args.operator)
        elif args.command == 'verify-infrastructure':
            from src.ml.bundle_evidence import verify_infrastructure
            result = verify_infrastructure(args.run_dir.resolve())
        elif args.command == 'finalize':
            from src.ml.bundle_evidence import finalize
            result = finalize(args.run_dir.resolve(), focused=args.focused, protected=args.protected, full=args.full)
        else:
            result = verify_activation_record(read_json(args.record))
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (ActivationError, OSError, ValueError, KeyError) as exc:
        print(f'Phase 8 blocked: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
