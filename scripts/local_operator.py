"""Personal-PC view of recorded scientific evidence. No new prediction cycle."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import local_operator as operator


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('status', 'predictions', 'new-predictions',
                                            'audit', 'settle', 'performance'))
    parser.add_argument('--league', choices=operator.LEAGUES)
    parser.add_argument('--fixture-id')
    parser.add_argument('--all', action='store_true', help='Show every matching recorded prediction')
    parser.add_argument('--limit', type=int, default=25)
    parser.add_argument('--full-replay', action='store_true', help='Reconstruct all Phase 19 pairs during audit')
    parser.add_argument('--json', action='store_true', help='Print the complete machine-readable result')
    args = parser.parse_args(argv)
    if args.command == 'settle' and not args.league:
        parser.error('settle requires --league')
    if args.command not in ('predictions', 'new-predictions') and (args.fixture_id or args.all or args.limit != 25):
        parser.error('prediction filters require the predictions command')
    if args.full_replay and args.command != 'audit':
        parser.error('--full-replay requires audit')
    if args.league and args.command not in ('predictions', 'new-predictions', 'settle'):
        parser.error('--league applies to predictions or settle')
    if args.limit < 1:
        parser.error('--limit must be positive')
    if not args.json:
        print('Checking sealed local evidence; this may take a few minutes.',
              file=sys.stderr, flush=True)
    try:
        if args.command == 'status':
            result = operator.snapshot()
        elif args.command == 'predictions':
            result = operator.predictions(league=args.league, fixture_id=args.fixture_id,
                                          all_rows=args.all, limit=args.limit)
        elif args.command == 'new-predictions':
            result = operator.new_predictions(league=args.league, fixture_id=args.fixture_id,
                                              all_rows=args.all, limit=args.limit)
        elif args.command == 'audit':
            result = operator.snapshot(audit=True, full_replay=args.full_replay)
        elif args.command == 'performance':
            result = operator.performance()
        else:
            result = operator.settle(args.league)
    except (ValueError, KeyError, TypeError, RuntimeError, OSError) as exc:
        print(f'Operation unavailable: {exc}', file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(result, sort_keys=True, default=str))
    elif args.command in ('status', 'audit'):
        p19 = result['phase19_integrity']
        if p19['verified']:
            info = p19['value']
            print('Phase 19 integrity:', 'VERIFIED' if info['verified'] else
                  f"BLOCKED: {len(info['changed'])} changed, {len(info['missing'])} missing, "
                  f"{len(info['production_changes'])} production trees changed")
        else:
            print('Phase 19 integrity:', p19['error'])
        p20 = result['phase20_integrity']
        if p20['verified']:
            info = p20['value']
            print('Phase 20 integrity:', 'VERIFIED' if info['verified'] else
                  f"BLOCKED: {', '.join(info.get('errors', [])[:3]) or 'boundary verification failed'}")
            boundary = info.get('approved_boundary')
            if boundary:
                print(f"  Approved boundary: {boundary['path']} | SHA-256 {boundary['sha256']}")
            if 'authorized_append_only_additions' in info:
                print(f"  Historical additions: {info['authorized_append_only_additions']} "
                      f"authorized append-only, {info['quarantined_historical_evidence']} "
                      f"quarantined, {info['terminated_incomplete_evidence']} terminated/incomplete; "
                      f"{info['unauthorized_additions']} unauthorized, "
                      f"{info['unknown_additions']} unknown, "
                      f"{info['unfinished_additions']} unfinished")
        else:
            print('Phase 20 integrity:', p20['error'])
        print('League  Cohort         Recorded  Settled  Pending  Current  Historical  Settlement')
        for row in result['leagues']:
            current = row['current'].get('value', {})
            history = row['historical'].get('value', {})
            pending = row['pending']
            if pending is None and row['settled'] is not None and row['recorded_predictions'] is not None:
                pending = row['recorded_predictions'] - row['settled']
            print(f"{row['league']:<6}  {current.get('cohort_id', '?'):<13}  "
                  f"{str(row['recorded_predictions']):>8}  {str(row['settled']):>7}  "
                  f"{str(pending):>7}  "
                  f"{'VERIFIED' if row['current_lineage_verified'] else 'BLOCKED':<8}  "
                  f"{history.get('state', 'UNVERIFIED'):<39}  {row['settlement_eligibility']}")
            if current:
                print(f"  Active model: {current['active_model']} | "
                      f"last archived source: {current['source_retrieved_at']}")
            for incident in history.get('incidents', []):
                print(f"  Historical {row['league']} event {incident['sequence']}: {incident['code']}")
            for label in ('current', 'historical', 'settlement'):
                if not row[label]['verified']:
                    print(f"  {row['league']} {label} verification: {row[label]['error']}")
            if args.command == 'audit':
                print(f"  {row['league']} journal head: {current.get('journal_head') or 'empty'}")
        print('Phase 19 new prediction cycle: UNAVAILABLE | Scheduling: UNAVAILABLE | Promotion: UNAVAILABLE')
    elif args.command in ('predictions', 'new-predictions'):
        for row in result['predictions']:
            labels = row['active']['class_order']
            active = ', '.join(f'{key}={value:.3f}' for key, value in
                               zip(labels, row['active']['final_probabilities']))
            candidate_labels = row['candidate']['class_order']
            candidate = ', '.join(f'{key}={value:.3f}' for key, value in
                                  zip(candidate_labels, row['candidate']['final_probabilities']))
            print(f"{row['league']} {row['fixture_id']} {row['kickoff_at']} "
                  f"{row['home_team']} v {row['away_team']} [{row['fixture_state']}]")
            print(f"  {row['state']} | active: {active}")
            print(f'  candidate: {candidate} | replay: {row["replay"]}')
        for error in result['errors']:
            print(f"{error['league']} BLOCKED: {error['error']}")
        print(f"Showing {len(result['predictions'])} of {result['total_matching']} recorded predictions")
        if result['total_matching'] == 0:
            print('NO RECORDED PREDICTION for the selected fixtures')
        if args.command == 'predictions' and (args.league == 'PL' or not args.league):
            print('PL: NO RECORDED PREDICTION')
        if args.command == 'predictions':
            print('Phase 19 new prediction cycle: UNAVAILABLE')
        else:
            print('New lineage settlement: UNAVAILABLE | Model promotion: UNAVAILABLE')
    elif args.command == 'performance':
        for row in result['leagues']:
            print(f"{row['league']}: {row['settled']} settled / {row['pairs']} recorded; "
                  f"{row['evidence']}")
            if row['descriptive_metrics']:
                print(json.dumps(row['descriptive_metrics'], sort_keys=True))
        print('Model promotion: UNAVAILABLE')
    else:
        print(result['reason'] if result['status'] == 'BLOCKED' else
              f"{result['result']['accepted']} new settlements; "
              f"{result['result']['pending']} predictions pending")
        if result.get('detail'):
            print(result['detail'])
        if result.get('unapproved_additions') is not None:
            print(f"Protected boundary: {result['unapproved_additions']} unapproved additions, "
                  f"{result['changed']} changed, {result['missing']} missing")
    if args.command == 'audit':
        if (not result['phase19_integrity']['verified'] or
            not result['phase19_integrity']['value']['verified'] or
            not result['phase20_integrity']['verified'] or
            not result['phase20_integrity']['value']['verified'] or
            any(not row['current_lineage_verified'] or
                not row['historical']['verified'] or
                not row['settlement']['verified'] for row in result['leagues'])):
            return 2
    if args.command in ('predictions', 'new-predictions') and result['errors']:
        return 2
    if args.command == 'performance' and any(not row['verified'] for row in result['leagues']):
        return 2
    return 2 if result.get('status') == 'BLOCKED' else 0


if __name__ == '__main__':
    raise SystemExit(main())
