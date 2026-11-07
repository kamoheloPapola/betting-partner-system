"""Operate the sealed Phase 19 one-shot collection; never schedule or settle."""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation import phase12_prospective as p
from src.evaluation import phase16_cohort
from src.evaluation import phase19_authorization as auth
from src.evaluation import phase19_batch as batch
from src.evaluation import phase19_collection as collection
from src.evaluation import phase19_integrity
from src.evaluation import phase19_pl_cycle
from src.evaluation import phase19_verifier


class VerificationProgress:
    """Diagnostic JSON on stderr; never part of the scientific result."""
    def __init__(self):
        self.started = time.monotonic()
        self.completed = {}
        self.total = None
        self.last_progress = None
        # The frozen index supplies an announced total only. Actual completion
        # comes from the verifier callbacks; this metadata cannot admit a pair.
        try:
            index = p.read(auth.RUN / 'pair_index.json')
            self.total = len(index['pairs'])
        except (OSError, ValueError, KeyError, TypeError):
            pass
        self.event('verification_started', None, 0, None, None)

    def event(self, event, league=None, completed=None, total=None, fixture_id=None):
        self.last_progress = datetime.now(timezone.utc).isoformat()
        if league is not None and completed is not None:
            self.completed[league] = completed
        print(json.dumps({'stream': 'phase19_verification_progress',
            'event': event, 'league': league, 'fixture_id': fixture_id,
            'league_completed_pairs': completed, 'league_total_pairs': total,
            'completed_pairs': sum(self.completed.values()),
            'announced_total_pairs': self.total,
            'elapsed_seconds': time.monotonic() - self.started,
            'last_progress_at_utc': self.last_progress}, sort_keys=True),
            file=sys.stderr, flush=True)

    def checkpoint(self, league, completed, total):
        self.event('checkpoint', league, completed, total)


def status():
    frozen = auth.verify()
    rows = []
    for item in frozen['cohorts']:
        league = item['league']
        raw = collection.ACQUISITION / league / 'recovery_matches.json'
        journal = (phase16_cohort.journal(item['cohort_id']) if league != 'PL' else [])
        rows.append({'league': league, 'cohort_id': item['cohort_id'],
            'source_status': p.read(raw)['status'] if raw.exists() else 'NOT_ACQUIRED',
            'pairs': sum(row['kind'] == 'pair' for row in journal),
            'observations': sum(row['kind'] == 'observation' for row in journal),
            'settlements': sum(row['kind'] == 'settlement' for row in journal),
            'journal_head': journal[-1]['record_sha256'] if journal else None,
            'horizon_end': item['end_at']})
    return {'authorization': p.identity(auth.AUTH), 'leagues': rows,
            'scheduler_authorized': False, 'activation_authorized': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('status', 'acquire', 'collect', 'verify', 'integrity'))
    parser.add_argument('--league', choices=('BL1', 'SA', 'FL1', 'PD', 'PL'))
    args = parser.parse_args()
    progress = None
    try:
        if args.command == 'status':
            result = status()
        elif args.command == 'acquire':
            p.check(args.league is None, 'ACQUISITION_REQUIRES_ALL_FIVE_LEAGUES')
            result = collection.acquire_all()
        elif args.command == 'collect':
            p.check(args.league is not None, 'COLLECTION_REQUIRES_EXPLICIT_LEAGUE')
            result = (phase19_pl_cycle.run() if args.league == 'PL' else
                      batch.collect_nonpl(args.league))
        elif args.command == 'verify':
            p.check(args.league is None, 'VERIFY_REQUIRES_COMPLETE_COHORT')
            progress = VerificationProgress()
            result = phase19_verifier.all_leagues(
                progress=progress.checkpoint, progress_event=progress.event)
        else:
            result = phase19_integrity.verify()
        print(json.dumps(result, sort_keys=True), flush=True)
        if progress:
            progress.event('verification_succeeded')
        return 0
    except (ValueError, KeyError, TypeError, RuntimeError, OSError) as exc:
        if progress:
            progress.event('verification_failed')
        print(json.dumps({'verified': False, 'error': str(exc)}, sort_keys=True))
        return 1
    except KeyboardInterrupt:
        if progress is None:
            raise
        progress.event('verification_interrupted')
        print(json.dumps({'verified': False, 'error': 'VERIFICATION_INTERRUPTED'}, sort_keys=True), flush=True)
        return 130
    except BaseException:
        if progress:
            progress.event('verification_failed')
        raise


if __name__ == '__main__':
    raise SystemExit(main())
