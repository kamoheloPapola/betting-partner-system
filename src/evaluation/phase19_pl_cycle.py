"""Deterministic no-admission evaluation of the frozen PL causal-feature boundary."""
from __future__ import annotations

import pandas as pd

from src.evaluation import phase12_prospective as p
from src.evaluation import phase17_pl_adapter as adapter
from src.evaluation import phase18_verifier
from src.evaluation import phase19_authorization as auth
from src.evaluation import phase19_collection as collection
from src.evaluation.phase12_operations import put

OUTPUT = auth.RUN / 'PL_cycle.json'
FEATURE_MATRIX = p.ROOT / 'data/features/feature_matrix.csv'


def reconstruct():
    authorization = auth.verify()
    assigned = next(row for row in authorization['cohorts'] if row['league'] == 'PL')
    metadata = collection.recover_network_attempt('PL')
    p.check(metadata['status'] == 'RETRIEVED' and metadata['http_status'] == 200 and
            p.instant(metadata['started_at']) > p.instant(authorization['frozen_at']),
            'PHASE19_PL_SOURCE_NOT_FRESH')
    raw = p.frozen.reference(metadata['raw'])
    sidecar = collection.ACQUISITION / 'PL/recovery_matches.source.json'
    source, events = adapter.normalize_snapshot(sidecar, raw)
    p.check(source['retrieved_at'] == metadata['retrieved_at'] and
            source['response_sha256'] == metadata['raw']['sha256'],
            'PHASE19_PL_SOURCE_SIDECAR_MISMATCH')
    current = p.instant(metadata['retrieved_at'])
    end = p.instant(assigned['end_at'])
    future = [event for event in events if event['status'] in ('TIMED', 'SCHEDULED') and
              current < p.instant(event['kickoff_at']) < end]
    frame = pd.read_csv(FEATURE_MATRIX, low_memory=False)
    frame = frame.loc[frame['league'].astype(str).eq('PL')].copy()
    frame['date'] = pd.to_datetime(frame['date'], utc=True, errors='coerce')
    ready = frame.loc[(frame['date'] > pd.Timestamp(current)) &
                      (frame['date'] < pd.Timestamp(end)) &
                      frame['home_score'].isna() & frame['away_score'].isna()]
    keyed = {str(value) for value in ready['match_hash']}
    eligible = [event for event in future if event['phase7p_canonical_fixture_id'] in keyed]
    p.check(not eligible, 'PHASE19_PL_CAPTURE_IMPLEMENTATION_REQUIRED')
    phase18_verifier.verify_pl_cohort()
    return p.seal({'schema_version': 'ffps.phase19_pl_cycle.v1',
        'authorization': p.identity(auth.AUTH), 'cohort': assigned['cohort'],
        'source': p.identity(collection.ACQUISITION / 'PL/recovery_matches.json'),
        'raw': metadata['raw'], 'sidecar': p.identity(sidecar),
        'normalized_sha256': p.digest(events), 'fixtures_observed': len(events),
        'future_within_horizon': len(future), 'eligible_with_causal_features': 0,
        'rejections': [{'fixture_id': event['fixture_id'],
                        'reason': 'NO_FROZEN_CAUSAL_FEATURE_ROW'} for event in future],
        'feature_matrix': p.identity(FEATURE_MATRIX),
        'feature_rows_in_window_without_outcome': len(ready),
        'pairs_admitted': 0, 'settlements': 0, 'journal_head': None,
        'horizon_end': assigned['end_at'], 'scheduler_authorized': False},
        'record_sha256')


def run():
    return put(OUTPUT, reconstruct())


def verify():
    saved = p.read(OUTPUT)
    p.frozen.sealed(saved, 'record_sha256')
    p.check(saved == reconstruct(), 'PHASE19_PL_REPLAY_MISMATCH')
    return saved
