"""One controlled, crash-safe source acquisition for the five sealed v2 cohorts."""
from __future__ import annotations

from pathlib import Path

from src.evaluation import phase11_current_source as source
from src.evaluation import phase12_prospective as p
from src.evaluation import phase19_authorization as auth
from src.evaluation.phase12_operations import put

LEAGUES = ('BL1', 'SA', 'FL1', 'PD', 'PL')
INTENT = auth.RUN / 'cycle_intent.json'
ACQUISITION = auth.RUN / 'acquisition'


def intent():
    authorization = auth.verify()
    p.check(tuple(item['league'] for item in authorization['cohorts']) == LEAGUES,
            'PHASE19_LEAGUE_SET_DRIFT')
    if INTENT.exists():
        row = p.read(INTENT)
        p.frozen.sealed(row, 'record_sha256')
        p.check(row['authorization'] == p.identity(auth.AUTH) and
                row['leagues'] == list(LEAGUES), 'PHASE19_INTENT_DRIFT')
        return row
    return put(INTENT, p.seal({'schema_version': 'ffps.phase19_cycle_intent.v1',
        'authorization': p.identity(auth.AUTH), 'leagues': list(LEAGUES),
        'started_at': p.now(), 'mode': 'CONTROLLED_ONE_SHOT',
        'scheduler_authorized': False}, 'record_sha256'))


def acquire(league):
    """Exactly one attempt per league; an interrupted archive is a blocker."""
    row = intent()
    p.check(league in LEAGUES, 'PHASE19_UNKNOWN_LEAGUE')
    directory = ACQUISITION / league
    meta = directory / 'primary_matches.json'
    raw = directory / 'primary_matches.raw'
    if meta.exists():
        record = p.read(meta)
        p.check(record['request'] == {'url':
            f'https://api.football-data.org/v4/competitions/{league}/matches',
            'params': {'season': 2026}}, 'PHASE19_REQUEST_DRIFT')
        if 'raw' in record:
            p.frozen.reference(record['raw'])
        p.check(p.instant(record['started_at']) > p.instant(row['started_at']) >
                p.instant(auth.verify()['frozen_at']), 'PHASE19_STALE_SOURCE')
        return record
    p.check(not raw.exists(), 'PHASE19_PARTIAL_ARCHIVE')
    key = source.credentials()['football_data_org']
    headers = {'X-Auth-Token': key} if key else {}
    record = source.fetch(directory, 'primary_matches',
        f'https://api.football-data.org/v4/competitions/{league}/matches',
        provider='football-data.org:v4', params={'season': 2026},
        headers=headers)
    p.check(p.instant(record['started_at']) > p.instant(row['started_at']) >
            p.instant(auth.verify()['frozen_at']), 'PHASE19_STALE_SOURCE')
    return record


def acquire_all():
    intent()
    results = {}
    for league in LEAGUES:
        try:
            row = acquire(league)
            results[league] = {'status': row['status'],
                               'http_status': row.get('http_status'),
                               'retrieved_at': row['retrieved_at'],
                               'raw_sha256': row.get('raw', {}).get('sha256')}
        except (ValueError, KeyError, TypeError, RuntimeError, OSError) as exc:
            results[league] = {'status': 'ACQUISITION_BLOCKED', 'reason': str(exc)}
    return results


def recover_network_attempt(league):
    """One preserved retry after the sandbox-blocked first connection attempt."""
    row = acquire(league)
    p.check(row['status'] == 'SOURCE_UNAVAILABLE' and
            row.get('error_type') == 'ConnectionError', 'PHASE19_RECOVERY_NOT_APPLICABLE')
    directory = ACQUISITION / league
    name = 'recovery_matches'
    meta = directory / (name + '.json')
    raw = directory / (name + '.raw')
    if meta.exists():
        result = p.read(meta)
        if 'raw' in result:
            p.frozen.reference(result['raw'])
        return result
    p.check(not raw.exists(), 'PHASE19_PARTIAL_RECOVERY_ARCHIVE')
    key = source.credentials()['football_data_org']
    result = source.fetch(directory, name,
        f'https://api.football-data.org/v4/competitions/{league}/matches',
        provider='football-data.org:v4', params={'season': 2026},
        headers={'X-Auth-Token': key} if key else {})
    p.check(p.instant(result['started_at']) > p.instant(intent()['started_at']) >
            p.instant(auth.verify()['frozen_at']), 'PHASE19_STALE_RECOVERY_SOURCE')
    return result
