"""Separate manual prospective lineage over the sealed Phase 11 causal adapter.

This lineage never writes under baselines/, data/, or src/ml/models/. A frozen
qualification is usable only while a fresh provider response proves that no
newly finished league match has made its causal observations stale.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path
import re

from src.evaluation import phase10_current_data as current
from src.evaluation import phase10_shadow as capture
from src.evaluation import phase12_prospective as p
from src.evaluation import phase16_cohort as cohort
from src.evaluation import phase19_authorization as phase19
from src.evaluation import phase19_batch
from src.evaluation import phase19_integrity
from src.evaluation import phase20_settlement as phase20
from src.evaluation import phase11_current_source as source
from src.ml.bundle_activation import state_lock

LEAGUES = ('BL1', 'SA', 'FL1', 'PD')
RUN = p.ROOT / 'prospective_evidence/prospective-20260924-v1'
AUTH = RUN / 'authorization.json'
IDENTITY = ('league', 'season', 'fixture_id', 'provider_fixture_id',
            'home_id', 'away_id', 'home_team', 'away_team', 'kickoff_at')


def _gate():
    p.check(phase19_integrity.verify()['verified'], 'PHASE19_PROTECTED_INTEGRITY_FAILED')
    result = phase20.integrity()
    p.check(result['verified'], 'PHASE20_PROTECTED_INTEGRITY_FAILED:' +
            ','.join(result.get('errors', [])[:3]))


def _directory(league):
    p.check(league in LEAGUES, 'NEW_LINEAGE_UNSUPPORTED_LEAGUE')
    return RUN / league


def _bindings():
    old = phase19.verify()
    result = {}
    for league in LEAGUES:
        assigned = next(item for item in old['cohorts'] if item['league'] == league)
        spec, protocol = cohort.binding(assigned['cohort_id'])
        qualification_path = (phase19_batch.REFRESH / league /
                              'assessments/assessment-001/qualification.json')
        q = p.read(qualification_path)
        p.frozen.sealed(q, 'qualification_sha256')
        p.check(spec['source_qualification'] == assigned['source_qualification'] and
                protocol['league'] == q['league'] == league and
                q['active_bundle_id'] == assigned['active_bundle_id'] and
                q['candidate_bundle_id'] == assigned['candidate_bundle_id'],
                'NEW_LINEAGE_FROZEN_BINDING_MISMATCH')
        result[league] = {'cohort': assigned['cohort'],
                          'protocol': assigned['protocol'],
                          'qualification': p.identity(qualification_path),
                          'active_bundle_id': assigned['active_bundle_id'],
                          'candidate_bundle_id': assigned['candidate_bundle_id']}
    return result


def authorize():
    """Freeze a separate lineage; do not mutate either prior authorization."""
    _gate()
    if AUTH.exists():
        return authorization()
    value = p.seal({'schema_version': 'ffps.next_prospective_authorization.v1',
        'created_at': p.now(), 'leagues': list(LEAGUES),
        'phase19_authorization': p.identity(phase19.AUTH),
        'implementation': p.identity(Path(__file__)),
        'source_adapter': p.identity(Path(source.__file__)),
        'bindings': _bindings(), 'manual_collection_authorized': True,
        'cross_lineage_policy': 'REPEATED_FORECAST_VINTAGE; UNIQUE_FIXTURE_OUTCOME_COUNTS_ONCE',
        'settlement_authorized': False, 'scheduling_authorized': False,
        'activation_authorized': False, 'promotion_authorized': False},
        'record_sha256')
    p.persist_new(AUTH, value)
    return value


def authorization():
    row = p.read(AUTH)
    p.frozen.sealed(row, 'record_sha256')
    p.check(row['schema_version'] == 'ffps.next_prospective_authorization.v1' and
            row['leagues'] == list(LEAGUES) and row['manual_collection_authorized'] is True and
            row['cross_lineage_policy'] ==
                'REPEATED_FORECAST_VINTAGE; UNIQUE_FIXTURE_OUTCOME_COUNTS_ONCE' and
            all(row[key] is False for key in ('settlement_authorized',
                'scheduling_authorized', 'activation_authorized', 'promotion_authorized')) and
            row['phase19_authorization'] == p.identity(phase19.AUTH) and
            row['implementation'] == p.identity(Path(__file__)) and
            row['source_adapter'] == p.identity(Path(source.__file__)) and
            row['bindings'] == _bindings(), 'NEW_LINEAGE_AUTHORIZATION_DRIFT')
    return row


def journal(league):
    directory = _directory(league) / 'journal'
    rows, previous = [], None
    for number, path in enumerate(sorted(directory.glob('*.json')), 1):
        p.check(path.name == f'{number:08d}.json', 'NEW_LINEAGE_JOURNAL_GAP')
        row = p.read(path)
        p.frozen.sealed(row, 'record_sha256')
        p.check(row['sequence'] == number and row['previous'] == previous and
                row['league'] == league and row['kind'] == 'pair' and
                row['authorization'] == p.identity(AUTH),
                'NEW_LINEAGE_JOURNAL_CHAIN_INVALID')
        rows.append(row)
        previous = row['record_sha256']
    p.check(len({r['payload']['pair']['fixture_id'] for r in rows}) == len(rows),
            'NEW_LINEAGE_DUPLICATE_FIXTURE')
    return rows


def _source_path(league, attempt):
    path = Path(attempt).resolve()
    p.check(path.parent.parent == (_directory(league) / 'acquisition').resolve() and
            path.name == 'matches.json' and
            re.fullmatch(r'attempt-\d{6}', path.parent.name) is not None,
            'NEW_LINEAGE_SOURCE_OUTSIDE_ARCHIVE')
    return path


def source_record(league, attempt):
    path = _source_path(league, attempt)
    row = p.read(path)
    p.check(row['schema_version'] == 'ffps.phase11_raw_source.v1' and
            row['provider'] == 'football-data.org:v4' and
            row['request'] == {'url': f'https://api.football-data.org/v4/competitions/{league}/matches',
                               'params': {'season': 2026}} and
            row['status'] == 'RETRIEVED' and row['http_status'] == 200 and
            'x-auth-token' in row['credential_headers_present'] and
            p.instant(row['started_at']) >= p.instant(authorization()['created_at']) and
            p.instant(row['started_at']) <= p.instant(row['retrieved_at']) <= p.instant(p.now()),
            'NEW_LINEAGE_SOURCE_PROVENANCE_INVALID')
    source._raw_bytes(row)
    return row


def acquire(league):
    _directory(league)
    authorization()
    _gate()  # Before an attempt directory or network operation.
    key = source.credentials()['football_data_org']
    p.check(bool(key), 'NEW_LINEAGE_PROVIDER_CREDENTIAL_REQUIRED')
    with state_lock(_directory(league) / 'operation.lock'):
        base = _directory(league) / 'acquisition'
        attempts = sorted(base.glob('attempt-*'))
        p.check(all(item.is_dir() and re.fullmatch(r'attempt-\d{6}', item.name)
                    for item in attempts), 'NEW_LINEAGE_ATTEMPT_DIRECTORY_INVALID')
        number = max((int(item.name[-6:]) for item in attempts), default=0) + 1
        directory = base / f'attempt-{number:06d}'
        row = source.fetch(directory, 'matches',
            f'https://api.football-data.org/v4/competitions/{league}/matches',
            provider='football-data.org:v4', params={'season': 2026},
            headers={'X-Auth-Token': key})
        return {'league': league, 'source': p.identity(directory / 'matches.json'),
                'status': row['status']}


def _context(league):
    assigned = authorization()['bindings'][league]
    q, index, _ = p.frozen.load_qualification(p.frozen.reference(assigned['qualification']))
    spec, protocol = cohort.binding(next(item['cohort_id'] for item in
        phase19.verify()['cohorts'] if item['league'] == league))
    p.check(assigned['protocol'] == p.identity(cohort._directory(spec['cohort_id']) /
                                             'prospective_protocol.json'),
            'NEW_LINEAGE_PROTOCOL_DRIFT')
    _, _, controller, _, _ = p.frozen.load_protocol(p.PHASE11, league)
    p.check(controller.legacy_id == assigned['active_bundle_id'] and
            controller.candidate_id == assigned['candidate_bundle_id'] and
            q['feature_readiness'] and q['assessment']['qualified'],
            'NEW_LINEAGE_MODEL_OR_INPUT_NOT_READY')
    return q, index, protocol, controller


def _events(league, metadata, q, protocol):
    events = p.current_events(metadata, protocol, q)
    p.check(len({e['fixture_id'] for e in events}) == len(events),
            'NEW_LINEAGE_DUPLICATE_PROVIDER_FIXTURE')
    frozen = {e['fixture_id']: e for e in q['assessment']['events']
              if e['season'] == protocol['season']}
    for event in events:
        if event['status'] != 'FINISHED':
            continue
        old = frozen.get(event['fixture_id'])
        p.check(old is not None and old['status'] == 'FINISHED' and
                all(old[key] == event[key] for key in IDENTITY) and
                all(old[key] == event[key] for key in ('home_goals', 'away_goals')),
                'NEW_LINEAGE_CAUSAL_INPUTS_STALE')
    return events


def _dates(start, end):
    try:
        first, last = date.fromisoformat(start), date.fromisoformat(end)
    except (ValueError, TypeError):
        raise ValueError('NEW_LINEAGE_DATE_INVALID') from None
    p.check(first <= last and (last - first).days <= 7,
            'NEW_LINEAGE_DATE_WINDOW_INVALID')
    return first, last


def collect(league, attempt, *, date_from, date_to):
    _directory(league)
    first, last = _dates(date_from, date_to)
    auth = authorization()
    _gate()  # Fresh check at the prediction publication boundary.
    path = _source_path(league, attempt)
    with state_lock(_directory(league) / 'operation.lock'):
        metadata = source_record(league, path)
        q, index, protocol, controller = _context(league)
        events = _events(league, metadata, q, protocol)
        targets = {row['fixture_id']: row for row in q['assessment']['upcoming']}
        rows = journal(league)
        existing = {row['payload']['pair']['fixture_id']: row for row in rows}
        active, candidate = [controller.owner.load(controller.bundle(bundle))
                             for bundle in (controller.legacy_id, controller.candidate_id)]
        accepted, already, rejected = [], [], []
        for event in events:
            kickoff = p.instant(event['kickoff_at'])
            if not first <= kickoff.date() <= last:
                continue
            fixture_id = event['fixture_id']
            if fixture_id in existing:
                prior = existing[fixture_id]['payload']['fresh_fixture']
                if all(event[key] == prior[key] for key in IDENTITY):
                    already.append(fixture_id)
                else:
                    rejected.append({'fixture_id': fixture_id,
                                     'reason': 'NEW_LINEAGE_RECORDED_FIXTURE_REVISED'})
                continue
            try:
                p.check(event['status'] in ('TIMED', 'SCHEDULED') and
                        p.instant(p.now()) < kickoff,
                        'NEW_LINEAGE_NOT_PREKICKOFF')
                target = targets.get(fixture_id)
                p.check(target is not None, 'NEW_LINEAGE_UNQUALIFIED_TARGET')
                p.check(all(event[key] == target[key] for key in IDENTITY),
                        'NEW_LINEAGE_FIXTURE_IDENTITY_OR_KICKOFF_CHANGED')
                at = p.now()
                p.check(p.instant(metadata['retrieved_at']) < p.instant(at) < kickoff,
                        'NEW_LINEAGE_NOT_PREKICKOFF')
                matrix, fixture = source.target_matrix(q['assessment'], target,
                    as_of=at, source_index=index)
                current.validate_features(matrix, fixture, q['ordered_features'], at)
                fixture = fixture.copy()
                fixture['kickoff_utc'] = target['kickoff_at']
                predicted_at = p.now()
                records = [capture.capture(matrix, fixture, suite, league, predicted_at)
                           for suite in (active, candidate)]
                pair = p.seal({'league': league, 'protocol_sha256': protocol['protocol_sha256'],
                    'fixture_id': fixture_id, 'created_at': p.now(),
                    'kickoff_at': target['kickoff_at'],
                    'qualification': auth['bindings'][league]['qualification'],
                    'active': records[0], 'candidate': records[1]}, 'pair_sha256')
                p.verify_pair_features(pair, q, index, protocol, controller)
                p.check(p.instant(metadata['retrieved_at']) <= p.instant(pair['created_at']) <
                        kickoff and p.instant(p.now()) < kickoff,
                        'NEW_LINEAGE_NOT_PREKICKOFF')
                item = p.seal({'sequence': len(rows) + 1,
                    'previous': rows[-1]['record_sha256'] if rows else None,
                    'league': league, 'kind': 'pair', 'created_at': p.now(),
                    'authorization': p.identity(AUTH),
                    'payload': {'pair': pair, 'fresh_source': p.identity(path),
                                'fresh_fixture': event}}, 'record_sha256')
                p.persist_new(_directory(league) / 'journal' /
                              f'{len(rows)+1:08d}.json', item)
                rows.append(item)
                accepted.append(fixture_id)
            except (ValueError, KeyError, TypeError, RuntimeError) as exc:
                rejected.append({'fixture_id': fixture_id, 'reason': str(exc)})
        return {'league': league, 'source': p.identity(path),
                'accepted': accepted, 'already_recorded': already,
                'rejected': rejected, 'journal_head': rows[-1]['record_sha256'] if rows else None,
                'total_recorded': len(rows), 'settlement_authorized': False}


def verify(league):
    authorization()
    q, index, protocol, controller = _context(league)
    rows = journal(league)
    for row in rows:
        value = row['payload']
        pair = value['pair']
        path = p.frozen.reference(value['fresh_source'])
        metadata = source_record(league, path)
        matches = [event for event in _events(league, metadata, q, protocol)
                   if event['fixture_id'] == pair['fixture_id']]
        p.check(len(matches) == 1 and matches[0] == value['fresh_fixture'] and
                matches[0]['status'] in ('TIMED', 'SCHEDULED') and
                p.instant(metadata['retrieved_at']) <= p.instant(pair['created_at']) <
                p.instant(pair['kickoff_at']) and
                all(matches[0][key] == pair['active']['fixture'].get(key)
                    for key in ('home_team', 'away_team')),
                'NEW_LINEAGE_SOURCE_OR_FIXTURE_REPLAY_FAILED')
        p.verify_pair_features(pair, q, index, protocol, controller)
    return {'league': league, 'verified': True, 'recorded': len(rows),
            'journal_head': rows[-1]['record_sha256'] if rows else None,
            'settlement_authorized': False, 'promotion_authorized': False}
