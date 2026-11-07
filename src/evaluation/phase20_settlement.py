"""Manual, append-only settlement of sealed Phase 19 prospective pairs."""
from __future__ import annotations

import json
from pathlib import Path
import re

import numpy as np

from src.evaluation import phase12_prospective as p
from src.evaluation import phase16_cohort as cohort
from src.evaluation import phase19_authorization as phase19
from src.evaluation import phase19_integrity
from src.evaluation import phase20_boundary
from src.evaluation import phase7_shadow as metrics
from src.evaluation.phase12_operations import put
from src.ml.bundle_activation import state_lock
from src.simulation.match_simulator import clamp_lambda

RUN = p.ROOT / 'baselines/phase20/phase20-controlled-settlement-20260923-v1'
AUTH = RUN / 'settlement_authorization.json'
APPROVED_BOUNDARY = p.ROOT / 'docs/phase20-boundary-approved-v1.json'
APPROVED_BOUNDARY_SHA256 = '1ed13b93c37bd18b17a52a256c4d8358abdb0269c8574e7e9f2fe44994b63876'
LEAGUES = ('BL1', 'SA', 'FL1', 'PD', 'PL')
IDENTITY = ('league', 'season', 'fixture_id', 'provider_fixture_id',
            'home_id', 'away_id', 'home_team', 'away_team', 'kickoff_at')


def _paths(league):
    p.check(league in LEAGUES, 'PHASE20_UNKNOWN_LEAGUE')
    return RUN / league


def _pairs(league, assignments=None):
    assignment = next(row for row in (assignments or phase19.verify()['cohorts'])
                      if row['league'] == league)
    rows = cohort.journal(assignment['cohort_id'])
    p.check(all(row['kind'] == 'pair' and row['league'] == league for row in rows),
            'PHASE19_PAIR_PREFIX_CHANGED')
    return rows


def authorize():
    """Freeze the manual scope and protected pre-settlement boundary once."""
    if AUTH.exists():
        return authorization()
    p.check(phase19_integrity.verify()['verified'], 'PHASE19_INTEGRITY_FAILED')
    index = p.read(phase19.RUN / 'pair_index.json')
    p.frozen.sealed(index, 'record_sha256')
    expected = {'BL1': 270, 'SA': 330, 'FL1': 261, 'PD': 311, 'PL': 0}
    heads = {}
    for league in LEAGUES:
        rows = _pairs(league)
        p.check(len(rows) == expected[league], 'PHASE19_PAIR_COUNT_DRIFT')
        heads[league] = rows[-1]['record_sha256'] if rows else None
    protected = {}
    for base in ('baselines', 'data', 'src/ml/models'):
        for path in sorted((p.ROOT / base).rglob('*')):
            if path.is_file() and not path.is_relative_to(RUN) and path.suffix not in ('.pyc', '.lock', '.tmp'):
                protected[path.relative_to(p.ROOT).as_posix()] = p.source.sha256_file(path)
    put(RUN / 'protected_before.json', protected)
    return put(AUTH, p.seal({'schema_version': 'ffps.phase20_settlement_authorization.v1',
        'frozen_at': p.now(), 'phase19_authorization': p.identity(phase19.AUTH),
        'phase19_pair_index': p.identity(phase19.RUN / 'pair_index.json'),
        'phase19_heads': heads, 'pair_counts': expected,
        'protected_before': p.identity(RUN / 'protected_before.json'),
        'leagues': list(LEAGUES), 'manual_result_acquisition_authorized': True,
        'manual_settlement_authorized': True, 'offline_verification_authorized': True,
        'scheduler_authorized': False, 'pair_creation_authorized': False,
        'training_authorized': False, 'calibration_authorized': False,
        'activation_authorized': False, 'promotion_authorized': False,
        'production_pointer_mutation_authorized': False,
        'provider_correction_policy': 'fail_closed'}, 'record_sha256'))


def authorization():
    row = p.read(AUTH)
    p.frozen.sealed(row, 'record_sha256')
    p.check(row['schema_version'] == 'ffps.phase20_settlement_authorization.v1' and
            row['leagues'] == list(LEAGUES) and row['manual_settlement_authorized'] is True and
            row['manual_result_acquisition_authorized'] is True and
            all(row[key] is False for key in ('scheduler_authorized', 'pair_creation_authorized',
                'training_authorized', 'calibration_authorized', 'activation_authorized',
                'promotion_authorized', 'production_pointer_mutation_authorized')) and
            row['provider_correction_policy'] == 'fail_closed' and
            row['phase19_authorization'] == p.identity(phase19.AUTH) and
            row['phase19_pair_index'] == p.identity(phase19.RUN / 'pair_index.json'),
            'PHASE20_AUTHORIZATION_INVALID')
    p.frozen.reference(row['protected_before'])
    assignments = phase19.verify()['cohorts']
    for league in LEAGUES:
        rows = _pairs(league, assignments)
        p.check(len(rows) == row['pair_counts'][league] and
                (rows[-1]['record_sha256'] if rows else None) == row['phase19_heads'][league],
                'PHASE19_PAIR_PREFIX_CHANGED')
    return row


def _source_path(league, attempt):
    path = Path(attempt).resolve()
    p.check(path.parent.parent == (_paths(league) / 'acquisition').resolve() and
            path.name == 'matches.json' and
            re.fullmatch(r'attempt-\d{6}', path.parent.name) is not None,
            'PHASE20_SOURCE_OUTSIDE_ARCHIVE')
    return path


def source_record(league, attempt, *, check_authorization=True):
    path = _source_path(league, attempt)
    row = p.read(path)
    if check_authorization:
        frozen_at = authorization()['frozen_at']
    else:
        frozen_at = p.read(AUTH)['frozen_at']
    p.check(row['schema_version'] == 'ffps.phase11_raw_source.v1' and
            row['provider'] == 'football-data.org:v4' and
            row['request'] == {'url': f'https://api.football-data.org/v4/competitions/{league}/matches',
                               'params': {'season': 2026}} and
            p.instant(row['started_at']) >= p.instant(frozen_at) and
            p.instant(row['retrieved_at']) >= p.instant(row['started_at']) and
            p.instant(row['retrieved_at']) <= p.instant(p.now()),
            'PHASE20_SOURCE_PROVENANCE_INVALID')
    p.check(row['status'] != 'RETRIEVED' or
            ('x-auth-token' in row['credential_headers_present'] and row['http_status'] == 200),
            'PHASE20_UNAUTHENTICATED_RESULT_SOURCE')
    if 'raw' in row:
        p.check(p.frozen.reference(row['raw']) == path.with_suffix('.raw'),
                'PHASE20_RAW_PATH_MISMATCH')
    return row


def acquire(league, *, authorized=False):
    p.check(authorized, 'PHASE20_OPERATOR_AUTHORIZATION_REQUIRED')
    authorization()
    gate = integrity()
    p.check(gate['verified'], 'PHASE20_PROTECTED_INTEGRITY_FAILED:' +
            ','.join(gate.get('errors', [])[:3]))
    base = _paths(league) / 'acquisition'
    with state_lock(_paths(league) / 'operation.lock'):
        attempts = sorted(base.glob('attempt-*'))
        p.check(all(re.fullmatch(r'attempt-\d{6}', path.name) and path.is_dir()
                    for path in attempts), 'PHASE20_ATTEMPT_DIRECTORY_INVALID')
        number = max((int(path.name[-6:]) for path in attempts), default=0) + 1
        directory = base / f'attempt-{number:06d}'
        p.check(not directory.exists(), 'PHASE20_ATTEMPT_COLLISION')
        from src.evaluation import phase11_current_source as source
        key = source.credentials()['football_data_org']
        p.check(bool(key), 'PHASE20_PROVIDER_CREDENTIAL_REQUIRED')
        row = source.fetch(directory, 'matches',
            f'https://api.football-data.org/v4/competitions/{league}/matches',
            provider='football-data.org:v4', params={'season': 2026},
            headers={'X-Auth-Token': key} if key else {})
        source_record(league, directory / 'matches.json')
        return {'source': p.identity(directory / 'matches.json'), 'status': row['status']}


def _events(league, source):
    p.check(source['status'] == 'RETRIEVED' and source['http_status'] == 200,
            'PHASE20_RESULT_SOURCE_UNAVAILABLE')
    raw = p.source._raw_bytes(source)
    if league == 'PL':
        # PL has no admitted pairs at this boundary; validate its archived scope.
        document = json.loads(raw)
        p.check(document.get('competition', {}).get('code') == 'PL' and
                document.get('filters', {}).get('season') == 2026 and
                isinstance(document.get('matches'), list) and
                document.get('resultSet', {}).get('count') == len(document['matches']),
                'PHASE20_PL_SOURCE_SCOPE_INVALID')
        p.check(not _pairs('PL'), 'PHASE20_PL_ADAPTER_REQUIRED_FOR_NEW_PAIRS')
        return []
    rows = _pairs(league)
    spec, protocol = cohort.binding(next(x for x in phase19.verify()['cohorts']
                                         if x['league'] == league)['cohort_id'])
    q = p.read(p.frozen.reference(rows[0]['payload']['pair']['qualification']))
    events = p.current_events(source, protocol, q)
    p.check(len({e['fixture_id'] for e in events}) == len(events),
            'PHASE20_DUPLICATE_SOURCE_FIXTURE')
    return events


def journal(league):
    directory = _paths(league) / 'journal'
    rows, previous = [], None
    for number, path in enumerate(sorted(directory.glob('*.json')), 1):
        p.check(path.name == f'{number:08d}.json', 'PHASE20_JOURNAL_GAP')
        row = p.read(path)
        p.frozen.sealed(row, 'record_sha256')
        p.check(row['sequence'] == number and row['previous'] == previous and
                row['league'] == league and row['kind'] == 'settlement',
                'PHASE20_JOURNAL_CHAIN_INVALID')
        rows.append(row)
        previous = row['record_sha256']
    return rows


def _score(pair, home, away):
    label = metrics.outcome_indices(np.array([home]), np.array([away]))
    result = {}
    for side in ('active', 'candidate'):
        record = pair[side]
        probabilities = np.asarray([record['prediction_output']['final_probabilities']], float)
        probability = metrics.probability_metric_contributions(label, probabilities)
        outputs = record['method_provenance']['descriptor']['ensemble']['member_outputs']
        lambdas = np.asarray([clamp_lambda(.6 * clamp_lambda(outputs[part+'_goals_lgbm']['effective']) +
                   .4 * clamp_lambda(outputs[part+'_goals_xgb']['effective']))
                   for part in ('home', 'away')], float)
        goals = metrics.goal_metric_contributions(np.asarray([home, away], float), lambdas)
        result[side] = {'prediction_sha256': p.digest(record),
            'log_loss': float(probability['log_loss'][0]),
            'multiclass_brier': float(probability['multiclass_brier'][0]),
            'joint_goal_squared_error': float(goals['squared_error'].mean()),
            'joint_goal_poisson_deviance': float(goals['poisson_deviance'].mean())}
    return result


def _match(pair_event, event, source):
    pair = pair_event['payload']['pair']
    original = pair_event['payload']['initial_observation']['fixture']
    p.check(all(event[key] == original[key] for key in IDENTITY),
            'PHASE20_FIXTURE_IDENTITY_OR_KICKOFF_CHANGED')
    p.check(event['status'] == 'FINISHED' and
            all(type(event[key]) is int and event[key] >= 0 for key in ('home_goals', 'away_goals')) and
            p.instant(event['kickoff_at']) < p.instant(source['started_at']) <=
            p.instant(source['retrieved_at']) <= p.instant(p.now()),
            'PHASE20_NOT_AUTHORITATIVE_FINAL')
    return pair


def _payload(pair_event, event, source_path, source, at):
    pair = _match(pair_event, event, source)
    return p.seal({'schema_version': 'ffps.phase20_settlement.v1',
        'league': pair['league'], 'phase19_pair_event': pair_event['record_sha256'],
        'pair_sha256': pair['pair_sha256'], 'fixture_id': pair['fixture_id'],
        'provider_fixture_id': event['provider_fixture_id'], 'kickoff_at': pair['kickoff_at'],
        'active_prediction_sha256': p.digest(pair['active']),
        'candidate_prediction_sha256': p.digest(pair['candidate']),
        'source': p.identity(source_path), 'raw': source['raw'],
        'normalized_result': event, 'home_goals': event['home_goals'],
        'away_goals': event['away_goals'], 'scoring_schema_version': 'ffps.phase12_metric_contributions.v1',
        'scores': _score(pair, event['home_goals'], event['away_goals']),
        'created_at': at}, 'settlement_sha256')


def settle(league, attempt, *, authorized=False):
    p.check(authorized, 'PHASE20_OPERATOR_AUTHORIZATION_REQUIRED')
    authorization()
    gate = integrity()
    p.check(gate['verified'], 'PHASE20_PROTECTED_INTEGRITY_FAILED:' +
            ','.join(gate.get('errors', [])[:3]))
    source_path = _source_path(league, attempt)
    with state_lock(_paths(league) / 'operation.lock'):
        source = source_record(league, source_path)
        events = _events(league, source)
        pairs = {row['payload']['pair']['fixture_id']: row for row in _pairs(league)}
        previous = journal(league)
        settled = {row['payload']['fixture_id']: row for row in previous}
        p.check(len(settled) == len(previous), 'PHASE20_DUPLICATE_SETTLEMENT')
        accepted, rejected = 0, []
        # Validate all candidate results before publishing any new event.
        pending = []
        for event in events:
            fixture_id = event['fixture_id']
            if fixture_id not in pairs:
                if event['status'] == 'FINISHED':
                    rejected.append({'fixture_id': fixture_id, 'reason': 'UNKNOWN_PHASE19_PAIR'})
                continue
            if event['status'] != 'FINISHED':
                continue
            try:
                pair = _match(pairs[fixture_id], event, source)
                if fixture_id in settled:
                    old = settled[fixture_id]['payload']
                    p.check(old['home_goals'] == event['home_goals'] and
                            old['away_goals'] == event['away_goals'] and
                            all(old['normalized_result'][key] == event[key] for key in IDENTITY) and
                            event['status'] == 'FINISHED', 'PHASE20_CONFLICTING_RESULT')
                else:
                    pending.append((pairs[fixture_id], event))
            except (ValueError, KeyError, TypeError) as exc:
                rejected.append({'fixture_id': fixture_id, 'reason': str(exc)})
        assessment = p.seal({'schema_version': 'ffps.phase20_result_assessment.v1',
            'league': league, 'source': p.identity(source_path),
            'rejected_results': rejected}, 'record_sha256')
        put(source_path.parent / 'assessment.json', assessment)
        p.check(not any(row['reason'] == 'PHASE20_CONFLICTING_RESULT' for row in rejected),
                'PHASE20_CONFLICTING_RESULT')
        for pair_event, event in pending:
            payload = _payload(pair_event, event, source_path, source, p.now())
            head = previous[-1]['record_sha256'] if previous else None
            row = p.seal({'sequence': len(previous)+1, 'previous': head,
                'league': league, 'kind': 'settlement', 'created_at': p.now(),
                'payload': payload})
            p.persist_new(_paths(league) / 'journal' / f'{len(previous)+1:08d}.json', row)
            previous.append(row)
            accepted += 1
        return {'league': league, 'accepted': accepted, 'rejected_results': rejected,
            'pairs': len(pairs), 'settled': len(previous), 'pending': len(pairs)-len(previous),
            'journal_head': previous[-1]['record_sha256'] if previous else None,
            'scientific_state': 'DECISION_EVIDENCE_INSUFFICIENT'}


def verify(league):
    authorization()
    pairs = {row['payload']['pair']['fixture_id']: row for row in _pairs(league)}
    rows = journal(league)
    seen, score_rows, source_cache = set(), [], {}
    for row in rows:
        value = row['payload']
        p.frozen.sealed(value, 'settlement_sha256')
        fixture_id = value['fixture_id']
        p.check(fixture_id in pairs and fixture_id not in seen and
                value['phase19_pair_event'] == pairs[fixture_id]['record_sha256'] and
                value['pair_sha256'] == pairs[fixture_id]['payload']['pair']['pair_sha256'],
                'PHASE20_ORPHAN_OR_DUPLICATE_SETTLEMENT')
        path = p.frozen.reference(value['source'])
        source = source_record(league, path, check_authorization=False)
        p.check(source['raw'] == value['raw'], 'PHASE20_SOURCE_HASH_CHANGED')
        source_key = value['source']['sha256']
        if source_key not in source_cache:
            source_cache[source_key] = _events(league, source)
        events = source_cache[source_key]
        matches = [event for event in events if event['fixture_id'] == fixture_id]
        p.check(len(matches) == 1 and matches[0] == value['normalized_result'],
                'PHASE20_RESULT_REPLAY_MISMATCH')
        pair_event = pairs[fixture_id]
        pair = _match(pair_event, matches[0], source)
        p.check(value['schema_version'] == 'ffps.phase20_settlement.v1' and
                value['league'] == league and
                value['provider_fixture_id'] == matches[0]['provider_fixture_id'] and
                value['kickoff_at'] == pair['kickoff_at'] and
                value['active_prediction_sha256'] == p.digest(pair['active']) and
                value['candidate_prediction_sha256'] == p.digest(pair['candidate']) and
                value['source'] == p.identity(path) and value['raw'] == source['raw'] and
                value['home_goals'] == matches[0]['home_goals'] and
                value['away_goals'] == matches[0]['away_goals'] and
                value['scoring_schema_version'] == 'ffps.phase12_metric_contributions.v1' and
                value['scores'] == _score(pair, matches[0]['home_goals'], matches[0]['away_goals']) and
                set(value) == {'schema_version', 'league', 'phase19_pair_event', 'pair_sha256',
                    'fixture_id', 'provider_fixture_id', 'kickoff_at', 'active_prediction_sha256',
                    'candidate_prediction_sha256', 'source', 'raw', 'normalized_result',
                    'home_goals', 'away_goals', 'scoring_schema_version', 'scores',
                    'created_at', 'settlement_sha256'} and
                p.instant(source['retrieved_at']) <=
                p.instant(value['created_at']) <= p.instant(row['created_at']),
                'PHASE20_SETTLEMENT_REPLAY_MISMATCH')
        seen.add(fixture_id)
        score_rows.append(value['scores'])
    rejected = 0
    for path in sorted((_paths(league) / 'acquisition').glob('attempt-*/assessment.json')):
        assessment = p.read(path)
        p.frozen.sealed(assessment, 'record_sha256')
        p.check(assessment['league'] == league and
                assessment['source'] == p.identity(path.parent / 'matches.json'),
                'PHASE20_ASSESSMENT_SOURCE_MISMATCH')
        rejected += len(assessment['rejected_results'])
    summary = {}
    for side in ('active', 'candidate'):
        summary[side] = {key: float(np.mean([x[side][key] for x in score_rows]))
                         for key in ('log_loss', 'multiclass_brier',
                                     'joint_goal_squared_error', 'joint_goal_poisson_deviance')} if rows else None
    return {'league': league, 'verified': True, 'pairs': len(pairs), 'settled': len(rows),
        'pending': len(pairs)-len(rows), 'rejected_results': rejected,
        'settled_fixture_ids': sorted(seen),
        'journal_head': rows[-1]['record_sha256'] if rows else None,
        'descriptive_metrics': summary, 'scientific_state': 'DECISION_EVIDENCE_INSUFFICIENT',
        'promotion_authorized': False}


def _replay_evidence_binding():
    """Keep the approved replay certificate bound without reimplementing replay."""
    approved = p.read(APPROVED_BOUNDARY)
    p.frozen.sealed(approved, 'record_sha256')
    evidence = approved['scientific_replay_evidence']
    basis = approved['approval_basis']
    p.check(evidence == basis['scientific_replay'], 'PHASE20_REPLAY_CERTIFICATE_BINDING_FAILED')
    certificate = p.read(p.frozen.reference(evidence))
    p.frozen.sealed(certificate, 'record_sha256')
    candidate = p.read(p.frozen.reference(basis['reviewed_candidate']))
    p.frozen.sealed(candidate, 'record_sha256')
    p.check(candidate['scientific_replay_evidence'] == evidence and
            candidate['additions'] == approved['additions'] and
            certificate['decision_state'] == 'REVIEW_ONLY' and
            certificate['approval_granted'] is False and
            certificate['candidate_outcomes'] == {'REPLAY_VERIFIED': 167} and
            certificate['phase19_pair_count'] == 1172,
            'PHASE20_REPLAY_CERTIFICATE_BINDING_FAILED')
    p.frozen.reference(certificate['proposal_v6'])
    p.frozen.reference(certificate['replay_script'])
    for source in certificate['phase12_verifiers']:
        p.frozen.reference(source)


def integrity():
    authorization()
    anchor = {'path': APPROVED_BOUNDARY.relative_to(p.ROOT).as_posix(),
              'sha256': APPROVED_BOUNDARY_SHA256}
    result = {}
    try:
        result = phase20_boundary.verify(p.ROOT, RUN, APPROVED_BOUNDARY,
                                         approved_sha256=APPROVED_BOUNDARY_SHA256)
        p.check(isinstance(result, dict) and type(result.get('verified')) is bool and
                isinstance(result.get('errors'), list) and
                (not result['verified'] or not result['errors']),
                'PHASE20_BOUNDARY_RESULT_INVALID')
        if result['verified']:
            _replay_evidence_binding()
    except Exception as exc:
        previous = result.get('errors') if isinstance(result, dict) else []
        result = {**(result if isinstance(result, dict) else {}), 'verified': False,
                  'errors': [*(previous if isinstance(previous, list) else []),
                             f'PHASE20_BOUNDARY_ERROR:{type(exc).__name__}:{exc}']}
    return {**result, 'approved_boundary': anchor,
            'protected_files': result.get('protected_existing', 0)}
