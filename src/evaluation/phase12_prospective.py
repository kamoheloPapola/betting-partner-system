"""Frozen Phase 11 experiment continuation. No production mutation API.

The authoritative cohort is the bound Phase 11 prefix plus this league's ordered
append-only journal. Pairs are sealed inside the atomic journal publication, so
there is no unsealed-pair recovery window. OS locks are released on process death.
"""
from __future__ import annotations

import copy
from pathlib import Path
import uuid

import numpy as np

from src.evaluation import phase10_current_data as p10
from src.evaluation import phase10_shadow as capture
from src.evaluation import phase11_current_source as source
from src.evaluation import phase11_shadow as frozen
from src.evaluation import phase11_evidence as evidence11
from src.evaluation import phase7_shadow as metrics7
from src.evaluation.league_prospective import validate_pair
from src.ml.bundle_activation import persist_new, state_lock

ROOT = source.ROOT
PHASE11 = ROOT / 'baselines/phase11/phase11-current-source-20260906-v1'
check, read, identity, digest = source.check, source.read, source.identity, source.sha256_json
now, instant = source.now, p10.instant


def seal(value, key='record_sha256'):
    return {**value, key: digest(value)}


def run_path(path):
    path = Path(path).resolve()
    base = ROOT / 'baselines/phase12'
    check(path.is_relative_to(base) and path != base, 'PHASE12_PATH_ESCAPE')
    return path


def snapshot():
    result = {}
    for base in ('baselines', 'data', 'src/ml/models'):
        for path in sorted((ROOT / base).rglob('*')):
            if path.is_file() and not path.is_relative_to(ROOT / 'baselines/phase12'):
                result[path.relative_to(ROOT).as_posix()] = source.sha256_file(path)
    return result


def prepare(path):
    path = run_path(path)
    check(not path.exists(), 'RUN_ALREADY_EXISTS')
    protected = snapshot()
    index = read(PHASE11 / 'artifact_index.json')
    for ref in index['artifacts'] + index['implementation_sources']:
        frozen.reference(ref)
    old = read(PHASE11 / 'protected_before.json')
    check(all(protected.get(k) == v for k, v in old.items()), 'PHASE11_PROTECTED_DRIFT')
    check(read(PHASE11 / 'offline_verification.json')['verified'], 'PHASE11_NOT_VERIFIED')
    leagues = {}
    for league in source.LEAGUES:
        protocol, pairs, settlements = frozen.verify_cohort(PHASE11, league)
        check(pairs and not settlements, 'INVALID_PHASE11_PREFIX')
        cohort = ROOT / protocol['cohort_path']
        leagues[league] = {'protocol': identity(PHASE11 / league / 'prospective_protocol.json'),
                           'prefix': [identity(p) for p in sorted((cohort / 'predictions').glob('*.json'))]}
    persist_new(path / 'protected_before.json', protected)
    persist_new(path / 'run_specification.json', seal({
        'schema_version': 'ffps.phase12_run.v1', 'created_at': now(), 'leagues': leagues,
        'phase11_index': identity(PHASE11 / 'artifact_index.json'),
        'cohort_policy': 'complete immutable Phase11 prefix plus ordered league continuation journal',
        'promotion_authorized': False, 'mixed_active_generations_authorized': False,
        'protected_before': identity(path / 'protected_before.json'),
        'production_inventory': {base: source._tree_snapshot(ROOT / base)
                                 for base in ('data', 'src/ml/models', 'baselines/phase7p')},
    }))
    return {'prepared': True, 'protected_count': len(protected), 'leagues': list(leagues)}


def specification(path):
    path = run_path(path)
    spec = read(path / 'run_specification.json')
    frozen.sealed(spec, 'record_sha256')
    check(list(spec['leagues']) == sorted(source.LEAGUES), 'LEAGUE_INVENTORY_DRIFT')
    frozen.reference(spec['phase11_index'])
    frozen.reference(spec['protected_before'])
    return spec


def protocol_context(path, league):
    spec = specification(path)
    check(league in source.LEAGUES, 'WRONG_LEAGUE')
    ref = spec['leagues'][league]['protocol']
    frozen.reference(ref)
    return frozen.load_protocol(PHASE11, league)


def journal(path, league):
    directory = run_path(path) / league / 'journal'
    events, previous = [], None
    for number, item in enumerate(sorted(directory.glob('*.json')), 1):
        check(item.name == f'{number:08d}.json', 'JOURNAL_MEMBER_MISSING_OR_REORDERED')
        event = read(item)
        frozen.sealed(event, 'record_sha256')
        check(event['sequence'] == number and event['previous'] == previous and
              event['league'] == league, 'JOURNAL_CHAIN_OR_LEAGUE_MISMATCH')
        check(event['kind'] in ('pair', 'settlement', 'incident'), 'UNKNOWN_JOURNAL_KIND')
        events.append(event)
        previous = event['record_sha256']
    # Check every previously published checkpoint, including older heads.
    for item in sorted((run_path(path) / league / 'checkpoints').glob('*.json')):
        checkpoint = read(item)
        frozen.sealed(checkpoint, 'record_sha256')
        count = checkpoint['journal_count']
        check(count <= len(events), 'JOURNAL_TAIL_REMOVED')
        check(checkpoint['journal_head'] == (events[count-1]['record_sha256'] if count else None),
              'CHECKPOINT_CHAIN_MISMATCH')
    return events


def append(path, league, kind, payload):
    """Caller owns the league OS lock. Atomic publication includes the seal."""
    events = journal(path, league)
    event = seal({'sequence': len(events)+1, 'previous': events[-1]['record_sha256'] if events else None,
                  'league': league, 'kind': kind, 'created_at': now(), 'payload': payload})
    persist_new(run_path(path) / league / 'journal' / f'{len(events)+1:08d}.json', event)
    return event


def current_events(metadata, protocol, qualification):
    check(metadata['provider'] == 'football-data.org:v4', 'UNAUTHORIZED_RESULT_SOURCE')
    request = metadata['request']
    check(request == {'url': f"https://api.football-data.org/v4/competitions/{protocol['league']}/matches",
                      'params': {'season': protocol['season']}}, 'RESULT_SOURCE_SCOPE_MISMATCH')
    check(instant(metadata['started_at']) <= instant(metadata['retrieved_at']) <= instant(now()),
          'INVALID_RETRIEVAL_TIME')
    events, _ = source._adapt_primary(metadata, protocol['league'], protocol['season'],
                                     qualification['assessment']['team_mappings'][str(protocol['season'])])
    return events


def same_fixture(event, target, *, admitted):
    for key in ('league', 'season', 'fixture_id', 'provider_fixture_id', 'home_id', 'away_id',
                'home_team', 'away_team'):
        check(event[key] == target[key], f'FIXTURE_IDENTITY_DISAGREEMENT:{key}')
    if admitted:
        check(instant(event['kickoff_at']) == instant(target['kickoff_at']), 'ADMITTED_KICKOFF_REVISION')


def settlement_value(pair, target, event, metadata, protocol):
    same_fixture(event, target, admitted=True)
    check(event['status'] == 'FINISHED', 'RESULT_NOT_FINISHED:' + str(event['status']))
    check(instant(pair['kickoff_at']) == instant(event['kickoff_at']) <
          instant(metadata['retrieved_at']) <= instant(now()), 'SETTLEMENT_BEFORE_KICKOFF')
    check(all(type(event[k]) is int and event[k] >= 0 for k in ('home_goals', 'away_goals')),
          'INVALID_SETTLEMENT_SCORE')
    return seal({'league': pair['league'], 'fixture_id': pair['fixture_id'],
                 'protocol_sha256': protocol['protocol_sha256'], 'prediction_pair_sha256': pair['pair_sha256'],
                 'created_at': now(), 'home_goals': event['home_goals'], 'away_goals': event['away_goals'],
                 'source': metadata, 'normalized_result': event}, 'settlement_sha256')


def verify_settlement(record, pair, target, protocol, qualification):
    frozen.sealed(record, 'settlement_sha256')
    check(record['league'] == pair['league'] and record['fixture_id'] == pair['fixture_id'] and
          record['protocol_sha256'] == protocol['protocol_sha256'] and
          record['prediction_pair_sha256'] == pair['pair_sha256'], 'SETTLEMENT_PAIR_SCOPE_MISMATCH')
    events = current_events(record['source'], protocol, qualification)
    matches = [e for e in events if e['fixture_id'] == pair['fixture_id']]
    check(len(matches) == 1, 'SETTLEMENT_FIXTURE_MISSING')
    event = matches[0]
    same_fixture(event, target, admitted=True)
    check(event == record['normalized_result'] and event['status'] == 'FINISHED', 'SETTLEMENT_RESULT_DRIFT')
    check([event['home_goals'], event['away_goals']] == [record['home_goals'], record['away_goals']],
          'SETTLEMENT_SCORE_DRIFT')
    check(instant(pair['kickoff_at']) < instant(record['source']['retrieved_at']) <=
          instant(record['created_at']) <= instant(now()), 'SETTLEMENT_TIME_DRIFT')


def pair_qualification(pair, original, original_index):
    if pair['qualification'] == original['qualification_ref']:
        return original['value'], original_index
    q, index, _ = frozen.load_qualification(frozen.reference(pair['qualification']))
    evidence11.replay_source_evidence(q)
    old = original['value']
    for field in ('league', 'active_bundle_id', 'candidate_bundle_id', 'active_sha256',
                  'candidate_sha256', 'ordered_features', 'feature_schema_sha256', 'phase9_deployment'):
        check(q[field] == old[field], 'REFRESH_CONTRACT_DRIFT:' + field)
    check(q['assessment']['season'] == old['assessment']['season'], 'REFRESH_SEASON_DRIFT')
    check(q['assessment']['team_mappings'] == old['assessment']['team_mappings'], 'REFRESH_MAPPING_DRIFT')
    return q, index


def verify_pair_features(pair, q, index, protocol, controller):
    validate_pair(pair, controller=controller, protocol=protocol)
    check(instant(protocol['start_at']) <= instant(pair['created_at']) < instant(pair['kickoff_at']),
          'INVALID_ADMISSION_TIME')
    targets = [e for e in q['assessment']['upcoming'] if e['fixture_id'] == pair['fixture_id']]
    check(len(targets) == 1, 'UNKNOWN_ADMITTED_TARGET')
    target = targets[0]
    check(target['kickoff_at'] == pair['kickoff_at'], 'PAIR_KICKOFF_DRIFT')
    for raw in index['sources']:
        source._raw_bytes(raw)
        check(instant(raw['retrieved_at']) <= instant(pair['created_at']), 'LATE_SOURCE_RETRIEVAL')
    matrix, fixture = source.target_matrix(q['assessment'], target, as_of=pair['created_at'], source_index=index)
    p10.validate_features(matrix, fixture, q['ordered_features'], pair['created_at'])
    expected = fixture[q['ordered_features']].to_numpy(float)
    for record in (pair['active'], pair['candidate']):
        check(record['fixture']['home_team'] == target['home_team'] and
              record['fixture']['away_team'] == target['away_team'], 'PAIR_ORIENTATION_MISMATCH')
        for values in record['feature_provenance']['model_inputs']:
            check(np.array_equal(np.asarray(values['feature_values']), expected), 'CAUSAL_FEATURE_REPLAY_MISMATCH')
    return target


def load_cohort(path, league):
    protocol, original_cohort, controller, original, index = protocol_context(path, league)
    evidence11.replay_source_evidence(original)
    spec = specification(path)['leagues'][league]
    check(spec['prefix'] == [identity(item) for item in sorted((original_cohort / 'predictions').glob('*.json'))],
          'INCOMPLETE_OR_REORDERED_PHASE11_PREFIX')
    pairs = [read(frozen.reference(ref)) for ref in spec['prefix']]
    events = journal(path, league)
    pairs.extend(e['payload'] for e in events if e['kind'] == 'pair')
    settlements = [e['payload'] for e in events if e['kind'] == 'settlement']
    check(len(pairs) == len({p['fixture_id'] for p in pairs}), 'DUPLICATE_PAIR')
    check(len(settlements) == len({s['fixture_id'] for s in settlements}), 'DUPLICATE_SETTLEMENT')
    by_id, targets = {}, {}
    original_context = {'qualification_ref': protocol['qualification'], 'value': original}
    qualifications = {}
    for pair in pairs:
        key = digest(pair['qualification'])
        if key not in qualifications:
            qualifications[key] = pair_qualification(pair, original_context, index)
        q, sources = qualifications[key]
        targets[pair['fixture_id']] = verify_pair_features(pair, q, sources, protocol, controller)
        by_id[pair['fixture_id']] = pair
    # Publication order is separately chained; pair capture order must be monotone.
    check([p['created_at'] for p in pairs] == sorted(p['created_at'] for p in pairs), 'PAIR_ADMISSION_REORDERED')
    for settled in settlements:
        check(settled['fixture_id'] in by_id, 'ORPHAN_SETTLEMENT')
        verify_settlement(settled, by_id[settled['fixture_id']], targets[settled['fixture_id']], protocol, original)
    seen = {p['fixture_id'] for p in pairs[:len(spec['prefix'])]}
    for event in events:
        payload = event['payload']
        if event['kind'] == 'pair':
            check(instant(payload['created_at']) <= instant(event['created_at']) < instant(payload['kickoff_at']),
                  'LATE_PAIR_PUBLICATION')
            seen.add(payload['fixture_id'])
        elif event['kind'] == 'settlement':
            check(payload['fixture_id'] in seen, 'SETTLEMENT_PRECEDES_PAIR')
    return protocol, controller, original, pairs, settlements, targets, events


def evaluate(protocol, pairs, settlements, as_of):
    """Only called after complete-cohort verification; no selection parameter."""
    by_id = {s['fixture_id']: s for s in settlements}
    completed = [p for p in pairs if p['fixture_id'] in by_id]
    check(len(completed) == len(settlements), 'INCOMPLETE_METRIC_JOIN')
    horizon_complete = instant(as_of) >= instant(protocol['end_at'])
    complete = len(completed) == len(pairs)
    reliability, contributions, summary = {}, {}, {}
    intervals = None
    if completed:
        from src.simulation.match_simulator import clamp_lambda
        actual = np.asarray([[by_id[p['fixture_id']]['home_goals'], by_id[p['fixture_id']]['away_goals']]
                             for p in completed], float)
        labels = metrics7.outcome_indices(actual[:, 0], actual[:, 1])
        for label in ('active', 'candidate'):
            probabilities, lambdas = [], []
            for pair in completed:
                record = pair[label]
                probabilities.append(record['prediction_output']['final_probabilities'])
                outputs = record['method_provenance']['descriptor']['ensemble']['member_outputs']
                lambdas.append([clamp_lambda(.6 * clamp_lambda(outputs[side+'_goals_lgbm']['effective']) +
                    .4 * clamp_lambda(outputs[side+'_goals_xgb']['effective'])) for side in ('home', 'away')])
            p = np.asarray(probabilities, float)
            losses = metrics7.probability_metric_contributions(labels, p)
            goals = metrics7.goal_metric_contributions(actual.reshape(-1), np.asarray(lambdas).reshape(-1))
            contributions[label] = {'log_loss': losses['log_loss'], 'multiclass_brier': losses['multiclass_brier'],
                'joint_goal_rmse': goals['squared_error'].reshape(-1, 2).mean(axis=1),
                'joint_goal_poisson_deviance': goals['poisson_deviance'].reshape(-1, 2).mean(axis=1)}
            reliability[label] = metrics7.summarize_probability_metrics(labels, p)
            summary[label] = {**reliability[label],
                'joint_goal_rmse': float(np.sqrt(contributions[label]['joint_goal_rmse'].mean())),
                'joint_goal_poisson_deviance': float(contributions[label]['joint_goal_poisson_deviance'].mean())}
        intervals = metrics7.paired_bootstrap(contributions['active'], contributions['candidate'],
            **protocol['bootstrap'], root_mean_metrics=['joint_goal_rmse'])
    decision = metrics7.apply_decision_rule(prospective_fixture_count=len(completed),
        horizon_complete=horizon_complete and complete, paired_intervals=intervals['metrics'] if intervals else None,
        active_reliability=reliability.get('active'), candidate_reliability=reliability.get('candidate'),
        provenance_verified=True)
    return {'fixture_ids': [p['fixture_id'] for p in completed], 'fixture_count': len(completed),
            'pending_fixture_ids': [p['fixture_id'] for p in pairs if p['fixture_id'] not in by_id],
            'cohort_completeness': complete, 'horizon_complete': horizon_complete,
            'metrics': summary, 'reliability': reliability, 'paired_intervals': intervals, 'decision': decision,
            'interim_descriptive_only': not (horizon_complete and complete)}


def incident(path, league, code, *, fixture_id=None, source_ref=None, violation=False):
    payload = {'code': code, 'fixture_id': fixture_id, 'source': source_ref, 'protocol_violation': violation}
    append(path, league, 'incident', payload)
    return payload


def is_violation(exc):
    return any(word in str(exc).upper() for word in (
        'DRIFT', 'MISMATCH', 'DISAGREEMENT', 'CONFLICT', 'REVISION', 'CONTAMINATION',
        'DUPLICATE', 'REORDER', 'REMOVED', 'CORRUPT', 'BUNDLE', 'POINTER', 'REPLAY', 'INCOMPATIBLE'))


def settle(path, league, context, metadata):
    protocol, _, q, pairs, settlements, targets, _ = context
    events = {e['fixture_id']: e for e in current_events(metadata, protocol, q)}
    existing = {s['fixture_id']: s for s in settlements}
    for pair in pairs:
        fixture = pair['fixture_id']
        event = events.get(fixture)
        try:
            if event is None:
                original = targets[fixture]
                replacements = [value for value in events.values() if all(
                    value[key] == original[key] for key in ('season', 'home_id', 'away_id'))]
                check(not replacements, 'FIXTURE_IDENTITY_DISAGREEMENT:provider_fixture_id')
            check(event is not None, 'FIXTURE_DISAPPEARED')
            same_fixture(event, targets[fixture], admitted=True)
            if fixture in existing:
                check(event['status'] == 'FINISHED' and
                      [event['home_goals'], event['away_goals']] ==
                      [existing[fixture]['home_goals'], existing[fixture]['away_goals']], 'CONFLICTING_SECOND_RESULT')
                continue
            if event['status'] != 'FINISHED':
                if instant(pair['kickoff_at']) < instant(metadata['retrieved_at']) or event['status'] not in ('TIMED', 'SCHEDULED'):
                    incident(path, league, 'AWAITING_RESULT:' + str(event['status']), fixture_id=fixture,
                             source_ref=metadata)
                continue
            value = settlement_value(pair, targets[fixture], event, metadata, protocol)
            append(path, league, 'settlement', value)
            # The scientific update following each settlement is immutable.
            write_checkpoint(path, league)
        except (ValueError, KeyError, TypeError, RuntimeError) as exc:
            incident(path, league, str(exc), fixture_id=fixture, source_ref=metadata, violation=is_violation(exc))
    return list(events.values())


def refresh(path, league, protocol):
    """Use the unchanged Phase 11 qualification workflow in a fresh acquisition run.

    This creates source assessments only; no second protocol is frozen.
    """
    acquisition = ROOT / 'baselines/phase11' / ('phase12-acquisition-' + uuid.uuid4().hex)
    persist_new(acquisition / 'run_specification.json', seal({
        'schema_version': 'ffps.phase12_source_refresh.v1', 'league': league, 'created_at': now(),
        'continuation': str(run_path(path).relative_to(ROOT)), 'protocol_sha256': protocol['protocol_sha256'],
        'source_contract_unchanged': True, 'promotion_authorized': False}))
    incident(path, league, 'SOURCE_REFRESH_STARTED', source_ref=identity(acquisition / 'run_specification.json'))
    source.acquire_league(acquisition, league, 'current')
    source.acquire_fotmob_current(acquisition, league, 'current')
    report = source.qualify_league(acquisition, league, 'current', 'current')
    check(report['ready'], 'SOURCE_REFRESH_BLOCKED:' + ','.join(report['blockers']))
    q, index, _ = frozen.load_qualification(frozen.reference(report['qualification']))
    evidence11.replay_source_evidence(q)
    return report['qualification'], q, index


def collect(path, league, context, refreshed):
    protocol, controller, original, pairs, _, _, _ = context
    ref, q, index = refreshed
    pair_qualification({'qualification': ref}, {'qualification_ref': protocol['qualification'], 'value': original}, index)
    existing = {p['fixture_id'] for p in pairs}
    active, candidate = [controller.owner.load(controller.bundle(bundle))
                         for bundle in (controller.legacy_id, controller.candidate_id)]
    for target in q['assessment']['upcoming']:
        fixture_id = target['fixture_id']
        if fixture_id in existing:
            continue
        if not instant(protocol['start_at']) <= instant(target['kickoff_at']) < instant(protocol['end_at']):
            continue
        try:
            at = now()
            check(instant(at) < instant(target['kickoff_at']), 'MISSED_PROSPECTIVE_CUTOFF')
            matrix, fixture = source.target_matrix(q['assessment'], target, as_of=at, source_index=index)
            p10.validate_features(matrix, fixture, q['ordered_features'], at)
            fixture = fixture.copy()
            fixture['kickoff_utc'] = target['kickoff_at']
            at = now()
            records = [capture.capture(matrix, fixture, suite, league, at) for suite in (active, candidate)]
            pair = seal({'league': league, 'protocol_sha256': protocol['protocol_sha256'],
                         'fixture_id': fixture_id, 'created_at': now(), 'kickoff_at': target['kickoff_at'],
                         'qualification': ref, 'active': records[0], 'candidate': records[1]}, 'pair_sha256')
            verify_pair_features(pair, q, index, protocol, controller)
            # Recheck artifact bytes and production identities immediately before admission.
            protocol_context(path, league)
            check(instant(now()) < instant(target['kickoff_at']), 'LATE_PAIR_PUBLICATION')
            append(path, league, 'pair', pair)
            existing.add(fixture_id)
            # Preserve Phase 11's earliest-uncollected-fixture operational batch.
            # This is a batch boundary, never an evidence/sample horizon limit.
            break
        except (ValueError, KeyError, TypeError, RuntimeError) as exc:
            incident(path, league, str(exc), fixture_id=fixture_id, source_ref=ref, violation=is_violation(exc))
            if is_violation(exc):
                break


def state_for(evaluation, pairs, incidents, at, temporarily_blocked=False):
    if any(i['protocol_violation'] for i in incidents):
        return 'PROTOCOL_VIOLATION_BLOCKED'
    if temporarily_blocked:
        return 'DATA_SOURCE_TEMPORARILY_BLOCKED'
    if evaluation['horizon_complete']:
        if not evaluation['cohort_completeness']:
            return 'COLLECTION_HORIZON_COMPLETE'
        if evaluation['decision']['phase8_promotion_review_authorized']:
            return 'DECISION_READY'
        return 'DECISION_EVIDENCE_INSUFFICIENT'
    pending = set(evaluation['pending_fixture_ids'])
    if any(p['fixture_id'] in pending and instant(p['kickoff_at']) < instant(at) for p in pairs):
        return 'AWAITING_SETTLEMENT'
    return 'PROSPECTIVE_COLLECTION_ACTIVE'


def write_checkpoint(path, league, *, discovery=None, temporarily_blocked=False):
    protocol, controller, _, pairs, settlements, _, events = load_cohort(path, league)
    at = now()
    evaluation = evaluate(protocol, pairs, settlements, at)
    incidents = [e['payload'] for e in events if e['kind'] == 'incident']
    rejected = {}
    for item in incidents:
        if item['fixture_id'] is not None:
            rejected.setdefault(item['code'], set()).add(item['fixture_id'])
    retrievals = [read(item) for item in (run_path(path) / league / 'retrievals').glob('*/matches.json')]
    successful = [item['retrieved_at'] for item in retrievals if item.get('status') == 'RETRIEVED']
    if discovery:
        discovery = copy.deepcopy(discovery)
        admitted = {pair['fixture_id'] for pair in pairs}
        remaining = [event for event in current_events(discovery['source'], protocol, context_qualification(path, league))
                     if event['fixture_id'] not in admitted and event['status'] in ('TIMED', 'SCHEDULED') and
                     instant(at) < instant(event['kickoff_at']) < instant(protocol['end_at'])]
        remaining.sort(key=lambda event: (event['kickoff_at'], event['fixture_id']))
        discovery['remaining_eligible_horizon'] = len(remaining)
        discovery['next_eligible_fixture'] = remaining[0] if remaining else None
    row = {'league': league, 'as_of': at, 'protocol': specification(path)['leagues'][league]['protocol'],
           'protocol_sha256': protocol['protocol_sha256'],
           'active': {k: controller.bundle(controller.legacy_id)[k] for k in ('bundle_id', 'aggregate_sha256')},
           'candidate': {k: controller.bundle(controller.candidate_id)[k] for k in ('bundle_id', 'aggregate_sha256')},
           'pairs': len(pairs), 'settled': len(settlements), 'pending': len(pairs)-len(settlements),
           'replay': 'VERIFIED', 'promotion': 'NOT_AUTHORIZED', 'evaluation': evaluation,
           'scientific_state': state_for(evaluation, pairs, incidents, at, temporarily_blocked),
           'temporary_source_block': temporarily_blocked,
           'horizon': {'start_at': protocol['start_at'], 'end_at': protocol['end_at'],
                       'complete': evaluation['horizon_complete'], 'sample_threshold': None,
                       'basis': 'frozen calendar horizon; no added sample threshold'},
           'journal_count': len(events), 'journal_head': events[-1]['record_sha256'] if events else None,
           'cohort_pair_hashes': [p['pair_sha256'] for p in pairs],
           'cohort_settlement_hashes': [s['settlement_sha256'] for s in settlements],
           'last_successful_source_retrieval': max(successful) if successful else None,
           'next_eligible_fixture': discovery['next_eligible_fixture'] if discovery else None,
           'rejected_by_reason': {key: len(values) for key, values in rejected.items()},
           'source_mismatch_count': sum(is_violation(item['code']) for item in incidents),
           'discovery': discovery, 'incidents': incidents}
    row = seal(row)
    persist_new(run_path(path) / league / 'checkpoints' / (row['record_sha256'] + '.json'), row)
    return row


def context_qualification(path, league):
    """Read the immutable mapping context without rerunning feature inference."""
    ref = read(frozen.reference(specification(path)['leagues'][league]['protocol']))['qualification']
    return read(frozen.reference(ref))


def operate(path):
    path = run_path(path)
    rows = []
    for league in source.LEAGUES:
        try:
            with state_lock(path / league / 'operation.lock'):
                context = load_cohort(path, league)
                protocol, _, q, pairs, _, _, events = context
                if any(e['kind'] == 'incident' and e['payload']['protocol_violation'] for e in events):
                    rows.append(write_checkpoint(path, league))
                    continue
                directory = path / league / 'retrievals' / uuid.uuid4().hex
                key = source.credentials()['football_data_org']
                metadata = source.fetch(directory, 'matches',
                    f'https://api.football-data.org/v4/competitions/{league}/matches',
                    provider='football-data.org:v4', params={'season': protocol['season']},
                    headers={'X-Auth-Token': key} if key else {})
                discovery, blocked = None, False
                try:
                    discovered = settle(path, league, context, metadata)
                    existing = {p['fixture_id'] for p in pairs}
                    eligible = [e for e in discovered if e['status'] in ('TIMED', 'SCHEDULED') and
                                instant(now()) < instant(e['kickoff_at']) < instant(protocol['end_at']) and
                                e['fixture_id'] not in existing]
                    discovery = {'source': metadata, 'eligible_fixture_ids': [e['fixture_id'] for e in eligible],
                                 'next_eligible_fixture': eligible[0] if eligible else None,
                                 'remaining_eligible_horizon': len(eligible)}
                    for event in discovered:
                        if event['fixture_id'] not in existing and instant(protocol['start_at']) <= instant(event['kickoff_at']) <= instant(now()):
                            incident(path, league, 'MISSED_PROSPECTIVE_CUTOFF', fixture_id=event['fixture_id'], source_ref=metadata)
                    violations = [e for e in journal(path, league) if e['kind'] == 'incident' and e['payload']['protocol_violation']]
                    if eligible and not violations:
                        refreshed = refresh(path, league, protocol)
                        collect(path, league, load_cohort(path, league), refreshed)
                except (ValueError, KeyError, TypeError, RuntimeError) as exc:
                    blocked = True
                    incident(path, league, str(exc), source_ref=metadata, violation=is_violation(exc))
                rows.append(write_checkpoint(path, league, discovery=discovery, temporarily_blocked=blocked))
        except (ValueError, KeyError, TypeError, RuntimeError, OSError) as exc:
            # A corrupted league must not prevent healthy leagues from operating.
            failure = seal({'league': league, 'as_of': now(), 'scientific_state': 'PROTOCOL_VIOLATION_BLOCKED',
                            'replay': 'FAILED', 'error': str(exc), 'promotion': 'NOT_AUTHORIZED'})
            persist_new(path / league / 'failures' / (failure['record_sha256'] + '.json'), failure)
            rows.append(failure)
    batch = seal({'as_of': now(), 'leagues': rows, 'promotion_authorized': False})
    persist_new(path / 'operations' / (batch['record_sha256'] + '.json'), batch)
    return batch
