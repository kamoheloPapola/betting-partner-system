"""Adversarial continuation tests; no production feature export or model fitting."""
import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.evaluation import phase12_prospective as p
from src.evaluation import phase12_evidence as evidence
from tests.unit.test_phase11_current_source import match, match_payload, mapped

AT = '2026-09-08T12:00:00+00:00'
KICKOFF = '2026-09-06T18:45:00+00:00'


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.setattr(p, 'ROOT', tmp_path)
    monkeypatch.setattr(p.source, 'ROOT', tmp_path)
    monkeypatch.setattr(p, 'identity', lambda path: {'path': Path(path).relative_to(tmp_path).as_posix(),
                       'sha256': p.source.sha256_file(path), 'size_bytes': Path(path).stat().st_size})
    monkeypatch.setattr(p, 'now', lambda: AT)
    return tmp_path / 'baselines/phase12/test'


def fixture_evidence(workspace):
    protocol = {'league': 'SA', 'season': 2026, 'protocol_sha256': 'protocol',
                'start_at': '2026-09-06T10:00:00+00:00', 'end_at': '2027-05-31T00:00:00+00:00',
                'bootstrap': {'resamples': 1000, 'seed': 42, 'confidence_level': .95}}
    q = {'assessment': {'team_mappings': {'2026': mapped()}}}
    raw = workspace / 'raw/matches.raw'
    raw.parent.mkdir(parents=True, exist_ok=True)
    raw.write_text(json.dumps(match_payload([match(kickoff=KICKOFF)])), encoding='utf-8')
    meta = {'provider': 'football-data.org:v4', 'status': 'RETRIEVED', 'http_status': 200,
            'started_at': AT, 'retrieved_at': AT, 'raw': p.identity(raw),
            'request': {'url': 'https://api.football-data.org/v4/competitions/SA/matches', 'params': {'season': 2026}}}
    target = p.current_events(meta, protocol, q)[0]
    pair = {'fixture_id': target['fixture_id'], 'league': 'SA', 'pair_sha256': 'pair', 'kickoff_at': KICKOFF}
    return protocol, q, meta, target, pair


def context(workspace):
    protocol, q, meta, target, pair = fixture_evidence(workspace)
    return (protocol, None, q, [pair], [], {pair['fixture_id']: target}, []), meta


def test_authoritative_settlement_round_trip(workspace):
    protocol, q, meta, target, pair = fixture_evidence(workspace)
    record = p.settlement_value(pair, target, target, meta, protocol)
    p.verify_settlement(record, pair, target, protocol, q)


@pytest.mark.parametrize('field,value', [('league', 'PD'), ('season', 2025), ('fixture_id', 'other'),
    ('provider_fixture_id', 999), ('home_id', 2), ('away_id', 1), ('home_team', 'BETA'), ('away_team', 'ALPHA')])
def test_settlement_identity_disagreements_block(workspace, field, value):
    protocol, _, meta, target, pair = fixture_evidence(workspace)
    changed = {**target, field: value}
    with pytest.raises(ValueError, match='FIXTURE_IDENTITY_DISAGREEMENT'):
        p.settlement_value(pair, target, changed, meta, protocol)


@pytest.mark.parametrize('status', ['TIMED', 'SCHEDULED', 'POSTPONED', 'SUSPENDED', 'CANCELLED', 'PAUSED', 'IN_PLAY'])
def test_unfinished_abandoned_cancelled_stale_status_never_settles(workspace, status):
    protocol, _, meta, target, pair = fixture_evidence(workspace)
    with pytest.raises(ValueError, match='RESULT_NOT_FINISHED'):
        p.settlement_value(pair, target, {**target, 'status': status}, meta, protocol)


def test_bl1_future_result_cannot_settle(workspace):
    protocol, _, meta, target, pair = fixture_evidence(workspace)
    target['kickoff_at'] = pair['kickoff_at'] = '2026-09-11T18:30:00+00:00'
    with pytest.raises(ValueError, match='SETTLEMENT_BEFORE_KICKOFF'):
        p.settlement_value(pair, target, target, meta, protocol)


def test_admitted_kickoff_revision_blocked_pending_revision_allowed(workspace):
    _, _, _, target, _ = fixture_evidence(workspace)
    revised = {**target, 'kickoff_at': '2026-09-07T18:45:00+00:00'}
    p.same_fixture(revised, target, admitted=False)
    with pytest.raises(ValueError, match='ADMITTED_KICKOFF_REVISION'):
        p.same_fixture(revised, target, admitted=True)


@pytest.mark.parametrize('bad', [-1, True, 1.1, None, '1'])
def test_manual_or_invalid_goals_rejected(workspace, bad):
    protocol, _, meta, target, pair = fixture_evidence(workspace)
    with pytest.raises(ValueError, match='INVALID_SETTLEMENT_SCORE'):
        p.settlement_value(pair, target, {**target, 'home_goals': bad}, meta, protocol)


@pytest.mark.parametrize('mutation', ['provider', 'url', 'season', 'bytes', 'retrieved_at'])
def test_source_authority_hash_scope_time(workspace, mutation):
    protocol, q, meta, _, _ = fixture_evidence(workspace)
    if mutation == 'provider':
        meta['provider'] = 'fotmob:web-api'
    elif mutation == 'url':
        meta['request']['url'] = 'https://untrusted.test/results'
    elif mutation == 'season':
        meta['request']['params']['season'] = 2025
    elif mutation == 'bytes':
        (p.ROOT / meta['raw']['path']).write_text('{}')
    else:
        meta['retrieved_at'] = '2026-09-09T00:00:00+00:00'
    with pytest.raises(ValueError):
        p.current_events(meta, protocol, q)


def test_duplicate_settlement_is_idempotent_conflict_is_permanent_incident(workspace, monkeypatch):
    ctx, meta = context(workspace)
    monkeypatch.setattr(p, 'write_checkpoint', lambda *a, **k: None)
    p.settle(workspace, 'SA', ctx, meta)
    stored = p.journal(workspace, 'SA')[0]['payload']
    ctx = (*ctx[:4], [stored], *ctx[5:])
    p.settle(workspace, 'SA', ctx, meta)
    assert len(p.journal(workspace, 'SA')) == 1
    changed = copy.deepcopy(stored['normalized_result'])
    changed['home_goals'] += 1
    monkeypatch.setattr(p, 'current_events', lambda *args: [changed])
    p.settle(workspace, 'SA', ctx, meta)
    events = p.journal(workspace, 'SA')
    assert len(events) == 2 and events[-1]['payload']['protocol_violation']
    assert events[0]['payload'] == stored


def test_disappeared_fixture_preserved_unsettled(workspace, monkeypatch):
    ctx, meta = context(workspace)
    monkeypatch.setattr(p, 'current_events', lambda *args: [])
    p.settle(workspace, 'SA', ctx, meta)
    assert p.journal(workspace, 'SA')[0]['payload']['code'] == 'FIXTURE_DISAPPEARED'


def test_replacement_provider_id_cannot_admit_the_same_fixture_again(workspace, monkeypatch):
    ctx, meta = context(workspace)
    original = next(iter(ctx[5].values()))
    replacement = {**original, 'fixture_id': 'SA_football_data_org_999', 'provider_fixture_id': 999}
    monkeypatch.setattr(p, 'current_events', lambda *args: [replacement])
    p.settle(workspace, 'SA', ctx, meta)
    entry = p.journal(workspace, 'SA')[0]
    assert entry['kind'] == 'incident'
    assert entry['payload']['code'] == 'FIXTURE_IDENTITY_DISAGREEMENT:provider_fixture_id'
    assert entry['payload']['protocol_violation']


def test_retrieval_survives_crash_before_append(workspace, monkeypatch):
    ctx, meta = context(workspace)
    real = p.append
    monkeypatch.setattr(p, 'append', lambda *a: (_ for _ in ()).throw(OSError('crash')))
    with pytest.raises(OSError):
        p.settle(workspace, 'SA', ctx, meta)
    assert p.current_events(meta, ctx[0], ctx[2])
    monkeypatch.setattr(p, 'append', real)
    monkeypatch.setattr(p, 'write_checkpoint', lambda *a, **k: None)
    p.settle(workspace, 'SA', ctx, meta)
    assert len(p.journal(workspace, 'SA')) == 1


def test_journal_atomic_seal_and_orphan_temp_recovery(workspace, monkeypatch):
    real_link = os.link
    monkeypatch.setattr(os, 'link', lambda *a: (_ for _ in ()).throw(OSError('crash before publication')))
    with pytest.raises(OSError):
        p.append(workspace, 'SA', 'incident', {'code': 'one'})
    assert not p.journal(workspace, 'SA')
    monkeypatch.setattr(os, 'link', real_link)
    event = p.append(workspace, 'SA', 'incident', {'code': 'one'})
    (workspace / 'SA/journal/orphan.tmp').write_text('partial uncommitted bytes')
    assert p.journal(workspace, 'SA') == [event]
    p.append(workspace, 'SA', 'incident', {'code': 'two'})
    assert len(p.journal(workspace, 'SA')) == 2


@pytest.mark.parametrize('mutation', ['missing', 'reorder', 'cross_league', 'hash', 'tail'])
def test_journal_corruption_and_member_removal_rejected(workspace, mutation):
    p.append(workspace, 'SA', 'incident', {'code': 'one'})
    second = p.append(workspace, 'SA', 'incident', {'code': 'two'})
    directory = workspace / 'SA/journal'
    if mutation == 'missing':
        (directory / '00000001.json').unlink()
    elif mutation == 'tail':
        p.persist_new(workspace / 'SA/checkpoints/head.json', p.seal({'journal_count': 2, 'journal_head': second['record_sha256']}))
        (directory / '00000002.json').unlink()
    else:
        event = p.read(directory / '00000002.json')
        if mutation == 'reorder':
            event['sequence'] = 1
        elif mutation == 'cross_league':
            event['league'] = 'PD'
        else:
            event['payload']['code'] = 'changed'
        (directory / '00000002.json').write_text(json.dumps(event))
    with pytest.raises(ValueError):
        p.journal(workspace, 'SA')


def test_os_lock_releases_after_exception(workspace):
    path = workspace / 'SA/operation.lock'
    with pytest.raises(RuntimeError):
        with p.state_lock(path):
            raise RuntimeError('crash unwind')
    with p.state_lock(path):
        assert path.exists()


def metric_pair(fixture_id, probabilities):
    record = {'prediction_output': {'final_probabilities': probabilities},
              'method_provenance': {'descriptor': {'ensemble': {'member_outputs': {
                  side + '_goals_' + model: {'effective': 1.2} for side in ('home', 'away') for model in ('lgbm', 'xgb')}}}}}
    return {'fixture_id': fixture_id, 'active': record, 'candidate': copy.deepcopy(record), 'kickoff_at': KICKOFF}


def test_all_settled_observations_enter_metrics_and_pending_stays_explicit(workspace):
    protocol, _, _, _, _ = fixture_evidence(workspace)
    pairs = [metric_pair('one', [.9, .05, .05]), metric_pair('two', [.01, .01, .98]), metric_pair('pending', [.3, .4, .3])]
    settled = [{'fixture_id': fixture, 'home_goals': 1, 'away_goals': 0} for fixture in ('one', 'two')]
    result = p.evaluate(protocol, pairs, settled, AT)
    assert result['fixture_ids'] == ['one', 'two'] and result['pending_fixture_ids'] == ['pending']
    assert result['metrics']['active']['log_loss'] == pytest.approx((-np.log(.9)-np.log(.01))/2)
    assert not result['decision']['phase8_promotion_review_authorized']
    assert result['interim_descriptive_only']
    assert p.evaluate(protocol, pairs, settled, AT) == result


def test_fixed_horizon_not_new_sample_threshold(workspace):
    protocol, _, _, _, _ = fixture_evidence(workspace)
    pair = metric_pair('one', [.4, .3, .3])
    settlements = [{'fixture_id': 'one', 'home_goals': 1, 'away_goals': 0}]
    interim = p.evaluate(protocol, [pair], settlements, AT)
    assert p.state_for(interim, [pair], [], AT) == 'PROSPECTIVE_COLLECTION_ACTIVE'
    final = p.evaluate(protocol, [pair], settlements, protocol['end_at'])
    assert final['horizon_complete'] and not final['interim_descriptive_only']
    assert p.state_for(final, [pair], [], protocol['end_at']) == 'DECISION_READY'


def test_states_independent_blocked_and_pending(workspace):
    protocol, _, _, _, _ = fixture_evidence(workspace)
    pairs = [metric_pair('one', [.4, .3, .3])]
    result = p.evaluate(protocol, pairs, [], AT)
    assert p.state_for(result, pairs, [], AT) == 'AWAITING_SETTLEMENT'
    assert p.state_for(result, pairs, [], AT, True) == 'DATA_SOURCE_TEMPORARILY_BLOCKED'
    assert p.state_for(result, pairs, [{'protocol_violation': True}], AT) == 'PROTOCOL_VIOLATION_BLOCKED'
    after = p.evaluate(protocol, pairs, [], protocol['end_at'])
    assert p.state_for(after, pairs, [], protocol['end_at']) == 'COLLECTION_HORIZON_COMPLETE'


def test_corrupt_league_does_not_prevent_other_leagues(workspace, monkeypatch):
    seen = []
    def corrupt(path, league):
        seen.append(league)
        raise ValueError('COHORT_CORRUPT')
    monkeypatch.setattr(p, 'load_cohort', corrupt)
    result = p.operate(workspace)
    assert seen == list(p.source.LEAGUES)
    assert len(result['leagues']) == 4
    assert all(row['scientific_state'] == 'PROTOCOL_VIOLATION_BLOCKED' for row in result['leagues'])


def test_duplicate_collection_never_regenerates_existing_predictions(workspace, monkeypatch):
    protocol = {'start_at': '2026-09-01T00:00:00Z', 'end_at': '2027-06-01T00:00:00Z', 'qualification': {}}
    owner = SimpleNamespace(load=lambda bundle: {})
    controller = SimpleNamespace(owner=owner, bundle=lambda bundle: {}, legacy_id='a', candidate_id='b')
    q = {'assessment': {'upcoming': [{'fixture_id': 'existing'}]}}
    monkeypatch.setattr(p, 'pair_qualification', lambda *a: None)
    monkeypatch.setattr(p.capture, 'capture', lambda *a: pytest.fail('existing prediction regenerated'))
    p.collect(workspace, 'SA', (protocol, controller, {}, [{'fixture_id': 'existing'}], [], {}, []), ({}, q, {}))
    assert not p.journal(workspace, 'SA')


@pytest.mark.parametrize('mutation', ['feature', 'rng', 'league', 'artifact', 'replay', 'late'])
def test_phase12_uses_existing_pair_invariants(monkeypatch, mutation):
    from src.ml import bundle_evidence
    protocol = p.read(p.PHASE11 / 'SA/prospective_protocol.json')
    pair = p.read(next((p.ROOT / protocol['cohort_path'] / 'predictions').glob('*.json')))
    bundles = {}
    for label in ('active', 'candidate'):
        record = pair[label]
        bundles[label] = {'members': {m['role']: {'sha256': m['artifact_sha256']}
                                     for m in record['model_provenance']['members']},
                          'features': record['feature_provenance']['model_inputs'][0]['ordered_feature_names']}
    controller = SimpleNamespace(root=p.ROOT, league='SA', legacy_id='active', candidate_id='candidate',
                                 bundle=lambda name: bundles[name])
    def historical(record, **kwargs):
        if mutation == 'replay':
            raise ValueError('PREDICTION_REPLAY_FAILED')
    monkeypatch.setattr(bundle_evidence, 'verify_historical_record', historical)
    if mutation == 'feature':
        pair['candidate']['feature_provenance']['model_inputs'][0]['feature_vector_sha256'] = 'different'
    elif mutation == 'rng':
        pair['candidate']['method_provenance']['descriptor']['simulation']['rng_state_before'] = {'different': True}
    elif mutation == 'league':
        pair['league'] = 'PD'
    elif mutation == 'artifact':
        pair['candidate']['model_provenance']['members'][0]['artifact_sha256'] = 'changed'
    elif mutation == 'late':
        pair['created_at'] = pair['kickoff_at']
    pair = p.seal({k: v for k, v in pair.items() if k != 'pair_sha256'}, 'pair_sha256')
    with pytest.raises((ValueError, RuntimeError)):
        p.validate_pair(pair, controller=controller, protocol=protocol)


def test_missing_fixed_horizon_cannot_produce_decision(workspace):
    protocol, _, _, _, _ = fixture_evidence(workspace)
    del protocol['end_at']
    with pytest.raises(KeyError):
        p.evaluate(protocol, [], [], AT)


def test_resealed_checkpoint_subset_is_rejected(workspace):
    protocol, _, _, _, _ = fixture_evidence(workspace)
    pairs = [dict(metric_pair('one', [.4, .3, .3]), pair_sha256='one'),
             dict(metric_pair('two', [.4, .3, .3]), pair_sha256='two')]
    row = p.seal({'journal_count': 0, 'cohort_pair_hashes': ['one'], 'cohort_settlement_hashes': []})
    with pytest.raises(ValueError, match='CHECKPOINT_COHORT_CHANGED'):
        evidence.verify_checkpoint(row, (protocol, None, None, pairs, [], {}, []), 2)
