"""Synthetic finals are confined to tmp_path; no production decision is created."""
from pathlib import Path
from threading import local
from types import SimpleNamespace
import pytest
from src.evaluation import phase13_authorization as a
from src.ml.phase13_bundle_activation import ProspectiveBundleController
from src.ml.bundle_activation import BundleController, ActivationError
from tests.unit.test_phase9_multi_league import deployment_material, isolated_leagues  # noqa: F401


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    p = a.p
    monkeypatch.setattr(p, 'ROOT', tmp_path)
    for key, folder in [('RUN', 'baselines/phase13/test'), ('NEXT', 'baselines/phase9/next'),
                        ('PRIOR', 'baselines/phase9/prior'), ('COHORT', 'baselines/phase12/test'),
                        ('ANCHOR', 'baselines/phase12/test/operational/report-000001')]:
        monkeypatch.setattr(a, key, tmp_path / folder)
    def identity(path):
        return {'path': Path(path).relative_to(tmp_path).as_posix(),
                'sha256': p.source.sha256_file(path), 'size_bytes': Path(path).stat().st_size}
    def reference(ref):
        path = tmp_path / ref['path']
        p.check(identity(path) == ref, 'HASH_MISMATCH')
        return path
    monkeypatch.setattr(p, 'identity', identity)
    monkeypatch.setattr(p.frozen, 'reference', reference)
    monkeypatch.setattr(p, 'now', lambda: '2027-06-02T00:00:00+00:00')
    for name in a.IMPLEMENTATION:
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text('isolated pinned implementation')
    run = {'leagues': {}}
    protocols = {}
    owners = {}
    for league in p.source.LEAGUES:
        protocol = {'league': league, 'protocol_sha256': league + '-protocol',
            'end_at': '2027-06-01T00:00:00+00:00',
            'bootstrap': {'resamples': 10000, 'confidence_level': .95, 'seed': 42}}
        target = tmp_path / league / 'protocol.json'
        p.persist_new(target, protocol)
        protocols[league] = protocol
        run['leagues'][league] = {'protocol': identity(target), 'prefix': []}
        old = p.seal({'league': league, 'prospective_contract': None,
            'active': {'bundle_id': league + '-active'}, 'candidate': {'bundle_id': league + '-candidate'},
            'evaluation_specification': {'specification_sha256': league + '-rules'}}, 'specification_sha256')
        p.persist_new(a.PRIOR / league / 'deployment_specification.json', old)
        owners[league] = SimpleNamespace(frozen=old, bundle=lambda id: {'bundle_id': id, 'hash': id})
        p.persist_new(a.ANCHOR / league / 'cohort_index.json',
                      {'league': league, 'prefix': [], 'ordered_journal': []})
    p.persist_new(a.ANCHOR / 'report.json', {'synthetic': True})
    p.persist_new(a.COHORT / 'run_specification.json', run)
    monkeypatch.setattr(p, 'specification', lambda _: run)
    monkeypatch.setattr(p.frozen, 'load_protocol', lambda _, league:
        (protocols[league], None, SimpleNamespace(owner=owners[league]), None, None))
    return SimpleNamespace(root=tmp_path, protocols=protocols, owners=owners)


def controller(league='SA'):
    a.install(league)
    target = a.NEXT / league / 'deployment_specification.json'
    return SimpleNamespace(league=league, specification_path=target, specification=lambda: a.p.read(target))


@pytest.mark.parametrize('league', ['SA', 'FL1', 'PD', 'BL1'])
def test_binding_deterministic_idempotent_and_missing(isolated, league):
    c = controller(league)
    before = {p: p.read_bytes() for p in isolated.root.rglob('*') if p.is_file()}
    assert a.construct(league) == a.construct(league)
    a.install(league)
    assert all(path.read_bytes() == data for path, data in before.items())
    contract, _ = a.verify_binding(c)
    assert contract['bundles']['candidate']['bundle_id'] == league + '-candidate'
    result = a.gate(c, a.RUN / league / 'decisions/final')
    assert result['status'] == 'BLOCKED_AWAITING_PROSPECTIVE_DECISION'
    assert not result['promotion_eligible'] and not result['activation_authorized']


@pytest.mark.parametrize('field', ['bundles', 'protocol', 'scientific_rules', 'lineage', 'phase8', 'decision'])
def test_resealed_contract_mismatch(isolated, field):
    c = controller()
    path = a.RUN / 'SA/prospective_contract.json'
    value = a.p.read(path)
    value[field] = {'forged': True}
    value.pop('contract_sha256')
    path.write_text(__import__('json').dumps(a.p.seal(value, 'contract_sha256')))
    with pytest.raises(ValueError):
        a.verify_binding(c)


def prediction(good):
    return {'prediction_output': {'final_probabilities': [.98, .01, .01] if good else [.1, .8, .1]},
        'method_provenance': {'descriptor': {'ensemble': {'member_outputs': {
            side + '_goals_' + model: {'effective': (2 if side == 'home' else .01) if good else 1}
            for side in ('home', 'away') for model in ('lgbm', 'xgb')}}}}}


def final_fixture(monkeypatch, isolated, classification='candidate'):
    c = controller()
    contract, installed = a.verify_binding(c)
    pairs = [{'fixture_id': 'synthetic-' + str(i), 'active': prediction(classification == 'active'),
              'candidate': prediction(classification != 'active' and classification != 'tie')} for i in range(5)]
    if classification == 'tie':
        for i, pair in enumerate(pairs):
            pair['active'] = prediction(i % 2 == 0)
            pair['candidate'] = prediction(i % 2 != 0)
    settlements = [{'fixture_id': row['fixture_id'], 'home_goals': 2, 'away_goals': 0} for row in pairs]
    context = [isolated.protocols['SA'], None, None, pairs, settlements, {}, []]
    monkeypatch.setattr(a.p, 'load_cohort', lambda *args: context)
    value = a.decision_material(contract, installed, '2027-06-01T01:00:00+00:00')
    directory = a.RUN / 'SA/decisions/isolated-final'
    return c, directory, value, context


def publish(directory, value):
    path = directory / 'prospective_result.json'
    a.p.persist_new(path, value)
    a.p.persist_new(directory / 'artifact_index.json', a.p.seal({'decision': a.p.identity(path)}))


@pytest.mark.parametrize('kind,expected', [('candidate', 'CANDIDATE_SUPPORTED_FOR_PROMOTION_REVIEW'),
                                        ('active', 'ACTIVE_SUPPORTED'), ('tie', 'CANDIDATE_INCONCLUSIVE')])
def test_science_is_separate_from_activation(isolated, monkeypatch, kind, expected):
    c, directory, value, _ = final_fixture(monkeypatch, isolated, kind)
    publish(directory, value)
    result = a.gate(c, directory)
    assert result['classification'] == expected
    assert result['promotion_eligible'] == (kind == 'candidate')
    assert result['activation_authorized'] is False
    assert result['candidate_bundle_id'] == 'SA-candidate'
    assert result['specification_fingerprint'] == 'SA-rules'


@pytest.mark.parametrize('field', ['league', 'contract', 'deployment_specification', 'protocol',
    'lineage', 'pairs', 'settlements', 'ordered_events', 'evaluation', 'status', 'schema_version'])
def test_forged_final_rejected(isolated, monkeypatch, field):
    c, directory, value, _ = final_fixture(monkeypatch, isolated)
    value[field] = 'forged-or-interim'
    value.pop('decision_sha256')
    publish(directory, a.p.seal(value, 'decision_sha256'))
    with pytest.raises(ValueError):
        a.gate(c, directory)


@pytest.mark.parametrize('failure', ['replay', 'review', 'correction', 'pending', 'horizon', 'reordered'])
def test_current_authoritative_evidence_invalidates_final(isolated, monkeypatch, failure):
    c, directory, value, context = final_fixture(monkeypatch, isolated)
    publish(directory, value)
    if failure == 'replay':
        def broken(*args):
            raise ValueError('REPLAY_MISMATCH')
        monkeypatch.setattr(a.p, 'load_cohort', broken)
    elif failure in ('review', 'correction'):
        context[-1].append({'kind': 'incident', 'payload': {'protocol_violation': failure == 'review',
                                                        'code': 'SETTLEMENT_CONFLICT'}})
    elif failure == 'pending':
        context[4].pop()
    elif failure == 'horizon':
        context[0]['end_at'] = '2028-01-01T00:00:00+00:00'
    else:
        context[3].reverse()
    with pytest.raises(ValueError):
        a.gate(c, directory)


def test_unsealed_final_and_interim_path_fail_closed(isolated, monkeypatch):
    c, directory, value, _ = final_fixture(monkeypatch, isolated)
    value['decision_sha256'] = 'bad'
    publish(directory, value)
    with pytest.raises(ValueError):
        a.gate(c, directory)
    with pytest.raises(ValueError, match='UNBOUND_DECISION_PATH'):
        a.gate(c, a.COHORT / 'SA/checkpoints')


def test_phase8_preflight_and_pointer_protection(isolated, monkeypatch):
    c = object.__new__(ProspectiveBundleController)
    c.league = 'SA'
    c.legacy_id = 'SA-active'
    c.lock = isolated.root / 'phase8.lock'
    c.pointer = isolated.root / 'pointer.json'
    c.pointer.write_text('unchanged')
    c._guard = local()
    calls = []
    monkeypatch.setattr(c, 'league_promotion_gate', lambda _: {'promotion_eligible': True, 'classification': 'CANDIDATE_SUPPORTED'})
    def preflight(expected):
        calls.append(expected)
        assert c._guard.held
        raise ValueError('serving workers must be stopped')
    monkeypatch.setattr(c, 'preflight', preflight)
    monkeypatch.setattr(c, '_transition', lambda *args, **kwargs: pytest.fail('transition before preflight'))
    with pytest.raises(ValueError, match='workers'):
        c.activate(isolated.root, expected_active='SA-active', operator='isolated-test')
    assert calls == ['SA-active'] and c.pointer.read_text() == 'unchanged'
    assert ProspectiveBundleController.rollback is BundleController.rollback
    assert ProspectiveBundleController.recover is BundleController.recover


def test_nested_cohort_guard_and_missing_operator(isolated):
    c = object.__new__(ProspectiveBundleController)
    c.league = 'SA'
    c._guard = local()
    with c.cohort_guard():
        with c.cohort_guard():
            assert c._guard.held
    assert not c._guard.held
    with pytest.raises(ActivationError, match='operator'):
        c.activate(isolated.root, expected_active='SA-active', operator='')


@pytest.mark.parametrize('label', ['active', 'candidate'])
def test_artifact_identity_changed_after_binding(isolated, label):
    c = controller()
    owner = isolated.owners['SA']
    original = owner.bundle
    owner.bundle = lambda id: {**original(id), 'hash': 'redesigned'} if id == 'SA-' + label else original(id)
    with pytest.raises(ValueError, match='CONTRACT_RECONSTRUCTION_MISMATCH'):
        a.verify_binding(c)


@pytest.mark.parametrize('mutation', ['protocol', 'horizon', 'bootstrap'])
def test_scientific_contract_changed_after_binding(isolated, mutation):
    c = controller()
    protocol = isolated.protocols['SA']
    if mutation == 'protocol':
        protocol['protocol_sha256'] = 'other'
    elif mutation == 'horizon':
        protocol['end_at'] = '2028-06-01T00:00:00+00:00'
    else:
        protocol['bootstrap']['resamples'] = 3
    # Real protocol loader validates the file hash and fixed bootstrap itself.
    # Updating the referenced test file reproduces an attempted frozen rewrite.
    path = isolated.root / 'SA/protocol.json'
    path.write_text(__import__('json').dumps(protocol))
    with pytest.raises(ValueError):
        a.verify_binding(c)


def test_partial_install_recovers_without_overwrite(isolated, monkeypatch):
    original = a.put
    def interrupted(path, value):
        if path.name == 'installation.json':
            raise OSError('simulated crash after spec publication')
        return original(path, value)
    monkeypatch.setattr(a, 'put', interrupted)
    with pytest.raises(OSError):
        a.install('SA')
    contract = (a.RUN / 'SA/prospective_contract.json').read_bytes()
    spec = (a.NEXT / 'SA/deployment_specification.json').read_bytes()
    monkeypatch.setattr(a, 'put', original)
    c = controller()
    assert (a.RUN / 'SA/prospective_contract.json').read_bytes() == contract
    assert (a.NEXT / 'SA/deployment_specification.json').read_bytes() == spec
    a.verify_binding(c)


def test_future_timestamp_and_missing_index_rejected(isolated, monkeypatch):
    c, directory, value, _ = final_fixture(monkeypatch, isolated)
    a.p.persist_new(directory / 'prospective_result.json', value)
    with pytest.raises(FileNotFoundError):
        a.gate(c, directory)
    contract, installed = a.verify_binding(c)
    with pytest.raises(ValueError, match='FIXED_HORIZON_NOT_COMPLETE'):
        a.decision_material(contract, installed, '2028-01-01T00:00:00+00:00')


def test_synthetic_gate_completes_real_phase8_transaction_in_private_state(isolated, isolated_leagues, monkeypatch):
    import src.ml.bundle_activation as p8
    import copy
    import json
    phase8 = isolated_leagues['SA']
    # Reuse real artifact identities, but all decision and pointer writes stay in tmp_path.
    old = copy.deepcopy(phase8.frozen)
    isolated.owners['SA'].frozen = old
    (a.PRIOR / 'SA/deployment_specification.json').write_text(json.dumps(old))
    c, directory, value, _ = final_fixture(monkeypatch, isolated)
    publish(directory, value)
    gate = a.gate(c, directory)
    assert gate['candidate_bundle_id'] == phase8.candidate_id
    monkeypatch.setattr(phase8, 'league_promotion_gate', lambda _: gate)
    before = p8.sha256_file(phase8.manifest)
    record = phase8.activate(directory, expected_active=phase8.legacy_id, operator='isolated synthetic test')
    p8.verify_activation_record(record)
    assert phase8.current()['bundle_id'] == phase8.candidate_id
    phase8.rollback(expected_active=phase8.candidate_id, operator='isolated synthetic test', reason='test recovery')
    assert phase8.current()['bundle_id'] == phase8.legacy_id
    assert p8.sha256_file(phase8.manifest) == before
