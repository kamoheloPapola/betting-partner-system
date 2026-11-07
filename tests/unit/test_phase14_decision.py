"""Terminal experiments use temporary roots only; real calendars never advance."""
import json
from pathlib import Path
import pytest
from src.evaluation import phase14_decision as d
from src.evaluation import phase14_verifier as v
from tests.unit.test_phase13_authorization import isolated, prediction  # noqa: F401


@pytest.fixture
def experiment(isolated, monkeypatch):
    a = d.a
    monkeypatch.setattr(a, 'RUN', a.RUN.with_name('phase13-authorization-20260915-v2'))
    monkeypatch.setattr(a, 'NEXT', a.NEXT.with_name('phase13-authorization-20260915-v2'))
    monkeypatch.setattr(d, 'RUN', isolated.root / 'baselines/phase14/test')
    contexts = {}
    for league in a.p.source.LEAGUES:
        a.install(league)
        pairs = [a.p.seal({'fixture_id': league + '-synthetic-' + str(i),
                 'active': prediction(False), 'candidate': prediction(True)}, 'pair_sha256') for i in range(5)]
        settlements = [a.p.seal({'fixture_id': pair['fixture_id'], 'home_goals': 2,
                                 'away_goals': 0}, 'settlement_sha256') for pair in pairs]
        contexts[league] = [isolated.protocols[league], None, None, pairs, settlements, {}, []]
    monkeypatch.setattr(a.p, 'load_cohort', lambda _, league: contexts[league])
    return contexts


@pytest.mark.parametrize('league', ['SA', 'FL1', 'PD', 'BL1'])
def test_terminal_publication_and_independent_verification(experiment, league):
    assert d.status(league)['state'] == 'READY_TO_FINALIZE'
    result = d.finalize(league)
    assert result['state'] == 'FINALIZED', result
    assert result['published'] and not result['activation_authorized']
    destination = d.a.RUN / league / 'decisions/final'
    assert v.verify(destination)['verified']
    before = {p.name: p.read_bytes() for p in destination.iterdir()}
    repeated = d.finalize(league)
    assert repeated['idempotent'] and not repeated['published']
    assert {p.name: p.read_bytes() for p in destination.iterdir()} == before
    assert d.status(league)['state'] == 'FINALIZED'


@pytest.mark.parametrize('kind,classification', [('active', 'ACTIVE_SUPPORTED'),
    ('inconclusive', 'CANDIDATE_INCONCLUSIVE'), ('candidate', 'CANDIDATE_SUPPORTED_FOR_PROMOTION_REVIEW')])
def test_all_scientific_outcomes(experiment, kind, classification):
    for i, pair in enumerate(experiment['SA'][3]):
        pair['active'] = prediction(kind == 'active' or (kind == 'inconclusive' and i % 2 == 0))
        pair['candidate'] = prediction(kind == 'candidate' or (kind == 'inconclusive' and i % 2 != 0))
    result = d.finalize('SA')
    assert result['state'] == 'FINALIZED', result
    assert result['publication']['classification'] == classification
    assert not result['publication']['activation_authorized']


def test_multiple_blockers_no_directory_created(experiment, monkeypatch):
    monkeypatch.setattr(d.a.p, 'now', lambda: '2026-09-15T00:00:00+00:00')
    experiment['SA'][4].pop()
    report = d.finalize('SA')
    assert {b['code'] for b in report['blockers']} == {'AWAITING_HORIZON', 'AWAITING_SETTLEMENTS'}
    assert report['reconstructed'] and not report['published']
    assert not (d.a.RUN / 'SA/decisions').exists()
    assert not list((d.a.RUN / 'SA').glob('.phase14-stage-*'))


def test_future_finalization_time(experiment):
    report = d.finalize('SA', finalized_at='2030-01-01T00:00:00+00:00')
    assert any(b['code'] == 'BLOCKED_FUTURE_FINALIZATION_TIME' for b in report['blockers'])
    assert not report['published']


@pytest.mark.parametrize('failure', ['MISSING_EVENT', 'REORDERED_EVENT', 'MISSING_PAIR',
    'MISSING_SETTLEMENT', 'ALTERED_SETTLEMENT', 'ACTIVE_PRODUCTION_BUNDLE_CHANGED',
    'FEATURE_SNAPSHOT_MISMATCH', 'RNG_MISMATCH', 'PHASE4_REPLAY_MISMATCH'])
def test_authoritative_reconstruction_failure(experiment, monkeypatch, failure):
    def reject(*args):
        raise ValueError(failure)
    monkeypatch.setattr(d.a.p, 'load_cohort', reject)
    result = d.finalize('SA')
    assert not result['published'] and result['blockers'][0]['detail'] == failure
    assert not (d.a.RUN / 'SA/decisions').exists()


@pytest.mark.parametrize('mutation', ['v1', 'absent_installation', 'specification', 'protocol',
    'candidate', 'active', 'lineage_anchor'])
def test_binding_rejections(experiment, monkeypatch, mutation):
    a = d.a
    if mutation == 'v1':
        monkeypatch.setattr(a, 'RUN', a.RUN.with_name('phase13-authorization-20260914-v1'))
    elif mutation == 'absent_installation':
        (a.RUN / 'SA/installation.json').unlink()
    elif mutation == 'specification':
        (a.NEXT / 'SA/deployment_specification.json').write_text('{}')
    elif mutation == 'protocol':
        (a.p.ROOT / 'SA/protocol.json').write_text('{}')
    elif mutation == 'lineage_anchor':
        (a.ANCHOR / 'SA/cohort_index.json').unlink()
    else:
        path = a.RUN / 'SA/prospective_contract.json'
        row = a.p.read(path)
        row['bundles'][mutation]['hash'] = 'redesigned'
        path.write_text(json.dumps(row))
    report = d.finalize('SA')
    assert not report['published'] and report['blockers']


@pytest.mark.parametrize('code,flag', [('CONFLICTING_SETTLEMENT', False), ('requires scientific review', True)])
def test_incident_refusal(experiment, code, flag):
    experiment['SA'][6].append(d.a.p.seal({'kind': 'incident',
        'payload': {'code': code, 'protocol_violation': flag}}))
    result = d.finalize('SA')
    assert result['state'] == 'BLOCKED_PROTOCOL_VIOLATION' and not result['published']


@pytest.mark.parametrize('failure', ['CHANGED_EVALUATION_METRIC', 'CHANGED_BOOTSTRAP_RESULT'])
def test_checkpoint_statistics_are_replayed(experiment, monkeypatch, failure):
    d.a.p.persist_new(d.a.COHORT / 'SA/checkpoints/test.json', {})
    def reject(*args):
        raise ValueError(failure)
    monkeypatch.setattr(d.e12, 'verify_checkpoint', reject)
    assert not d.finalize('SA')['published']


def test_publisher_evaluation_must_match_frozen_material(experiment, monkeypatch):
    original = d.a.decision_material
    def wrong(*args):
        value = original(*args)
        value['evaluation']['fixture_count'] = 123
        return value
    monkeypatch.setattr(d.a, 'decision_material', wrong)
    report = d.finalize('SA')
    assert not report['published']
    assert 'EVALUATION_RECONSTRUCTION_MISMATCH' in str(report)


def test_mutation_before_directory_commit_refused(experiment, monkeypatch):
    original = d.a.decision_material
    def racing(*args):
        result = original(*args)
        d.a.p.persist_new(d.a.COHORT / 'SA/journal/unexpected.json', {})
        return result
    monkeypatch.setattr(d.a, 'decision_material', racing)
    result = d.finalize('SA')
    assert not result['published'] and 'JOURNAL_MUTATION' in str(result)
    assert not (d.a.RUN / 'SA/decisions').exists()


def test_conflicting_publication_and_later_append_invalidate(experiment):
    assert d.finalize('SA')['published']
    destination = d.a.RUN / 'SA/decisions/final'
    original = (destination / 'prospective_result.json').read_bytes()
    conflict = d.finalize('SA', finalized_at='2027-06-01T04:00:00+00:00')
    assert not conflict['published'] and 'CONFLICTING_IMMUTABLE_PUBLICATION' in str(conflict)
    experiment['SA'][6].append(d.a.p.seal({'kind': 'incident',
        'payload': {'code': 'source retrieval preserved', 'protocol_violation': False}}))
    with pytest.raises(ValueError):
        v.verify(destination)
    assert (destination / 'prospective_result.json').read_bytes() == original
    successor = d.finalize('SA', publication_id='successor-1')
    assert successor['published']


def test_crash_before_atomic_rename_recovers(experiment, monkeypatch):
    original = d.publish_directory
    def crash(*args):
        raise OSError('injected crash before directory commit')
    monkeypatch.setattr(d, 'publish_directory', crash)
    assert not d.finalize('SA')['published']
    assert not (d.a.RUN / 'SA/decisions/final').exists()
    stages = list((d.a.RUN / 'SA').glob('.phase14-stage-*'))
    assert len(stages) == 1
    with pytest.raises(ValueError):
        v.verify(stages[0])
    monkeypatch.setattr(d, 'publish_directory', original)
    assert d.finalize('SA')['published']


def test_atomic_directory_is_never_replaced(experiment, tmp_path):
    left, right = tmp_path / 'stage', tmp_path / 'destination'
    left.mkdir()
    right.mkdir()
    (right / 'protected').write_text('immutable')
    with pytest.raises(OSError):
        d.publish_directory(left, right)
    assert (right / 'protected').read_text() == 'immutable'


def test_real_os_lock_prevents_competing_process(experiment, tmp_path):
    import subprocess
    import sys
    lock = d.a.COHORT / 'SA/operation.lock'
    code = 'from pathlib import Path; from src.ml.bundle_activation import state_lock; ' + \
           '\nwith state_lock(Path(' + repr(str(lock)) + ')): print("unexpected lock")'
    with d.a.p.state_lock(lock):
        child = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, timeout=30)
    assert child.returncode != 0 and 'busy' in child.stderr


def test_independent_verifier_rejects_tampered_index(experiment):
    assert d.finalize('SA')['published']
    directory = d.a.RUN / 'SA/decisions/final'
    path = directory / 'artifact_index.json'
    row = d.a.p.read(path)
    row['decision']['sha256'] = '0' * 64
    row.pop('record_sha256')
    path.write_bytes(d.canonical_bytes(d.a.p.seal(row)))
    with pytest.raises(ValueError, match='INDEX_MISMATCH'):
        v.verify(directory)


def test_crash_after_directory_commit_retries_identically(experiment, monkeypatch):
    original = d.publish_directory
    def crash(stage, destination):
        original(stage, destination)
        raise OSError('injected process loss after atomic commit')
    monkeypatch.setattr(d, 'publish_directory', crash)
    result = d.finalize('SA')
    assert result['publication_exists']
    destination = d.a.RUN / 'SA/decisions/final'
    assert sorted(p.name for p in destination.iterdir()) == ['artifact_index.json', 'prospective_result.json']
    monkeypatch.setattr(d, 'publish_directory', original)
    assert d.finalize('SA')['idempotent']


def test_independent_readiness_prefix_reconstruction(experiment, monkeypatch):
    context = experiment['SA']
    for kind, rows in [('pair', context[3]), ('settlement', context[4])]:
        context[6].extend(d.a.p.seal({'kind': kind, 'payload': row,
            'created_at': '2026-09-14T00:00:00+00:00'}) for row in rows)
    monkeypatch.setattr(d.a.p, 'now', lambda: '2026-09-15T00:00:00+00:00')
    report = d.a.p.seal(d.finalize('SA'))
    assert v.verify_readiness(report)['verified']
    report['ordered_event_hashes'].pop()
    report.pop('record_sha256')
    with pytest.raises(ValueError):
        v.verify_readiness(d.a.p.seal(report))


def test_phase14_has_no_activation_or_training_calls():
    import ast
    for module in (d, v):
        tree = ast.parse(Path(module.__file__).read_text())
        assert not any(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and
                       node.func.attr in ('activate', 'rollback', 'fit', 'calibrate') for node in ast.walk(tree))


def test_missing_incident_source_refused(experiment):
    experiment['SA'][6].append(d.a.p.seal({'kind': 'incident', 'payload': {
        'code': 'source retrieval', 'protocol_violation': False,
        'source': {'path': 'missing-response.json', 'sha256': '0' * 64, 'size_bytes': 1}}}))
    report = d.finalize('SA')
    assert not report['published'] and report['state'] == 'BLOCKED_LINEAGE_INVALID'
