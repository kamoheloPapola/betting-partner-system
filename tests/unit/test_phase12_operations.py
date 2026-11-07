"""Operational recovery/isolation tests; inherited scientific tests remain intact."""
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.evaluation import phase12_operations as op
from src.evaluation import phase12_operations_evidence as oe
from src.evaluation import phase12_prospective as p
from tests.unit.test_phase12_prospective import workspace  # noqa: F401


@pytest.fixture
def operational(workspace, monkeypatch):
    def reference(ref):
        path = p.ROOT / ref['path']
        p.check(p.identity(path) == ref, 'HASH_MISMATCH')
        return path
    monkeypatch.setattr(p.frozen, 'reference', reference)
    monkeypatch.setattr(op, 'initialize', lambda path: {})
    monkeypatch.setattr(op, 'integrity', lambda path: {'unchanged': True})
    p.persist_new(op.root(workspace) / 'configuration.json', {})
    monkeypatch.setattr(op.time, 'sleep', lambda delay: None)
    return workspace


def controller(bound=False):
    return SimpleNamespace(owner=SimpleNamespace(frozen={'prospective_contract': {} if bound else None}))


def test_ready_science_does_not_authorize_deployment():
    evaluation = {'horizon_complete': True, 'cohort_completeness': True,
                  'decision': {'phase8_promotion_review_authorized': True}}
    assert p.state_for(evaluation, [], [], '2027-06-01T00:00:00+00:00') == 'DECISION_READY'
    assert op.authorization(controller())['state'] == 'BLOCKED_MISSING_PROSPECTIVE_CONTRACT'
    assert not op.authorization(controller(True))['activation_authorized_by_this_operation']


@pytest.mark.parametrize('status,retry', [('SOURCE_UNAVAILABLE', True), (429, True), (503, True), (401, False), (404, False)])
def test_retry_only_transport(status, retry):
    metadata = {'status': status} if isinstance(status, str) else {'http_status': status}
    assert op.transient(metadata) is retry


@pytest.mark.parametrize('statuses,count,success', [(['SOURCE_UNAVAILABLE', 'RETRIEVED'], 2, True),
    (['SOURCE_UNAVAILABLE'] * 3, 3, False), (['HTTP_ERROR'], 1, False)])
def test_preserved_source_attempts(operational, monkeypatch, statuses, count, success):
    calls = []
    monkeypatch.setattr(p.source, 'credentials', lambda: {'football_data_org': None})
    def fetch(directory, *args, **kwargs):
        value = {'status': statuses[len(calls)]}
        calls.append(directory)
        p.persist_new(directory / 'matches.json', value)
        return value
    monkeypatch.setattr(p.source, 'fetch', fetch)
    if success:
        assert op.retrieve(operational, 'SA', {'season': 2026})['status'] == 'RETRIEVED'
    else:
        with pytest.raises(ValueError, match='SOURCE_UNAVAILABLE'):
            op.retrieve(operational, 'SA', {'season': 2026})
    assert len(set(calls)) == count
    assert len(p.journal(operational, 'SA')) == count - int(success)


def minimal_seal(path, directory, outcomes, preserved):
    row = p.seal({'artifacts': [p.identity(directory / 'intent.json')], 'leagues': outcomes})
    op.put(directory / 'report.json', row)
    return {'report': p.identity(directory / 'report.json'), 'duplicate': False}


def test_identical_key_references_seal_without_sources(operational, monkeypatch):
    calls = []
    monkeypatch.setattr(op, 'process_league', lambda path, directory, league: calls.append(league) or {'league': league})
    monkeypatch.setattr(oe, 'seal_report', minimal_seal)
    first = op.run(operational, 'fixed')
    again = op.run(operational, 'fixed')
    assert again['duplicate'] and again['report'] == first['report']
    assert calls == list(p.source.LEAGUES)
    assert len(op.reports(operational)) == 1


def test_one_league_replay_failure_does_not_stop_others(operational, monkeypatch):
    calls = []
    def process(path, directory, league):
        calls.append(league)
        if league == 'SA':
            raise ValueError('REPLAY_MISMATCH')
        return {'league': league}
    monkeypatch.setattr(op, 'process_league', process)
    monkeypatch.setattr(oe, 'seal_report', minimal_seal)
    result = op.run(operational, 'isolation')
    assert calls == list(p.source.LEAGUES)
    report = p.read(p.frozen.reference(result['report']))
    assert report['leagues'][0]['replay'] == 'FAILED'


def test_crash_after_report_artifact_resumes_same_intent(operational, monkeypatch):
    calls = []
    def process(path, directory, league):
        file = directory / league / 'outcome.json'
        if file.exists():
            return p.read(file)
        calls.append(league)
        return op.put(file, {'league': league})
    def crash(path, directory, rows, preserved):
        op.put(directory / 'partial.json', {'complete': False})
        raise KeyboardInterrupt()
    monkeypatch.setattr(op, 'process_league', process)
    monkeypatch.setattr(oe, 'seal_report', crash)
    with pytest.raises(KeyboardInterrupt):
        op.run(operational, 'original')
    monkeypatch.setattr(oe, 'seal_report', minimal_seal)
    op.run(operational, 'later-hour')
    assert calls == list(p.source.LEAGUES)
    assert len(op.reports(operational)) == 1
    assert p.read(op.reports(operational)[0] / 'intent.json')['key'] == 'original'


def test_concurrent_busy_never_becomes_scientific_violation(operational, monkeypatch):
    @contextmanager
    def busy(path):
        raise RuntimeError('state is busy; retry')
        yield
    monkeypatch.setattr(p, 'state_lock', busy)
    with pytest.raises(RuntimeError, match='busy'):
        op.run(operational, 'concurrent')
    assert not op.reports(operational)


def test_crash_before_second_intent_preserves_previous_report_link(operational, monkeypatch):
    monkeypatch.setattr(op, 'process_league', lambda path, directory, league: {'league': league})
    monkeypatch.setattr(oe, 'seal_report', minimal_seal)
    first = op.run(operational, 'first')
    pending = op.root(operational) / 'report-000002'
    pending.mkdir()
    op.run(operational, 'second')
    assert p.read(pending / 'intent.json')['previous_report'] == first['report']


def test_immutable_publication_refuses_conflict(operational):
    file = operational / 'immutable.json'
    op.put(file, {'a': 1})
    op.put(file, {'a': 1})
    with pytest.raises(ValueError, match='CONFLICT'):
        op.put(file, {'a': 2})


def worker(operational, monkeypatch, events=None):
    protocol = {'season': 2026, 'start_at': '2026-09-01T00:00:00+00:00', 'end_at': '2027-06-01T00:00:00+00:00'}
    event = {'fixture_id': 'future', 'status': 'TIMED', 'kickoff_at': '2026-09-11T18:30:00+00:00'}
    def context(*args):
        journal = p.journal(operational, 'BL1')
        pairs = [e['payload'] for e in journal if e['kind'] == 'pair']
        return (protocol, controller(), {}, pairs, [], {}, journal)
    monkeypatch.setattr(p, 'load_cohort', context)
    monkeypatch.setattr(op, 'retrieve', lambda *args: {'source': 'new'})
    monkeypatch.setattr(p, 'settle', lambda *args: [event] if events is None else events)
    monkeypatch.setattr(op, 'refreshed', lambda *args: ({}, {}, {}))
    def collect(*args):
        p.append(operational, 'BL1', 'pair', {'fixture_id': 'future'})
    monkeypatch.setattr(p, 'collect', collect)
    def checkpoint(*args, **kwargs):
        ctx = context()
        return {'pairs': len(ctx[3]), 'settled': 0, 'pending': len(ctx[3]), 'replay': 'VERIFIED'}
    monkeypatch.setattr(p, 'write_checkpoint', checkpoint)
    return operational / 'operational/report-000001'


def test_crash_after_pair_before_checkpoint_never_regenerates(operational, monkeypatch):
    directory = worker(operational, monkeypatch)
    checkpoint = p.write_checkpoint
    monkeypatch.setattr(p, 'write_checkpoint', lambda *a, **k: (_ for _ in ()).throw(KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):
        op.process_league(operational, directory, 'BL1')
    assert sum(e['kind'] == 'pair' for e in p.journal(operational, 'BL1')) == 1
    monkeypatch.setattr(p, 'write_checkpoint', checkpoint)
    row = op.process_league(operational, directory, 'BL1')
    assert row['newly_admitted'] == 1 and row['newly_settled'] == 0
    assert len(p.journal(operational, 'BL1')) == 1


def test_empty_provider_no_new_fixtures_or_results(operational, monkeypatch):
    directory = worker(operational, monkeypatch, events=[])
    row = op.process_league(operational, directory, 'BL1')
    assert row['newly_admitted'] == row['newly_settled'] == 0
    assert row['checkpoint']['replay'] == 'VERIFIED'


def test_missed_cutoff_preserved_once_never_collected(operational, monkeypatch):
    past = {'fixture_id': 'missed', 'status': 'TIMED', 'kickoff_at': '2026-09-07T00:00:00+00:00'}
    directory = worker(operational, monkeypatch, events=[past])
    row = op.process_league(operational, directory, 'BL1')
    row2 = op.process_league(operational, directory.parent / 'report-000002', 'BL1')
    assert row['newly_admitted'] == row2['newly_admitted'] == 0
    assert len(p.journal(operational, 'BL1')) == 1


def test_source_failure_preserves_zero_settlements(operational, monkeypatch):
    directory = worker(operational, monkeypatch)
    monkeypatch.setattr(op, 'retrieve', lambda *args: (_ for _ in ()).throw(ValueError('SOURCE_UNAVAILABLE')))
    row = op.process_league(operational, directory, 'BL1')
    assert row['error'] == 'SOURCE_UNAVAILABLE'
    assert row['newly_admitted'] == row['newly_settled'] == 0


def test_numbered_report_gap_rejected(operational):
    (op.root(operational) / 'report-000002').mkdir()
    with pytest.raises(ValueError, match='REMOVED_OR_REORDERED'):
        op.reports(operational)


def test_scheduler_has_no_alternate_activation_or_training_path():
    source = (Path(__file__).resolve().parents[2] / 'scripts/register_phase12_task.ps1').read_text()
    assert 'IgnoreNew' in source and 'StartWhenAvailable' in source
    assert 'run_phase12_operations.py' in source and 'nightly_pipeline.py' not in source
    assert 'LogonType Interactive' in source


@pytest.mark.parametrize('tamper', [None, 'source', 'outcome', 'remove_report'])
def test_report_seal_and_offline_hash_verification(operational, monkeypatch, tamper):
    monkeypatch.setattr(p.source, 'LEAGUES', ('BL1',))
    directory = worker(operational, monkeypatch, events=[])
    monkeypatch.setattr(p, 'specification', lambda path: {'leagues': {'BL1': {'prefix': []}}})
    old_checkpoint = p.write_checkpoint
    monkeypatch.setattr(p, 'write_checkpoint', lambda *a, **kw: {
        **old_checkpoint(*a, **kw), 'journal_count': 0, 'cohort_pair_hashes': [],
        'cohort_settlement_hashes': [], 'scientific_state': 'PROSPECTIVE_COLLECTION_ACTIVE'})
    monkeypatch.setattr(oe.evidence, 'verify_checkpoint', lambda *a: None)
    source = operational / 'archived.json'
    p.persist_new(source, {'source': 'archived'})
    result = op.run(operational, 'verify')
    assert not result['degraded']
    receipt = oe.verify(operational)
    assert receipt['verified'] and receipt['leagues'][0]['settled'] == 0
    if tamper is None:
        return
    if tamper == 'source':
        source.write_text('{}')
    elif tamper == 'outcome':
        (directory / 'BL1/outcome.json').write_text('{}')
    else:
        (directory / 'report.json').unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        oe.verify(operational)
