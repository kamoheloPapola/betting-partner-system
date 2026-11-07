"""Progress observes real verifier stages without replacing scientific checks."""
import json
import sys

import pytest

from scripts import run_phase19_collection as cli
from src.evaluation import phase19_verifier as verifier


def metadata(monkeypatch):
    monkeypatch.setattr(cli.p, 'read', lambda path: {'pairs': [1, 2]})
    monkeypatch.setattr(sys, 'argv', ['run_phase19_collection.py', 'verify'])


def events(capsys):
    captured = capsys.readouterr()
    return json.loads(captured.out), [json.loads(x) for x in captured.err.splitlines()]


def test_cli_progress_stays_off_scientific_stdout(monkeypatch, capsys):
    metadata(monkeypatch)
    scientific = {'leagues': [{'league': 'BL1', 'verified': True}], 'record_sha256': 'same'}
    def replay(*, progress, progress_event):
        progress_event('league_started', 'BL1', 0, 2, None)
        progress_event('pair_started', 'BL1', 0, 2, 'first')
        progress_event('pair_verified', 'BL1', 1, 2, 'first')
        progress_event('pair_started', 'BL1', 1, 2, 'second')
        progress_event('pair_verified', 'BL1', 2, 2, 'second')
        progress('BL1', 2, 2)
        return scientific
    monkeypatch.setattr(cli.phase19_verifier, 'all_leagues', replay)
    assert cli.main() == 0
    result, progress = events(capsys)
    assert result == scientific
    assert progress[0]['event'] == 'verification_started'
    assert progress[-1]['event'] == 'verification_succeeded'
    assert progress[-1]['completed_pairs'] == 2
    assert progress[-1]['announced_total_pairs'] == 2
    assert [x['fixture_id'] for x in progress if x['event'] == 'pair_verified'] == ['first', 'second']
    assert all(x['last_progress_at_utc'] and x['elapsed_seconds'] >= 0 for x in progress)


@pytest.mark.parametrize('error,exit_code,terminal', [
    (ValueError('PHASE19_PAIR_BINDING_DRIFT'), 1, 'verification_failed'),
    (KeyboardInterrupt(), 130, 'verification_interrupted'),
])
def test_failure_and_interruption_do_not_claim_completed_pair(monkeypatch, capsys, error, exit_code, terminal):
    metadata(monkeypatch)
    def fail(*, progress, progress_event):
        progress_event('pair_started', 'BL1', 0, 2, 'first')
        raise error
    monkeypatch.setattr(cli.phase19_verifier, 'all_leagues', fail)
    assert cli.main() == exit_code
    result, progress = events(capsys)
    assert result['verified'] is False
    assert progress[-1]['event'] == terminal
    assert progress[-1]['completed_pairs'] == 0
    assert not any(x['event'] in ('verification_succeeded', 'pair_verified') for x in progress)
    assert result['error'] == ('VERIFICATION_INTERRUPTED' if exit_code == 130 else str(error))


def test_missing_progress_metadata_does_not_change_scientific_result(monkeypatch, capsys):
    metadata(monkeypatch)
    monkeypatch.setattr(cli.p, 'read', lambda path: (_ for _ in ()).throw(OSError('metadata unavailable')))
    monkeypatch.setattr(cli.phase19_verifier, 'all_leagues', lambda **kwargs: {'verified': True})
    assert cli.main() == 0
    result, progress = events(capsys)
    assert result == {'verified': True}
    assert progress[-1]['announced_total_pairs'] is None


def test_nonverification_interrupt_behavior_is_unchanged(monkeypatch, capsys):
    monkeypatch.setattr(sys, 'argv', ['run_phase19_collection.py', 'status'])
    monkeypatch.setattr(cli, 'status', lambda: (_ for _ in ()).throw(KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):
        cli.main()
    captured = capsys.readouterr()
    assert captured.out == captured.err == ''


def test_progress_counters_isolate_leagues(monkeypatch, capsys):
    metadata(monkeypatch)
    progress = cli.VerificationProgress()
    progress.event('pair_verified', 'BL1', 1, 1, 'bl1')
    progress.event('pair_started', 'SA', 0, 1, 'sa')
    progress.event('pair_verified', 'SA', 1, 1, 'sa')
    captured = [json.loads(x) for x in capsys.readouterr().err.splitlines()]
    assert [x['completed_pairs'] for x in captured] == [0, 1, 1, 2]


def test_all_leagues_forwards_progress_without_using_stored_results(monkeypatch):
    auth = {'scheduler_authorized': False, 'settlement_authorized': False,
            'activation_authorized': False, 'cohorts': [{'league': 'PL', 'source_qualification': {}}]}
    monkeypatch.setattr(verifier.auth, 'verify', lambda: auth)
    monkeypatch.setattr(verifier, '_acquisition', lambda league: None)
    monkeypatch.setattr(verifier.p, 'read', lambda path: {'retrieved_at': 'time'})
    monkeypatch.setattr(verifier.pl, 'verify', lambda: {
        'raw': {}, 'fixtures_observed': 0, 'future_within_horizon': 0,
        'rejections': [], 'horizon_end': 'end'})
    calls = []
    observer = lambda *args: calls.append(args)
    checkpoint = lambda *args: None
    def nonpl(league, progress=None, *, progress_event=None):
        assert progress is checkpoint and progress_event is observer
        progress_event('pair_verified', league, 1, 1, league + '-fixture')
        return {'league': league, 'cohort_id': league, 'verified': True}
    monkeypatch.setattr(verifier, 'nonpl', nonpl)
    monkeypatch.setattr(verifier, '_stored_nonpl', lambda league: pytest.fail('stored replay used'))
    result = verifier.all_leagues(progress=checkpoint, progress_event=observer)
    assert len(result['leagues']) == 5
    assert [x[1] for x in calls if x[0] == 'pair_verified'] == ['BL1', 'SA', 'FL1', 'PD']
    assert calls[-1] == ('league_verified', 'PL', 0, 0, None)
