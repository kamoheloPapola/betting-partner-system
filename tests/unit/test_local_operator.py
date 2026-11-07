"""The local interface must compose evidence without opening a new authority path."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib

import pytest

from scripts.local_operator import main
from src import local_operator as op


def _mock_snapshot(monkeypatch):
    monkeypatch.setattr(op, '_authority', lambda: ({'cohorts': []}, {'leagues': []}))
    monkeypatch.setattr(op, '_current', lambda league, *args, **kwargs: {
        'cohort_id': league + '-clean-001', 'recorded_predictions': 0 if league == 'PL' else 1,
        'active_model': league + '-active', 'candidate_model': league + '-candidate',
        'journal_head': None, 'source_retrieved_at': '2026-09-23T00:00:00+00:00',
        'replay': 'VERIFIED'})
    monkeypatch.setattr(op, '_historical', lambda league: {
        'state': 'NO_PHASE12_COHORT' if league == 'PL' else
                 'BLOCKED_PROTOCOL_VIOLATION' if league in ('BL1', 'SA', 'FL1') else
                 'NO_RECORDED_PROTOCOL_VIOLATION_NOT_CLEARANCE',
        'incidents': ([{'sequence': 102, 'code': 'ARTIFACT_DRIFT'}] if league == 'FL1' else [])})
    monkeypatch.setattr(op.phase19_integrity, 'verify', lambda: {'verified': True})
    monkeypatch.setattr(op.phase20, 'authorization', lambda: {'manual_settlement_authorized': True})
    monkeypatch.setattr(op.phase20, 'integrity', lambda: {
        'verified': False, 'added': ['older-file'], 'changed': [], 'missing': []})
    monkeypatch.setattr(op.phase20, 'verify', lambda league: {
        'verified': True, 'pairs': 0 if league == 'PL' else 1,
        'settled': 0, 'pending': 0 if league == 'PL' else 1,
        'settled_fixture_ids': [], 'descriptive_metrics': {'active': None, 'candidate': None}})


def test_status_keeps_clean_cohorts_and_historical_blockers_separate(monkeypatch):
    _mock_snapshot(monkeypatch)
    first = op.snapshot()
    second = op.snapshot()
    assert first == second
    assert op.snapshot(audit=True) == op.snapshot(audit=True)
    assert [row['league'] for row in first['leagues']] == list(op.LEAGUES)
    assert all(row['current_lineage_verified'] for row in first['leagues'])
    assert all(row['settlement_eligibility'] == 'BLOCKED_PHASE20_INTEGRITY'
               for row in first['leagues'][:4])
    assert first['leagues'][2]['historical']['value']['incidents'][0]['code'] == 'ARTIFACT_DRIFT'
    assert first['leagues'][4]['recorded_predictions'] == 0
    assert first['new_prediction_cycle_authorized'] is False
    assert first['scheduling_authorized'] is False
    assert first['promotion_authorized'] is False


def test_audit_full_replay_dispatches_to_existing_verifier(monkeypatch):
    _mock_snapshot(monkeypatch)
    requested = []

    def current(league, *args, **kwargs):
        requested.append((league, kwargs['audit'], kwargs['full_replay']))
        return {'cohort_id': league + '-clean-001',
                'recorded_predictions': 0 if league == 'PL' else 1}

    monkeypatch.setattr(op, '_current', current)
    op.snapshot(audit=True, full_replay=True)
    assert requested == [(league, True, True) for league in op.LEAGUES]


def test_read_only_views_preserve_scientific_evidence_bytes():
    files = [op.phase19.AUTH,
             op.cohort._directory('BL1-clean-001') / 'journal/00000001.json',
             op.phase20.AUTH,
             op.HISTORICAL / 'FL1/journal/00000102.json']
    before = [hashlib.sha256(path.read_bytes()).hexdigest() for path in files]
    auth, report = op._authority()
    assert op._current('BL1', auth, report)['recorded_predictions'] == 270
    assert op._historical('FL1')['incidents'][-1]['code'] == 'ARTIFACT_DRIFT'
    assert op.phase20.journal('BL1') == []
    after = [hashlib.sha256(path.read_bytes()).hexdigest() for path in files]
    assert after == before


def test_one_current_league_failure_does_not_hide_other_predictions(monkeypatch):
    at = datetime.now(timezone.utc)
    assignments = [{'league': league, 'cohort_id': league + '-clean-001'} for league in op.LEAGUES]
    monkeypatch.setattr(op, '_authority', lambda: ({'cohorts': assignments}, {}))
    monkeypatch.setattr(op.phase19_integrity, 'verify', lambda: {'verified': True})

    def current(league, *args):
        if league == 'SA':
            raise ValueError('SA_REPLAY_FAILED')
        return {'cohort_id': league + '-clean-001', 'recorded_predictions': 0 if league == 'PL' else 1}
    monkeypatch.setattr(op, '_current', current)

    def journal(cohort_id):
        if cohort_id == 'PL-clean-001':
            return []
        league = cohort_id.split('-')[0]
        output = {'class_order': ['home_win', 'draw', 'away_win'],
                  'final_probabilities': [0.5, 0.3, 0.2]}
        pair = {'fixture_id': league + '_fixture_1',
                'kickoff_at': (at + timedelta(days=1)).isoformat(),
                'created_at': (at - timedelta(days=1)).isoformat(),
                'pair_sha256': league + '-pair',
                'active': {'fixture': {'home_team': 'HOME', 'away_team': 'AWAY'},
                           'prediction_output': output},
                'candidate': {'prediction_output': output}}
        return [{'kind': 'pair', 'record_sha256': league + '-event',
                 'payload': {'pair': pair}}]
    monkeypatch.setattr(op.cohort, 'journal', journal)
    monkeypatch.setattr(op.phase20, 'verify', lambda league: {'settled_fixture_ids': []})
    result = op.predictions(all_rows=True, now=at)
    assert [row['league'] for row in result['predictions']] == ['BL1', 'FL1', 'PD']
    assert result['errors'] == [{'league': 'SA', 'error': 'ValueError: SA_REPLAY_FAILED'}]
    assert result['pl_state'] == 'NO RECORDED PREDICTION'
    assert all(row['state'] == 'RECORDED PROSPECTIVE PREDICTION' for row in result['predictions'])

    monkeypatch.setattr(op, '_current', lambda league, *args: {
        'cohort_id': league + '-clean-001',
        'recorded_predictions': 0 if league == 'PL' else 1})
    restored = op.predictions(all_rows=True, now=at)
    assert {row['league'] for row in restored['predictions']} == {'BL1', 'SA', 'FL1', 'PD'}
    assert restored['errors'] == []


def test_false_integrity_blocks_settlement_before_acquisition(monkeypatch):
    monkeypatch.setattr(op, '_authority', lambda: ({}, {}))
    monkeypatch.setattr(op, '_current', lambda *args: {'recorded_predictions': 1})
    monkeypatch.setattr(op.phase19_integrity, 'verify', lambda: {'verified': True})
    monkeypatch.setattr(op.phase20, 'authorization', lambda: {})
    monkeypatch.setattr(op.phase20, 'integrity', lambda: {
        'verified': False, 'added': ['file'], 'changed': [], 'missing': []})
    monkeypatch.setattr(op.phase20, 'acquire', lambda *args, **kwargs: pytest.fail('acquired despite gate'))
    monkeypatch.setattr(op.phase20, 'settle', lambda *args, **kwargs: pytest.fail('settled despite gate'))
    result = op.settle('FL1')
    assert result['status'] == 'BLOCKED'
    assert result['reason'] == 'SETTLEMENT BLOCKED: INTEGRITY'
    assert result['unapproved_additions'] == 1


def test_status_shows_approved_boundary_and_uncleared_history(monkeypatch, capsys):
    _mock_snapshot(monkeypatch)
    monkeypatch.setattr(op.phase20, 'integrity', lambda: {
        'verified': True, 'protected_existing': 19176,
        'authorized_append_only_additions': 167,
        'quarantined_historical_evidence': 12,
        'terminated_incomplete_evidence': 4,
        'unauthorized_additions': 0, 'unknown_additions': 0,
        'unfinished_additions': 0, 'changed': [], 'missing': [], 'errors': [],
        'approved_boundary': {'path': 'docs/phase20-boundary-approved-v1.json',
                              'sha256': 'approved-hash'}})
    result = op.snapshot()
    assert all(row['settlement_eligibility'] == 'ELIGIBLE' for row in result['leagues'][:4])
    assert result['leagues'][2]['historical']['value']['incidents'][0]['code'] == 'ARTIFACT_DRIFT'
    assert main(['status']) == 0
    output = capsys.readouterr().out
    assert '12 quarantined' in output
    assert '4 terminated/incomplete' in output
    assert 'ARTIFACT_DRIFT' in output
    assert 'approved-hash' in output


def test_settle_delegates_to_phase20_only_after_all_gates_pass(monkeypatch):
    monkeypatch.setattr(op, '_authority', lambda: ({}, {}))
    monkeypatch.setattr(op, '_current', lambda *args: {'recorded_predictions': 1})
    monkeypatch.setattr(op.phase19_integrity, 'verify', lambda: {'verified': True})
    monkeypatch.setattr(op.phase20, 'authorization', lambda: {})
    monkeypatch.setattr(op.phase20, 'integrity', lambda: {
        'verified': True, 'added': [], 'changed': [], 'missing': []})
    monkeypatch.setattr(op.phase20, 'verify', lambda league: {'verified': True})
    calls = []

    def acquire(league, *, authorized=False):
        calls.append(('acquire', league, authorized))
        return {'status': 'RETRIEVED', 'source': {'path': 'tmp/result.json'}}

    def settle(league, path, *, authorized=False):
        calls.append(('settle', league, path, authorized))
        return {'accepted': 1, 'pending': 0}

    monkeypatch.setattr(op.phase20, 'acquire', acquire)
    monkeypatch.setattr(op.phase20, 'settle', settle)
    result = op.settle('BL1')
    assert result['status'] == 'SETTLED'
    assert calls == [('acquire', 'BL1', True),
                     ('settle', 'BL1', op.p.ROOT / 'tmp/result.json', True)]


def test_operator_exposes_no_new_collection_scheduling_or_promotion_commands(capsys):
    with pytest.raises(SystemExit) as err:
        main(['new-predict'])
    assert err.value.code == 2
    with pytest.raises(SystemExit):
        main(['schedule'])
    with pytest.raises(SystemExit):
        main(['promote'])
    with pytest.raises(SystemExit) as help_exit:
        main(['--help'])
    assert help_exit.value.code == 0
    help_text = capsys.readouterr().out
    for command in ('status', 'predictions', 'audit', 'settle', 'performance'):
        assert command in help_text


def test_audit_returns_blocked_when_integrity_value_is_false(monkeypatch, capsys):
    _mock_snapshot(monkeypatch)
    assert main(['audit']) == 2
    assert 'Phase 20 integrity: BLOCKED' in capsys.readouterr().out


def test_performance_has_no_measured_zero_when_unsettled(monkeypatch):
    monkeypatch.setattr(op.phase20, 'authorization', lambda: {
        'pair_counts': {league: 1 for league in op.LEAGUES}})
    monkeypatch.setattr(op, '_settlement_status', lambda league: {
        'settled': 0, 'pending': None, 'settled_fixture_ids': []})
    rows = op.performance()['leagues']
    assert all(row['descriptive_metrics'] is None and
               row['evidence'] == 'NO SETTLED EVIDENCE' for row in rows)
