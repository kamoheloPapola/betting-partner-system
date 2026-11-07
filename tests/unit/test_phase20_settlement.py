"""Focused behavior of the isolated Phase 20 settlement journal."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from src.evaluation import phase20_settlement as s


def _boundary_pass():
    return {'verified': True, 'protected_existing': 19176,
            'changed': [], 'missing': [], 'errors': [],
            'authorized_append_only_additions': 167,
            'quarantined_historical_evidence': 12,
            'terminated_incomplete_evidence': 4,
            'unauthorized_additions': 0, 'unknown_additions': 0,
            'unfinished_additions': 0}


def test_live_integrity_uses_fixed_approved_boundary_and_preserves_counts(monkeypatch, tmp_path):
    monkeypatch.setattr(s, 'authorization', lambda: {})
    monkeypatch.setattr(s, 'RUN', tmp_path / 'phase20')
    calls = []
    def verify(root, run, contract, *, approved_sha256):
        calls.append((root, run, contract, approved_sha256))
        return _boundary_pass()
    monkeypatch.setattr(s.phase20_boundary, 'verify', verify)
    monkeypatch.setattr(s, '_replay_evidence_binding', lambda: calls.append('replay-bound'))
    result = s.integrity()
    assert result['verified'] is True
    assert result['protected_existing'] == 19176
    assert (result['authorized_append_only_additions'],
            result['quarantined_historical_evidence'],
            result['terminated_incomplete_evidence']) == (167, 12, 4)
    assert calls == [(s.p.ROOT, s.RUN, s.APPROVED_BOUNDARY,
                      s.APPROVED_BOUNDARY_SHA256), 'replay-bound']
    assert not s.RUN.exists()  # Integrity alone publishes no settlement evidence.


@pytest.mark.parametrize('failure', ['boundary_false', 'verifier_exception', 'certificate_drift'])
def test_live_integrity_fails_closed_without_fallback(monkeypatch, failure):
    monkeypatch.setattr(s, 'authorization', lambda: {})
    if failure == 'boundary_false':
        monkeypatch.setattr(s.phase20_boundary, 'verify', lambda *args, **kwargs: {
            **_boundary_pass(), 'verified': False, 'errors': ['BOUNDARY_ADDITION_DRIFT']})
    elif failure == 'verifier_exception':
        def broken(*args, **kwargs):
            raise RuntimeError('BOUNDARY_CANNOT_COMPLETE')
        monkeypatch.setattr(s.phase20_boundary, 'verify', broken)
    else:
        monkeypatch.setattr(s.phase20_boundary, 'verify', lambda *args, **kwargs: _boundary_pass())
        monkeypatch.setattr(s, '_replay_evidence_binding', lambda: (_ for _ in ()).throw(
            ValueError('CERTIFICATE_DRIFT')))
    result = s.integrity()
    assert result['verified'] is False
    assert result['approved_boundary']['sha256'] == s.APPROVED_BOUNDARY_SHA256
    assert any('BOUNDARY_ADDITION_DRIFT' in error or 'BOUNDARY_CANNOT_COMPLETE' in error or
               'CERTIFICATE_DRIFT' in error for error in result['errors'])


def test_direct_acquisition_checks_integrity_before_creating_attempt(monkeypatch, tmp_path):
    monkeypatch.setattr(s, 'RUN', tmp_path / 'phase20')
    monkeypatch.setattr(s, 'authorization', lambda: {})
    monkeypatch.setattr(s, 'integrity', lambda: {'verified': False})
    with pytest.raises(ValueError, match='PHASE20_PROTECTED_INTEGRITY_FAILED'):
        s.acquire('PD', authorized=True)
    assert not (s.RUN / 'PD/acquisition').exists()


def test_phase12_pd_pair_cannot_enter_phase20_pair_population(monkeypatch):
    phase19_pair = {"kind": "pair", "league": "PD", "record_sha256": "phase19-head"}
    phase12_pair118 = {"kind": "pair", "league": "PD", "record_sha256": "phase12-event-118"}
    seen = []
    monkeypatch.setattr(s.phase19, "verify", lambda: {"cohorts": [
        {"league": "PD", "cohort_id": "PD-clean-001"}]})
    def phase19_journal(cohort_id):
        seen.append(cohort_id)
        assert cohort_id == "PD-clean-001"
        return [phase19_pair]
    monkeypatch.setattr(s.cohort, "journal", phase19_journal)
    assert s._pairs("PD") == [phase19_pair]
    assert phase12_pair118 not in s._pairs("PD")
    assert seen == ["PD-clean-001", "PD-clean-001"]


def _fixture(monkeypatch):
    temporary = TemporaryDirectory(dir=s.p.ROOT / 'tmp')
    monkeypatch.setattr(s, 'RUN', Path(temporary.name))
    kickoff = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
    identity = {'league': 'BL1', 'season': 2026, 'fixture_id': 'BL1_football_data_org_1',
        'provider_fixture_id': 1, 'home_id': 1, 'away_id': 2,
        'home_team': 'HOME', 'away_team': 'AWAY', 'kickoff_at': kickoff}
    initial = {**identity, 'status': 'TIMED', 'home_goals': None, 'away_goals': None}
    pair = {'league': 'BL1', 'fixture_id': identity['fixture_id'], 'kickoff_at': kickoff,
            'pair_sha256': 'pair-hash', 'active': {'model': 'active'},
            'candidate': {'model': 'candidate'}}
    pair_event = {'record_sha256': 'event-hash', 'payload': {'pair': pair,
                  'initial_observation': {'fixture': initial}}}
    raw = s.RUN / 'BL1/acquisition/attempt-000001/matches.raw'
    raw.parent.mkdir(parents=True)
    raw.write_bytes(b'fixture')
    source_path = raw.with_suffix('.json')
    source_path.write_text('{}', encoding='utf-8')
    source = {'started_at': s.p.now(), 'retrieved_at': s.p.now(), 'raw': s.p.identity(raw)}
    events = [{**identity, 'status': 'FINISHED', 'home_goals': 2, 'away_goals': 1}]
    monkeypatch.setattr(s, 'authorization', lambda: {'frozen_at': s.p.now()})
    monkeypatch.setattr(s, 'integrity', lambda: {'verified': True})
    monkeypatch.setattr(s, '_pairs', lambda league: [pair_event] if league == 'BL1' else [])
    monkeypatch.setattr(s, 'source_record', lambda league, path, **kwargs: source)
    monkeypatch.setattr(s, '_events', lambda league, row: events)
    monkeypatch.setattr(s, '_score', lambda pair, home, away: {
        side: {'prediction_sha256': side, 'log_loss': 0.1,
               'multiclass_brier': 0.2, 'joint_goal_squared_error': 0.3,
               'joint_goal_poisson_deviance': 0.4}
        for side in ('active', 'candidate')})
    return temporary, source_path, events, source


def test_manual_settlement_replays_and_duplicate_is_noop(monkeypatch):
    temporary, path, events, _ = _fixture(monkeypatch)
    with temporary:
        with pytest.raises(ValueError, match='OPERATOR_AUTHORIZATION_REQUIRED'):
            s.settle('BL1', path)
        assert s.settle('BL1', path, authorized=True)['accepted'] == 1
        assert s.settle('BL1', path, authorized=True)['accepted'] == 0
        fresh = s.RUN / 'BL1/acquisition/attempt-000002/matches.json'
        fresh.parent.mkdir(parents=True)
        fresh.write_text('{}', encoding='utf-8')
        assert s.settle('BL1', fresh, authorized=True)['accepted'] == 0
        result = s.verify('BL1')
        assert (result['pairs'], result['settled'], result['pending']) == (1, 1, 0)
        assert result['promotion_authorized'] is False
        events[0]['home_goals'] = 3
        corrected = s.RUN / 'BL1/acquisition/attempt-000003/matches.json'
        corrected.parent.mkdir(parents=True)
        corrected.write_text('{}', encoding='utf-8')
        with pytest.raises(ValueError, match='CONFLICTING_RESULT'):
            s.settle('BL1', corrected, authorized=True)
        assert (corrected.parent / 'assessment.json').exists()
        assert len(s.journal('BL1')) == 1


def test_nonfinal_and_empty_pl_are_clean_noops(monkeypatch):
    temporary, path, events, _ = _fixture(monkeypatch)
    with temporary:
        events[0].update(status='POSTPONED', home_goals=None, away_goals=None)
        result = s.settle('BL1', path, authorized=True)
        assert (result['accepted'], result['pending']) == (0, 1)
        assert s.verify('PL')['pairs'] == s.verify('PL')['settled'] == 0


def test_unknown_fixture_cannot_cross_league(monkeypatch):
    temporary, _, _, _ = _fixture(monkeypatch)
    with temporary:
        other = s.RUN / 'SA/acquisition/attempt-000001/matches.json'
        other.parent.mkdir(parents=True)
        other.write_text('{}', encoding='utf-8')
        result = s.settle('SA', other, authorized=True)
        assert result['accepted'] == result['pairs'] == 0
        assert result['rejected_results'][0]['reason'] == 'UNKNOWN_PHASE19_PAIR'
        assert s.verify('SA')['rejected_results'] == 1
        assert s.journal('BL1') == []


@pytest.mark.parametrize('change', [
    {'away_goals': None},
    {'kickoff_at': (datetime.now(timezone.utc) + timedelta(days=1)).isoformat()},
])
def test_invalid_final_result_is_rejected(monkeypatch, change):
    temporary, path, events, _ = _fixture(monkeypatch)
    with temporary:
        events[0].update(change)
        result = s.settle('BL1', path, authorized=True)
        assert result['accepted'] == 0
        assert result['pending'] == 1
        assert len(result['rejected_results']) == 1
        assert s.journal('BL1') == []


def test_journal_corruption_fails_closed(monkeypatch):
    temporary, path, _, _ = _fixture(monkeypatch)
    with temporary:
        s.settle('BL1', path, authorized=True)
        (s.RUN / 'BL1/journal/stranded.tmp').write_text('partial', encoding='utf-8')
        assert s.settle('BL1', path, authorized=True)['accepted'] == 0
        member = s.RUN / 'BL1/journal/00000001.json'
        member.write_text('{}', encoding='utf-8')
        with pytest.raises((ValueError, KeyError)):
            s.verify('BL1')
