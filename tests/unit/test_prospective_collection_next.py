"""Focused fail-closed checks for the separate prospective lineage."""
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from src.evaluation import prospective_collection_next as next_collection


def _fixture(status='TIMED'):
    return {'league': 'BL1', 'season': 2026,
            'fixture_id': 'BL1_football_data_org_123', 'provider_fixture_id': 123,
            'home_id': 1, 'away_id': 2, 'home_team': 'HOME', 'away_team': 'AWAY',
            'kickoff_at': '2026-10-09T18:30:00+00:00', 'status': status,
            'home_goals': None, 'away_goals': None}


def test_newly_finished_result_blocks_stale_qualified_inputs(monkeypatch):
    old = _fixture()
    fresh = {**old, 'status': 'FINISHED', 'home_goals': 1, 'away_goals': 0}
    monkeypatch.setattr(next_collection.p, 'current_events',
                        lambda metadata, protocol, qualification: [fresh])
    qualification = {'assessment': {'events': [old]}}
    with pytest.raises(ValueError, match='NEW_LINEAGE_CAUSAL_INPUTS_STALE'):
        next_collection._events('BL1', {}, qualification, {'season': 2026})


def test_post_kickoff_first_capture_appends_nothing(monkeypatch, tmp_path):
    event = _fixture()
    controller = SimpleNamespace(legacy_id='active', candidate_id='candidate',
        bundle=lambda name: name, owner=SimpleNamespace(load=lambda bundle: None))
    monkeypatch.setattr(next_collection, 'authorization',
        lambda: {'bindings': {'BL1': {'qualification': {}}}})
    monkeypatch.setattr(next_collection, '_gate', lambda: None)
    monkeypatch.setattr(next_collection, '_source_path', lambda league, path: tmp_path / 'matches.json')
    monkeypatch.setattr(next_collection, 'state_lock', lambda path: nullcontext())
    monkeypatch.setattr(next_collection, 'source_record',
        lambda league, path: {'retrieved_at': '2026-10-09T17:00:00+00:00'})
    monkeypatch.setattr(next_collection, '_context',
        lambda league: ({'assessment': {'upcoming': [event]}}, {}, {}, controller))
    monkeypatch.setattr(next_collection, '_events', lambda *args: [event])
    monkeypatch.setattr(next_collection, 'journal', lambda league: [])
    monkeypatch.setattr(next_collection.p, 'identity', lambda path: {'path': str(path)})
    monkeypatch.setattr(next_collection.p, 'now', lambda: '2026-10-09T18:31:00+00:00')

    result = next_collection.collect('BL1', tmp_path / 'matches.json',
                                     date_from='2026-10-09', date_to='2026-10-09')
    assert result['accepted'] == []
    assert result['rejected'] == [{'fixture_id': event['fixture_id'],
                                   'reason': 'NEW_LINEAGE_NOT_PREKICKOFF'}]
    assert not list(tmp_path.rglob('journal/*.json'))


def test_identical_rerun_adds_no_pair(monkeypatch, tmp_path):
    event = _fixture()
    prior = {'payload': {'pair': {'fixture_id': event['fixture_id']},
                         'fresh_fixture': event}, 'record_sha256': 'old-head'}
    controller = SimpleNamespace(legacy_id='active', candidate_id='candidate',
        bundle=lambda name: name, owner=SimpleNamespace(load=lambda bundle: None))
    monkeypatch.setattr(next_collection, 'authorization',
        lambda: {'bindings': {'BL1': {'qualification': {}}}})
    monkeypatch.setattr(next_collection, '_gate', lambda: None)
    monkeypatch.setattr(next_collection, '_source_path', lambda league, path: tmp_path / 'matches.json')
    monkeypatch.setattr(next_collection, 'state_lock', lambda path: nullcontext())
    monkeypatch.setattr(next_collection, 'source_record',
        lambda league, path: {'retrieved_at': '2026-09-24T10:00:00+00:00'})
    monkeypatch.setattr(next_collection, '_context',
        lambda league: ({'assessment': {'upcoming': [event]}}, {}, {}, controller))
    monkeypatch.setattr(next_collection, '_events', lambda *args: [event])
    monkeypatch.setattr(next_collection, 'journal', lambda league: [prior])
    monkeypatch.setattr(next_collection.p, 'identity', lambda path: {'path': str(path)})
    monkeypatch.setattr(next_collection.p, 'now', lambda: '2026-09-24T10:01:00+00:00')

    result = next_collection.collect('BL1', tmp_path / 'matches.json',
                                     date_from='2026-10-09', date_to='2026-10-09')
    assert result['accepted'] == []
    assert result['already_recorded'] == [event['fixture_id']]
    assert result['journal_head'] == 'old-head'
    assert not list(tmp_path.rglob('journal/*.json'))
