"""Focused guards for the one-shot Phase 19 operational boundary."""
import copy

import pytest

from src.evaluation import phase12_prospective as p
from src.evaluation import phase19_authorization as authorization
from src.evaluation import phase19_batch as batch
from src.evaluation import phase19_collection as collection
from src.evaluation import phase19_pl_cycle as pl


def test_frozen_authorization_and_one_shot_intent_reconstruct():
    frozen = authorization.verify()
    intent = collection.intent()
    assert frozen['collection_mode'] == 'CONTROLLED_ONE_SHOT'
    assert [row['league'] for row in frozen['cohorts']] == list(collection.LEAGUES)
    assert frozen['scheduler_authorized'] is False
    assert frozen['settlement_authorized'] is False
    assert frozen['activation_authorized'] is False
    assert intent['authorization'] == p.identity(authorization.AUTH)


@pytest.mark.parametrize('league', collection.LEAGUES)
def test_archived_source_reuse_is_idempotent_without_network(league, monkeypatch):
    monkeypatch.setattr(collection.source, 'fetch', lambda *a, **k: (_ for _ in ()).throw(
        AssertionError('unexpected second fetch')))
    initial = collection.acquire(league)
    recovered = collection.recover_network_attempt(league)
    assert initial['status'] == 'SOURCE_UNAVAILABLE'
    assert recovered['status'] == 'RETRIEVED'
    p.frozen.reference(recovered['raw'])


def test_stale_and_out_of_horizon_fixtures_fail_closed():
    spec = {'start_at': '2026-09-23T00:00:00+00:00',
            'end_at': '2027-01-01T00:00:00+00:00'}
    source = {'retrieved_at': '2026-09-23T01:00:00+00:00'}
    fixture = {'status': 'TIMED', 'kickoff_at': '2026-10-01T12:00:00+00:00'}
    batch.check_target(fixture, spec, source, '2026-09-23T02:00:00+00:00')
    for target, retrieved, at in (
        ({**fixture, 'status': 'POSTPONED'}, source, '2026-09-23T02:00:00+00:00'),
        ({**fixture, 'kickoff_at': '2026-09-22T12:00:00+00:00'}, source,
         '2026-09-23T02:00:00+00:00'),
        ({**fixture, 'kickoff_at': '2027-01-02T12:00:00+00:00'}, source,
         '2026-09-23T02:00:00+00:00'),
        (fixture, {'retrieved_at': '2026-09-23T03:00:00+00:00'},
         '2026-09-23T02:00:00+00:00')):
        with pytest.raises(ValueError, match='PHASE19_FIXTURE_NOT_CAUSALLY_ELIGIBLE'):
            batch.check_target(target, spec, retrieved, at)


def test_pl_fresh_raw_replays_to_explicit_zero_admissions():
    value = pl.verify()
    assert value['fixtures_observed'] == 380
    assert value['future_within_horizon'] == len(value['rejections'])
    assert value['eligible_with_causal_features'] == 0
    assert value['horizon_end'] == '2027-01-01T00:00:00+00:00'
    assert value['pairs_admitted'] == value['settlements'] == 0


def test_pl_modified_archive_identity_rejected(monkeypatch):
    original = collection.recover_network_attempt
    def altered(league):
        row = copy.deepcopy(original(league))
        row['raw']['sha256'] = '0' * 64
        return row
    monkeypatch.setattr(collection, 'recover_network_attempt', altered)
    with pytest.raises(ValueError, match='ARTIFACT_DRIFT|SOURCE_BYTES_DRIFT|REFERENCE'):
        pl.reconstruct()


@pytest.mark.parametrize('league', ('BL1', 'SA', 'FL1', 'PD'))
def test_discovery_and_qualified_source_versions_agree_on_fixture_facts(league):
    assignment = next(row for row in authorization.verify()['cohorts'] if row['league'] == league)
    _, qualification, index = batch.qualification(league)
    protocol = p.read(p.frozen.reference(assignment['protocol']))
    discovered = p.current_events(collection.recover_network_attempt(league), protocol, qualification)
    qualified = p.current_events(index['sources'][3], protocol, qualification)
    stable = ('league', 'season', 'fixture_id', 'provider_fixture_id', 'home_id',
              'away_id', 'home_team', 'away_team', 'kickoff_at', 'status',
              'home_goals', 'away_goals')
    first = {row['fixture_id']: row for row in discovered}
    second = {row['fixture_id']: row for row in qualified}
    assert len(first) == len(discovered) == len(qualified) == len(second)
    assert first.keys() == second.keys()
    assert all(all(first[key][field] == second[key][field] for field in stable)
               for key in first)
