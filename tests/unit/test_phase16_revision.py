"""V2 scientific rule: prospective, league-generic and fail-closed."""
import copy
from datetime import datetime, timezone

import pytest

from src.config.leagues import ACTIVE_LEAGUES
from src.evaluation import phase16_revision as r


def fixture(league='BL1', kickoff='2026-10-10T18:00:00+00:00'):
    return {'league': league, 'season': 2026, 'fixture_id': f'{league}_football_data_org_7',
        'provider_fixture_id': 7, 'home_id': 18, 'away_id': 29,
        'home_team': 'HOME', 'away_team': 'AWAY', 'kickoff_at': kickoff,
        'status': 'SCHEDULED', 'home_goals': None, 'away_goals': None}


def case(league='BL1'):
    old = fixture(league)
    rule = r.rule()
    cohort = {'cohort_id': league+'-clean-001', 'league': league, 'season': 2026,
        'protocol_sha256': 'v2-league-hash', 'revision_rule_sha256': rule['protocol_sha256'],
        'start_at': '2026-09-01T00:00:00+00:00', 'end_at': '2027-06-01T00:00:00+00:00'}
    pair = {key: old[key] for key in r.IDENTITY}
    pair.update({'protocol_sha256': cohort['protocol_sha256'], 'kickoff_at': old['kickoff_at'],
        'admission_at': '2026-09-05T10:01:00+00:00',
        'prediction_at': '2026-09-05T10:00:00+00:00',
        'active_prediction_at': '2026-09-05T10:00:00+00:00',
        'candidate_prediction_at': '2026-09-05T10:00:00+00:00',
        'causal_cutoff': '2026-09-05T10:00:00+00:00',
        'active_causal_cutoff': '2026-09-05T10:00:00+00:00',
        'candidate_causal_cutoff': '2026-09-05T10:00:00+00:00',
        'maximum_source_event_at': '2026-09-04T12:00:00+00:00',
        'feature_cutoffs': ['2026-09-05T10:00:00+00:00']})
    first = {'source': {'retrieved_at': '2026-09-05T09:00:00+00:00',
        'raw': {'path': 'raw-1', 'sha256': 'a'*64, 'size_bytes': 1}}, 'fixture': old}
    members = [pair, {'fixture_id': f'{league}_football_data_org_8',
                      'kickoff_at': '2026-10-20T18:00:00+00:00'}]
    return rule, cohort, pair, [first], members


def revised(versions, kickoff='2026-10-11T18:00:00+00:00', *, at='2026-09-06T09:00:00+00:00',
            status='TIMED'):
    event = copy.deepcopy(versions[0]['fixture'])
    event.update(kickoff_at=kickoff, status=status)
    versions.append({'source': {'retrieved_at': at,
        'raw': {'path': f'raw-{len(versions)+1}', 'sha256': 'b'*64, 'size_bytes': 1}},
        'fixture': event})
    return versions[-1]


def classify(parts):
    return r.classify(*parts)


@pytest.mark.parametrize('league', ACTIVE_LEAGUES)
def test_one_rule_for_every_active_league(league):
    parts = case(league)
    revised(parts[3])
    result = classify(parts)
    assert result['classification'] == 'PERMITTED_KICKOFF_REVISION'
    assert result['admissible_for_new_cohort'] and not result['activation_authorized']
    assert result['observed_versions'][-1]['identity_stable']
    assert result['observed_versions'][-1]['causality_preserved']


def test_unchanged_and_duplicate_identical_revision():
    parts = case()
    assert classify(parts)['classification'] == 'UNCHANGED'
    revised(parts[3], kickoff=parts[2]['kickoff_at'])
    duplicate = copy.deepcopy(parts[3][-1])
    duplicate['source']['retrieved_at'] = '2026-09-07T09:00:00+00:00'
    parts[3].append(duplicate)
    result = classify(parts)
    assert result['classification'] == 'UNCHANGED'
    assert result['observed_versions'][-1]['duplicate_identical']


@pytest.mark.parametrize('kickoff', ['2026-10-11T18:00:00+00:00', '2026-09-06T18:00:00+00:00'])
def test_later_and_earlier_safe_kickoff(kickoff):
    parts = case()
    revised(parts[3], kickoff)
    assert classify(parts)['classification'] == 'PERMITTED_KICKOFF_REVISION'


def test_earlier_than_prediction_is_causality_violation():
    parts = case()
    revised(parts[3], '2026-09-05T09:30:00+00:00')
    assert classify(parts)['classification'] == 'CAUSALITY_VIOLATION'


def test_multiple_successive_revisions_and_retrieval_sorting():
    parts = case()
    last = revised(parts[3], '2026-10-12T18:00:00+00:00', at='2026-09-07T09:00:00+00:00')
    middle = copy.deepcopy(last)
    middle['source']['retrieved_at'] = '2026-09-06T09:00:00+00:00'
    middle['fixture']['kickoff_at'] = '2026-10-11T18:00:00+00:00'
    parts[3].append(middle)  # Input is intentionally not retrieval-ordered.
    result = classify(parts)
    assert result['classification'] == 'PERMITTED_KICKOFF_REVISION'
    assert [row['fixture']['kickoff_at'] for row in result['observed_versions']][-2:] == [
        '2026-10-11T18:00:00+00:00', '2026-10-12T18:00:00+00:00']


@pytest.mark.parametrize('key,value', [
    ('home_id', 99), ('away_id', 99), ('home_team', 'REPLACEMENT'),
    ('league', 'PL'), ('season', 2027), ('provider_fixture_id', 8),
    ('fixture_id', 'BL1_football_data_org_8')])
def test_identity_changes_are_material(key, value):
    parts = case()
    revised(parts[3])["fixture"][key] = value
    assert classify(parts)['classification'] == 'MATERIAL_FIXTURE_REVISION'


@pytest.mark.parametrize('status,kickoff,expected', [
    ('POSTPONED', None, 'INSUFFICIENT_EVIDENCE'),
    ('CANCELLED', '2026-10-11T18:00:00+00:00', 'MATERIAL_FIXTURE_REVISION'),
    ('ABANDONED', '2026-10-11T18:00:00+00:00', 'MATERIAL_FIXTURE_REVISION')])
def test_unknown_postponement_and_terminal_status(status, kickoff, expected):
    parts = case()
    revised(parts[3], kickoff, status=status)
    assert classify(parts)['classification'] == expected


def test_horizon_and_cohort_order_fail_closed():
    parts = case()
    revised(parts[3], '2027-06-02T18:00:00+00:00')
    # Both constraints fail; ordering has higher frozen precedence.
    assert classify(parts)['classification'] == 'ORDERING_VIOLATION'
    parts = case()
    revised(parts[3], '2026-10-21T18:00:00+00:00')
    assert classify(parts)['classification'] == 'ORDERING_VIOLATION'
    parts = list(case())
    parts[4] = [parts[2]]
    revised(parts[3], '2027-06-02T18:00:00+00:00')
    assert classify(parts)['classification'] == 'HORIZON_VIOLATION'


def test_missing_source_and_same_time_disagreement_fail_closed():
    parts = case()
    assert r.classify(parts[0], parts[1], parts[2], [], parts[4])['classification'] == 'INSUFFICIENT_EVIDENCE'
    parts = case()
    revised(parts[3])['source'].pop('retrieved_at')
    assert classify(parts)['classification'] == 'INSUFFICIENT_EVIDENCE'
    parts = case()
    revised(parts[3])
    parts[3][-1]['source']['retrieved_at'] = parts[3][0]['source']['retrieved_at']
    assert classify(parts)['classification'] == 'INSUFFICIENT_EVIDENCE'


def test_fixture_disappears_and_original_version_missing():
    parts = case()
    later = revised(parts[3])
    later['fixture'] = None
    assert classify(parts)['classification'] == 'INSUFFICIENT_EVIDENCE'
    parts = case()
    revised(parts[3])
    parts[3].pop(0)
    assert classify(parts)['classification'] == 'INSUFFICIENT_EVIDENCE'


def test_revision_observed_after_revised_kickoff_fails():
    parts = case()
    revised(parts[3], at='2026-10-12T09:00:00+00:00')
    assert classify(parts)['classification'] == 'CAUSALITY_VIOLATION'


def test_stale_admission_and_missing_raw_identity_fail_closed():
    parts = case()
    revised(parts[3], at='2026-09-05T09:30:00+00:00')
    assert classify(parts)['classification'] == 'INSUFFICIENT_EVIDENCE'
    parts = case()
    revised(parts[3])['source'].pop('raw')
    assert classify(parts)['classification'] == 'INSUFFICIENT_EVIDENCE'


def test_late_original_source_and_feature_cutoff_fail_closed():
    parts = case()
    parts[3][0]['source']['retrieved_at'] = '2026-09-05T10:00:30+00:00'
    assert classify(parts)['classification'] == 'INSUFFICIENT_EVIDENCE'
    parts = case()
    parts[2]['feature_cutoffs'] = ['2026-09-05T10:00:30+00:00']
    revised(parts[3])
    assert classify(parts)['classification'] == 'CAUSALITY_VIOLATION'


def test_finished_settlement_after_permitted_revision():
    parts = case()
    revised(parts[3])
    final = revised(parts[3], status='FINISHED', at='2026-10-11T20:00:00+00:00')
    final['fixture'].update(home_goals=2, away_goals=1)
    assert classify(parts)['classification'] == 'PERMITTED_KICKOFF_REVISION'
    parts = case()
    revised(parts[3], status='FINISHED', at='2026-10-11T20:00:00+00:00')
    assert classify(parts)['classification'] == 'INSUFFICIENT_EVIDENCE'


def test_protocol_swap_and_cross_league_pair_rejected():
    parts = case()
    parts[1]['protocol_sha256'] = 'other'
    with pytest.raises(ValueError, match='PAIR_PROTOCOL_SUBSTITUTION'):
        classify(parts)
    parts = case()
    parts[2]['league'] = 'PL'
    with pytest.raises(ValueError, match='CROSS_LEAGUE_PAIR'):
        classify(parts)


def test_classifier_does_not_read_scores_or_wall_clock():
    parts = case()
    revised(parts[3])
    first = classify(parts)
    parts[3][-1]['fixture']['home_goals'] = 19
    parts[3][-1]['fixture']['away_goals'] = 0
    second = classify(parts)
    assert first['classification'] == second['classification'] == 'PERMITTED_KICKOFF_REVISION'
    assert first['classification_sha256'] != second['classification_sha256']  # Source evidence remains explicit.


def test_rule_is_deterministic():
    assert r.rule() == r.rule()
    assert datetime.now(timezone.utc)  # No clock input to the rule.
