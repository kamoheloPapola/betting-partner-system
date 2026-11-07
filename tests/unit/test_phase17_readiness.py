import copy
import json

import pytest

from src.evaluation import phase16_cohort, phase17_pl_adapter as adapter
from src.evaluation import phase17_readiness as readiness


def _examples():
    source, document = adapter.load_raw()
    aliases = json.loads(adapter.ALIAS.read_text(encoding='utf-8'))['aliases']
    return source, document['matches'], aliases


def test_pl_archived_source_replays_byte_equivalent_and_preserves_identity():
    first_source, first = adapter.normalize_snapshot()
    second_source, second = adapter.normalize_snapshot()
    assert first_source == second_source and first == second
    assert len(first) == 380 and len({row['fixture_id'] for row in first}) == 380
    assert {'TIMED', 'FINISHED'} <= {row['status'] for row in first}
    assert any('COVENTRY' in (row['home_team'], row['away_team']) for row in first)
    assert any('HULL' in (row['home_team'], row['away_team']) for row in first)
    assert adapter.normalize_versions([(adapter.SOURCE_2026, adapter.RAW_2026, 560542)] * 2)[0]['fixture'] == first[0]


def test_pl_fixture_revision_keeps_stable_provider_id_and_changes_date_identity():
    source, examples, aliases = _examples()
    original = next(row for row in examples if row['status'] == 'TIMED')
    changed = copy.deepcopy(original)
    changed['utcDate'] = '2026-09-25T20:00:00Z'
    before = adapter.normalize_event(original, source, aliases)
    after = adapter.normalize_event(changed, source, aliases)
    assert after['fixture_id'] == before['fixture_id']
    assert after['provider_fixture_id'] == before['provider_fixture_id']
    assert after['kickoff_at'] != before['kickoff_at']
    assert after['phase7p_canonical_fixture_id'] != before['phase7p_canonical_fixture_id']


@pytest.mark.parametrize('status', ['POSTPONED', 'CANCELLED', 'ABANDONED', 'SUSPENDED'])
def test_pl_terminal_status_is_normalized_without_a_result(status):
    source, examples, aliases = _examples()
    example = copy.deepcopy(next(row for row in examples if row['status'] == 'TIMED'))
    example['status'] = status
    normalized = adapter.normalize_event(example, source, aliases)
    assert normalized['status'] == status
    assert normalized['home_goals'] is None and normalized['away_goals'] is None


def test_pl_finished_score_and_reversed_orientation_are_explicit():
    source, examples, aliases = _examples()
    original = next(row for row in examples if row['status'] == 'FINISHED')
    result = adapter.normalize_event(original, source, aliases)
    assert result['home_goals'] == original['score']['fullTime']['home']
    assert result['away_goals'] == original['score']['fullTime']['away']
    reversed_event = copy.deepcopy(original)
    reversed_event['homeTeam'], reversed_event['awayTeam'] = original['awayTeam'], original['homeTeam']
    reversed_result = adapter.normalize_event(reversed_event, source, aliases)
    assert reversed_result['home_id'] == result['away_id']
    assert reversed_result['away_id'] == result['home_id']
    assert reversed_result['fixture_id'] == result['fixture_id']
    assert reversed_result['home_team'] != result['home_team']


@pytest.mark.parametrize('mutation', [
    lambda row: row.update(utcDate=None),
    lambda row: row.update(status='UNRECOGNIZED'),
    lambda row: row['competition'].update(code='SA'),
    lambda row: row['homeTeam'].update(id=None),
    lambda row: row['score']['fullTime'].update(home=1),
])
def test_pl_malformed_source_fails_closed(mutation):
    source, examples, aliases = _examples()
    row = copy.deepcopy(next(row for row in examples if row['status'] == 'TIMED'))
    mutation(row)
    with pytest.raises((RuntimeError, ValueError, TypeError, KeyError)):
        adapter.normalize_event(row, source, aliases)


def test_pl_source_identity_conflict_rejected(tmp_path):
    source = json.loads(adapter.SOURCE_2026.read_text(encoding='utf-8'))
    source['provider'] = 'unapproved-provider'
    altered = tmp_path / 'source.json'
    altered.write_text(json.dumps(source), encoding='utf-8')
    with pytest.raises(ValueError, match='PL_RAW_SOURCE_IDENTITY_MISMATCH'):
        adapter.load_raw(altered, adapter.RAW_2026)


def test_pl_requires_a_scientific_protocol_before_cohort_creation():
    with pytest.raises(ValueError, match='UNQUALIFIED_CURRENT_SOURCE_ADAPTER'):
        phase16_cohort.initialize('PL-clean-001', 'PL')
    assert readiness.verify_pl()['classification'] == 'PL_ADAPTER_OFFLINE_QUALIFIED'
    result = readiness.status()
    assert result['scientific_state'] == 'PHASE17_PARTIAL_READY_PL_PROTOCOL_BLOCKED'
    assert all(row['pairs'] == row['settlements'] == 0 for row in result['leagues'])
    assert sum(row['cohort'] is not None for row in result['leagues']) == 4
