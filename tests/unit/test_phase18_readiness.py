"""Phase 18 authority boundary and empty PL cohort."""
import copy
from datetime import timedelta

import pytest

from src.evaluation import phase12_prospective as p
from src.evaluation import phase16_cohort, phase16_revision
from src.evaluation import phase17_readiness as prior
from src.evaluation import phase18_authority as authority
from src.evaluation import phase18_cohort as cohort
from src.evaluation import phase18_verifier as verifier


def test_pl_contract_adopts_only_frozen_phase7p_boundary():
    value = authority.verify_pl_contract()
    assert value['end_at'] == '2027-01-01T00:00:00+00:00'
    assert value['horizon_authority_phase'] == 'Phase7P'
    assert value['phase11_pl_protocol_exists'] is False
    assert value['revision_rule'] == p.identity(phase16_cohort.RULE_FILE)
    assert value['phase17_source_qualification'] == p.identity(prior.RUN / 'pl_source_qualification.json')
    assert value['initial_pair_prefix'] == value['initial_settlement_prefix'] == []
    assert value['historical_pl_pair_import_authorized'] is False


@pytest.mark.parametrize('league', ('BL1', 'SA', 'FL1', 'PD'))
def test_phase11_is_still_supported_authority_and_existing_cohort_unchanged(league):
    old_ref = p.identity(p.PHASE11 / league / 'prospective_protocol.json')
    resolved = authority.resolve(old_ref)
    assert resolved['authority_type'] == 'phase11_frozen_protocol'
    assert resolved['league'] == league
    cohort_id = prior.COHORT_IDS[league]
    path = phase16_cohort.RUN / 'cohorts' / cohort_id / 'cohort_specification.json'
    before = p.identity(path)
    assert cohort.initialize(cohort_id, old_ref)['cohort_id'] == cohort_id
    assert p.identity(path) == before


def test_pl_uses_same_generic_initializer_and_has_empty_independent_cohort():
    ref = p.identity(authority.PL_CONTRACT)
    assert authority.resolve(ref)['authority_type'] == 'phase18_explicit_pl_contract'
    before = verifier.verify_pl_cohort()
    assert cohort.initialize('PL-clean-001', ref)['cohort_id'] == 'PL-clean-001'
    assert verifier.verify_pl_cohort() == before
    result = verifier.all_leagues()
    assert result['scientific_state'] == 'PHASE18_ALL_LEAGUES_V2_READY_FOR_PROSPECTIVE_COLLECTION'
    assert len(result['leagues']) == 5
    assert all(row['pairs'] == row['settlements'] == 0 and row['journal_head'] is None
               for row in result['leagues'])
    assert len({row['cohort']['path'] for row in result['leagues']}) == 5


def test_pl_contract_cannot_spawn_a_second_cohort():
    with pytest.raises(ValueError, match='PL_CONTRACT_BOUND_TO_SINGLE_COHORT'):
        cohort.initialize('PL-clean-002', p.identity(authority.PL_CONTRACT))
    assert not (authority.RUN / 'cohorts/PL-clean-002').exists()


def test_phase7p_specification_or_arbitrary_artifact_is_not_a_cohort_authority():
    for path in (prior.PHASE7_SPEC, prior.RUN / 'pl_source_qualification.json'):
        with pytest.raises(ValueError, match='UNSUPPORTED_PROSPECTIVE_HORIZON_AUTHORITY'):
            authority.resolve(p.identity(path))


def test_modified_phase7p_reference_hash_fails_closed(tmp_path, monkeypatch):
    altered = copy.deepcopy(p.read(authority.PHASE7P_REFERENCE))
    altered['file_sha256'] = '0' * 64
    path = tmp_path / 'reference.json'
    path.write_text(__import__('json').dumps(altered), encoding='utf-8')
    monkeypatch.setattr(authority, 'PHASE7P_REFERENCE', path)
    with pytest.raises(ValueError, match='PHASE7P_FROZEN_REFERENCE_DRIFT'):
        authority._frozen_pl_inputs()


def test_modified_pl_contract_and_arbitrary_end_fail_closed(monkeypatch):
    original = p.read(authority.PL_CONTRACT)
    altered = p.seal({**{k:v for k,v in original.items() if k != 'contract_sha256'},
                      'end_at': '2027-06-01T00:00:00+00:00'}, 'contract_sha256')
    real_read = p.read
    monkeypatch.setattr(p, 'read', lambda path: altered if path == authority.PL_CONTRACT else real_read(path))
    with pytest.raises(ValueError, match='PL_CONTRACT_RECONSTRUCTION_MISMATCH'):
        authority.verify_pl_contract()


def test_pl_creation_cannot_predate_contract_freeze(monkeypatch):
    directory = authority.RUN / 'cohorts/PL-clean-001'
    intent = p.read(directory / 'creation_intent.json')
    frozen = p.instant(authority.verify_pl_contract()['frozen_at'])
    altered = p.seal({**{k:v for k,v in intent.items() if k != 'record_sha256'},
        'created_at': (frozen - timedelta(minutes=1)).isoformat(),
        'start_at': (frozen - timedelta(minutes=1) + timedelta(seconds=2)).isoformat()})
    real_read = p.read
    monkeypatch.setattr(p, 'read', lambda path: altered if path == directory / 'creation_intent.json' else real_read(path))
    with pytest.raises(ValueError, match='PL_CREATION_INTENT_OR_START_INVALID'):
        verifier.verify_pl_cohort()


def test_historical_pl_prediction_cannot_enter_new_cohort():
    spec = p.read(authority.RUN / 'cohorts/PL-clean-001/cohort_specification.json')
    old = next((p.ROOT / 'baselines/phase7er/phase7er-qualified-source-refresh-79110f9a-20260905-v2/phase7p_resumed_cohort/predictions').glob('*.json'))
    with pytest.raises(ValueError, match='CROSS_PROTOCOL_OR_HISTORICAL_PAIR'):
        cohort.validate_pair_origin(spec, p.read(old))


def test_pl_real_cohort_binds_the_unchanged_generic_v2_classifier():
    spec = p.read(authority.RUN / 'cohorts/PL-clean-001/cohort_specification.json')
    rule = phase16_cohort.load_rule()
    assert rule == phase16_revision.rule()
    assert spec['revision_rule_sha256'] == rule['protocol_sha256']
    assert spec['end_at'] == authority.END
