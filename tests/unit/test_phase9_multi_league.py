"""Real per-league evidence plus cross-league failure injection in private stores."""
import copy
import json

import numpy as np
import pytest

from src.config.leagues import ACTIVE_LEAGUES
from src.evaluation.multi_league_integrity import ROOT, read, active_bundle
from src.ml.league_integrity import (LeagueModelBundle, verify_prediction, verify_training,
                                   verify_field_authority, derive_state)
from src.predictions.provenance import prediction_fingerprint, record_integrity_hash, sha256_json

RUN = ROOT/'baselines/phase9/phase9-multi-league-20260905-v1'
INVENTORY = read(RUN/'league_inventory.json')
ROWS = {r['league']:r for r in INVENTORY['leagues']}


@pytest.mark.parametrize('league',ACTIVE_LEAGUES)
def test_every_production_league_has_exact_artifacts_and_ordered_schema(league):
    from src.predictions.provenance import sha256_file
    row = ROWS[league]
    names = row['feature_schema']['ordered_feature_names']
    assert len(names) == len(set(names)) > 0
    assert not any(t in n.lower() for n in names for t in ('odds','bookmaker','implied_probability'))
    for member in row['active_goal_models'].values():
        assert sha256_file(ROOT/member['path']) == member['sha256']
        assert member['ordered_feature_names'] == names
        assert member['schema_sha256'] == sha256_json(names)


@pytest.mark.parametrize('league',ACTIVE_LEAGUES)
def test_each_production_league_deterministic_provenance_replays(league):
    from src.ml.bundle_evidence import verify_historical_record
    path = RUN/league/'final_active_prediction_provenance.json'
    if league == 'PL':
        path = RUN/league/'prediction_provenance.json'
    record = read(path)
    result = verify_prediction(record,bundle=active_bundle(ROWS[league]),root=ROOT,verify_referenced_artifacts=False)
    assert result['verified'],result
    assert verify_historical_record(record,root=ROOT,replay=True)['replay']['reproduced']
    for call in record['feature_provenance']['model_inputs']:
        assert np.isfinite(call['feature_values']).all()
        assert call['model_calibrator'] == {'applied':False,'declared':False,'reference':None}
    assert record['data_provenance']['temporal_cutoff_verified']


def test_league_identity_participates_in_provenance_fingerprint():
    record = read(RUN/'BL1/prediction_provenance.json')
    changed = copy.deepcopy(record)
    changed['fixture']['league'] = 'SA'
    assert prediction_fingerprint(changed) != prediction_fingerprint(record)


def test_provenance_cannot_validate_another_league_even_with_resealed_fingerprint():
    record = read(RUN/'BL1/prediction_provenance.json')
    result = verify_prediction(record,bundle=active_bundle(ROWS['SA']),root=ROOT,verify_referenced_artifacts=False)
    assert not result['verified']
    assert 'prediction league mismatch' in result['errors']


def test_lineage_cannot_cross_leagues():
    core = read(RUN/'BL1/lineage/home_goals_lgbm.json')
    result = verify_training(core,bundle=active_bundle(ROWS['SA']),role='home_goals_lgbm',root=ROOT)
    assert not result['assessment_verified']
    assert 'training lineage league mismatch' in result['errors']


def test_candidate_cannot_be_attached_to_wrong_league():
    value = read(RUN/'BL1/candidate/candidate_bundle.json')
    with pytest.raises(ValueError,match='league mismatch'):
        LeagueModelBundle.from_dict(value,expected_league='SA')


def test_feature_schema_substitution_fails_even_with_resealed_bundle():
    value = read(RUN/'BL1/candidate/candidate_bundle.json')
    value['members']['home_goals_lgbm']['schema_sha256'] = sha256_json(['wrong_feature'])
    value['bundle_sha256'] = sha256_json({k:v for k,v in value.items() if k != 'bundle_sha256'})
    with pytest.raises(ValueError,match='feature schema'):
        LeagueModelBundle.from_dict(value,expected_league='BL1')


@pytest.mark.parametrize('league',[x for x in ACTIVE_LEAGUES if x != 'PL'])
def test_candidate_lineage_partitions_reproduction_and_paired_evaluation(league):
    from src.evaluation.league_integrity_verifier import verify_candidate,verify_evaluation
    bundle,checks = verify_candidate(RUN/league,ROWS[league])
    assert all(c['lineage_verified'] and c['repeat_predictions_reproduced'] for c in checks.values())
    assert verify_evaluation(RUN/league,ROWS[league],bundle,replay=False)['metrics_verified']


def test_provider_field_authority_is_league_scoped():
    authority = {'league':'BL1','source':'football-data.co.uk:csv','fields':['home_corners'],'overlap_rows':20,'mismatches':0}
    assert verify_field_authority(authority,expected_league='BL1',source=authority['source'],field='home_corners')
    with pytest.raises(ValueError,match='league mismatch'):
        verify_field_authority(authority,expected_league='PL',source=authority['source'],field='home_corners')
    authority['mismatches'] = 1
    with pytest.raises(ValueError,match='exact historical overlap'):
        verify_field_authority(authority,expected_league='BL1',source=authority['source'],field='home_corners')


def test_inventory_contains_complete_production_configuration():
    assert set(INVENTORY['production_leagues']) == set(ACTIVE_LEAGUES)
    assert {r['league'] for r in INVENTORY['leagues'] if r['production_enabled']} == set(ACTIVE_LEAGUES)
    assert set(INVENTORY['discovery_sources']['feature_pipeline']) == set(ACTIVE_LEAGUES)


def test_batch_verifier_reports_each_failure_instead_of_skipping(tmp_path):
    from src.evaluation.league_integrity_verifier import verify_run
    for filename in ('league_inventory.json','multi_league_integrity_matrix.json'):
        (tmp_path/filename).write_bytes((RUN/filename).read_bytes())
    (tmp_path/'protected_before.json').write_text('{}')
    result = verify_run(tmp_path,require_index=False,replay=False)
    assert not result['verified']
    assert set(result['league_results']) == set(ACTIVE_LEAGUES)
    assert all(not r['verified'] for r in result['league_results'].values())


def test_batch_audit_records_failures_and_continues_all_leagues(tmp_path,monkeypatch):
    import src.evaluation.multi_league_integrity as audit
    (tmp_path/'league_inventory.json').write_bytes((RUN/'league_inventory.json').read_bytes())
    inspected = []
    def fail(directory,row):
        inspected.append(row['league'])
        raise RuntimeError('isolated missing artifact')
    monkeypatch.setattr(audit,'audit_league',fail)
    result = audit.assess(tmp_path)
    assert inspected[0] == 'BL1'
    assert set(inspected) == set(ACTIVE_LEAGUES)
    assert all(r['execution'] == 'EXECUTION_PROVENANCE_BLOCKED' for r in result['reports'])


def test_state_cannot_claim_promotion_without_its_own_prerequisites():
    assert derive_state(execution=True,active_lineage=False,candidate=True,shadow=True,data_ready=False) == 'SHADOW_EVALUATION_READY'
    with pytest.raises(ValueError,match='prerequisites'):
        derive_state(execution=True,active_lineage=False,candidate=True,shadow=True,data_ready=False,promotion=True)


@pytest.fixture(scope='module')
def deployment_material():
    from src.ml.league_bundle_activation import LeagueBundleController
    output = {}
    for league in ('BL1','SA'):
        controller = LeagueBundleController(RUN/league/'deployment_specification.json',league=league)
        bundles = {b:controller.bundle(b) for b in (controller.legacy_id,controller.candidate_id)}
        smokes = {b:controller.smoke(value) for b,value in bundles.items()}
        output[league] = (bundles,smokes)
    return output


@pytest.fixture
def isolated_leagues(tmp_path,monkeypatch,deployment_material):
    from src.ml.league_bundle_activation import LeagueBundleController
    output = {}
    for league in ('BL1','SA'):
        c = LeagueBundleController(RUN/league/'deployment_specification.json',league=league)
        c.pointer = tmp_path/league/'active.json'
        c.history = tmp_path/league/'history'
        c.lock = tmp_path/league/'lock'
        c.pending = tmp_path/league/'pending.json'
        bundles,smokes = deployment_material[league]
        monkeypatch.setattr(c,'bundle',lambda name,bundles=bundles:copy.deepcopy(bundles[name]))
        def smoke(bundle,state=None,smokes=smokes):
            record = copy.deepcopy(smokes[bundle['bundle_id']])
            if state is not None:
                for member in record['model_provenance']['members']:
                    member['active_pointer_fingerprint'] = state['activation_fingerprint']
                record['prediction_fingerprint'] = prediction_fingerprint(record)
                record['record_integrity_sha256'] = record_integrity_hash(record)
            return record
        monkeypatch.setattr(c,'smoke',smoke)
        output[league] = c
    return output


def authorize_test_boundary(monkeypatch):
    import src.ml.bundle_activation as p8
    def gate(evidence,*,controller):
        return {'promotion_eligible':True,'classification':p8.SUPPORTED,'league':controller.league,
                'candidate_bundle_id':controller.candidate_id,
                'specification_fingerprint':controller.frozen['evaluation_specification']['specification_sha256'],
                'final_decision_sha256':'test-boundary-only'}
    monkeypatch.setattr(p8,'promotion_gate',gate)


def test_phase8_cannot_activate_bl1_using_pl_evidence(isolated_leagues):
    import src.ml.bundle_activation as p8
    c = isolated_leagues['BL1']
    with pytest.raises(p8.ActivationError,match='league mismatch'):
        c.activate(ROOT/p8.PHASE7,expected_active=c.legacy_id,operator='isolated test')
    assert not c.pointer.exists() and not c.pending.exists()


def test_phase8_own_historical_evidence_still_blocks_activation(isolated_leagues):
    import src.ml.bundle_activation as p8
    c = isolated_leagues['BL1']
    with pytest.raises(p8.ActivationError,match='PROMOTION_NOT_ELIGIBLE'):
        c.activate(RUN/'BL1',expected_active=c.legacy_id,operator='isolated test')
    assert not c.pointer.exists()


def test_phase8_cross_league_evaluation_rejected(isolated_leagues):
    import src.ml.bundle_activation as p8
    with pytest.raises(p8.ActivationError,match='league mismatch'):
        isolated_leagues['SA'].league_promotion_gate(RUN/'BL1')


def test_independent_activation_and_rollback_state(isolated_leagues,monkeypatch):
    authorize_test_boundary(monkeypatch)
    bl,sa = isolated_leagues['BL1'],isolated_leagues['SA']
    bl.activate(RUN/'BL1',expected_active=bl.legacy_id,operator='isolated test')
    assert not sa.pointer.exists()
    sa.activate(RUN/'SA',expected_active=sa.legacy_id,operator='isolated test')
    sa_before = sa.pointer.read_bytes()
    bl.rollback(expected_active=bl.candidate_id,operator='isolated test',reason='independence test')
    assert bl.current()['bundle_id'] == bl.legacy_id
    assert sa.pointer.read_bytes() == sa_before
    assert sa.current()['bundle_id'] == sa.candidate_id


def test_wrong_league_pointer_fails_closed(isolated_leagues,monkeypatch):
    import src.ml.bundle_activation as p8
    authorize_test_boundary(monkeypatch)
    bl = isolated_leagues['BL1']
    bl.activate(RUN/'BL1',expected_active=bl.legacy_id,operator='isolated test')
    state = read(bl.pointer)
    state['league'] = 'SA'
    bl.pointer.write_text(json.dumps(p8.seal(state)))
    with pytest.raises(p8.ActivationError,match='league mismatch'):
        bl.current()


def test_league_post_replace_failure_recovers_only_its_previous_bundle(isolated_leagues,monkeypatch):
    import src.ml.bundle_activation as p8
    authorize_test_boundary(monkeypatch)
    bl,sa = isolated_leagues['BL1'],isolated_leagues['SA']
    original = bl.smoke
    def fail_candidate(bundle,state=None):
        if state is not None and bundle['bundle_id'] == bl.candidate_id:
            raise RuntimeError('injected post-replace failure')
        return original(bundle,state)
    monkeypatch.setattr(bl,'smoke',fail_candidate)
    with pytest.raises(p8.ActivationError,match='previous verified bundle restored'):
        bl.activate(RUN/'BL1',expected_active=bl.legacy_id,operator='isolated test')
    assert bl.current()['bundle_id'] == bl.legacy_id
    assert not bl.pending.exists() and not sa.pointer.exists()


@pytest.mark.parametrize('league',[x for x in ACTIVE_LEAGUES if x != 'PL'])
def test_unqualified_leagues_cannot_start_prospective_collection(league):
    from src.ml.league_bundle_activation import LeagueBundleController
    from src.evaluation.league_prospective import verify_protocol
    from src.ml.bundle_activation import ActivationError
    controller = LeagueBundleController(RUN/league/'deployment_specification.json',league=league)
    with pytest.raises(ActivationError,match='not frozen'):
        verify_protocol(controller)


def test_prospective_pair_cannot_be_reconstructed_after_kickoff():
    from types import SimpleNamespace
    from src.evaluation.league_prospective import validate_pair
    from src.ml.bundle_activation import ActivationError
    protocol = {'protocol_sha256':'frozen','frozen_at':'2026-01-01T00:00:00Z'}
    pair = {'league':'BL1','protocol_sha256':'frozen','created_at':'2026-02-01T16:00:00Z',
            'kickoff_at':'2026-02-01T15:00:00Z'}
    pair['pair_sha256'] = sha256_json(pair)
    with pytest.raises(ActivationError,match='retrospective reconstruction'):
        validate_pair(pair,controller=SimpleNamespace(league='BL1'),protocol=protocol)


def test_prospective_settlement_without_prediction_cannot_be_added():
    from src.evaluation.league_prospective import append_settlement
    from src.ml.league_bundle_activation import LeagueBundleController
    from src.ml.bundle_activation import ActivationError
    controller = LeagueBundleController(RUN/'BL1/deployment_specification.json',league='BL1')
    with pytest.raises(ActivationError,match='not frozen'):
        append_settlement(controller=controller,fixture_id='invented',home_goals=1,away_goals=0,result_source={})
