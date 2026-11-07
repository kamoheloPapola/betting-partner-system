"""Phase 8 failure injection uses private temporary state, never live pointers."""
import copy
import json
from pathlib import Path

import pytest

import src.ml.bundle_activation as p8


def bound_smoke(smoke, state=None):
    from src.predictions.provenance import prediction_fingerprint, record_integrity_hash
    value = copy.deepcopy(smoke)
    if state is not None:
        for member in value['model_provenance']['members']:
            member['active_pointer_fingerprint'] = state['activation_fingerprint']
        value['prediction_fingerprint'] = prediction_fingerprint(value)
        value['record_integrity_sha256'] = record_integrity_hash(value)
    return value


@pytest.fixture(scope='module')
def real_bundles():
    controller = p8.BundleController()
    return {name: controller.bundle(name) for name in (p8.LEGACY_ID, p8.CANDIDATE_ID)}


@pytest.fixture(scope='module')
def smokes(real_bundles):
    controller = p8.BundleController()
    return {name: controller.smoke(bundle) for name, bundle in real_bundles.items()}


@pytest.fixture
def isolated(tmp_path, monkeypatch, real_bundles, smokes):
    controller = p8.BundleController()
    controller.pointer = tmp_path / 'active_pl_bundle.json'
    controller.history = tmp_path / 'history'
    controller.lock = tmp_path / 'lock'
    controller.pending = tmp_path / 'pending.json'
    monkeypatch.setattr(controller, 'bundle', lambda name: copy.deepcopy(real_bundles[name]))
    monkeypatch.setattr(controller, 'smoke', lambda bundle, state=None: bound_smoke(smokes[bundle['bundle_id']], state))
    return controller


@pytest.fixture
def authorized(monkeypatch):
    """Only the transition unit boundary is stubbed; no production bypass exists."""
    gate = {'promotion_eligible': True, 'classification': p8.SUPPORTED, 'status': 'PROMOTION_ELIGIBLE'}
    monkeypatch.setattr(p8, 'promotion_gate', lambda *args, **kwargs: gate)


def activate(controller):
    return controller.activate(p8.ROOT / p8.PHASE7, expected_active=p8.LEGACY_ID, operator='isolated unit test')


def test_current_inconclusive_evidence_blocks_activation(isolated):
    with pytest.raises(p8.ActivationError, match='PROMOTION_NOT_ELIGIBLE'):
        activate(isolated)
    assert not isolated.pointer.exists()
    assert not isolated.pending.exists()
    assert not isolated.history.exists()


def test_real_candidate_hashes_and_training_lineage(real_bundles):
    bundle = real_bundles[p8.CANDIDATE_ID]
    assert bundle['aggregate_sha256'] == p8.CANDIDATE_HASH
    assert set(bundle['members']) == set(p8.ROLES)
    assert all(item['classification'] == 'TRAINING_LINEAGE_VERIFIED' for item in bundle['lineage'].values())


def test_dry_run_never_mutates_pointer_or_manifest(isolated):
    before = p8.sha256_file(isolated.manifest)
    dry = isolated.dry_run(p8.ROOT / p8.PHASE7)
    assert dry['classification'] == 'PHASE8_READY_ACTIVATION_BLOCKED_BY_PHASE7'
    assert dry['preflight']['status'] == 'ACTIVATION_PREFLIGHT_VERIFIED'
    assert not dry['eligibility']['promotion_eligible']
    assert not isolated.pointer.exists() and not isolated.pending.exists()
    assert before == p8.sha256_file(isolated.manifest)


def test_dry_run_does_not_activate_even_when_eligible(isolated, authorized):
    dry = isolated.dry_run(p8.ROOT / p8.PHASE7)
    assert dry['eligibility']['promotion_eligible']
    assert not isolated.pointer.exists()


def test_wrong_candidate_rejected():
    with pytest.raises(p8.ActivationError, match='wrong candidate'):
        p8.BundleController().bundle('wrong')


@pytest.mark.parametrize('mutation,match', [
    ('member', 'aggregate mismatch'), ('partial', 'aggregate mismatch'),
    ('lineage', 'lineage assessment failed'), ('schema', 'feature-schema mismatch'),
    ('calibrator', 'calibrator insertion'), ('member_bytes', 'member hash mismatch'),
])
def test_preflight_integrity_failures(monkeypatch, mutation, match):
    original = p8.read_json

    def modified(path):
        value = original(path)
        if path.name == 'candidate_bundle.json':
            if mutation == 'member':
                value['models'][0]['artifact']['sha256'] = '0' * 64
            if mutation == 'partial':
                value['models'].pop()
        if path.name == 'manifest.json':
            if mutation == 'calibrator':
                value['home_goals_v1.0.0_PL']['calibrator_filename'] = 'evil.json'
            if mutation == 'schema':
                value['home_goals_v1.0.0_PL']['features'] = ['wrong'] * 30
        if path.name == 'prediction_provenance.json' and 'phase4' in str(path) and mutation == 'member_bytes':
            value['model_provenance']['members'][0]['artifact_sha256'] = '0' * 64
        return value

    monkeypatch.setattr(p8, 'read_json', modified)
    if mutation == 'lineage':
        import src.ml.phase6_controlled_training as p6
        monkeypatch.setattr(p6, 'verify_candidate_assessment', lambda *a, **k: {'verified': False})
    with pytest.raises(p8.ActivationError, match=match):
        p8.BundleController().bundle(p8.LEGACY_ID if mutation in ('schema', 'calibrator', 'member_bytes') else p8.CANDIDATE_ID)


def test_unexpected_active_state_blocks_mutation(isolated, authorized):
    with pytest.raises(p8.ActivationError, match='unexpected current'):
        isolated.preflight('unexpected')
    assert not isolated.pointer.exists()


def test_atomic_write_failure_preserves_exact_old_bytes(tmp_path, monkeypatch):
    path = tmp_path / 'pointer.json'
    path.write_bytes(b'old bundle')
    def failure(*args):
        raise OSError('injected atomic replace failure')
    monkeypatch.setattr(p8.os, 'replace', failure)
    with pytest.raises(OSError):
        p8.atomic_write(path, b'new bundle')
    assert path.read_bytes() == b'old bundle'
    assert list(tmp_path.iterdir()) == [path]


def test_activation_and_rollback_restore_all_four(isolated, authorized, real_bundles):
    activation = activate(isolated)
    assert isolated.current()['bundle_id'] == p8.CANDIDATE_ID
    assert activation['loaded_model_hashes'] == {r: m['sha256'] for r, m in real_bundles[p8.CANDIDATE_ID]['members'].items()}
    rollback = isolated.rollback(expected_active=p8.CANDIDATE_ID, operator='test', reason='technical rollback rehearsal')
    assert isolated.current()['bundle_id'] == p8.LEGACY_ID
    assert rollback['loaded_model_hashes'] == {r: m['sha256'] for r, m in real_bundles[p8.LEGACY_ID]['members'].items()}
    assert p8.verify_activation_record(activation)['status'] == 'VERIFIED'
    assert p8.verify_activation_record(rollback)['status'] == 'VERIFIED'
    assert rollback['authorization']['activation_record_fingerprint'] == activation['activation_fingerprint']
    assert len(list(isolated.history.glob('*.json'))) == 4


def test_rollback_target_must_verify_before_pointer_write(isolated, authorized, monkeypatch):
    def failed_smoke(bundle, state=None):
        if bundle['bundle_id'] == p8.LEGACY_ID:
            raise p8.ActivationError('rollback smoke failure')
        return {}
    monkeypatch.setattr(isolated, 'smoke', failed_smoke)
    with pytest.raises(p8.ActivationError, match='rollback smoke'):
        activate(isolated)
    assert not isolated.pointer.exists()


def test_post_switch_failure_automatically_rolls_back(isolated, authorized, monkeypatch, smokes):
    def smoke(bundle, state=None):
        if state is not None and bundle['bundle_id'] == p8.CANDIDATE_ID:
            raise p8.ActivationError('injected post-activation failure')
        return bound_smoke(smokes[bundle['bundle_id']], state)
    monkeypatch.setattr(isolated, 'smoke', smoke)
    with pytest.raises(p8.ActivationError, match='previous verified bundle restored'):
        activate(isolated)
    assert isolated.current()['bundle_id'] == p8.LEGACY_ID
    assert not isolated.pending.exists()


def test_crash_journal_blocks_serving_and_recovery_restores(isolated, authorized, monkeypatch, smokes):
    def crash(bundle, state=None):
        if state is not None:
            raise KeyboardInterrupt('simulated process termination')
        return copy.deepcopy(smokes[bundle['bundle_id']])
    monkeypatch.setattr(isolated, 'smoke', crash)
    with pytest.raises(KeyboardInterrupt):
        activate(isolated)
    with pytest.raises(p8.ActivationError, match='unfinished transition'):
        isolated.current()
    monkeypatch.setattr(isolated, 'smoke', lambda bundle, state=None: bound_smoke(smokes[bundle['bundle_id']], state))
    record = isolated.recover(operator='test recovery')
    assert record['schema_version'] == p8.ROLLBACK_SCHEMA
    assert isolated.current()['bundle_id'] == p8.LEGACY_ID


def test_missing_pointer_after_activation_cannot_fallback(isolated, authorized):
    activate(isolated)
    isolated.pointer.unlink()
    with pytest.raises(p8.ActivationError, match='pointer missing'):
        isolated.current()


def test_loaded_models_reload_to_new_bytes(isolated, authorized, real_bundles):
    old = isolated.production_suite()
    activate(isolated)
    new = isolated.production_suite()
    for role, slot in p8.SLOTS.items():
        assert old[slot] is not new[slot]
        assert new[slot]._ffps_provenance['artifact_sha256'] == real_bundles[p8.CANDIDATE_ID]['members'][role]['sha256']
        assert old[slot]._ffps_provenance['artifact_sha256'] != new[slot]._ffps_provenance['artifact_sha256']


def test_stale_suite_and_partial_runtime_are_rejected(isolated, authorized, monkeypatch):
    suite = isolated.production_suite()
    # Guard obtains the same isolated controller used by the runtime store.
    monkeypatch.setattr(p8, 'BundleController', lambda *args: isolated)
    with p8.production_bundle_guard(suite):
        pass
    suite['mh_goals']._ffps_provenance['artifact_sha256'] = '0' * 64
    with pytest.raises(p8.ActivationError, match='registry/runtime mismatch'):
        with p8.production_bundle_guard(suite):
            pass
    activate(isolated)
    with pytest.raises(p8.ActivationError, match='stale production suite'):
        with p8.production_bundle_guard(suite):
            pass


@pytest.mark.parametrize('field', ['loaded_model_hashes', 'operator', 'new_bundle', 'authorization'])
def test_activation_record_tampering_detected(isolated, authorized, field):
    record = activate(isolated)
    record[field] = 'tampered'
    with pytest.raises(p8.ActivationError, match='fingerprint'):
        p8.verify_activation_record(record)


def test_rollback_record_tampering_detected(isolated, authorized):
    activate(isolated)
    record = isolated.rollback(expected_active=p8.CANDIDATE_ID, operator='test', reason='incident')
    record['resulting_state']['bundle_id'] = p8.CANDIDATE_ID
    with pytest.raises(p8.ActivationError, match='fingerprint'):
        p8.verify_activation_record(record)


def test_smoke_provenance_identifies_bundles(real_bundles, smokes):
    for name, record in smokes.items():
        p8.verify_smoke(record, real_bundles[name], root=p8.ROOT)
        assert record['prediction_output']['class_order'] == ['home_win', 'draw', 'away_win']
        assert record['prediction_output']['final_probabilities'] == record['prediction_output']['raw_probabilities']


def test_smoke_wrong_bundle_provenance_fails(real_bundles, smokes):
    with pytest.raises(p8.ActivationError, match='loaded hashes mismatch'):
        p8.verify_smoke(smokes[p8.LEGACY_ID], real_bundles[p8.CANDIDATE_ID], root=p8.ROOT)


def test_real_production_loader_uses_same_phase4_legacy_identity():
    from src.cli.commands.prediction import _load_prediction_models
    suite = _load_prediction_models('PL')
    assert all(suite[slot]._ffps_provenance['active_bundle_id'] == p8.LEGACY_ID for slot in p8.SLOTS.values())
    assert all(isinstance(suite[slot], p8.GenerationBoundModel) for slot in p8.SLOTS.values())


def test_records_cannot_be_overwritten(tmp_path):
    path = tmp_path / 'record.json'
    p8.persist_new(path, p8.seal({'value': 1}))
    with pytest.raises(FileExistsError):
        p8.persist_new(path, p8.seal({'value': 2}))
    assert p8.read_json(path)['value'] == 1


def test_no_force_option_or_api_override():
    import inspect
    import subprocess
    import sys
    assert 'force' not in inspect.signature(p8.BundleController.activate).parameters
    result = subprocess.run([sys.executable, str(p8.ROOT / 'scripts/run_phase8_bundle_activation.py'), 'activate', '--force-promote'], capture_output=True)
    assert result.returncode == 2


def test_forged_result_is_rejected_before_preflight(monkeypatch, isolated):
    original = p8.read_json
    def forged(path):
        result = original(path)
        if path.name == 'prospective_result.json':
            result['decision']['classification'] = p8.SUPPORTED
            result['decision']['phase8_promotion_review_authorized'] = True
        return result
    monkeypatch.setattr(p8, 'read_json', forged)
    with pytest.raises(p8.ActivationError, match='forged Phase 7 result'):
        activate(isolated)
    assert not isolated.pointer.exists()


def test_rehashed_support_claim_cannot_bypass_fixed_horizon(monkeypatch, isolated):
    original = p8.read_json
    def forged(path):
        result = original(path)
        if path.name == 'prospective_result.json':
            result['decision'].update(classification=p8.SUPPORTED, phase8_promotion_review_authorized=True)
        elif path.name in ('run_manifest.json', 'offline_verification.json'):
            result['final_classification'] = p8.SUPPORTED
            if path.name == 'run_manifest.json':
                result['phase8_promotion_review_authorized'] = True
        return result
    monkeypatch.setattr(p8, 'read_json', forged)
    with pytest.raises(p8.ActivationError, match='fixed horizon not complete'):
        activate(isolated)


def test_wrong_candidate_evidence_fails(monkeypatch, isolated):
    original = p8.read_json
    def forged(path):
        result = original(path)
        if path.name == 'run_manifest.json':
            result['candidate_bundle_id'] = 'wrong'
        return result
    monkeypatch.setattr(p8, 'read_json', forged)
    with pytest.raises(p8.ActivationError, match='wrong candidate evidence'):
        activate(isolated)


def test_prior_frozen_evidence_unchanged():
    snapshot = p8.read_json(p8.ROOT / 'baselines/phase8/phase8-controlled-pl-bundle-20260905-v1/protected_before.json')
    assert {path: p8.sha256_file(p8.ROOT / path) for path in snapshot} == snapshot


def test_actual_generation_proxy_rejects_stale_prediction(isolated, authorized):
    import pandas as pd
    old = isolated.production_suite()
    features = old['meta_goals']['features']
    frame = pd.DataFrame([[1.] * 30], columns=features)
    old['mh_goals'].predict(frame)
    activate(isolated)
    with pytest.raises(p8.ActivationError, match='stale production suite'):
        old['mh_goals'].predict(frame)
    isolated.production_suite()['mh_goals'].predict(frame)


def test_proxy_rejects_runtime_calibrator(isolated):
    import pandas as pd
    suite = isolated.production_suite()
    suite['mh_goals'].meta['calibrator_filename'] = 'injected'
    frame = pd.DataFrame([[1.] * 30], columns=suite['meta_goals']['features'])
    with pytest.raises(p8.ActivationError, match='metadata/calibrator mismatch'):
        suite['mh_goals'].predict(frame)


def test_actual_transition_smoke_without_inference_stub(isolated, authorized, monkeypatch):
    # Real bytes, ensemble, simulator, capture and replay in a temporary pointer
    # store. Only the scientific evidence gate is simulated for the rehearsal.
    monkeypatch.setattr(isolated, 'smoke', p8.BundleController.smoke.__get__(isolated))
    record = activate(isolated)
    p8.verify_activation_record(record)
    rollback = isolated.rollback(expected_active=p8.CANDIDATE_ID, operator='test', reason='real smoke rehearsal')
    p8.verify_activation_record(rollback)


def test_named_immutable_record_schemas(isolated, authorized, tmp_path):
    activation = p8.ModelBundleActivationRecord.from_dict(activate(isolated))
    path = tmp_path / 'activation.json'
    activation.save(path)
    assert p8.ModelBundleActivationRecord.load(path).payload == activation.payload
    rollback = isolated.rollback(expected_active=p8.CANDIDATE_ID, operator='test', reason='schema rehearsal')
    p8.ModelBundleRollbackRecord.from_dict(rollback)
    with pytest.raises(p8.ActivationError, match='schema mismatch'):
        p8.ModelBundleActivationRecord.from_dict(rollback)


def test_cross_process_lock_blocks_mutation_and_releases(tmp_path):
    import subprocess
    import sys
    path = tmp_path / 'shared.lock'
    code = 'import sys; from pathlib import Path; from src.ml.bundle_activation import state_lock;\nwith state_lock(Path(sys.argv[1])): print("acquired")'
    with p8.state_lock(path):
        busy = subprocess.run([sys.executable, '-c', code, str(path)], capture_output=True)
    free = subprocess.run([sys.executable, '-c', code, str(path)], capture_output=True)
    assert busy.returncode != 0 and b'busy' in busy.stderr
    assert free.returncode == 0


def test_phase7p_frozen_execution_contract_remains_operational():
    from scripts.run_phase7p_prospective import verify_frozen_contract
    result = verify_frozen_contract()
    assert result['verified'], result


@pytest.fixture
def future_evidence(tmp_path, monkeypatch):
    """Test the final-evidence adapter with simulated *unit* verifier outputs.

    Existing Phase 7/7P tests cover actual cohort verification and statistics;
    this fixture exercises envelope hashes, complete exports and decision binding.
    """
    from datetime import datetime, timezone
    import src.evaluation.phase7p_prospective as prospective
    import src.ml.bundle_evidence as evidence_module
    from src.evaluation.phase7_shadow import apply_decision_rule
    spec = p8.BundleController().specification()
    controller = p8.BundleController(tmp_path)
    monkeypatch.setattr(controller, 'specification', lambda: spec)
    class FutureClock(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2027, 1, 3, tzinfo=timezone.utc)
    monkeypatch.setattr(p8, 'datetime', FutureClock)
    metrics = {'pl_1x2': {'active': {'top_label_ece': .1, 'classwise_ece': .1}, 'candidate': {'top_label_ece': .09, 'classwise_ece': .09}}}
    intervals = {name: {'point_delta': -.02, 'lower': -.04, 'upper': -.001} for name in ('log_loss', 'multiclass_brier', 'joint_goal_rmse', 'joint_goal_poisson_deviance')}
    aggregate = {'fixture_count': 1, 'metrics': metrics, 'paired_intervals': {'metrics': intervals}}
    decision = apply_decision_rule(prospective_fixture_count=1, horizon_complete=True, paired_intervals=intervals,
                                  active_reliability=metrics['pl_1x2']['active'], candidate_reliability=metrics['pl_1x2']['candidate'], provenance_verified=True)
    predictions = [{'active_prediction_provenance': {'label': 'active'}, 'candidate_prediction_provenance': {'label': 'candidate'}}]
    settlements = [{'label': 'settlement'}]
    cohort = tmp_path / 'baselines/phase7p/phase7p-prospective-shadow-79110f9a-20260829-v1/cohort'
    p8.persist_new(cohort / 'predictions/a.json', predictions[0])
    p8.persist_new(cohort / 'settlements/a.json', settlements[0])
    monkeypatch.setattr(prospective, 'eligible_pairs', lambda *a, **k: [{}])
    monkeypatch.setattr(prospective, 'aggregate_eligible', lambda *a, **k: aggregate)
    checked = []
    monkeypatch.setattr(evidence_module, 'verify_historical_record', lambda value, **k: checked.append(value['label']))
    run = tmp_path / 'final_phase7'
    run.mkdir()
    files = {'evaluation_specification.json': spec, 'prospective_predictions.json': predictions,
             'prospective_settlements.json': settlements,
             'prospective_result.json': {**aggregate, 'decision': decision, 'status': 'FINALIZED_FIXED_HORIZON'}}
    for name, value in files.items():
        (run / name).write_text(json.dumps(value))
    artifacts = [{'path': str((run / name).relative_to(tmp_path)), 'sha256': p8.sha256_file(run / name), 'size_bytes': (run / name).stat().st_size} for name in files]
    index = {'artifacts': artifacts, 'aggregate_sha256': p8.sha256_json(artifacts)}
    p8.persist_new(run / 'artifact_index.json', index)
    manifest = {'schema_version': 'ffps.phase7_run_manifest.v1', 'status': 'complete', 'finalized_at': '2027-01-02T00:00:00+00:00',
                'specification_fingerprint': spec['specification_fingerprint'], 'candidate_bundle_id': p8.CANDIDATE_ID,
                'candidate_bundle_aggregate_sha256': p8.CANDIDATE_HASH, 'active_bundle_sha256': spec['active']['model_bundle_sha256'],
                'artifact_index_sha256': p8.sha256_file(run / 'artifact_index.json'), 'artifact_aggregate_sha256': index['aggregate_sha256'],
                'final_classification': p8.SUPPORTED, 'phase8_promotion_review_authorized': True, 'prospective_fixture_count': 1}
    p8.persist_new(run / 'run_manifest.json', manifest)
    p8.persist_new(run / 'offline_verification.json', {'status': 'VERIFIED', 'errors': [], 'final_classification': p8.SUPPORTED, 'specification_fingerprint': spec['specification_fingerprint']})
    return controller, run, aggregate, checked, cohort


def test_future_final_evidence_can_be_eligible_without_activation(future_evidence):
    controller, run, _, checked, _ = future_evidence
    gate = p8.promotion_gate(run, controller=controller)
    assert gate['promotion_eligible'] and gate['classification'] == p8.SUPPORTED
    assert checked == ['active', 'candidate']
    assert not controller.pointer.exists()


def test_future_claim_must_match_recomputed_metrics(future_evidence):
    controller, run, aggregate, _, _ = future_evidence
    aggregate['metrics']['pl_1x2']['candidate']['top_label_ece'] = .2
    with pytest.raises(p8.ActivationError, match='forged prospective metrics'):
        p8.promotion_gate(run, controller=controller)


def test_future_export_cannot_cherry_pick_cohort(future_evidence):
    controller, run, _, _, cohort = future_evidence
    p8.persist_new(cohort / 'predictions/omitted.json', {'omitted': True})
    with pytest.raises(p8.ActivationError, match='authoritative prospective cohort'):
        p8.promotion_gate(run, controller=controller)


def test_future_bad_phase4_provenance_blocks_eligibility(future_evidence, monkeypatch):
    import src.ml.bundle_evidence as evidence_module
    controller, run, _, _, _ = future_evidence
    def failure(*args, **kwargs):
        raise p8.ActivationError('historical Phase 4 verification failed')
    monkeypatch.setattr(evidence_module, 'verify_historical_record', failure)
    with pytest.raises(p8.ActivationError, match='Phase 4'):
        p8.promotion_gate(run, controller=controller)


def test_historical_source_archive_keeps_frozen_phase4_verifiable():
    from src.ml.bundle_evidence import verify_historical_record
    path = p8.ROOT / 'baselines/phase4/phase4-pl-1x2-provenance-79110f9a-20260827-v1/prediction_provenance.json'
    result = verify_historical_record(p8.read_json(path), replay=True)
    assert result['overall_phase4_trusted'] and result['replay']['reproduced']
    assert result['archived_source_paths'] == ['src/ml/registry.py']


def test_legacy_registry_mutator_cannot_change_pl_bundle(monkeypatch):
    from src.ml.registry import ModelRegistry
    registry = ModelRegistry()
    original = registry.manifest
    try:
        registry.manifest = copy.deepcopy(p8.read_json(p8.BundleController().manifest))
        registry.manifest['active_models']['home_goals_PL'] = 'candidate_insertion'
        with pytest.raises(p8.ActivationError, match='Phase 8 activation workflow'):
            registry._save_manifest_file()
    finally:
        registry.manifest = original


def test_serving_process_quiescence_blocks_old_python_workers():
    import subprocess
    import sys
    child = subprocess.Popen([sys.executable, '-c', 'print("ready", flush=True); input()'], stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE, text=True, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    try:
        assert child.stdout.readline().strip() == 'ready'
        with pytest.raises(p8.ActivationError, match='serving processes must stop'):
            p8.runtime_quiescence()
    finally:
        child.communicate('\n', timeout=10)


def test_loaded_runtime_version_mismatch_fails(real_bundles):
    bundle = copy.deepcopy(real_bundles[p8.CANDIDATE_ID])
    bundle['members']['home_goals_lgbm']['meta']['sklearn_version'] = '0.0.0'
    with pytest.raises(p8.ActivationError, match='sklearn runtime'):
        p8.BundleController().load(bundle)


def test_nonfinite_runtime_output_fails(isolated, monkeypatch):
    import pandas as pd
    suite = isolated.production_suite()
    monkeypatch.setattr(suite['mh_goals']._model, 'predict', lambda frame: [float('nan')])
    with pytest.raises(p8.ActivationError, match='non-finite production model output'):
        suite['mh_goals'].predict(pd.DataFrame([[1.] * 30], columns=suite['meta_goals']['features']))


def test_failed_transition_replace_keeps_exact_bootstrap_state(isolated, authorized, monkeypatch):
    def failure(*args):
        raise OSError('injected replace failure')
    monkeypatch.setattr(p8, 'atomic_write', failure)
    with pytest.raises(p8.ActivationError, match='transition failed'):
        activate(isolated)
    assert not isolated.pointer.exists() and not isolated.pending.exists()
    assert isolated.current()['bundle_id'] == p8.LEGACY_ID
    records = [p8.read_json(path) for path in isolated.history.glob('*.json') if not path.name.endswith('.intent.json')]
    assert len(records) == 1 and records[0]['status'] == 'ABORTED_UNCHANGED'
