"""Phase 8: one PL bundle reference, verified transitions and crash recovery.

The legacy manifest remains an immutable bootstrap input. Once present,
active_pl_bundle.json exclusively owns PL goal resolution, including rollback.
OS advisory locking covers model inference calls and complete transitions.
An unfinished journal blocks serving until the explicit recovery operation.
"""
from __future__ import annotations

import copy
import io
import json
import os
import sys
import threading
import uuid
from dataclasses import dataclass
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping

from src.predictions.provenance import canonical_bytes, sha256_bytes, sha256_file, sha256_json

ROOT = Path(__file__).resolve().parents[2]
PHASE6 = Path('baselines/phase6/phase6-controlled-pl-goals-79110f9a-20260829-v1')
PHASE7 = Path('baselines/phase7/phase7-active-vs-candidate-shadow-79110f9a-20260829-v2')
CANDIDATE_HASH = '7fc4d5600bde7e96901bbba9896873cbdee3ba6bd7e0189b496aea1fe3583ae2'
CANDIDATE_ID = 'phase6-evaluation-only-pl-goals-7fc4d5600bde7e96'
LEGACY_ID = 'active-production-pl-goals-phase4-frozen'
SUPPORTED = 'CANDIDATE_SUPPORTED_FOR_PROMOTION_REVIEW'
ROLES = ('home_goals_lgbm', 'away_goals_lgbm', 'home_goals_xgb', 'away_goals_xgb')
SLOTS = dict(zip(ROLES, ('mh_goals', 'ma_goals', 'mh_goals_xgb', 'ma_goals_xgb')))
NAMES = dict(zip(ROLES, ('home_goals', 'away_goals', 'home_goals_xgb', 'away_goals_xgb')))
ACTIVATION_SCHEMA = 'ffps.model_bundle_activation_record.v1'
ROLLBACK_SCHEMA = 'ffps.model_bundle_rollback_record.v1'
POINTER_SCHEMA = 'ffps.active_pl_bundle.v1'
_LOCK = threading.RLock()


class ActivationError(RuntimeError):
    pass


def require(condition: Any, message: str) -> None:
    if not condition:
        raise ActivationError(message)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding='utf-8'))


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def runtime_quiescence() -> dict[str, Any]:
    """Require maintenance downtime for the supported serving interpreter.

    Old processes cannot adopt new Python source on disk. Refuse the transition
    while another process uses this venv/base interpreter. Newly launched FFPS
    processes load Phase 8 and contend on the same state lock. No process is killed.
    """
    processes = []
    if os.name == 'nt':
        import ctypes
        from ctypes import wintypes as w
        class Entry(ctypes.Structure):
            _fields_ = [('dwSize', w.DWORD), ('cntUsage', w.DWORD), ('th32ProcessID', w.DWORD),
                        ('th32DefaultHeapID', ctypes.c_size_t), ('th32ModuleID', w.DWORD),
                        ('cntThreads', w.DWORD), ('th32ParentProcessID', w.DWORD),
                        ('pcPriClassBase', w.LONG), ('dwFlags', w.DWORD), ('szExeFile', w.WCHAR * 260)]
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.CreateToolhelp32Snapshot.argtypes = [w.DWORD, w.DWORD]
        kernel.CreateToolhelp32Snapshot.restype = w.HANDLE
        kernel.Process32FirstW.argtypes = kernel.Process32NextW.argtypes = [w.HANDLE, ctypes.POINTER(Entry)]
        kernel.OpenProcess.argtypes = [w.DWORD, w.BOOL, w.DWORD]
        kernel.OpenProcess.restype = w.HANDLE
        kernel.QueryFullProcessImageNameW.argtypes = [w.HANDLE, w.DWORD, w.LPWSTR, ctypes.POINTER(w.DWORD)]
        kernel.CloseHandle.argtypes = [w.HANDLE]
        snapshot = kernel.CreateToolhelp32Snapshot(2, 0)
        require(snapshot != w.HANDLE(-1).value, 'cannot inspect serving process state')
        try:
            entry = Entry()
            entry.dwSize = ctypes.sizeof(entry)
            more = kernel.Process32FirstW(snapshot, ctypes.byref(entry))
            while more:
                name = entry.szExeFile.lower()
                if name in ('python.exe', 'pythonw.exe', 'uvicorn.exe', 'gunicorn.exe'):
                    handle = kernel.OpenProcess(0x1000, False, entry.th32ProcessID)
                    if handle:
                        try:
                            length = w.DWORD(32768)
                            buffer = ctypes.create_unicode_buffer(length.value)
                            require(kernel.QueryFullProcessImageNameW(handle, 0, buffer, ctypes.byref(length)), 'cannot inspect Python executable identity')
                            processes.append({'pid': entry.th32ProcessID, 'parent_pid': entry.th32ParentProcessID, 'executable': buffer.value})
                        finally:
                            kernel.CloseHandle(handle)
                    else:
                        require(ctypes.get_last_error() == 87, 'cannot inspect a running Python process')
                more = kernel.Process32NextW(snapshot, ctypes.byref(entry))
        finally:
            kernel.CloseHandle(snapshot)
    else:
        require(Path('/proc').is_dir(), 'serving-process inspection is unsupported on this host')
        for directory in Path('/proc').iterdir():
            if not directory.name.isdigit():
                continue
            try:
                executable = str((directory / 'exe').resolve(strict=True))
                if 'python' in Path(executable).name.lower():
                    processes.append({'pid': int(directory.name), 'parent_pid': None, 'executable': executable})
            except FileNotFoundError:
                continue
    normalize = lambda value: os.path.normcase(str(Path(value).resolve()))
    interpreters = {normalize(sys.executable), normalize(getattr(sys, '_base_executable', sys.executable))}
    own = {os.getpid()}
    # Windows venv's small launcher stays alive as our parent. It executes no
    # Python code itself; other launcher/worker processes still block promotion.
    for process in processes:
        if process['pid'] == os.getppid() and normalize(process['executable']) == normalize(ROOT / 'venv/Scripts/python.exe'):
            own.add(process['pid'])
    workers = [p for p in processes if p['pid'] not in own and (
        normalize(p['executable']) in interpreters or Path(p['executable']).is_relative_to(ROOT / 'venv'))]
    require(not workers, 'serving processes must stop before activation: ' + ', '.join(str(p['pid']) for p in workers))
    return {'status': 'VERIFIED', 'policy': 'supported FFPS interpreter quiescent; restart serving after transition',
            'other_serving_processes': [], 'operator_process_id': os.getpid()}


def seal(value: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(value))
    result.pop('activation_fingerprint', None)
    result['activation_fingerprint'] = sha256_json(result)
    return result


def verify_seal(value: Mapping[str, Any]) -> None:
    require(value.get('activation_fingerprint') == seal(value)['activation_fingerprint'], 'activation fingerprint mismatch')


def persist_new(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        with temporary.open('xb') as stream:
            stream.write(canonical_bytes(value))
            stream.flush()
            os.fsync(stream.fileno())
        # Hard-link publication is atomic and refuses an existing destination.
        # A crash can leave an unreferenced temp, never a partial evidence record.
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    require(read_json(path) == value, 'immutable record readback mismatch')


def atomic_write(path: Path, payload: bytes) -> None:
    """Same-filesystem replace; failure before replace leaves old bytes intact."""
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        with temporary.open('xb') as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        if os.name != 'nt':
            fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def state_lock(path: Path) -> Iterator[None]:
    """Crash-released OS lock. Contention fails closed; lock file is reusable."""
    with _LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('a+b') as stream:
            stream.seek(0, os.SEEK_END)
            if not stream.tell():
                stream.write(b'0')
                stream.flush()
            stream.seek(0)
            try:
                if os.name == 'nt':
                    import msvcrt
                    msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise ActivationError('PL bundle state is busy; retry the request') from exc
            try:
                yield
            finally:
                stream.seek(0)
                if os.name == 'nt':
                    msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


class BundleController:
    league = 'PL'
    legacy_id = LEGACY_ID
    candidate_id = CANDIDATE_ID
    pointer_schema = POINTER_SCHEMA

    def __init__(self, root: Path = ROOT):
        self.root = root.resolve()
        self.models = self.root / 'src/ml/models'
        self.manifest = self.models / 'manifest.json'
        self.pointer = self.models / 'active_pl_bundle.json'
        self.history = self.models / 'pl_bundle_history'
        self.lock = self.models / 'pl_bundle.lock'
        self.pending = self.models / 'pl_bundle_pending.json'

    def specification(self) -> dict[str, Any]:
        from src.evaluation.phase7_shadow import verify_specification
        from src.evaluation.phase7p_prospective import FROZEN_SPECIFICATION_FINGERPRINT
        spec = read_json(self.root / PHASE7 / 'evaluation_specification.json')
        require(verify_specification(spec)['verified'], 'invalid Phase 7 specification')
        require(spec['specification_fingerprint'] == FROZEN_SPECIFICATION_FINGERPRINT, 'wrong frozen specification')
        return spec

    def bundle(self, bundle_id: str) -> dict[str, Any]:
        spec = self.specification()
        features = spec['feature_contract']['ordered_feature_names']
        require(len(features) == 30 and sha256_json(features) == spec['feature_contract']['ordered_feature_schema_sha256'], 'feature-schema mismatch')
        require(not any(any(t in f.lower() for t in ('odds', 'bookmaker', 'implied_probability')) for f in features), 'bookmaker dependency')
        if bundle_id == LEGACY_ID:
            manifest = read_json(self.manifest)
            record = read_json(self.root / 'baselines/phase4/phase4-pl-1x2-provenance-79110f9a-20260827-v1/prediction_provenance.json')
            members = {}
            for item in record['model_provenance']['members']:
                role = item['role']
                require(manifest['active_models'].get(item['registry_slot']) == item['manifest_key'], 'legacy manifest pointer changed')
                meta = copy.deepcopy(manifest[item['manifest_key']])
                require((self.models / meta['filename']).resolve() == (self.root / item['source_location']).resolve(), 'legacy artifact path changed')
                members[role] = {'path': item['source_location'], 'sha256': item['artifact_sha256'], 'size_bytes': item['size_bytes'], 'meta': meta, 'provenance': copy.deepcopy(item)}
            aggregate = spec['active']['model_bundle_sha256']
            lineage = {'classification': 'LEGACY_TRAINING_LINEAGE_UNPROVEN'}
        else:
            require(bundle_id == CANDIDATE_ID, 'wrong candidate bundle')
            from src.ml.phase6_controlled_training import verify_candidate_assessment
            from src.ml.training_lineage import ModelTrainingLineageRecord, verify_lineage_record
            candidate = read_json(self.root / PHASE6 / 'candidate_bundle.json')
            material = {key: candidate[key] for key in ('training_specification_sha256', 'training_partition_identity', 'validation_partition_identity', 'models')}
            require(candidate['bundle_id'] == bundle_id and candidate['bundle_aggregate_sha256'] == CANDIDATE_HASH == sha256_json(material), 'candidate aggregate mismatch')
            require(len(candidate['models']) == 4 and {m['role'] for m in candidate['models']} == set(ROLES), 'partial candidate bundle')
            scores = read_json(self.root / PHASE6 / 'integration/integration_result.json')['descriptive_guard_calibration_scores']
            members, lineage = {}, {}
            for item in candidate['models']:
                role = item['role']
                require(item['feature_schema_sha256'] == sha256_json(features), 'candidate feature-schema mismatch')
                for label in ('assessment', 'lineage'):
                    ref = item[label]
                    require(sha256_file(self.root / ref['path']) == ref['sha256'], f'{role} {label} hash mismatch')
                assessment = read_json(self.root / item['assessment']['path'])
                require(verify_candidate_assessment(assessment, root=self.root)['verified'], 'training lineage assessment failed')
                core = ModelTrainingLineageRecord.load(self.root / item['lineage']['path'])
                require(verify_lineage_record(core, root=self.root)['lineage_verified'] and core.fingerprint == item['lineage']['fingerprint'], 'training lineage failed')
                require(assessment['role'] == role and assessment['artifact']['artifact_sha256'] == item['artifact']['sha256'], 'lineage artifact binding mismatch')
                lineage[role] = {**item['lineage'], 'assessment': item['assessment'], 'classification': 'TRAINING_LINEAGE_VERIFIED'}
                members[role] = {**item['artifact'], 'meta': {
                    'name': NAMES[role], 'version': bundle_id, 'features': features,
                    'sklearn_version': candidate['runtime_identity']['dependencies']['scikit-learn'],
                    'type': 'lgbm_regressor_poisson' if role.endswith('lgbm') else 'xgb_regressor_count_poisson',
                    'metrics': {'calibration_score': scores[role]},
                }, 'provenance': {
                    'artifact_sha256': item['artifact']['sha256'], 'source_location': item['artifact']['path'], 'size_bytes': item['artifact']['size_bytes'],
                    'manifest_key': f'{bundle_id}:{role}', 'registry_slot': f'ACTIVE_BUNDLE:{role}',
                    'training_lineage': {'classification': 'proven', 'lineage_fingerprint': core.fingerprint, 'lineage_path': item['lineage']['path']},
                }}
            aggregate = CANDIDATE_HASH
        require(set(members) == set(ROLES), 'partial bundle')
        for role, item in members.items():
            payload = (self.root / item['path']).read_bytes()
            require(sha256_bytes(payload) == item['sha256'] and len(payload) == item['size_bytes'], f'{role} member hash mismatch')
            require(item['meta'].get('features') == features, 'feature-schema mismatch')
            require(not item['meta'].get('calibrator_filename'), 'calibrator insertion')
        return {'bundle_id': bundle_id, 'aggregate_sha256': aggregate, 'members': members, 'lineage': lineage, 'features': features}

    def current(self, *, allow_pending: bool = False) -> dict[str, Any]:
        require(allow_pending or not self.pending.exists(), 'unfinished transition; recover before serving')
        if not self.pointer.exists():
            # A deleted pointer after the first transition must never silently fall back.
            for path in self.history.glob('*.json'):
                if path.name.endswith('.intent.json'):
                    continue
                record = read_json(path)
                verify_activation_record(record, root=self.root)
                require(record['status'] == 'ABORTED_UNCHANGED', 'active pointer missing despite transition history')
            return {**self.pointer_scope(), 'bundle_id': self.legacy_id, 'generation': 0, 'activation_fingerprint': None}
        state = read_json(self.pointer)
        verify_seal(state)
        require(state.get('schema_version') == self.pointer_schema, 'invalid pointer schema')
        require(state.get('league', 'PL') == self.league, 'active pointer league mismatch')
        require(state.get('bundle_id') in (self.legacy_id, self.candidate_id), 'unknown active bundle')
        if not allow_pending:
            record = read_json(self.history / (state['transition_id'] + '.json'))
            verify_activation_record(record, root=self.root)
            require(record['resulting_state'] == state and record['status'] == 'VERIFIED', 'pointer/history disagreement')
        return state

    def load(self, bundle: Mapping[str, Any], state: Mapping[str, Any] | None = None) -> dict[str, Any]:
        import joblib
        import pickle
        import sklearn
        import warnings
        from sklearn.exceptions import InconsistentVersionWarning
        suite = {}
        for role in ROLES:
            item = bundle['members'][role]
            path = self.root / item['path']
            payload = path.read_bytes()
            require(sha256_bytes(payload) == item['sha256'], 'loaded bytes mismatch')
            require(not item['meta'].get('sklearn_version') or item['meta']['sklearn_version'] == sklearn.__version__, 'incompatible sklearn runtime')
            with warnings.catch_warnings():
                warnings.simplefilter('error', InconsistentVersionWarning)
                model = joblib.load(io.BytesIO(payload)) if path.suffix == '.joblib' else pickle.load(io.BytesIO(payload))
            require(not getattr(model, '_ffps_calibrator', None), 'embedded calibrator insertion')
            model.meta = copy.deepcopy(item['meta'])
            model._ffps_provenance = copy.deepcopy(item['provenance'])
            model._ffps_provenance.update(active_bundle_id=bundle['bundle_id'], bundle_aggregate_sha256=bundle['aggregate_sha256'])
            model._ffps_provenance['bundle_loader_sha256'] = sha256_file(Path(__file__))
            if state is not None:
                model._ffps_provenance['active_pointer_fingerprint'] = state.get('activation_fingerprint')
            suite[SLOTS[role]] = GenerationBoundModel(model, self, dict(state), bundle, role) if state is not None else model
        suite['meta_goals'] = suite['mh_goals'].meta
        suite['meta_goals_xgb'] = suite['mh_goals_xgb'].meta
        if state is not None:
            suite['_ffps_bundle_state'] = dict(state)
            suite['_ffps_bundle_root'] = str(self.root)
        return suite

    def production_suite(self) -> dict[str, Any]:
        with state_lock(self.lock):
            state = self.current()
            return self.load(self.bundle(state['bundle_id']), state)

    def smoke(self, bundle: Mapping[str, Any], state: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Approved frozen technical fixture; no data refresh or performance scoring."""
        import pandas as pd
        from src.cli.commands.prediction import _calc_goals
        from src.predictions.provenance import PredictionCapture, mark_output_reproduced, replay_pl_1x2, verify_record
        from src.simulation.match_simulator import MatchSimulator, DEFAULT_N_SIMULATIONS
        frozen = read_json(self.root / PHASE6 / 'integration/prediction_provenance.json')
        inputs = frozen['feature_provenance']['model_inputs'][0]
        fixture = pd.Series(dict(zip(inputs['ordered_feature_names'], inputs['feature_values'])))
        identity = frozen['fixture']
        for key in ('home_team', 'away_team', 'match_hash'):
            fixture[key] = identity[key]
        fixture['date'] = identity['kickoff_identity']
        fixture['league'] = 'PL'
        simulator = MatchSimulator(n_simulations=DEFAULT_N_SIMULATIONS, seed=42, league='PL', seed_scope='phase8_deployment_smoke')
        # This isolated smoke runs while the transition owns the state lock.
        suite = self.load(bundle)
        if state is not None:
            for slot in SLOTS.values():
                suite[slot]._ffps_provenance['active_pointer_fingerprint'] = state['activation_fingerprint']
        # Transition owns the OS lock. The isolated invocation cannot recurse into it.
        suite.pop('_ffps_bundle_state', None)
        with PredictionCapture(prediction_timestamp=identity['prediction_timestamp'], data_provenance=frozen['data_provenance']) as capture:
            probabilities = _calc_goals(fixture, suite, 'Phase8Smoke', simulator, league='PL', rl_weights=None, use_simulator=True)
            record = capture.finalize(fixture=fixture, league='PL', probabilities=probabilities, simulator=simulator)
        record = mark_output_reproduced(record, replay_pl_1x2(record, root=self.root))
        verified = verify_record(record, root=self.root, verify_referenced_artifacts=True)
        require(verified.get('overall_phase4_trusted'), f'smoke Phase 4 verification failed: {verified}')
        result = record.to_dict()
        verify_smoke(result, bundle, root=self.root)
        return result

    def preflight(self, expected_active: str) -> dict[str, Any]:
        runtime = runtime_quiescence() if self.pointer == self.models / f'active_{self.league.lower()}_bundle.json' else {'status': 'ISOLATED_STATE_STORE'}
        state = self.current()
        require(state['bundle_id'] == expected_active, 'unexpected current active bundle')
        rollback = self.bundle(state['bundle_id'])
        require(state['generation'] == 0 or state.get('aggregate_sha256') == rollback['aggregate_sha256'], 'active pointer aggregate mismatch')
        candidate = self.bundle(self.candidate_id)
        return {'status': 'ACTIVATION_PREFLIGHT_VERIFIED', 'previous_state': state,
                'candidate': candidate, 'rollback_target': rollback,
                'candidate_smoke': self.smoke(candidate), 'rollback_smoke': self.smoke(rollback),
                'manifest_sha256': sha256_file(self.manifest), 'runtime_quiescence': runtime}

    def dry_run(self, evidence: Path, expected_active: str = LEGACY_ID) -> dict[str, Any]:
        before = self.pointer.read_bytes() if self.pointer.exists() else None
        manifest_before = sha256_file(self.manifest)
        with state_lock(self.lock):
            gate = promotion_gate(evidence, controller=self)
            preflight = self.preflight(expected_active)
        require(before == (self.pointer.read_bytes() if self.pointer.exists() else None), 'dry-run mutated pointer')
        require(manifest_before == sha256_file(self.manifest), 'dry-run mutated manifest')
        return seal({'schema_version': 'ffps.phase8_dry_run.v1', 'timestamp': now(),
                     'classification': 'PHASE8_READY_ACTIVATION_BLOCKED_BY_PHASE7' if not gate['promotion_eligible'] else 'PHASE8_READY_FOR_AUTHORIZED_ACTIVATION',
                     'eligibility': gate, 'preflight': preflight,
                     'manifest_before_sha256': manifest_before, 'manifest_after_sha256': sha256_file(self.manifest),
                     'pointer_before': before.decode() if before else None, 'pointer_after': before.decode() if before else None,
                     'production_activation_performed': False})

    def activate(self, evidence: Path, *, expected_active: str, operator: str) -> dict[str, Any]:
        require(bool(operator.strip()), 'operator required')
        with state_lock(self.lock):
            gate = promotion_gate(evidence, controller=self)
            require(gate['promotion_eligible'], 'PROMOTION_NOT_ELIGIBLE: ' + gate['classification'])
            require(expected_active == self.legacy_id, 'candidate already active')
            preflight = self.preflight(expected_active)
            return self._transition(preflight['candidate'], preflight['rollback_target'], preflight['previous_state'],
                                    schema=ACTIVATION_SCHEMA, authorization=gate, preflight=preflight, operator=operator)

    def rollback(self, *, expected_active: str, operator: str, reason: str) -> dict[str, Any]:
        require(bool(operator.strip()) and bool(reason.strip()), 'operator and rollback reason required')
        with state_lock(self.lock):
            if self.pointer == self.models / f'active_{self.league.lower()}_bundle.json':
                runtime_quiescence()
            state = self.current()
            require(state['bundle_id'] == expected_active, 'unexpected current active bundle')
            require(state['bundle_id'] == self.candidate_id, 'no candidate activation to roll back')
            activation = read_json(self.history / (state['transition_id'] + '.json'))
            verify_activation_record(activation, root=self.root)
            previous = self.bundle(activation['previous_bundle']['bundle_id'])
            require(previous == activation['previous_bundle'], 'rollback target drift')
            # Do not require broken current candidate bytes to load during an incident.
            preflight = {'status': 'ACTIVATION_PREFLIGHT_VERIFIED', 'rollback_smoke': self.smoke(previous)}
            return self._transition(previous, activation['new_bundle'], state, schema=ROLLBACK_SCHEMA,
                                    authorization={'activation_record_fingerprint': activation['activation_fingerprint'], 'reason': reason},
                                    preflight=preflight, operator=operator)

    def _transition(self, target: dict[str, Any], previous: dict[str, Any], state: dict[str, Any], *, schema: str,
                    authorization: dict[str, Any], preflight: dict[str, Any], operator: str) -> dict[str, Any]:
        require(self.current() == state, 'active pre-state changed')
        transition_id = str(uuid.uuid4())
        pointer = seal({'schema_version': self.pointer_schema, **self.pointer_scope(), 'bundle_id': target['bundle_id'], 'aggregate_sha256': target['aggregate_sha256'],
                        'generation': state['generation'] + 1, 'transition_id': transition_id, 'previous_pointer_fingerprint': state.get('activation_fingerprint')})
        original = self.pointer.read_bytes() if self.pointer.exists() else None
        intent = seal({'schema_version': 'ffps.phase8_transition_intent.v1', 'transition_id': transition_id, 'timestamp': now(),
                       'record_schema': schema, 'previous_state': state, 'requested_state': pointer,
                       'original_pointer': original.decode() if original else None,
                       'previous_bundle': previous, 'new_bundle': target, 'authorization': authorization,
                       'preflight': preflight, 'operator': operator, 'manifest_before_sha256': sha256_file(self.manifest)})
        persist_new(self.history / (transition_id + '.intent.json'), intent)
        persist_new(self.pending, intent)
        try:
            atomic_write(self.pointer, canonical_bytes(pointer))
            require(self.current(allow_pending=True) == pointer, 'registry/runtime mismatch')
            reloaded = self.bundle(pointer['bundle_id'])
            require(reloaded == target, 'post-activation artifact mismatch')
            smoke = self.smoke(reloaded, pointer)
            record = self._complete(intent, pointer, smoke, 'VERIFIED')
        except Exception as exc:
            # Always restore a verified previous bundle. Failure here leaves the
            # pending journal and blocks production until explicit recovery.
            self._recover_locked(reason=f'{type(exc).__name__}: {exc}')
            raise ActivationError('transition failed; previous verified bundle restored') from exc
        self.pending.unlink()
        return record

    def _complete(self, intent: dict[str, Any], state: dict[str, Any], smoke: dict[str, Any], status: str) -> dict[str, Any]:
        from src.predictions.provenance import code_runtime_identity
        record = seal({'schema_version': intent['record_schema'], 'activation_id': intent['transition_id'], 'timestamp': now(),
                       'previous_bundle': intent['previous_bundle'], 'new_bundle': intent['new_bundle'],
                       'previous_state': intent['previous_state'], 'resulting_state': state,
                       'authorization': intent['authorization'], 'preflight': intent['preflight'],
                       'operator': intent['operator'], 'authorization_classification': 'EXPLICIT_ADMINISTRATIVE_OPERATION',
                       'manifest_before_sha256': intent['manifest_before_sha256'], 'manifest_after_sha256': sha256_file(self.manifest),
                       'pointer_before_sha256': sha256_bytes(intent['original_pointer'].encode()) if intent['original_pointer'] is not None else None,
                       'pointer_after_sha256': sha256_file(self.pointer) if self.pointer.exists() else None,
                       'loaded_model_hashes': {m['role']: m['artifact_sha256'] for m in smoke['model_provenance']['members']},
                       'smoke_prediction_provenance': smoke, 'source_revision': code_runtime_identity(), 'status': status,
                       'intent_fingerprint': intent['activation_fingerprint']})
        verify_activation_record(record, root=self.root)
        persist_new(self.history / (intent['transition_id'] + '.json'), record)
        verify_activation_record(read_json(self.history / (intent['transition_id'] + '.json')), root=self.root)
        return record

    def _recover_locked(self, *, reason: str) -> dict[str, Any]:
        intent = read_json(self.pending)
        verify_seal(intent)
        if self.pointer.exists():
            on_disk = self.current(allow_pending=True)
            recovery_path = self.history / (on_disk['transition_id'] + '.json')
            if recovery_path.exists():
                recovery_record = read_json(recovery_path)
                if recovery_record.get('authorization', {}).get('failed_intent_fingerprint') == intent['activation_fingerprint']:
                    verify_activation_record(recovery_record, root=self.root)
                    require(recovery_record['resulting_state'] == on_disk, 'recovery state mismatch')
                    self.pending.unlink()
                    return recovery_record
        # Crash after committed evidence but before journal cleanup: commit is complete.
        completed = self.history / (intent['transition_id'] + '.json')
        if completed.exists():
            record = read_json(completed)
            verify_activation_record(record, root=self.root)
            require(self.current(allow_pending=True) == record['resulting_state'], 'committed state mismatch')
            self.pending.unlink()
            return record
        previous = self.bundle(intent['previous_bundle']['bundle_id'])
        require(previous == intent['previous_bundle'], 'recovery target drift')
        actual_pointer = self.pointer.read_text(encoding='utf-8') if self.pointer.exists() else None
        if actual_pointer == intent['original_pointer']:
            # No replace occurred. Preserve the exact prior pointer bytes and
            # record a failed attempt, without inventing an active candidate.
            smoke = self.smoke(previous, intent['previous_state'] if intent['previous_state']['generation'] else None)
            record = self._complete(intent, intent['previous_state'], smoke, 'ABORTED_UNCHANGED')
            self.pending.unlink()
            return record
        recovery_id = str(uuid.uuid4())
        recovered = seal({'schema_version': self.pointer_schema, **self.pointer_scope(), 'bundle_id': previous['bundle_id'], 'aggregate_sha256': previous['aggregate_sha256'],
                          'generation': intent['requested_state']['generation'] + 1, 'transition_id': recovery_id,
                          'previous_pointer_fingerprint': intent['requested_state']['activation_fingerprint']})
        recovery = seal({**intent, 'transition_id': recovery_id, 'record_schema': ROLLBACK_SCHEMA,
                         'original_pointer': actual_pointer,
                         'previous_bundle': intent['new_bundle'], 'new_bundle': previous,
                         'previous_state': intent['requested_state'], 'requested_state': recovered,
                         'authorization': {'failed_intent_fingerprint': intent['activation_fingerprint'], 'reason': reason}})
        persist_new(self.history / (recovery_id + '.intent.json'), recovery)
        self.smoke(previous)
        atomic_write(self.pointer, canonical_bytes(recovered))
        record = self._complete(recovery, recovered, self.smoke(previous, recovered), 'VERIFIED')
        self.pending.unlink()
        return record

    def recover(self, *, operator: str) -> dict[str, Any]:
        require(bool(operator.strip()), 'operator required')
        with state_lock(self.lock):
            if self.pointer == self.models / f'active_{self.league.lower()}_bundle.json':
                runtime_quiescence()
            require(self.pending.exists(), 'no unfinished transition')
            return self._recover_locked(reason='explicit crash recovery by ' + operator)

    def pointer_scope(self) -> dict[str, Any]:
        # Preserve the frozen PL v1 pointer representation exactly.
        return {}


def verify_smoke(record: Mapping[str, Any], bundle: Mapping[str, Any], *, root: Path, historical: bool = False) -> None:
    from src.predictions.provenance import verify_record
    if historical:
        from src.ml.bundle_evidence import verify_historical_record
        require(verify_historical_record(record,root=root).get('overall_phase4_trusted'), 'historical smoke provenance failure')
    else:
        require(verify_record(record, root=root, verify_referenced_artifacts=True).get('overall_phase4_trusted'), 'smoke provenance failure')
    members = record['model_provenance']['members']
    require({m['role']: m['artifact_sha256'] for m in members} == {r: m['sha256'] for r, m in bundle['members'].items()}, 'smoke loaded hashes mismatch')
    require(all(m.get('active_bundle_id') == bundle['bundle_id'] for m in members), 'smoke bundle identity mismatch')
    require(all(c['ordered_feature_names'] == bundle['features'] and not c['model_calibrator']['applied'] for c in record['feature_provenance']['model_inputs']), 'smoke feature/calibrator mismatch')
    method = record['method_provenance']['descriptor']
    require(method['ensemble']['lgbm_weight'] == .6 and method['ensemble']['xgb_weight'] == .4, 'ensemble contract mismatch')
    require(not method['calibrator']['applied'] and not method['post_processing'] and not method['match_outcome_classifier']['used'] and method['rl_weights'] is None, 'unexpected inference path')
    frozen = read_json(root / PHASE6 / 'integration/prediction_provenance.json')['method_provenance']['descriptor']
    for key in ('implementation', 'n_simulations', 'max_goals', 'top_scorelines', 'tempo_sigma', 'league'):
        require(method['simulation'].get(key) == frozen['simulation'].get(key), f'simulator contract changed: {key}')


def verify_activation_record(record: Mapping[str, Any], *, root: Path = ROOT) -> dict[str, Any]:
    verify_seal(record)
    require(record.get('schema_version') in (ACTIVATION_SCHEMA, ROLLBACK_SCHEMA), 'unsupported activation schema')
    require(record.get('status') in ('VERIFIED', 'ABORTED_UNCHANGED'), 'transition not verified')
    aborted = record['status'] == 'ABORTED_UNCHANGED'
    target = record['previous_bundle'] if aborted else record['new_bundle']
    league = target.get('league', 'PL')
    require(record['previous_bundle'].get('league', 'PL') == record['new_bundle'].get('league', 'PL'), 'transition crosses leagues')
    require(record['resulting_state'].get('league', 'PL') == league, 'transition pointer league mismatch')
    if league != 'PL' and record['schema_version'] == ACTIVATION_SCHEMA:
        authorization = record['authorization']
        require(authorization.get('league') == league, 'activation authorization league mismatch')
        require(authorization.get('candidate_bundle_id') == target['bundle_id'], 'activation authorization candidate mismatch')
        require(authorization.get('specification_fingerprint') == target['evaluation_specification_sha256'], 'activation evaluation mismatch')
        require(bool(authorization.get('final_decision_sha256')), 'activation final decision identity absent')
    require(record['resulting_state']['bundle_id'] == target['bundle_id'], 'resulting active state mismatch')
    if aborted:
        require(record['resulting_state'] == record['previous_state'] and record['pointer_before_sha256'] == record['pointer_after_sha256'], 'aborted transition changed state')
    else:
        require(record['resulting_state']['aggregate_sha256'] == target['aggregate_sha256'], 'resulting bundle aggregate mismatch')
        verify_seal(record['resulting_state'])
        require(record['resulting_state']['transition_id'] == record['activation_id'], 'activation ID mismatch')
        require(record['resulting_state']['generation'] == record['previous_state']['generation'] + 1, 'generation mismatch')
        require(record['resulting_state']['previous_pointer_fingerprint'] == record['previous_state'].get('activation_fingerprint'), 'history chain mismatch')
        require(record['pointer_after_sha256'] == sha256_bytes(canonical_bytes(record['resulting_state'])), 'pointer bytes mismatch')
    require(record['manifest_before_sha256'] == record['manifest_after_sha256'], 'legacy manifest mutated')
    require(record['preflight'].get('status') == 'ACTIVATION_PREFLIGHT_VERIFIED', 'preflight not verified')
    if record['schema_version'] == ACTIVATION_SCHEMA:
        require(record['authorization'].get('promotion_eligible') is True and record['authorization'].get('classification') == SUPPORTED, 'activation lacks scientific authority')
    require(record['loaded_model_hashes'] == {r: m['sha256'] for r, m in target['members'].items()}, 'loaded identity mismatch')
    # Historical records remain verifiable even when a later incident damages a
    # model file. Byte existence is checked afresh by deployment/rollback preflight.
    from src.predictions.provenance import verify_record
    smoke = record['smoke_prediction_provenance']
    require(smoke['fixture']['league'] == league, 'activation smoke league mismatch')
    require(verify_record(smoke, root=root, verify_referenced_artifacts=False).get('overall_phase4_trusted'), 'record smoke provenance failure')
    require({m['role']: m['artifact_sha256'] for m in smoke['model_provenance']['members']} == record['loaded_model_hashes'], 'record smoke hash mismatch')
    require(all(m.get('active_bundle_id') == target['bundle_id'] for m in smoke['model_provenance']['members']), 'record smoke bundle mismatch')
    require(all(m.get('active_pointer_fingerprint') == record['resulting_state']['activation_fingerprint'] for m in smoke['model_provenance']['members']), 'record smoke pointer mismatch')
    return {'status': 'VERIFIED', 'activation_fingerprint': record['activation_fingerprint']}


@dataclass(frozen=True)
class ModelBundleActivationRecord:
    """Immutable canonical serialization of a verified v1 activation."""
    payload: bytes

    @classmethod
    def from_dict(cls, value: Mapping[str, Any], *, root: Path = ROOT):
        expected = ROLLBACK_SCHEMA if cls is ModelBundleRollbackRecord else ACTIVATION_SCHEMA
        require(value.get('schema_version') == expected, 'record schema mismatch')
        verify_activation_record(value, root=root)
        return cls(canonical_bytes(value))

    @classmethod
    def load(cls, path: Path, *, root: Path = ROOT):
        return cls.from_dict(read_json(path), root=root)

    def to_dict(self):
        return json.loads(self.payload)

    def save(self, path: Path):
        persist_new(path, self.to_dict())


@dataclass(frozen=True)
class ModelBundleRollbackRecord(ModelBundleActivationRecord):
    """Same transition bindings with an explicit rollback authorization link."""


def promotion_gate(evidence: Path, *, controller: BundleController) -> dict[str, Any]:
    """Resolve existing Phase 7 evidence; independently recompute a future final.

    Future finalized runs use the existing run/index/spec/result layout and add
    indexed prospective_predictions.json and prospective_settlements.json arrays.
    Self-reported approval, offline status, and metrics cannot authorize a switch.
    """
    if controller.league != 'PL':
        return controller.league_promotion_gate(evidence)
    from src.evaluation.phase7_shadow import apply_decision_rule
    from src.evaluation.phase7p_prospective import aggregate_eligible, eligible_pairs
    spec = controller.specification()
    manifest = read_json(evidence / 'run_manifest.json')
    submitted_spec = read_json(evidence / 'evaluation_specification.json')
    require(submitted_spec == spec, 'promotion evidence specification mismatch')
    require(manifest.get('status') == 'complete' and manifest.get('schema_version') == 'ffps.phase7_run_manifest.v1', 'Phase 7 evidence not finalized')
    require(manifest.get('specification_fingerprint') == spec['specification_fingerprint'], 'Phase 7 fingerprint mismatch')
    require(manifest.get('candidate_bundle_id') == CANDIDATE_ID and manifest.get('candidate_bundle_aggregate_sha256') == CANDIDATE_HASH, 'wrong candidate evidence')
    require(manifest.get('active_bundle_sha256') == spec['active']['model_bundle_sha256'], 'wrong active control evidence')
    index = read_json(evidence / 'artifact_index.json')
    require(sha256_file(evidence / 'artifact_index.json') == manifest['artifact_index_sha256'], 'Phase 7 index hash mismatch')
    require(sha256_json(index['artifacts']) == index['aggregate_sha256'] == manifest['artifact_aggregate_sha256'], 'Phase 7 index aggregate mismatch')
    indexed = set()
    for item in index['artifacts']:
        path = (controller.root / item['path']).resolve()
        require(path.is_relative_to(evidence.resolve()), 'evidence index escapes run directory')
        require(sha256_file(path) == item['sha256'] and path.stat().st_size == item['size_bytes'], 'Phase 7 indexed artifact mismatch')
        indexed.add(path)
    for name in ('evaluation_specification.json', 'prospective_result.json'):
        require((evidence / name).resolve() in indexed, 'decision/specification not indexed')
    offline = read_json(evidence / 'offline_verification.json')
    require(offline.get('status') == 'VERIFIED' and not offline.get('errors') and offline.get('specification_fingerprint') == spec['specification_fingerprint'], 'Phase 7 offline verification failed')
    result = read_json(evidence / 'prospective_result.json')
    decision = result['decision']
    require(decision.get('classification') == manifest.get('final_classification') == offline.get('final_classification'), 'forged Phase 7 result')
    require(decision.get('phase8_promotion_review_authorized') == manifest.get('phase8_promotion_review_authorized') and decision.get('human_override_applied') is False, 'human override or authorization mismatch')
    supported = decision['classification'] == SUPPORTED
    if supported:
        end = datetime.fromisoformat(spec['horizon']['end_date']).replace(tzinfo=timezone.utc)
        from datetime import timedelta
        end += timedelta(days=1)
        finalized = datetime.fromisoformat(manifest['finalized_at'].replace('Z', '+00:00'))
        require(datetime.now(timezone.utc) >= end and end <= finalized <= datetime.now(timezone.utc), 'fixed horizon not complete')
        require(result.get('status') == 'FINALIZED_FIXED_HORIZON', 'prospective evidence is not final')
        for name in ('prospective_predictions.json', 'prospective_settlements.json'):
            require((evidence / name).resolve() in indexed, 'final prospective records not indexed')
        predictions = read_json(evidence / 'prospective_predictions.json')
        settlements = read_json(evidence / 'prospective_settlements.json')
        # The finalized export must contain the entire immutable collection,
        # not a favorable subset selected from already settled fixtures.
        cohort = controller.root / 'baselines/phase7p/phase7p-prospective-shadow-79110f9a-20260829-v1/cohort'
        for kind, exported in (('predictions', predictions), ('settlements', settlements)):
            stored = [read_json(path) for path in sorted((cohort / kind).glob('*.json'))]
            require(sorted(sha256_json(v) for v in exported) == sorted(sha256_json(v) for v in stored), 'final export differs from authoritative prospective cohort')
        pairs = eligible_pairs(predictions, settlements, specification=spec)
        require(len(pairs) == len(predictions) == len(settlements) > 0, 'invalid or incomplete prospective cohort')
        from src.ml.bundle_evidence import verify_historical_record
        for prediction in predictions:
            for key in ('active_prediction_provenance', 'candidate_prediction_provenance'):
                verify_historical_record(prediction[key], root=controller.root)
        aggregate = aggregate_eligible(predictions, settlements, specification=spec)
        require(result['fixture_count'] == manifest['prospective_fixture_count'] == aggregate['fixture_count'], 'prospective count mismatch')
        require(result['metrics'] == aggregate['metrics'] and result['paired_intervals'] == aggregate['paired_intervals'], 'forged prospective metrics')
        recomputed = apply_decision_rule(prospective_fixture_count=len(pairs), horizon_complete=True,
            paired_intervals=aggregate['paired_intervals']['metrics'], active_reliability=aggregate['metrics']['pl_1x2']['active'],
            candidate_reliability=aggregate['metrics']['pl_1x2']['candidate'], provenance_verified=True)
        require(recomputed == decision and recomputed['classification'] == SUPPORTED, 'scientific decision recomputation failed')
    else:
        require(decision['classification'] in ('CANDIDATE_INCONCLUSIVE', 'ACTIVE_SUPPORTED') and not decision['phase8_promotion_review_authorized'], 'invalid Phase 7 classification')
    return {'status': 'PROMOTION_ELIGIBLE' if supported else 'PROMOTION_NOT_ELIGIBLE', 'promotion_eligible': supported,
            'classification': decision['classification'], 'offline_verification_status': 'VERIFIED',
            'evidence_path': str(evidence.resolve()), 'evidence_manifest_sha256': sha256_file(evidence / 'run_manifest.json'),
            'specification_fingerprint': spec['specification_fingerprint'], 'candidate_bundle_id': CANDIDATE_ID,
            'candidate_bundle_aggregate_sha256': CANDIDATE_HASH, 'active_control_identity': spec['active']['model_bundle_sha256']}


@contextmanager
def production_bundle_guard(suite: Mapping[str, Any]) -> Iterator[None]:
    state = suite.get('_ffps_bundle_state')
    if state is None:  # Explicit evaluation-only suites retain their frozen path.
        yield
        return
    from src.ml.league_bundle_activation import production_controller
    controller = production_controller(state.get('league', 'PL'), root=Path(suite['_ffps_bundle_root']))
    require(controller is not None, 'production bundle owner missing')
    with state_lock(controller.lock):
        require(controller.current() == state, 'stale production suite; reload bundle')
        bundle = controller.bundle(state['bundle_id'])
        for role, slot in SLOTS.items():
            model = suite[slot]
            require(model._ffps_provenance['artifact_sha256'] == bundle['members'][role]['sha256'], 'registry/runtime mismatch')
            require(model.meta == bundle['members'][role]['meta'], 'runtime metadata/calibrator mismatch')
        yield
        require(controller.current() == state, 'active pointer changed during inference')


class GenerationBoundModel:
    """A loaded object becomes non-servable after any pointer generation change.

    All members are materialized before the prediction calculation begins. A
    change during loading or between model calls therefore fails the request,
    rather than blending members. Completed in-flight calls keep their explicit
    old bundle provenance; the next call must resolve the new generation.
    """
    def __init__(self, model, controller, state, bundle, role):
        self._model = model
        self._controller = controller
        self._state = state
        self._bundle = bundle
        self._role = role

    def __getattr__(self, name):
        return getattr(self._model, name)

    @property
    def meta(self):
        return self._model.meta

    def predict(self, frame, *args, **kwargs):
        import numpy as np
        with state_lock(self._controller.lock):
            require(self._controller.current() == self._state, 'stale production suite; reload bundle')
            expected = self._bundle['members'][self._role]
            require(self._model.meta == expected['meta'], 'runtime metadata/calibrator mismatch')
            require(self._model._ffps_provenance['artifact_sha256'] == expected['sha256'], 'registry/runtime mismatch')
            require(sha256_file(self._controller.root / expected['path']) == expected['sha256'], 'runtime artifact drift')
            require(list(frame.columns) == self._bundle['features'], 'runtime feature-schema mismatch')
            require(np.isfinite(frame.to_numpy(dtype=float)).all(), 'non-finite production feature vector')
            output = self._model.predict(frame, *args, **kwargs)
            require(np.isfinite(np.asarray(output, dtype=float)).all(), 'non-finite production model output')
            return output
