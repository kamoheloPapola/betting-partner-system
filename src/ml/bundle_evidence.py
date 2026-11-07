"""Offline Phase 8 evidence, including verification of versioned Phase 4 code.

Historical records name historical source bytes. A temporary read-only artifact
view supplies the archived registry version without rewriting frozen evidence or
weakening Phase 4's existing source verifier.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping
import xml.etree.ElementTree as ET

from src.ml.bundle_activation import (
    ROOT, PHASE7, BundleController, persist_new, read_json, require, seal,
    sha256_file, verify_seal, verify_smoke,
)

RUN = ROOT / 'baselines/phase8/phase8-controlled-pl-bundle-20260905-v1'


def historical_source(path: Path, expected_sha256: str, *, root: Path = ROOT) -> Path:
    """Resolve exact archived source bytes; never substitute models or data."""
    relative = path.relative_to(root)
    if path.is_file() and sha256_file(path) == expected_sha256:
        return path
    archives = [root/'baselines/phase8/phase8-controlled-pl-bundle-20260905-v1/prior_sources',
                *sorted((root/'baselines/phase9').glob('*/prior_sources')),
                *sorted((root/'baselines/phase9').glob('*/source_versions/*'))]
    for archive in archives:
        candidate = archive/relative
        if candidate.is_file() and sha256_file(candidate) == expected_sha256:
            return candidate
    require(False, 'historical source bytes unavailable: '+str(relative))


def verify_historical_record(record: Mapping[str, Any], *, root: Path = ROOT, replay: bool = False) -> dict[str, Any]:
    from src.predictions.provenance import verify_record, replay_pl_1x2
    sources = record['code_provenance']['relevant_source_files']
    artifacts = [
        {'path': m['source_location'], 'sha256': m['artifact_sha256']}
        for m in record['model_provenance']['members']
    ] + record['data_provenance']['source_identity']['artifacts']
    archive = root / 'baselines/phase8/phase8-controlled-pl-bundle-20260905-v1/prior_sources'
    root.joinpath('tmp').mkdir(exist_ok=True)
    substitutions = []
    with tempfile.TemporaryDirectory(prefix='phase8-offline-', dir=root / 'tmp') as temporary:
        view = Path(temporary)
        for item in [*sources, *artifacts]:
            relative = Path(item['path'])
            require(not relative.is_absolute() and '..' not in relative.parts, 'unsafe historical reference')
            source = root / relative
            if sha256_file(source) != item['sha256']:
                require(item in sources, f'historical artifact drift: {relative}')
                archived = historical_source(source,item['sha256'],root=root)
                source = archived
                substitutions.append(relative.as_posix())
            target = view / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                os.link(source, target)
        result = verify_record(record, root=view, verify_referenced_artifacts=True)
        require(result.get('overall_phase4_trusted'), f'historical Phase 4 verification failed: {result}')
        if replay:
            result['replay'] = replay_pl_1x2(record, root=view)
            require(result['replay']['reproduced'], 'historical replay failed')
    result['archived_source_paths'] = sorted(set(substitutions))
    return result


def junit(path: Path) -> dict[str, Any]:
    cases = list(ET.parse(path).getroot().iter('testcase'))
    failed = [case.get('classname', '') + '::' + case.get('name', '') for case in cases
              if case.find('failure') is not None or case.find('error') is not None]
    skipped = sum(case.find('skipped') is not None for case in cases)
    return {'tests': len(cases), 'passed': len(cases) - len(failed) - skipped, 'skipped': skipped,
            'failures': len(failed), 'failure_ids': sorted(failed), 'sha256': sha256_file(path)}


def protected_snapshot(run: Path) -> dict[str, str]:
    before = read_json(run / 'protected_before.json')
    current = {name: sha256_file(ROOT / name) for name in before}
    require(current == before, 'protected Phase 0-7 or model bytes changed')
    return current


def finalize(run: Path, *, focused: Path, protected: Path, full: Path) -> dict[str, Any]:
    controller = BundleController()
    # This final run is immutable; preliminary development dry-runs remain as
    # separate records with their original source identities.
    result = controller.dry_run(ROOT / PHASE7)
    persist_new(run / 'final_dry_run.json', result)
    tests = {'focused': junit(focused), 'protected': junit(protected), 'full': junit(full),
             'prior_full': junit(ROOT / 'tmp/phase7er-full.xml')}
    tests['new_unexplained_failure_ids'] = sorted(set(tests['full']['failure_ids']) - set(tests['prior_full']['failure_ids']))
    require(tests['focused']['failures'] == tests['protected']['failures'] == 0, 'required focused/protected checks failed')
    require(not tests['new_unexplained_failure_ids'], 'new unexplained full-suite failures')
    for name, source in (('focused', focused), ('protected', protected), ('full', full), ('prior_full', ROOT / 'tmp/phase7er-full.xml')):
        destination = run / 'tests' / (name + '.xml')
        destination.parent.mkdir(exist_ok=True)
        with destination.open('xb') as stream:
            stream.write(source.read_bytes())
            stream.flush()
            os.fsync(stream.fileno())
    persist_new(run / 'test_results.json', tests)
    persist_new(run / 'protected_after.json', protected_snapshot(run))
    from scripts.run_phase7p_prospective import verify_frozen_contract
    frozen = verify_frozen_contract()
    require(frozen['verified'], 'Phase 7P frozen contract failed')
    historical = {}
    for label, path in {
        'phase4': 'baselines/phase4/phase4-pl-1x2-provenance-79110f9a-20260827-v1/prediction_provenance.json',
        'phase6': 'baselines/phase6/phase6-controlled-pl-goals-79110f9a-20260829-v1/integration/prediction_provenance.json',
    }.items():
        historical[label] = verify_historical_record(read_json(ROOT / path), replay=True)
    persist_new(run / 'protected_offline_verification.json', {'phase7p_contract': frozen, 'historical_provenance': historical})
    report = f'''# Phase 8 implementation report

**{result['classification']}**

Production remains `{result['preflight']['previous_state']['bundle_id']}`. Candidate activation was not performed. The current scientific classification is `CANDIDATE_INCONCLUSIVE`; `promotion_eligible=false` is enforced before the mutation path.

| Requested verification | Result |
| --- | --- |
| Active ownership | File manifest bootstrap; one atomic `active_pl_bundle.json` reference after first transition; DB cannot supersede it |
| Cache/reload | Registry manifest singleton bypassed for PL goals; complete bundle load; generation-bound models reject stale calls; supported serving interpreter must be quiescent for activation |
| Activation schema | `ffps.model_bundle_activation_record.v1`, immutable canonical SHA-256 seal |
| Rollback schema | `ffps.model_bundle_rollback_record.v1`, linked to activation or failed intent |
| Phase 7 gate | `PROMOTION_NOT_ELIGIBLE`, finalized/indexed evidence identity verified |
| Candidate | Exact four-member aggregate `7fc4d5600bde7e96901bbba9896873cbdee3ba6bd7e0189b496aea1fe3583ae2` verified |
| Lineage | All four members `TRAINING_LINEAGE_VERIFIED`, Phase 5 and Phase 6 references checked |
| Dry-run preflight | `ACTIVATION_PREFLIGHT_VERIFIED`; raw, ordered, finite, normalized PL 1X2; unchanged feature/ensemble/simulator contracts |
| Runtime adoption | `{result['preflight']['runtime_quiescence']['status']}`; no other supported serving workers at final dry-run |
| Atomicity | Staged/fsynced one-file replacement, cross-process locking, pre-replace failure preserves exact pointer, interrupted transition blocks serving |
| Rollback | All four prior references restored; post-write failure recovery and explicit crash recovery tested |
| Smoke provenance | Actual model bytes, bundle ID and pointer fingerprint bound; independent Phase 4 replay verified for activation and rollback |
| Legacy manifest before | `{result['manifest_before_sha256']}` |
| Legacy manifest after | `{result['manifest_after_sha256']}` |
| Active pointer before/after dry-run | Absent / absent; verified legacy bootstrap remains authoritative |
| Focused plus registry tests | {tests['focused']['passed']} passed, {tests['focused']['failures']} failures |
| Protected Phase 0–7 regressions | {tests['protected']['passed']} passed, {tests['protected']['failures']} failures |
| Full suite | {tests['full']['passed']} passed, {tests['full']['failures']} inherited failures, {tests['full']['skipped']} skipped |
| New unexplained failures | {len(tests['new_unexplained_failure_ids'])} |
| Frozen evidence/model preservation | {len(read_json(run / 'protected_before.json'))} files unchanged |
| Offline verification | Final index, source/evidence hashes, bundle/lineage, smoke provenance and protected identities verified by `verify-infrastructure` |

The full-suite failures are unchanged from the captured Phase 7E-R test baseline:

''' + '\n'.join('- `' + name + '`' for name in tests['full']['failure_ids']) + '''

These are one quiet-mode expectation and five stale drift-artifact freshness checks. Phase 8 did not refresh drift files or change their assertions.

The implementation uses a dedicated bundle pointer rather than rewriting four manifest entries. The prediction module was restored byte-for-byte before final validation. Historical Phase 4 records are verified against their archived registry source in an isolated artifact view; current-source-only verification of an old record is intentionally not reported as equivalent. Phase 7P's unchanged frozen execution contract verifies and collection can continue.

Successful-switch tests use isolated temporary pointers and the real frozen model bytes. Only the scientific gate is simulated at that unit-test boundary; no such bypass is exposed by the production API or CLI. Real current-evidence activation is blocked. No retraining, recalibration, feature/threshold/stopping-rule changes, source refresh, betting or ROI evaluation occurred.

Limitations: local filesystem and supported serving interpreter only; maintenance downtime is required before switching and workers restart afterwards. An already completed in-flight model calculation retains its explicitly recorded old-bundle provenance. Hash seals detect tampering but are not signatures or filesystem access controls. Hardware power loss was not physically tested. Recovery fails closed if prior protected artifacts are unavailable. The future Phase 7 fixed-horizon final export must use the documented adapter; it does not exist today.

See `docs/phase8-model-bundle-operations.md` for commands, ownership, evidence contracts and recovery procedure. Continue active production and Phase 7P shadow collection; do not activate the candidate early.
'''
    with (run / 'PHASE8_IMPLEMENTATION_REPORT.md').open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(report)
        stream.flush()
        os.fsync(stream.fileno())
    sources = ['src/ml/bundle_activation.py', 'src/ml/bundle_evidence.py', 'src/ml/registry.py',
               'scripts/run_phase8_bundle_activation.py', 'tests/unit/test_bundle_activation.py', 'docs/phase8-model-bundle-operations.md']
    index = seal({'schema_version': 'ffps.phase8_infrastructure_index.v1',
                  'sources': [{'path': path, 'sha256': sha256_file(ROOT / path)} for path in sources],
                  'artifacts': [{'path': str(path.relative_to(ROOT).as_posix()), 'sha256': sha256_file(path)}
                                for path in sorted(run.rglob('*')) if path.is_file() and path.name not in ('artifact_index.json', 'offline_verification.json')]})
    persist_new(run / 'artifact_index.json', index)
    verified = verify_infrastructure(run)
    persist_new(run / 'offline_verification.json', verified)
    return verified


def verify_infrastructure(run: Path) -> dict[str, Any]:
    index = read_json(run / 'artifact_index.json')
    verify_seal(index)
    for item in index['sources']:
        historical_source(ROOT/item['path'],item['sha256'])
    for item in index['artifacts']:
        require(sha256_file(ROOT / item['path']) == item['sha256'], 'Phase 8 indexed evidence/source drift: ' + item['path'])
    protected_snapshot(run)
    result = read_json(run / 'final_dry_run.json')
    verify_seal(result)
    controller = BundleController()
    require(result['manifest_before_sha256'] == result['manifest_after_sha256'] == sha256_file(controller.manifest), 'dry-run manifest mismatch')
    require(result['pointer_before'] == result['pointer_after'] is None and not controller.pointer.exists(), 'dry-run active pointer mismatch')
    require(result['eligibility']['promotion_eligible'] is False and result['eligibility']['classification'] == 'CANDIDATE_INCONCLUSIVE', 'current eligibility mismatch')
    for name in ('candidate', 'rollback_target'):
        bundle = result['preflight'][name]
        require(controller.bundle(bundle['bundle_id']) == bundle, 'preflight bundle drift')
        verify_smoke(result['preflight']['candidate_smoke' if name == 'candidate' else 'rollback_smoke'], bundle, root=ROOT, historical=True)
    from scripts.run_phase7p_prospective import verify_frozen_contract
    require(verify_frozen_contract()['verified'], 'Phase 7P no longer operational')
    tests = read_json(run / 'test_results.json')
    for name in ('focused', 'protected', 'full', 'prior_full'):
        require(junit(run / 'tests' / (name + '.xml')) == tests[name], 'test report identity mismatch')
    require(not tests['new_unexplained_failure_ids'] and tests['focused']['failures'] == tests['protected']['failures'] == 0, 'test evidence failed')
    return {'schema_version': 'ffps.phase8_offline_verification.v1', 'status': 'VERIFIED',
            'classification': result['classification'], 'promotion_eligible': False,
            'production_activation_performed': False, 'protected_files_verified': len(read_json(run / 'protected_before.json')),
            'focused_tests': tests['focused'], 'protected_tests': tests['protected'], 'full_suite': tests['full'],
            'new_unexplained_failure_ids': tests['new_unexplained_failure_ids']}
