"""Filesystem capability and fail-closed checks use isolated fixtures only."""
from pathlib import Path
import json
import os
import subprocess
import sys
import tempfile

import pytest

from scripts import run_read_only_preflight as runner
from src.ml import bundle_activation as publication


def test_private_temporary_output_replaces_publishes_and_cleans_up(tmp_path):
    selected = tmp_path / '.tmp/verification'
    result = runner.verify_temp_directory(selected, root=tmp_path)
    assert result['verified'] and result['cleanup']
    assert result['atomic_replace'] and result['hard_link_publication']
    assert result['existing_destination_refused']
    assert list(selected.iterdir()) == []


def test_temp_location_outside_local_workspace_is_rejected(tmp_path):
    forbidden = tmp_path / 'baselines/phase19'
    with pytest.raises(ValueError, match='OUTSIDE_LOCAL_TMP'):
        runner.verify_temp_directory(forbidden, root=tmp_path)
    assert not forbidden.exists()


def test_unavailable_temp_location_has_no_fallback(tmp_path):
    selected = tmp_path / 'tmp/blocked'
    selected.parent.mkdir()
    selected.write_bytes(b'unavailable directory')
    with pytest.raises(FileExistsError):
        runner.verify_temp_directory(selected, root=tmp_path)
    assert selected.read_bytes() == b'unavailable directory'
    assert not (tmp_path / '.tmp').exists()


def test_denied_hard_link_fails_environment_check_and_cleans_fixture(tmp_path, monkeypatch):
    selected = tmp_path / '.tmp/verification'
    def denied(*args):
        raise PermissionError('injected hard-link denial')
    monkeypatch.setattr(runner.os, 'link', denied)
    with pytest.raises(PermissionError, match='hard-link denial'):
        runner.verify_temp_directory(selected, root=tmp_path)
    assert list(selected.iterdir()) == []


def test_atomic_assessment_replacement_uses_supported_filesystem(tmp_path):
    assessment = tmp_path / 'assessment.json'
    assessment.write_bytes(b'old-assessment')
    publication.atomic_write(assessment, b'complete-new-assessment')
    assert assessment.read_bytes() == b'complete-new-assessment'
    assert list(tmp_path.iterdir()) == [assessment]


def test_failed_replacement_preserves_existing_assessment(tmp_path, monkeypatch):
    assessment = tmp_path / 'assessment.json'
    assessment.write_bytes(b'old-assessment')
    def denied(*args):
        raise PermissionError('injected replacement denial')
    monkeypatch.setattr(publication.os, 'replace', denied)
    with pytest.raises(PermissionError, match='replacement denial'):
        publication.atomic_write(assessment, b'new-assessment')
    assert assessment.read_bytes() == b'old-assessment'
    assert list(tmp_path.iterdir()) == [assessment]


def test_immutable_publication_failure_preserves_existing_bytes(tmp_path, monkeypatch):
    assessment = tmp_path / 'assessment.json'
    assessment.write_bytes(b'old-assessment')
    def denied(*args):
        raise PermissionError('injected publication denial')
    monkeypatch.setattr(publication.os, 'link', denied)
    with pytest.raises(PermissionError, match='publication denial'):
        publication.persist_new(assessment, {'new': 'assessment'})
    assert assessment.read_bytes() == b'old-assessment'
    assert list(tmp_path.iterdir()) == [assessment]


def test_feature_pipeline_uses_selected_temp_and_cleans_output(tmp_path, monkeypatch):
    from src.evaluation import phase10_current_data as current
    from src.evaluation import phase11_current_source as source
    from src.features import pipeline
    selected = tmp_path / '.tmp/verification'
    runner.verify_temp_directory(selected, root=tmp_path)
    for key in ('TMPDIR', 'TEMP', 'TMP'):
        monkeypatch.setenv(key, str(selected))
    monkeypatch.setattr(tempfile, 'tempdir', None)
    original = pipeline.DATA_DIR
    generated = source.pipeline(current.base_frame('BL1'), 'BL1')
    assert len(generated) > 0
    assert pipeline.DATA_DIR == original
    assert Path(tempfile.gettempdir()).resolve() == selected.resolve()
    assert list(selected.iterdir()) == []


def test_child_environment_is_explicit_and_does_not_change_parent(tmp_path):
    before = {key: os.environ.get(key) for key in ('TMPDIR', 'TEMP', 'TMP')}
    env = runner.child_environment(tmp_path)
    assert all(env[key] == str(tmp_path) for key in before)
    assert {key: os.environ.get(key) for key in before} == before


def test_zero_exit_cannot_conceal_failed_verification():
    assert not runner.result_verified('phase20-integrity', {'verified': False})
    assert not runner.result_verified('phase19-verify', {'leagues': []})
    assert not runner.result_verified('operator-status', {})


def test_runner_exposes_only_existing_read_only_commands():
    assert runner.CHECKS == {
        'phase19-integrity': ['scripts/run_phase19_collection.py', 'integrity'],
        'phase19-verify': ['scripts/run_phase19_collection.py', 'verify'],
        'phase19-status': ['scripts/run_phase19_collection.py', 'status'],
        'phase20-integrity': ['scripts/run_phase20_settlement.py', 'integrity'],
        'phase20-verify': ['scripts/run_phase20_settlement.py', 'verify'],
        'phase20-boundary': ['scripts/verify_phase20_boundary.py', '--authorization',
            'docs/phase20-boundary-approved-v1.json', '--approved-sha256',
            '1ed13b93c37bd18b17a52a256c4d8358abdb0269c8574e7e9f2fe44994b63876'],
        'new-lineage-status': ['scripts/run_prospective_collection_next.py', 'status'],
        'operator-status': ['scripts/local_operator.py', 'status', '--json'],
    }


def test_redirected_temp_root_cannot_admit_protected_destination(tmp_path):
    root = tmp_path / 'repo'
    target = root / 'baselines'
    target.mkdir(parents=True)
    marker = target / 'protected.json'
    marker.write_bytes(b'unchanged')
    link = root / 'tmp'
    if os.name == 'nt':
        command = "New-Item -ItemType Junction -Path '{}' -Target '{}' -ErrorAction Stop | Out-Null".format(
            str(link).replace("'", "''"), str(target).replace("'", "''"))
        subprocess.run(['powershell.exe', '-NoProfile', '-NonInteractive', '-Command', command], check=True)
    else:
        link.symlink_to(target, target_is_directory=True)
    try:
        with pytest.raises(ValueError, match='OUTSIDE_LOCAL_TMP'):
            runner.local_path(link / 'verification', root=root)
        assert marker.read_bytes() == b'unchanged'
        assert not (target / 'verification').exists()
    finally:
        if os.name == 'nt':
            link.rmdir()  # Remove only this own junction; never recurse into its target.
        else:
            link.unlink()
    assert marker.read_bytes() == b'unchanged'


@pytest.fixture
def isolated_runner(tmp_path, monkeypatch):
    root = tmp_path / 'repo'
    root.mkdir()
    monkeypatch.setattr(runner, 'ROOT', root)
    monkeypatch.setattr(runner, 'TEMP_ROOT', root / '.tmp/verification')
    monkeypatch.setattr(runner.sys, 'prefix', str(root / 'venv'))
    # local_path's default root is bound at import time; use the fixture root.
    local_path = runner.local_path
    monkeypatch.setattr(runner, 'local_path', lambda path, root=root: local_path(path, root))
    return root


def test_missing_archives_stop_before_any_child(isolated_runner, monkeypatch):
    monkeypatch.setattr(runner, 'verify_temp_directory', lambda *_: {'verified': True})
    monkeypatch.setattr(runner, 'execute', lambda *_: pytest.fail('child started with missing archives'))
    output = isolated_runner / 'tmp/report'
    assert runner.main(['verify', '--output-directory', str(output)]) == 1
    failure = json.loads((output / 'results.json').read_text())['results']['failure']
    assert 'READ_ONLY_ARCHIVED_INPUT_MISSING' in failure['error']


def test_cleanup_and_operation_errors_both_survive_report(isolated_runner, monkeypatch):
    real_temporary = tempfile.TemporaryDirectory
    class CleanupFailure(real_temporary):
        def cleanup(self):
            super().cleanup()
            raise PermissionError('injected cleanup denial')
    def denied(*_):
        raise PermissionError('injected operation denial')
    monkeypatch.setattr(runner.tempfile, 'TemporaryDirectory', CleanupFailure)
    monkeypatch.setattr(runner.os, 'link', denied)
    monkeypatch.setattr(runner, 'verify_temp_directory', lambda path: runner_probe(path, root=isolated_runner))
    monkeypatch.setattr(runner, 'execute', lambda *_: pytest.fail('child started after filesystem failure'))
    output = isolated_runner / 'tmp/report'
    assert runner.main(['verify', '--output-directory', str(output)]) == 1
    failure = json.loads((output / 'results.json').read_text())['results']['failure']
    assert 'injected operation denial' in failure['traceback']
    assert 'injected cleanup denial' in failure['traceback']
    assert list((isolated_runner / '.tmp/verification').iterdir()) == []


runner_probe = runner.verify_temp_directory


def test_failed_child_stops_remaining_dispatch(isolated_runner, monkeypatch):
    monkeypatch.setattr(runner, 'verify_temp_directory', lambda *_: {'verified': True})
    monkeypatch.setattr(runner, 'archived_read_only_inputs', lambda: None)
    called = []
    def failed(label, *_):
        called.append(label)
        return {'exit_code': 0, 'verified': False}
    monkeypatch.setattr(runner, 'execute', failed)
    output = isolated_runner / 'tmp/report'
    assert runner.main(['verify', '--output-directory', str(output)]) == 1
    assert called == ['phase19-integrity']
    saved = json.loads((output / 'results.json').read_text())
    assert saved['results']['phase19-integrity']['verified'] is False
    assert saved['not_started'] == list(runner.CHECKS)[1:]


def test_successful_dispatch_runs_all_exact_commands(isolated_runner, monkeypatch):
    monkeypatch.setattr(runner, 'verify_temp_directory', lambda *_: {'verified': True})
    monkeypatch.setattr(runner, 'archived_read_only_inputs', lambda: None)
    called = []
    def passed(label, command, output, env):
        called.append((label, command))
        assert all(env[key] == str(runner.TEMP_ROOT) for key in ('TMPDIR', 'TEMP', 'TMP'))
        return {'exit_code': 0, 'verified': True}
    monkeypatch.setattr(runner, 'execute', passed)
    output = isolated_runner / 'tmp/report'
    assert runner.main(['verify', '--output-directory', str(output)]) == 0
    assert called == list(runner.CHECKS.items())
    saved = json.loads((output / 'results.json').read_text())
    assert saved['verified'] is True and saved['not_started'] == []


@pytest.mark.parametrize('operation', ['fsync', 'replace'])
def test_probe_denied_sync_or_replace_fails_and_cleans(tmp_path, monkeypatch, operation):
    selected = tmp_path / '.tmp/verification'
    def denied(*_):
        raise PermissionError('injected ' + operation + ' denial')
    monkeypatch.setattr(runner.os, operation, denied)
    with pytest.raises(PermissionError, match=operation + ' denial'):
        runner.verify_temp_directory(selected, root=tmp_path)
    assert list(selected.iterdir()) == []


@pytest.mark.parametrize('label', ['phase19-verify', 'phase20-verify', 'phase19-status', 'new-lineage-status', 'operator-status'])
def test_complete_successful_outputs_are_accepted(label):
    leagues = runner.ALL_LEAGUES[:-1] if label == 'new-lineage-status' else runner.ALL_LEAGUES
    wrapped = {'verified': True, 'value': {'verified': True}}
    value = {'leagues': [
        {'league': league, 'verified': True, 'pairs': 0, 'observations': 0, 'settlements': 0,
         'recorded': 0, 'current_lineage_verified': True, 'current': wrapped,
         'historical': wrapped, 'settlement': wrapped} for league in leagues],
        'phase19_integrity': wrapped, 'phase20_integrity': wrapped}
    assert runner.result_verified(label, value) is True


def test_operator_nested_failure_is_rejected():
    wrapped = {'verified': True, 'value': {'verified': True}}
    value = {'leagues': [{'league': league, 'current_lineage_verified': True,
              'current': wrapped, 'historical': wrapped, 'settlement': wrapped}
              for league in runner.ALL_LEAGUES],
             'phase19_integrity': wrapped,
             'phase20_integrity': {'verified': True, 'value': {'verified': False}}}
    assert runner.result_verified('operator-status', value) is False


@pytest.mark.parametrize('label', ['phase19-verify', 'phase20-verify', 'phase19-status', 'new-lineage-status'])
@pytest.mark.parametrize('value', [
    {'leagues': [{'league': 'BL1', 'verified': True}]},
    {'verified': False, 'leagues': [{'league': x, 'verified': True} for x in ('BL1', 'SA', 'FL1', 'PD', 'PL')]},
    {'leagues': [{'league': 'BL1', 'verified': True}] * 5},
    {'leagues': [True]},
])
def test_incomplete_failed_duplicate_or_malformed_results_are_rejected(label, value):
    assert runner.result_verified(label, value) is False


@pytest.mark.parametrize('code,payload,expected', [
    (7, '{"verified":true}', False),
    (0, '{"verified":false}', False),
    (0, '{"verified":true,"error":"failed"}', False),
    (0, '[]', False),
    (0, 'invalid JSON', False),
    (0, '{"verified":true}', True),
])
def test_execute_preserves_real_exit_and_rejects_false_results(tmp_path, code, payload, expected):
    output = tmp_path / 'output'
    output.mkdir()
    program = tmp_path / 'isolated_child.py'
    program.write_text('import sys\nprint(' + repr(payload) + ')\nprint("diagnostic", file=sys.stderr)\nsys.exit(' + str(code) + ')\n')
    result = runner.execute('phase20-integrity', [str(program)], output, runner.child_environment(tmp_path))
    assert result['exit_code'] == code
    assert result['verified'] is expected
    assert json.loads((output / 'phase20-integrity.result.json').read_text()) == result
    assert (output / 'phase20-integrity.stdout.log').read_text().strip() == payload
    assert (output / 'phase20-integrity.stderr.log').read_text().strip() == 'diagnostic'
