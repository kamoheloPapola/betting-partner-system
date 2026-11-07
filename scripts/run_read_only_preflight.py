"""Run FFPS read-only checks with explicit, tested local temporary storage.

No acquisition, collection, authorization, settlement, or repair interface exists.
The immutable implementation files remain unchanged. A filesystem capability
failure stops this runner; changing TMPDIR cannot make denied hard links safe.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import traceback

ROOT = Path(__file__).resolve().parents[1]
TEMP_ROOT = ROOT / '.tmp/read-only-verification'
APPROVED_HASH = '1ed13b93c37bd18b17a52a256c4d8358abdb0269c8574e7e9f2fe44994b63876'
TEST_FILES = [f'tests/unit/test_{name}.py' for name in (
    'prospective_collection_next', 'phase19_collection', 'phase20_boundary',
    'phase20_settlement', 'local_operator', 'read_only_preflight')]
CHECKS = {
    'phase19-integrity': ['scripts/run_phase19_collection.py', 'integrity'],
    'phase19-verify': ['scripts/run_phase19_collection.py', 'verify'],
    'phase19-status': ['scripts/run_phase19_collection.py', 'status'],
    'phase20-integrity': ['scripts/run_phase20_settlement.py', 'integrity'],
    'phase20-verify': ['scripts/run_phase20_settlement.py', 'verify'],
    'phase20-boundary': ['scripts/verify_phase20_boundary.py', '--authorization',
        'docs/phase20-boundary-approved-v1.json', '--approved-sha256', APPROVED_HASH],
    'new-lineage-status': ['scripts/run_prospective_collection_next.py', 'status'],
    'operator-status': ['scripts/local_operator.py', 'status', '--json'],
}
ALL_LEAGUES = ('BL1', 'SA', 'FL1', 'PD', 'PL')


def local_path(path: Path, root: Path = ROOT) -> Path:
    value = path.resolve()
    # Resolve the candidate, not the allowed bases: a redirected tmp base must
    # not turn protected evidence or an external volume into an allowed root.
    root = root.resolve()
    if not any(value.is_relative_to(root / name) for name in ('tmp', '.tmp')):
        raise ValueError('READ_ONLY_WORKSPACE_OUTSIDE_LOCAL_TMP:' + str(value))
    return value


def verify_temp_directory(path: Path, root: Path = ROOT) -> dict:
    """Test actual private-temp write, replacement, publication, and cleanup.

    Every file touched belongs to this newly created probe. Any operation failure
    propagates; there is no fallback or ACL mutation, and cleanup stays enabled.
    """
    path = local_path(path, root)
    path.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='ffps-io-check-', dir=path) as name:
        directory = Path(name)
        original = directory / 'original.json'
        replacement = directory / 'replacement.tmp'
        linked = directory / 'published.json'
        original.write_bytes(b'old')
        with replacement.open('xb') as stream:
            stream.write(b'complete-new-content')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(replacement, original)
        if original.read_bytes() != b'complete-new-content':
            raise RuntimeError('READ_ONLY_ATOMIC_REPLACE_READBACK_FAILED')
        os.link(original, linked)
        if linked.read_bytes() != original.read_bytes():
            raise RuntimeError('READ_ONLY_PUBLICATION_READBACK_FAILED')
        try:
            os.link(original, linked)
        except FileExistsError:
            pass
        else:
            raise RuntimeError('READ_ONLY_PUBLICATION_OVERWROTE_EXISTING_FILE')
    if directory.exists():
        raise RuntimeError('READ_ONLY_TEMP_CLEANUP_FAILED')
    return {'verified': True, 'temporary_directory': str(path),
            'file_write': True, 'atomic_replace': True, 'hard_link_publication': True,
            'existing_destination_refused': True, 'cleanup': True}


def child_environment(path: Path) -> dict:
    env = os.environ.copy()
    env.update({key: str(path) for key in ('TMPDIR', 'TEMP', 'TMP')})
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    return env


def result_verified(label: str, value: dict) -> bool:
    if not isinstance(value, dict) or 'error' in value or value.get('errors') or \
            value.get('verified', True) is not True:
        return False
    if label in ('phase19-verify', 'phase20-verify', 'phase19-status',
                 'new-lineage-status', 'operator-status'):
        rows = value.get('leagues')
        expected = set(ALL_LEAGUES[:-1] if label == 'new-lineage-status' else ALL_LEAGUES)
        if not isinstance(rows, list) or len(rows) != len(expected) or \
                not all(isinstance(row, dict) and isinstance(row.get('league'), str) for row in rows) or \
                {row['league'] for row in rows} != expected:
            return False
    if label in ('phase19-verify', 'phase20-verify'):
        return all(row.get('verified') is True and 'error' not in row and not row.get('errors') for row in rows)
    if label == 'operator-status':
        def wrapped(item, *, nested=False):
            return isinstance(item, dict) and item.get('verified') is True and 'error' not in item and (
                not nested or isinstance(item.get('value'), dict) and
                item['value'].get('verified') is True and 'error' not in item['value'] and
                not item['value'].get('errors'))
        return all(wrapped(value.get(key), nested=True)
                   for key in ('phase19_integrity', 'phase20_integrity')) and all(
                       row.get('current_lineage_verified') is True and
                       wrapped(row.get('current')) and wrapped(row.get('historical')) and
                       wrapped(row.get('settlement'), nested=True)
                       for row in rows)
    if label in ('phase19-status', 'new-lineage-status'):
        fields = ('recorded',) if label == 'new-lineage-status' else ('pairs', 'observations', 'settlements')
        return all(type(row.get(field)) is int and row[field] >= 0
                   for row in rows for field in fields) and all(
                       'error' not in row and row.get('verified', True) is True for row in rows)
    return value.get('verified') is True


def archived_read_only_inputs():
    """Refuse before old verifiers can initialize a missing one-shot archive."""
    base = ROOT / 'baselines/phase19/phase19-controlled-collection-20260923-v1'
    paths = [base / 'cycle_intent.json', base / 'PL_cycle.json']
    for league in ('BL1', 'SA', 'FL1', 'PD', 'PL'):
        paths.extend(base / 'acquisition' / league / name
                     for name in ('primary_matches.json', 'recovery_matches.json'))
    paths.append(base / 'acquisition/PL/recovery_matches.source.json')
    missing = [str(path.relative_to(ROOT)) for path in paths if not path.is_file()]
    if missing:
        raise RuntimeError('READ_ONLY_ARCHIVED_INPUT_MISSING:' + ','.join(missing))


def execute(label: str, args: list[str], output: Path, env: dict) -> dict:
    command = [sys.executable, *args]
    print(json.dumps({'command_started': label, 'command': command}), flush=True)
    with (output / (label + '.stdout.log')).open('x', encoding='utf-8') as stdout, \
         (output / (label + '.stderr.log')).open('x', encoding='utf-8') as stderr:
        process = subprocess.run(command, cwd=ROOT, env=env, stdout=stdout, stderr=stderr, check=False)
    result = {'command': command, 'exit_code': process.returncode}
    if label == 'tests':
        result['verified'] = process.returncode == 0
    else:
        try:
            value = json.loads((output / (label + '.stdout.log')).read_text(encoding='utf-8'))
            if not isinstance(value, dict):
                raise ValueError('READ_ONLY_COMMAND_JSON_OBJECT_REQUIRED')
            result.update({'value': value, 'verified': process.returncode == 0 and result_verified(label, value)})
        except (ValueError, TypeError, KeyError) as exc:
            result.update({'verified': False, 'parse_error': str(exc)})
    (output / (label + '.result.json')).write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({'command_finished': label, 'exit_code': process.returncode,
                      'verified': result['verified']}), flush=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('environment', 'tests', 'verify'))
    parser.add_argument('--output-directory', type=Path, required=True)
    args = parser.parse_args(argv)
    output = local_path(args.output_directory)
    output.mkdir(parents=True, exist_ok=False)
    results = {}
    try:
        if Path(sys.prefix).resolve() != (ROOT / 'venv').resolve():
            raise RuntimeError('READ_ONLY_REPOSITORY_VENV_REQUIRED')
        results['environment'] = verify_temp_directory(TEMP_ROOT)
        env = child_environment(TEMP_ROOT)
        if args.mode in ('tests', 'verify'):
            archived_read_only_inputs()
        if args.mode == 'tests':
            results['tests'] = execute('tests', ['-m', 'pytest', *TEST_FILES, '-q',
                '--basetemp', str(output / 'pytest-temp'), '--junitxml', str(output / 'tests.xml')], output, env)
        elif args.mode == 'verify':
            for label, command in CHECKS.items():
                results[label] = execute(label, command, output, env)
                if results[label]['verified'] is not True:
                    break
    except (OSError, ValueError, RuntimeError) as exc:
        results['failure'] = {'verified': False, 'error': str(exc),
                              'type': type(exc).__name__,
                              'traceback': ''.join(traceback.format_exception(exc))}
    verified = all(value.get('verified') is True for value in results.values())
    not_started = [label for label in CHECKS if label not in results] if args.mode == 'verify' else []
    (output / 'results.json').write_text(json.dumps({'verified': verified, 'results': results,
        'not_started': not_started}, indent=2), encoding='utf-8')
    print(json.dumps({'verified': verified, 'output_directory': str(output)}), flush=True)
    return 0 if verified else 1


if __name__ == '__main__':
    raise SystemExit(main())
