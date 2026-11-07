"""Run isolated acceptance/publication unit tests; no native qualification.

Usage: python -I -S -B scripts/test_phase20_publication.py
All output is diagnostic stdout/stderr. No report or acceptance file is written.
"""
from __future__ import annotations

import ctypes
import json
import os
from pathlib import Path
import runpy
import sys
import unittest
import zoneinfo


ROOT = Path(__file__).resolve().parents[1]
TESTS = ('test_phase20_local_acceptance.py', 'test_phase20_publication.py',
         'test_phase20_windows_io.py')


def _audit(event, args):
    if event == 'open':
        mode, flags = args[1], args[2]
        if ((isinstance(mode, str) and any(c in mode for c in 'wax+'))
                or (isinstance(flags, int) and flags &
                    (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND))):
            raise RuntimeError('P20_UNIT_WRITE_PROHIBITED')
    if event in {'os.remove', 'os.rename', 'os.rmdir', 'os.mkdir', 'os.link',
                 'os.symlink', 'os.chmod', 'os.truncate', 'os.utime',
                 'subprocess.Popen', 'os.system', 'socket.__new__',
                 'socket.connect', 'socket.bind', 'ctypes.dlopen', 'ctypes.dlsym'}:
        raise RuntimeError('P20_UNIT_EFFECT_PROHIBITED:' + event)


def main():
    if not (sys.flags.isolated and sys.flags.no_site and sys.dont_write_bytecode):
        raise SystemExit('Run with -I -S -B; site startup and bytecode must be disabled.')
    # ctypes is initialized above before forbidding further DLL binding. The
    # Windows adapter tests must supply function doubles, never actual WinDLLs.
    sys.addaudithook(_audit)
    sys.path.insert(0, str(ROOT))
    zoneinfo.reset_tzpath([str(ROOT / 'venv/Lib/site-packages/tzdata/zoneinfo')])
    suite = unittest.TestSuite()
    counts = {}
    for name in TESTS:
        scope = runpy.run_path(str(ROOT / 'tests/unit' / name))
        count = 0
        for obj in scope.values():
            if isinstance(obj, type) and issubclass(obj, unittest.TestCase) and obj is not unittest.TestCase:
                cases = unittest.defaultTestLoader.loadTestsFromTestCase(obj)
                count += cases.countTestCases()
                suite.addTests(cases)
        counts[name] = count
    print(json.dumps({'scope': 'UNIT_AND_INJECTED_API_DOUBLES_ONLY',
                      'python': sys.version, 'platform': sys.platform,
                      'pointer_bits': ctypes.sizeof(ctypes.c_void_p) * 8,
                      'tests': counts}), flush=True)
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    print(json.dumps({'run': result.testsRun, 'failures': len(result.failures),
                      'errors': len(result.errors), 'skipped': len(result.skipped),
                      'native_qualification': 'NOT_RUN', 'package_capture': 'NOT_CAPTURED'}), flush=True)
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    raise SystemExit(main())
