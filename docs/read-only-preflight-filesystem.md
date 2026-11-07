# Read-only scientific verification temporary storage

Use the existing repository virtual environment from the FFPS root. This
runner invokes only the documented integrity, offline verification, and status
commands. It exposes no acquisition, prediction collection, authorization,
settlement, scheduling, or repair operation.

```powershell
.\venv\Scripts\python.exe scripts/run_read_only_preflight.py environment --output-directory tmp/preflight-environment-review-01
.\venv\Scripts\python.exe scripts/run_read_only_preflight.py tests --output-directory tmp/preflight-tests-review-01
.\venv\Scripts\python.exe scripts/run_read_only_preflight.py verify --output-directory tmp/preflight-verification-review-01
```

Choose a new output directory for each invocation; existing output directories
are refused. Output paths must resolve inside repository `tmp/` or `.tmp/`.
Full stdout, stderr, real exit codes, parsed verification results, and test
JUnit XML are retained there. A successful exit code cannot conceal a false,
incomplete, or malformed verification result; league outputs must include the
complete expected population (five original leagues, four for new-lineage status).
Verification runs the fixed commands sequentially in separate processes and
stops after the first failed command, retaining completed results and listing
unstarted commands in `not_started`. Existing archived Phase 19 inputs must be
present before tests or verification start, so missing one-shot archives are
not initialized by this runner.

The explicit temporary-output base is `.tmp/read-only-verification`. Before
any child command starts, a private temporary fixture at this exact base must
pass file creation, flushed complete-content writing, same-directory atomic
replacement, hard-link publication, refusal of an existing publication target,
readback, and cleanup. Only the children receive `TMPDIR`, `TEMP`, and `TMP`
pointing to that tested base. Parent and user-wide environment settings stay
unchanged. There is no fallback and no ACL change.

Resolved output and temp paths are checked against the literal repository temp
bases: redirecting a base into protected evidence or another filesystem cannot
make that destination admissible. Probe source, replacement and publication
target are siblings in one private directory, preserving same-filesystem
operations. Errors include their complete chained traceback in `results.json`,
so a cleanup error does not hide the original operation failure.

This is a trusted local operator workflow, not an OS security sandbox. Do not
run it alongside writers that remove/redirect archives or temp paths after the
checks. Pytest's existing shared `tmp_path` fixture uses `.tmp/test-fixtures`
separately from `--basetemp` and has best-effort outer fixture cleanup. The
runner's private probe and feature-output cleanup are explicitly checked;
successful tests alone do not prove all shared fixture directories were removed.

## Diagnosed execution-context limitation

The October 9 investigation retained controlled probe results under
`tmp/permission-investigation-20261009`:

- In restricted execution, Python selected the packaged-app `AC/Temp`
  directory. A mode-0700 directory could be created there, but file creation
  and directory removal failed. Mode-0777 permitted file writes and cleanup,
  but replacement and hard-link publication were denied.
- At repository `tmp/` and `.tmp/`, both directory modes permitted writes,
  atomic replacement, and cleanup in restricted execution. Hard-link
  publication still failed with `WinError 5` in both locations.
- The same script and existing Python 3.14.2 virtual environment, run outside
  restricted execution, passed every operation at all three locations and
  both directory modes. Its default was ordinary `AppData/Local/Temp`, rather
  than packaged-app Temp; the two repository base paths were identical across
  contexts.

The original feature errors first occurred at CSV file creation, then during
`TemporaryDirectory` cleanup (`rmdir` and its `chmod` error handler). Directory
creation itself had succeeded. The six isolated settlement errors were at
`os.link` in immutable `persist_new`, not at `os.replace`. The previous
execution report's description of those failures as atomic renames was
imprecise; its original diagnostics are preserved.

These controls identify an execution-context limitation. They do not establish
an antivirus product, a particular ACL entry, or a specific Windows policy as
the lower-level cause. An explicit local temp directory fixes the observed
temp-output location problem, but cannot grant missing hard-link capability.

If the runner's environment check reports a hard-link denial, stop. Use an
ordinary local PowerShell process with the existing venv, or an explicitly
approved execution outside the restricted context, for these exact isolated
tests and read-only checks. This changes the process context, not scientific
rules or user-wide security settings. Do not weaken immutable publication,
replace it with direct writes, or modify hash-bound scientific implementation
files. The runner's gate must pass in the selected context before checks run.

Review the execution report and diff after successful verification. Prospective
acquisition and collection remain a separate task requiring their own fresh
source, stale-input, and pre-kickoff gates. Preflight success grants no collection,
settlement, scheduling, activation or promotion authorization.
