# Phase 11 current-source operations

Phase 11 reads production models and frozen history but writes only into a new immutable evidence run under `baselines/phase11`. It has no training, calibration, activation, rollback, or promotion command.

The current source contract composes three explicitly identified sources. Football-data.org owns fixture identity, kickoff, status, and goals. The HTTPS no-www football-data.co.uk CSV is the primary corner source. FotMob's web API fills current corner gaps only after exact overlap qualification. Every response, including provider errors, is stored with its retrieval time and SHA-256; credentials and authorization values are never stored.

Run acquisition and qualification separately for each league. A qualification writes a new numbered assessment and never replaces an earlier attempt. Freeze only the exact `qualification.json` selected for that league. Collection considers the earliest still-future eligible fixture, rebuilds its causal features from frozen history plus archived completed observations, and writes an append-only active/candidate pair with common RNG.

Example commands:

```powershell
.\venv\Scripts\python.exe scripts\run_phase11_current_source.py --run baselines\phase11\<run> acquire --league SA --attempt sa-qualified
.\venv\Scripts\python.exe scripts\run_phase11_current_source.py --run baselines\phase11\<run> acquire-fotmob-current --league SA --attempt sa-fotmob-qualified
.\venv\Scripts\python.exe scripts\run_phase11_current_source.py --run baselines\phase11\<run> qualify --league SA --source-attempt sa-qualified --fotmob-attempt sa-fotmob-qualified
.\venv\Scripts\python.exe scripts\run_phase11_current_source.py --run baselines\phase11\<run> freeze --league SA --qualification baselines\phase11\<run>\SA\assessments\assessment-001\qualification.json
.\venv\Scripts\python.exe scripts\run_phase11_current_source.py --run baselines\phase11\<run> collect-one --league SA
.\venv\Scripts\python.exe scripts\run_phase11_current_source.py --run baselines\phase11\<run> verify
```

The BL1 adapter handles the exact frozen identifier `M'GLADBACH` by substituting a collision-checked safe sentinel only inside feature grouping, then restoring the canonical label. The global sanitizer remains unchanged and continues to reject arbitrary apostrophes, path syntax, control characters, and other disallowed input.

If a source returns an error, stale or malformed payload, wrong league or season, changed identity, missing statistic, disagreement, or late retrieval, preserve the attempt and leave the league `DATA_BLOCKED`. Never convert failures to empty data and never construct a retrospective prediction pair.

The final status matrix permits only `PROSPECTIVE_COLLECTION_ACTIVE` or `DATA_BLOCKED`. Promotion remains a later, league-specific Phase 8 review after the frozen prospective horizon and settlement are complete.
