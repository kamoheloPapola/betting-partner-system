# Phase 20 controlled prospective settlement

Phase 20 settles only existing Phase 19 admitted pairs. Its immutable
authorization, source archives, and per-league settlement journals live under
`baselines/phase20/phase20-controlled-settlement-20260923-v1`. The Phase 19
pair journals and production artifacts remain unchanged. A settlement binds
the Phase 19 pair event, exact provider response, final score, and Phase 12
metric contributions. The Phase 20 journal is the source of truth for settled
versus pending pairs.

The operator must explicitly freeze the manual authorization first:

```powershell
.\venv\Scripts\python.exe scripts/run_phase20_settlement.py authorize --authorize-settlement
```

For one league, archive a fresh football-data.org season response and settle
the archived source path returned by `acquire`:

```powershell
.\venv\Scripts\python.exe scripts/run_phase20_settlement.py acquire --league BL1 --authorize-settlement
.\venv\Scripts\python.exe scripts/run_phase20_settlement.py settle --league BL1 --source baselines/phase20/phase20-controlled-settlement-20260923-v1/BL1/acquisition/attempt-000001/matches.json --authorize-settlement
```

Use the same commands separately for SA, FL1, PD, and PL. Each attempt is
retained, including HTTP and connection failures. An interrupted attempt with
an orphan raw file is retained; the next invocation starts a new numbered
attempt. Settlement accepts only a `FINISHED` result with complete full-time
goals, exact provider fixture identity, unchanged teams, and unchanged kickoff.
Non-final matches remain pending. A corrected final score for an already
settled pair fails closed and its archived source and assessment remain for
audit; no accepted settlement is rewritten. A rerun of an identical final
result appends nothing. PL currently has zero admitted pairs and therefore
zero settlements. A later legitimate PL pair would require a separately
verified PL settlement adapter before settlement.

Audit without network access:

```powershell
.\venv\Scripts\python.exe scripts/run_phase20_settlement.py verify
.\venv\Scripts\python.exe scripts/run_phase20_settlement.py integrity
```

The verifier reconstructs every settlement from the sealed pair and archived
result, checks the journal chain, and reports pairs, settled, pending,
rejected-results, and descriptive active/candidate metrics. It always reports
`DECISION_EVIDENCE_INSUFFICIENT`. Scheduling, pair creation, model changes,
activation, and promotion are outside this authorization.

The initial implementation evidence is in `test_report.json`,
`test_scope_addendum.json`, and `run_report.json` under the Phase 20 run
directory. The initial state has no
post-match results and therefore no live settlements; the scoring path is
covered by focused tests, while a genuine settlement must still be observed
before treating prospective settlement as scientifically demonstrated.

Phase 19's test-scope correction remains authoritative. Of its six
deselections, two were obsolete empty-cohort assertions and four were Phase 18
initializer/authority parametrizations omitted to avoid repeating full
live-cohort replay. The literal all-tests attempt was interrupted and was
incomplete, not passing.

The repository-wide test command created 117 additive files in older Phase
11/12 baseline locations. Each was absent from the frozen Phase 20 protected
snapshot and created during that test run. Those test outputs were removed
after path and timestamp checks; the final protected-tree comparison passes.
