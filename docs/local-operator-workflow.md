# Local Operator Workflow v1

This is a personal-PC command for inspecting the existing Phase 19 predictions
and their evidence. It does not create predictions or grant scientific authority.
Run it from the repository root with the repository Python environment:

```powershell
.\venv\Scripts\python.exe scripts\local_operator.py status
.\venv\Scripts\python.exe scripts\local_operator.py predictions --league BL1
.\venv\Scripts\python.exe scripts\local_operator.py predictions --all
.\venv\Scripts\python.exe scripts\local_operator.py audit
.\venv\Scripts\python.exe scripts\local_operator.py audit --full-replay
.\venv\Scripts\python.exe scripts\local_operator.py performance
.\venv\Scripts\python.exe scripts\local_operator.py settle --league BL1
```

`--json` is available for each command. `predictions` defaults to the first 25
matching recorded pairs in kickoff order. Use `--league`, `--fixture-id`,
`--limit`, or `--all` to inspect more. Each row names its clean cohort, fixture,
kickoff, active and candidate 1X2 outputs, pair and event hashes, and whether
the fixture is upcoming, kicked off without an accepted result, or settled.
`PL` reports no admitted prospective predictions.

`status` checks the sealed Phase 19 authorization and report, current cohort
heads and stored replay certificates, Phase 19 protected integrity, the Phase
20 authorization and protected integrity, Phase 20 settlement journals, and
historical Phase 12 journals. It shows the historical blocker beside the
current clean cohort. `audit` adds the existing deeper source and settlement
verifiers; `audit --full-replay` reconstructs the Phase 19 predictions rather
than relying on their stored replay certificates. These checks can take time
because the frozen integrity code hashes many local files. `audit` exits with
a blocked status when any integrity or replay check fails, while still showing
the per-league results.

`settle --league` first checks the current Phase 19 evidence and Phase 20
authorization and integrity. If the Phase 20 integrity result is false, it
stops **before** result acquisition and prints added, changed, and missing
counts. It does not treat a CLI exit code of zero as proof of integrity. Only
when the separate protected-boundary decision has made the existing Phase 20
gate genuinely pass will the same command call Phase 20's authorized manual
acquisition and settlement functions. It never settles an unrecorded pair.

`performance` reads Phase 20's existing descriptive scoring result. With no
settled pairs it says `NO SETTLED EVIDENCE` and shows no measured metrics.
There is no new prediction, scheduling, training, activation, or promotion
command. Phase 19 permits only its already archived one-shot collection; a
repeat daily prediction cycle needs a separate scientific authorization.

The operational lineage decision is in
`docs/local-operator-eligibility-decision.md`. The unresolved FL1 historical
incident remains documented in
`docs/fl1-phase12-artifact-drift-adjudication.md`.
