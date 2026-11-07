# Phase 9 multi-league integrity

The production league set comes from `src/config/leagues.py`, reconciled against
the registry, canonical builder divisions, feature pipeline and actual datasets.
The first run discovers PL, BL1, FL1, PD and SA. Global fallback artifacts are
registry fallback identities, not additional competitions.

Run the inventory before training or migration:

```powershell
venv/Scripts/python.exe scripts/run_phase9_multi_league.py inventory --run-dir baselines/phase9/<new-run>
venv/Scripts/python.exe scripts/run_phase9_multi_league.py assess --run-dir baselines/phase9/<new-run> --league BL1
venv/Scripts/python.exe scripts/run_phase9_multi_league.py qualify --run-dir baselines/phase9/<new-run> --league BL1
venv/Scripts/python.exe scripts/run_phase9_multi_league.py assess --run-dir baselines/phase9/<new-run> --all-production-leagues
venv/Scripts/python.exe scripts/run_phase9_multi_league.py qualify --run-dir baselines/phase9/<new-run> --all-production-leagues
venv/Scripts/python.exe scripts/run_phase9_multi_league.py prepare-matrix --run-dir baselines/phase9/<new-run>
```

Inventory and evidence records use exclusive creation. A failed qualification is
retained as a numbered attempt. A failed partial fit directory requires a new
evidence attempt; it is never overwritten. PL qualification is reference-only:
its Phase 6 candidate and continuing Phase 7P cohort are not retrained or reset.

`LeagueModelBundle` adds competition ownership around existing artifacts. It
does not alter their estimator classes, feature engineering or ensemble formula.
The existing `ffps.prediction_provenance.v1` and
`ffps.model_training_lineage.v1` records remain the execution and lineage
contracts. League-aware verification checks their expected bundle, ordered
schema and artifact SHA join. Identical schemas do not transfer trust.

Candidates use the Phase 6 fresh-estimator functions and unchanged predeclared
configuration. Finite feature eligibility is frozen without consulting outcomes;
excluded cold-start fixture IDs are recorded. Dated season cohorts determine
partitions independently per league. The latest two observed cohorts are held
out; the preceding cohort is calibration-only, the one before that validation,
and earlier cohorts training. Actual date ranges and fixture lists are frozen
before fitting. All fitting inputs and outputs remain in the league evidence
directory. Historical comparisons are diagnostic and cannot authorize activation.

Provider overlap is checked separately for each league on exact date and ordered
canonical home/away team joins. Scores and corners require exact discrete equality.
All compared fields retain raw-source hashes and normalized row indices. Historical
overlap is not evidence of current-season publication timing or team coverage.
The initial non-PL rollout therefore has no admitted prospective protocol.

`league_prospective.py` provides league-scoped append-only pair/settlement
operations and fixed-horizon final verification. Admission requires a separately
frozen, league-qualified current-data contract. A pair must be written before
kickoff; a settlement must reference that pair and a matching completed result.
The final gate compares the complete authoritative cohort with its export,
replays both predictions and recomputes the existing Phase 7 zero-margin rule.
No non-PL protocol or synthetic prospective fixture is created by this rollout.

Phase 8 retains its original transaction journal, locking, quiescence,
generation checks, complete-bundle switch and rollback. Non-PL operations require
both `--league` and `--league-specification` on the existing administrative CLI.
The specification binds the league, active/candidate identities, evaluation and
prospective contract. There is no mass-activation command. Until a league passes
its own prospective gate, activation is rejected before pointer mutation.

Production routing retains the existing manifest bootstrap until a league has
an explicit pointer. An activated league uses its own `active_<league>_bundle.json`,
history, pending journal and lock. Missing pointers with transition history,
cross-league references and unfinished transitions fail closed. Direct registry
mutation cannot replace the protected four-model bootstrap for any production
league. PL retains its original specification and pointer schema.

After tests and finalization, the matrix is verified entirely offline:

```powershell
venv/Scripts/python.exe scripts/run_phase9_multi_league.py finalize --run-dir baselines/phase9/<new-run> --focused tmp/phase9-focused.xml --protected tmp/phase9-protected.xml --full tmp/phase9-full.xml
venv/Scripts/python.exe scripts/run_phase9_multi_league.py verify --run-dir baselines/phase9/phase9-multi-league-20260905-v1
```

The verifier includes every currently configured production league, checks the
artifact index and protected originals, reconstructs fitting partitions, replays
repeat-model validation predictions and every historical paired simulation, and
recomputes the metrics and mechanical state matrix. Any missing league or failed
verification is returned explicitly with a nonzero exit code. Historical source
versions are resolved by exact hash from immutable archives; model and data
substitution is never allowed.
