# Phase 12 prospective operations

Phase 12 runs the existing frozen SA, FL1, PD and BL1 experiments. It has no training, calibration, candidate replacement, activation, rollback or production-pointer command. PL Phase 7P remains separate and read-only.

The authoritative league cohort consists of the exact Phase 11 prefix referenced by the immutable Phase 12 run specification, followed by every entry in that league's ordered continuation journal. The original protocol, its cohort path, and its original predictions remain byte-identical. The continuation is necessary because the Phase 11 run is sealed, indexes its complete file inventory, and explicitly does not accept settlements. Never copy an original pair into a new prediction or select an evaluation subset.

## Normal operation

```powershell
.\venv\Scripts\python.exe scripts\run_phase12_prospective.py --run baselines\phase12\phase12-prospective-20260908-v1 run
.\venv\Scripts\python.exe scripts\run_phase12_prospective.py --run baselines\phase12\phase12-prospective-20260908-v1 status
.\venv\Scripts\python.exe scripts\run_phase12_prospective.py --run baselines\phase12\phase12-prospective-20260908-v1 verify
```

The `run` command checks each league independently, retrieves its authoritative current fixture population, settles admitted FINISHED fixtures, discovers future fixtures, refreshes the qualified source evidence, admits the earliest uncollected qualifying fixture, replays the complete cohort and emits an immutable checkpoint. The collection batch follows Phase 11's earliest-fixture policy. This is an operational batch size, not a new scientific horizon or sample threshold. Repeat operational passes to drain the queue; fixture metadata and sources are retrieved afresh each time. Multiple remaining fixtures at one kickoff require enough passes before that kickoff. No background schedule is installed by this implementation.

The frozen calendar endpoints are authoritative. The decision bootstrap uses each protocol's 10,000 resamples, seed 42 and 95% confidence level. Phase 7's log loss, multiclass Brier score, joint goal RMSE, Poisson deviance, top-label and classwise reliability, and zero-margin decision rule are unchanged. Metrics include every settled admitted pair; all pending pairs remain explicit. Incomplete horizons or pending outcomes cannot authorize a decision.

## Source evidence

Football-data.org owns fixture identity, kickoff, status and goals. The no-www HTTPS football-data.co.uk CSV remains the primary corner source. FotMob remains only the exactly qualified current corner gap-fill source. Source refresh calls the original Phase 11 acquisition and qualification functions in a fresh `baselines/phase11/phase12-acquisition-<id>` directory. Those directories contain source assessments, not new experimental protocols. Their references are recorded in the Phase 12 journal and sealed report. Refresh must preserve the frozen mappings, season, feature schema, active identity and candidate identity. No source qualification rule is redesigned or broadened.

Raw responses, including HTTP failures, are archived before interpretation. Credentials remain local and are never written into evidence. A transport failure is not an empty fixture list or a result. Fresh retrieval is required before every new pair. Post-match settlement requires a FINISHED primary event with matching league, season, stable fixture/team identifiers, home/away orientation and exact admitted kickoff. Source bytes and retrieval time are bound to the settlement.

SA, FL1 and PD's September 6 pairs may settle only from such post-match evidence. BL1's original September 11 18:30 UTC pair must remain pending until authoritative completion is retrieved.

## Revisions, blocking and recovery

Before admission, legitimate provider kickoff metadata may change. After admission, any kickoff change fails closed because the existing contract provides no rule permitting alteration of frozen prediction timing. The observation stays in the authoritative cohort, and the league is blocked for a scientific-contract decision. The same applies to identity/orientation disagreements and conflicting second results. Settlements are never overwritten or removed. Postponed, suspended, cancelled, abandoned, missing and stale-status fixtures remain unsettled with explicit incidents. A source outage blocks that operation while other leagues continue.

The read-only Phase 10 shadow controller verifies production registry identities and the Phase 9 bundles before inference and again before publication. An active pointer or artifact change blocks continuation: these protocols do not authorize mixed active generations. BL1 continues through the exact Phase 11 `M'GLADBACH` sentinel. The global sanitizer and feature semantics are unchanged.

Each league owns an OS advisory lock under its Phase 12 directory. Process termination releases the lock automatically; do not delete lock files. Publication uses the existing atomic hard-link, fsync and exclusive-create primitive. A journal entry contains its payload and seal in one atomic publication. A crash cannot publish an unsealed pair; an orphan `.tmp` is uncommitted and never admitted. If a process dies after publication but before a checkpoint, the next pass replays the sealed journal and resumes. Retrievals left by a crash remain archived; the next pass may retrieve a new response. Existing predictions are skipped, never regenerated. An equal result is idempotent; a conflicting result blocks. Journal sequence, previous hash, and every checkpoint head detect missing, reordered or removed committed records.

Do not manually repair corrupted records, remove unfavorable pairs, or change protocol/model bytes to unblock a league. Scientific ambiguity requires a separate contract decision.

## Sealed reporting and verification

After focused and protected tests pass, run the full suite through the existing feature-export isolation guard. The expected JUnit files are `tmp/phase12-focused.xml`, `tmp/phase12-protected.xml` and `tmp/phase12-full.xml`.

```powershell
.\venv\Scripts\python.exe scripts\run_phase12_prospective.py --run baselines\phase12\phase12-prospective-20260908-v1 finalize
.\venv\Scripts\python.exe scripts\run_phase12_prospective.py --run baselines\phase12\phase12-prospective-20260908-v1 verify
```

Each finalization creates a numbered immutable report package under `reports/report-NNN`, with status matrix, cohort/pair/settlement/source indexes, rejection ledger, horizon progress, metrics/reliability, readiness, protected hashes, indexed test results, implementation/run report and offline verification. Later operational passes append to the same authoritative cohort and preserve previous packages. Never create a new run merely to reset evidence progress. `prepare` is a one-time initialization, and refuses an existing directory.

Offline verification does not retrieve sources or regenerate predictions. It reparses archived source bytes, reconstructs causal features, verifies Phase 4 replay/common RNG/model identities, checks every original and continued pair, joins every settlement, recomputes all checkpoint metrics from their complete journal prefixes and verifies scientific states. It also verifies sealed artifact indexes, historical protected files, exact production-data/model inventories and PL Phase 7P. Any new unexplained test failure or replay/integrity failure blocks sealing.

The current Phase 9 deployment specifications lack a `prospective_contract` binding for Phase 8 activation. Phase 12 exposes this authorization gap but changes neither deployment specifications nor gates. `DECISION_READY` means the frozen scientific support condition has been met and an evidence package is available for the separate Phase 8 process; it never activates a model.
