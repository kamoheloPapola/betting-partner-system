# Phase 19 controlled collection

Phase 19 is one authorized prospective collection cycle for `BL1`, `SA`,
`FL1`, `PD`, and `PL`. Its immutable authorization is
`baselines/phase19/phase19-controlled-collection-20260923-v1/collection_authorization.json`.
The Phase 16 v2 rule, Phase 18 PL endpoint, five cohort specifications, model
bundles, and production pointers are unchanged. Scheduling, settlement,
training, calibration, activation, and promotion remain unauthorized.

The cycle first archived one football-data.org 2026 match response per league.
The sandboxed connection attempts failed before receiving bytes and are retained
as `primary_matches.json`. A bounded authenticated recovery request, made with
network access, archived each exact response as `recovery_matches.raw` and
`recovery_matches.json`. Do not overwrite either attempt. A rerun recognizes
them and must not fetch or backdate another response.

For the four non-PL leagues, the qualified causal refresh is under
`baselines/phase11/phase19-qualified-refresh-20260923-v1`. This additive source
lineage uses the unchanged Phase 11 composition: football-data.org matches and
teams, no-www football-data.co.uk CSV corners, and the qualified FotMob gap
filler. The frozen cohort still binds to its original Phase 11 qualification;
each new pair points to its additional immutable, replayable qualification.
The refresh must be ready and its model, mapping, and feature schema identities
must equal the frozen contract before admission.

The non-PL collector is `src/evaluation/phase19_batch.py`. It considers every
fixture in each refreshed qualification's `upcoming` set. It checks the exact
fresh source event, horizon and pre-kickoff ordering, causal feature state,
active/candidate model records, common RNG, Phase 4 replay, and v2 revision
classification. It holds the league cohort lock and atomically publishes each
sealed pair event. A rejected fixture receives a sealed per-fixture reason
under the Phase 19 rejection ledger. A crash leaves a valid journal prefix;
the next run verifies the prefix and skips already admitted fixture identities.

For PL, `src/evaluation/phase19_pl_cycle.py` replays the fresh raw provider
snapshot through the Phase 17 adapter and tests every future fixture against
the frozen Phase 7P causal feature matrix. At this acquisition no fixture had
an outcome-free feature row in the fixed window, so PL legitimately admitted
zero pairs. Its 2027-01-01 UTC endpoint is unchanged. A later cycle may not
import an old Phase 7P prediction into the new cohort.

To inspect the archived input without network access:

```powershell
.\venv\Scripts\python.exe -c "from src.evaluation.phase19_authorization import verify; print(verify()['record_sha256'])"
.\venv\Scripts\python.exe -c "from src.evaluation.phase19_pl_cycle import verify; print(verify()['record_sha256'])"
.\venv\Scripts\python.exe scripts/run_phase19_collection.py status
```

To resume an interrupted non-PL batch against the **same archived
qualification**:

```powershell
.\venv\Scripts\python.exe scripts/run_phase19_collection.py collect --league BL1
```

Run once for each league separately; one league failure must not alter another
league's journal. The command never fetches, settles, or promotes. Do not
repeat qualification acquisition under a new timestamp to fill missed pairs.

After all batches finish, run the Phase 19 offline verifier and integrity check:

```powershell
.\venv\Scripts\python.exe scripts/run_phase19_collection.py verify
.\venv\Scripts\python.exe scripts/run_phase19_collection.py integrity
```

The offline verifier uses only sealed raw/source and journal evidence. It
requires every frozen `upcoming` fixture to be either paired or explicitly
rejected, and reconstructs every pair's causal features and prediction replay.
Phase 18's original empty-cohort verifier remains a historical readiness
check; it should be run against the empty precollection state, while Phase 19
verifies the live append-only state.

The sealed run is under
`baselines/phase19/phase19-controlled-collection-20260923-v1`. Its
`collection_authorization.json`, raw acquisition records, per-league verifier
records, pair and source indexes, rejection ledger, horizon report, test XML,
protected comparison, and `run_report.json` form the review package.
`PHASE19_IMPLEMENTATION_REPORT.md` is the human-readable status matrix.
No fixture in this cycle has a Phase 19 settlement; metrics and promotion
decisions therefore remain unavailable. In particular, PL's future fixtures
have no frozen causal feature rows and must remain explicit rejections, not
retrospective predictions.

For a fresh independent audit, run `verify` above. It replays every admitted
prediction and can take substantial time. The sealed report reuses the
per-league replay certificates only after checking each current journal head,
source hash, and qualification hash; changing a journal invalidates that
report. The protected and repository-wide regression selections explicitly
exclude six cases. Two are obsolete Phase 17/18 empty-cohort assertions. The
other four are Phase 18 initializer/authority parametrizations omitted because
each repeats a full live-cohort replay; they are not obsolete assertions.
The repository-wide XML covers the selected tests, with inherited failures
classified separately from new regressions. The exact exclusions are recorded
in the run report. An attempted literal all-tests run was stopped at 69%
during expensive historical regression work; it was incomplete, not passing.

Read `PHASE19_TEST_SCOPE_CORRECTION.md` and `artifact_index_addendum.json`
alongside the base run report. They clarify that two exclusions are obsolete
empty-cohort assertions; the four parametrized Phase 18 initializer cases
were omitted because they repeat full live-cohort replay. The interrupted
all-tests log does not identify its exact in-flight test.
