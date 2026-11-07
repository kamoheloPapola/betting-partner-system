# Separate prospective collection

This manual lineage is rooted at `prospective_evidence/prospective-20260924-v1`.
It is outside the protected Phase 19 and Phase 20 evidence trees. It records
new active and candidate predictions for BL1, SA, FL1, and PD only. PL,
settlement, scheduling, activation, and promotion are unsupported.

Many fixtures are also in the frozen Phase 19 population. These are distinct
forecast vintages for the same eventual match, **not additional independent
fixture outcomes**. A later model decision must count each fixture outcome once
and analyze repeated forecasts separately.

The first version reuses the frozen Phase 19 qualified causal feature inputs and
the Phase 11 feature adapter. Each acquisition archives a fresh, authenticated
football-data.org season response. Before capture, the collection compares all
finished league results in that response with the qualified input snapshot. A
new or revised finished result blocks collection with
`NEW_LINEAGE_CAUSAL_INPUTS_STALE`. This rule is intentional: after the first
new league result, another source qualification is needed before later fixtures
can be collected. This workflow does not authorize a qualification refresh.
Stop the affected league and retain the exact diagnostic when inputs are stale.
Every individual feature row must also validate.

The original October 9–10, 2026 example window has expired. Never reuse it as
a current collection instruction. As of October 12, retained September 23
authenticated sources identify October 16–18 UTC as a provisional candidate
window only. Current eligibility, actual kickoff times, causal freshness and
pre-kickoff publication remain unestablished. See
`tmp/collection-preparation-20261012-01/window-evidence.json` and the refreshed
preparation brief in that directory. Select execution dates from a fresh
authenticated response after a separate explicit request authorizes live work.

Run from the repository root with the virtual environment active. Dates are
UTC fixture dates. Preparation and read-only preflight do not authorize live
acquisition or collection. Reuse the existing authorization; do not run
`authorize`, recreate it, or expand its contract in the date-refresh task.
Confirm it through the read-only status interface:

```powershell
.\venv\Scripts\python.exe scripts/run_prospective_collection_next.py status
```

Before any separately authorized live attempt, run the complete corrected
read-only preflight in the documented execution context, require all eight
verified child results, and compare protected state with the retained baseline.
Start with BL1. Continue FL1, PD and SA only as their independent gates permit.

For each league, archive fresh fixture data, then use the **returned** source
path to collect only the dates selected from validated source evidence. The
following is a future live-work example, not an instruction to run in preparation:

```powershell
$league = 'BL1'
$raw = & .\venv\Scripts\python.exe scripts/run_prospective_collection_next.py acquire --league $league
$acquisitionExit = $LASTEXITCODE
# Retain stdout, stderr and UTC start/completion times before parsing.
if ($acquisitionExit -ne 0) { throw 'Acquisition failed' }
$acq = $raw | ConvertFrom-Json
if ($acq.status -ne 'RETRIEVED' -or -not $acq.source.path) { throw 'Acquisition failed' }
$source = $acq.source.path
# Set dateFrom/dateTo only after inspecting the fresh source, exact UTC kickoff,
# existing qualified targets and all eligibility gates. Do not infer eligibility
# from a date range. The existing inclusive window limit is eight calendar days.
& .\venv\Scripts\python.exe scripts/run_prospective_collection_next.py collect --league $league --source $source --date-from $dateFrom --date-to $dateTo
$collectionExit = $LASTEXITCODE
$collectionCompletedUtc = [DateTimeOffset]::UtcNow.ToString('o')
if ($collectionExit -ne 0) { throw 'Collection blocked; retain diagnostic' }
& .\venv\Scripts\python.exe scripts/run_prospective_collection_next.py verify --league $league
$verificationExit = $LASTEXITCODE
if ($verificationExit -ne 0) { throw 'Offline verification failed' }
# Also require parsed verified:true; retain the result and native exit.
& .\venv\Scripts\python.exe scripts/local_operator.py new-predictions --league $league
$operatorExit = $LASTEXITCODE
if ($operatorExit -ne 0) { throw 'Local Operator check failed' }
```

Repeat `acquire` and `collect` separately for FL1, PD, and SA. A repeated
`collect` with the same source and date window adds no second pair for any
already recorded fixture. Require publication of every accepted pair before its
kickoff. Each pair binds the
frozen qualification and protocol, the fresh fixture source and identity, both
model outputs with their provenance, and an immutable journal sequence and hash.
Offline `verify` rebuilds causal features and replays each prediction. The
fresh source is validated again at collection and verification. Provider
corrections to already recorded fixture identity are rejected.

For each newly accepted fixture, compare parsed UTC retrieval, both prediction,
pair and journal timestamps with the actual source kickoff. Retain successful
collection completion before that kickoff as a publication upper bound;
timestamp fields alone do not prove when publication completed. Stop if that
proof cannot be established. Repeat collection with the identical source and
dates only after the first attempt passes its gates; require zero additional
accepted pairs and unchanged journal count/head, then rerun offline verification.
An exit of zero can include rejected fixtures: report accepted, rejected and
already-recorded IDs separately. Retain failures rather than calling them
successful collection. Finish with protected verification and before/after
comparison, preserving historical classifications and the original 1,172 pairs.

The Phase 20 settlement command accepts only the frozen Phase 19 pair
population. It must **never** be used for this new lineage. A separate future
settlement contract will join these pairs to post-match evidence.
