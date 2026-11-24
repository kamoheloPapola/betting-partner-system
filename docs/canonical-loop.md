# Canonical local football loop — active since 2026-11-24

Production adopted the verified v6 serving model on 2026-11-24.
`BettingPartnerCanonical` runs daily at 18:00 SAST. The first automated run
exited 0 and recorded a successful heartbeat. The legacy Nightly task is
disabled. The canonical loop operates separately from the legacy CSV,
corner/card, automatic-retraining and promotion routes. [Adoption receipt](canonical-adoption-2026-11-24.json)

## Current evidence and limits

The fixed E recipe was refitted for the original 22 folds / 8,332 fixtures.
All E prediction hashes matched the accepted run. E minus R log loss was
-0.0170577219, with the frozen bootstrap 95% interval
[-0.0214601715, -0.0125694467], passing the original 0.006 margin.
R predictions were reused only after all 17,241 unsealed input payloads matched
and the original prediction/lineage hashes were verified.
The locked training replay excludes 2024/25 results. The serving seed uses admitted 2024/25 results for state updates under the [v6 bridge receipt](../data/canonical/serving/bridge-v6-2026-11-22/proofs/history_bridge/history-bridge.json).

The shared canonical E17 feature code matched frozen E17 on every unsealed
fixture (tolerance 1e-12). This is formula parity, not proof of complete live
history. Historical case/name variants create multiple states per official ID;
other old clubs have no unique official match. Only 33 states were bound,
219 were left unbound. A partial seed is not eligible for activation.

The bridge opened the final-test and confirmatory windows on November 22, 2026,
for feature and rating state construction. Those windows are no longer
untouched holdouts for the bridged checkpoint. The opening receipt and ledger
are under `tmp/canonical-history-bridge`. The adjacent reconstructed fixture-ID
manifest identifies the rows admitted by the pinned reader.
Candidate 2024 deduplication and 2025 relabeling exist inside NEXT-38 only.
The four ambiguous PD rows remain excluded, counted and retained as opaque bytes.

## Offline E17 history bridge

The bridge advances the frozen June 2, 2024 seed. It reads full-time goals from
authoritative CSV rows through the 2024/25 season, after exact reconciliation
with reviewed fixture IDs. For 2024/25, the provider archive supplies identity,
date and status. Starting in 2025/26, the provider archive supplies the goals.
Official roster and round-robin checks apply when the season's next July 1 is
on or before `through_exclusive` and its last fixture is after the inclusive
seed day. The bridge takes the last date from the pinned source manifest and
checks it against the rows. A 20-club league requires 38 matches per club, and
an 18-club league requires 34. At the November 15 cutoff, 2024/25 and 2025/26
require official coverage. The authoritative 2023/24 files end by the seed
day, so the receipt records their round-robin check by CSV label as a
non-blocking audit. For 2024/25, every CSV row must match exactly one provider
fixture by date and both official team IDs.

Beginning June 3, 2024, only `FINISHED` fixtures can advance state.
`provider.STATE_UPDATING_STATUS` supplies that rule to the bridge and the live
provider projection, history and settlement. In the bridge, `AWARDED` and
`CANCELLED` account for a scheduled pair without a state update or score
projection. Matching CSV rows get `excluded_awarded` or `excluded_cancelled`.
A provider-authoritative void is recorded directly from provider metadata.
The preflight, opening ledger and candidate receipt list each exclusion in
league, season, date and official team ID order. Completed season summaries
report the void count separately from the total accounted fixtures. An open
result such as `POSTPONED` or `SUSPENDED`, an unmatched replay CSV row, a
finished and void fixture for the same ordered pair, or a missing pair in a
completed season blocks the bridge.
Cold-start clubs begin with zero frozen matches and carry only state accumulated from admitted bridge replay.

### Accepted bridge debt

- **2026-11-22 — Pre–June 3, 2024 awarded fixtures.** The frozen seed has no
  recorded awarded-fixture disposition. The bridge applies the owner's
  [November 21 policy, formally approved November 22](../tmp/bridge-v6-approved-2026-11-22/formal-approval.json)
  from June 3 onward. A few fixtures may have been treated differently
  across the cutoff. Revisit when the original freeze can be audited or
  rebuilt with authoritative status evidence.
- **2026-11-16 — Espanyol.** The
  [owner addendum](../tmp/BL1-T3-offline-dependency-manifest-binding-20261027-01/NEXT-38l-binding-review-01/revision-02/review-addendum/espanyol-debt-addendum.json)
  retained the published `ESPANYOL` state. The June 4, 2023 fixture affected
  Almería under the original ID while the carried Espanyol state excludes
  it under `ESPANOL`. Revisit if Track B is regenerated or the frozen PD
  artifacts are reopened.
- **2026-11-16 — Seed-end stale threshold.** The
  [approved binding rule](../tmp/BL1-T3-offline-dependency-manifest-binding-20261027-01/NEXT-38k2-catalog-hardening-01/owner-decision-addendum.json)
  measures staleness at the common seed end and was retained in the
  [v6 approval](../tmp/bridge-v6-approved-2026-11-22/formal-approval.json).
  Parma cold-started after 1,183 days to re-entry. Levante and Schalke carried
  after 1,184 and 1,191 days. Keeping the common cutoff preserves the
  reviewed mapping. Revisit at the next owner-reviewed seed or stale-rule
  revision.

The bridge resolves each endpoint of pre-seed CSV rows independently. A
resolved endpoint remains evidence that the club played before the cutoff.
The receipt names unresolved endpoints with source, line, date and side.
An exact normalized match against an official ID, name or reviewed label in
the same league's tail, replay, verified rosters or post-seed captures blocks
preflight, including matches in captures after `through_exclusive`.

From the checkout root, pass the independently reviewed SHA-256 of the plan to
the preflight command:

    .\venv\Scripts\python.exe -B scripts\run_history_bridge.py preflight <plan-relative-path> --sha256 <reviewed-sha256> --identity-review-sha256 <approved-review-sha256> --approval-diff <diff-relative-path> --approval-diff-sha256 <diff-sha256>

The plan uses `schema_version: 1`. Its references use checkout-relative
`path` and `sha256` fields.

The runner verifies the separately pinned approval diff against the signed
draft. Only the plan status and the approved identity-review path and hash
can change. The approved identity review can add the reviewer, approval date
and formal sign-off reference and advance its status. Its reviewed data must
match the draft.

- `leagues`, `through_exclusive` and `observed_as_of` define coverage and
  the actual observation cutoff. Use `2026-11-15` to include November 14.
- `model`, `recipe`, `feature_implementation`, `bridge_reader`,
  `source_manifest` and `owner_grant` pin the frozen inputs and state-only
  opening grant. `offline_verification` pins a PASS report for the same
  bridge-reader, feature-implementation and provider-implementation digests.
  `recipe` pins Freeze C's `FREEZE_C.md` at SHA-256
  `12c3e64c7bdd76c05bbf1ef28fc846cef3a18bbfd34d437fb093c0162dbcbebe`.
  It contains the frozen E17 recipe and Elo rule. The Stage C
  `protocol-c.json` (`b65df9a9…`) is the experiment declaration.
  `history_bridge._plan` requires the exact `FROZEN_RECIPE` path and digest
  before checking the file. `run_bridge` checks the same file again after replay.
  CSV entries already carry `max_date`. For completed provider-authoritative
  seasons from 2025/26, add `provider_season_max_dates` to the same pinned
  source manifest. Each entry contains `league`, `season` and `max_date`.
  The date must match the latest fixture in that season's pinned capture.
- `identity_review` references the separately approved review record. It uses
  `schema_version: 4` and requires `blockers: []`. There was no persisted
  version 3 bridge record to carry forward.
  The initial v6 draft recorded the owner's wording in `owner_decisions` under
  `STATED_PENDING_FORMAL_APPROVAL`. The separate approved review used for the
  opening records `APPROVED_FOR_BRIDGE`, the reviewer, leagues, cutoff and matching
  model and source-manifest pins. It also pins `seed_appearance_sources` (the
  frozen input inventory with `canonical_inputs`) and `stale_rule.register`
  and `stale_rule.proposals` (the owner decision and cutoff evidence).
  Its `seed_bindings` contains reviewed
  `league`, `provider_id` and `seed_label` records. Every replaying ID needs
  one review record. The exact label binds one frozen state, including for
  clubs absent in 2024/25 that return later. Optional `seed_aliases` supplies
  additional reviewed identity labels. Choose the latest evidenced chain
  across distinct seasons. Overlapping or same-season chains block the
  bridge except for the recorded Espanyol fixture debt. For PD provider ID 80,
  the binding's `proposal_evidence` must name the pinned Espanyol debt addendum.
  The reader requires both evidenced chains. The excluded chain minus
  the carried chain must contain exactly the fixture and count in the owner
  decision. The carried label must equal the pinned published binding and its
  last seed appearance must be on or before the excluded chain's last
  appearance. Every other chain must satisfy the ordinary distinct-season
  rule, with the carried label latest among those remaining. The one-sided
  update remains in the frozen seed. Revisit the debt
  if Track B is regenerated or the frozen PD artifacts are reopened.
  A null label requires `finding: NO_USABLE_STATE` and a `reason` for
  each replaying ID. `NO_SEED_LABEL` requires no matching seed state and no
  resolved pre-seed appearance. `STALE_BEYOND_THRESHOLD` requires a verified
  latest chain ending strictly before the inclusive cutoff, currently
  2021-06-02. Record its `evidence` as `last_seed_appearance`, `threshold` and
  the appearance inventory's `source_sha256`. Preflight verifies those
  values from pinned metadata before projecting scores. A stale proposal
  flagged `REVIEWED_SEED_ALIAS_REQUIRED` also needs a reviewed `seed_aliases`
  entry to establish the state identity. The unapproved review sheet supplies
  proposals for owner review.
  `csv_labels` binds exact `league`, `season` and `label` to
  `provider_id` for 2023/24 and 2024/25. Those are the bridge's two
  authoritative CSV seasons. Earlier processed files supply frozen seed appearance
  evidence independently. From 2025/26, provider fixtures supply official IDs.
  `tail_fixtures` declares reviewed June 3–30 official fixture
  metadata by `id`, `league`, `day`, `home_id`, `away_id` and `status: FINISHED`,
  with optional `home_name` and `away_name`. Reviewed void tail fixtures
  can use `AWARDED` or `CANCELLED`. Use an empty list when there are
  no tail fixtures. The receipt's `tail_fixtures_by_league` includes a zero
  for each league with no admitted tail rows. Version 4 requires a
  `tail_expected_counts` entry for every league. The bridge compares those
  counts with the reconciled CSV metadata before any scores are projected.
  `completed_season_rosters` supplies `league`, `season` and `clubs`, an
  inventory of `provider_id` and `name`, for exactly the league-seasons that
  meet both official coverage conditions.
- `provider_sources` selects one capture `manifest` and `expected_fixtures`
  count for each `league` and `season`. The capture must reach the earlier
  of the bridge cutoff and the end of the selected season. Any unresolved
  result before the cutoff blocks the bridge. Each league's final requested
  season also requires a pinned older `gap_baseline` capture and its expected
  fixture count. Refreshed captures must account for all baseline fixture IDs
  before the cutoff with the same teams.

Generate the unapproved v6 review packet under `tmp` with
`scripts/build_canonical_mapping_review.py <packet> PD 81 --offline-verification <PASS-report>`.
The report must pin the current bridge reader, features and provider rule. The packet's plan
uses the reader's v1 plan envelope and pins the v4 identity draft and a
draft copy of the source manifest with provider season endings. Keep the
packet out of preflight until its evidence and identity record are reviewed.
The v6 bridge carried Barcelona's existing frozen state forward. The owner
approved the Barcelona live release on 2026-11-22, citing the bridge receipt's
SHA-256. [Pinned owner release](../data/canonical/serving/bridge-v6-2026-11-22/proofs/barcelona_release/barcelona-release.json)
The Espanyol fixture comparison documents both chains and the one-sided debt.
The generator and reader apply the same exception checks. The recipe and owner
evidence currently resolve to pinned files in untracked `tmp`. A fresh checkout
needs those exact files supplied separately.

Preflight checks the evidence and reconciliation without projecting goals or
writing files. Once preflight passes, the state-only opening uses the same plan
and digest:

    .\venv\Scripts\python.exe -B scripts\run_history_bridge.py build <plan-relative-path> --sha256 <reviewed-sha256> --identity-review-sha256 <approved-review-sha256> --approval-diff <diff-relative-path> --approval-diff-sha256 <diff-sha256> --operator <operator>

Build reserves the one-time opening in `tmp/canonical-history-bridge/access-ledger.json`
before projecting goals. A successful build writes `candidate-seed.json` and
`history-bridge.json` in that packet. An interrupted opening retains its ledger
and cannot be retried through this command. The candidate is for separate review
before any model admission. Its `identity_key: provider_id` means post-seed
state and result updates use official IDs. The reviewed CSV labels serve only
to reconcile source metadata. The command does not evaluate or retrain a model.

## Read-only operator commands

From the canonical workspace, using the existing virtual environment:

    .\venv\Scripts\python.exe -B scripts\run_canonical.py status
    .\venv\Scripts\python.exe -B scripts\run_canonical.py forecasts --league PL
    .\venv\Scripts\python.exe -B scripts\run_canonical.py serve

The dashboard binds 127.0.0.1:8765. It exposes read-only records and health.
It does not generate a forecast on demand. A missing database is shown as
NEEDS_SETUP; reading status does not create one.

The `team_mapping` section of `status` shows rejection reasons for forecast
candidates and post-seed history. Any rejection turns health at least yellow.
A rejected history row withholds its league for that run, and `withheld_leagues`
lists the league and reason codes. If the mapping fails to load or resolve,
the run still records results by
official fixture ID and settles existing forecasts. It returns
`SETTLEMENT_ONLY`, reports red health, and leaves forecast freshness unchanged.

The run fetches all five leagues for the seasons spanned by the adopted seed
cutoff and the current season, plus older seasons with unresolved frozen
forecasts. The results transaction commits first. Before writing forecasts,
the run checks each required payload and its fetch-completion timestamp.
A fixture is due three hours after kickoff, allowing time for play, stoppages
and publication. Due finished fixtures must be present in mapped history
with their scores. Awarded and cancelled fixtures are counted as exclusions
and postponed fixtures as deferrals. An unresolved or omitted due fixture
withholds forecasts for its league. Missing, empty or malformed payloads,
parser rejections, duplicate fixture IDs and regressions from `FINISHED`
block the whole run. Results commit first. Any withheld league keeps health
red and the alert names the affected leagues. The receipt lists each league's
written count, withheld candidates and the raw timing metadata for unresolved
fixtures. Earlier results already in the seed are excluded from feature
replay. The offline checkpoint remains a review artifact until adoption.

**2026-11-22 — Scope completeness per league.** Two rehearsal blocks on
provider statuses still marked `TIMED` past the margin prompted this decision.
Keep the three-hour margin. If the gate withholds forecasts more than twice
in the first four weeks of live running, review provider timing evidence and
revisit the margin and whether per-club scoping is justified.

### Unresolved-fixture recovery

Review the fixture ID and status in the blocked run. After confirming the
disposition, the owner can approve a JSON waiver for a manual retry.
The document uses `schema_version` 1 and `status` `APPROVED`. It records
`league`, `season`, `fixture_id`, `kickoff_utc`, `provider_status`,
`approved_by`, `reason`, `approved_utc` and `expires_utc`.
Use the exact fixture and status from the blocked run. Approval lasts at most
24 hours. It covers only the named fixture in that league. A second unresolved
fixture can still withhold its own league. Supply the document and its
separately recorded SHA-256 to the normal run command.

    python -B scripts/run_canonical.py run --fixture-waiver <waiver.json> --fixture-waiver-sha256 <sha256>

The waiver applies when the retry sees the same due fixture and unresolved
provider status within the approval interval. It records the reason, pin and
fetch timestamp in the run receipt, and health reports the use of the
waiver. The fixture contributes no result to feature history until an
ordinary run receives a valid result. Every other fixture and payload must
pass the normal checks. Provider parse failures, duplicates, missing data and
regressions from `FINISHED` still block. Keep the scheduled command on the
ordinary path and remove the waiver flags when the fixture resolves.
List waiver-affected forecasts with `SELECT fixture_id, run_id, waiver_sha256 FROM forecasts WHERE waiver_sha256 IS NOT NULL;`.

The legacy Nightly and daily workflows no longer invoke the old settlement or
drift writers. The tracked outcomes CSV and rolling drift JSON are historical
records. Production settlement and health come from canonical SQLite. The old
Phase 19 operator is historical evidence.

## Activation sequence after the foundation is complete

The owner-approved Barcelona mapping was published on 2026-11-22 with a
separately pinned approval diff. Production adopted v6 on 2026-11-24.
[Publication receipt](../data/canonical/mapping-publication-receipts/ca90b5da9cb846a895591be9b8742c27.json)

The Barcelona publication takes a separately pinned approval diff. Record
the signed sheet, pending decision and bridge receipt as `path` and
`sha256` references in the formal approval. Record `status: APPROVED`,
`approved_by` and `approved_on`. Derive the approved decision from the pending
decision by changing its status and adding the approver and date. The diff
uses `schema_version: 1`, `formal_approval`, `approved_decision` and `changes`.
Use `provider_team_mapping.release_decision_changes` to render the changes.
The publisher verifies the diff pin, re-derives the original baseline, and
publishes the exact approved candidate with a publication receipt.

    python -B -m src.canonical.provider_team_mapping publish --baseline-sha256 <baseline> --candidate-sha256 <candidate> --release-diff <diff-relative-path> --release-diff-sha256 <diff-sha256>

After publication, exercise the forecast gate on an isolated scratch database
with a due unfinished fixture, then with a clean fetch. Verify zero forecasts
and red health on the blocked run. Deliver a real test notification to the
operator and record their confirmation. Before activating production, create
a SQLite backup and open a restored copy in scratch to verify integrity and
the business records. Model rollback changes the active model. Restore is
the recovery path for incorrect insert-only forecast rows.

1. Complete the historical provider-ID crosswalk and continuous verified prior
   history, without opening sealed outcomes prematurely. Rebuild the fixed
   feature inputs and repeat the locked R/E evaluation if their identities change.
   No new model structure or tuning round is authorized by this implementation.
2. Produce a new model artifact and hash-pinned gate evidence. The model must pass
   both provider-identity and history-continuity gates. Editing a status label is
   not evidence. The v6 builder derives both labels after verifying the approved
   identity review and all ten completed league-seasons.
   Kamohelo Godfrey Papola approved separating training provenance from
   serving bindings effective 2026-11-22. The archived
   `model.training.source_manifest_sha256` identifies the locked replay's
   `candidate-manifest.json`. Adoption verifies that retrospective link.
   Use `model.serving.source_manifest_sha256` for the live source authority
   and `model.serving.team_mapping` for the published `mapping_sha256` and
   all four `accepted_input_sha256` pins. The `decision_register` input pin
   identifies the serving identity register. Adoption matches both serving
   proof digests to reviewed pins. The live consumer matches them to the
   corresponding `validation.foundation_proofs` references and reads those
   files by path and hash. Adoption verifies that the embedded seed hashes to
   the bridge receipt's candidate-seed digest. The bridge baseline mapping is
   a separately pinned foundation proof. A changed mapping also requires an
   independently pinned owner release citing that receipt and allowing exactly
   the Barcelona row transition. The mapping still has to pass the exact
   publication checks and contain no `OWNER_SEED_WINDOW_ONLY` bindings.
   Publication receipts qualify by mapping hash and reviewed inputs because the receipt ID adds no independent integrity guarantee.

   For v6, the serving source-authority digest is
   `4fa8fa0fd8c0021a94ad41be9ad2f20b4a9eaa954239907ed95a502a8f0249a3`,
   the derivative consumed under the approved bridge plan. The reviewed
   NEXT-38d base `73dd6951…` supplies the same file authority and dispositions.
   The derivative adds the base reference and five provider season-end dates.
   The [authority review](../data/canonical/serving/bridge-v6-2026-11-22/source-authority-review.json)
   records the candidate's exact diff and bridge approval chain. The
   [adoption receipt](canonical-adoption-2026-11-24.json) records the production
   launch decision for that artifact.
3. Inspect a model by its full artifact path and observed SHA-256:

       .\venv\Scripts\python.exe -B scripts\run_canonical.py check-model "<artifact>" --sha256 "<sha256>"

4. Record explicit adoption, then run once in the owner's private key-enabled
   terminal. Never paste the key in chat or a command argument:

       .\venv\Scripts\python.exe -B scripts\run_canonical.py adopt "<artifact>" --sha256 "<sha256>" --reason "<owner decision>"
       .\venv\Scripts\python.exe -B scripts\run_canonical.py run

   The key is read from FOOTBALL_DATA_API_KEY; no dotenv file is loaded. A missing
   key fails before creating the database or making a request. A rejected model
   fails before provider access. A complete five-league response is validated
   before a business-data transaction commits.
5. After a successful smoke run, optionally protect the launching terminal's key
   for the current Windows user and register one canonical daily task:

       .\venv\Scripts\python.exe -B scripts\run_canonical.py configure-key
       .\scripts\register_canonical_task.ps1 -DailyAt 18:00

   No key is included in scheduler arguments. DPAPI encryption is scoped to the
   Windows account; the encrypted file cannot serve as a portable key backup.
   configure-key refuses an existing file. Credential rotation needs deliberate
   operator handling; no automatic overwrite or secret deletion is implemented.

Registration uses an interactive, limited user token, catches up after sleep,
ignores overlapping invocations, and limits execution to 15 minutes. It refuses
to replace an existing task or register while the legacy task is enabled.
Registration completed on 2026-11-24. The first automated run at 08:27 SAST
returned exit code 0, recorded `SENT` with kind `SUCCESS`, and appeared in the
production check's event log. The permanent daily trigger is 18:00 SAST.

### External heartbeat

The temporary Healthchecks rehearsal passed on 2026-11-23. The five-minute
check with two minutes of grace logged the scratch success, a missing-key
failure with a null `run_id`, and recovery. It went down at 12:52 SAST after
seven minutes without a ping. The operator confirmed the failure email in
Inbox. The Healthchecks event log records recovery and the absence
transition. Use operator confirmation as evidence of inbox delivery. Both
successful receipts recorded `SENT` with kind `SUCCESS` while health showed
`MISSED_FORECASTS`.

Use a 24-hour period and three hours of grace for production. Route alerts to
the tested email integration. The service must accept an
HTTPS GET at its base success URL and at the same path with `/fail` appended,
as in the [Healthchecks Pinging API](https://healthchecks.io/docs/http_api/).
For Healthchecks, allow GET requests and use URL-based outcome detection.
The URL path carries the alert signal even if the service discards the body.
The rehearsal ping log displayed the JSON body for GET on all three pings.

The production base ping URL is set in the scheduled Windows user's
environment as `CANONICAL_HEARTBEAT_URL`. Keep the URL private. The canonical
CLI reads its process environment directly. Keep the value out of dotenv,
task arguments, logs, and chat. Check inheritance using the heartbeat
state in its actual run receipt. `scheduler-ready` also prints the masked
configuration state for the process that invokes it. On an error it returns
`NOT_READY`, the exception class, and a nonzero exit code. Canonical errors
also include their code. Configure a valid URL before registration. An absent
or invalid URL returns `HEARTBEAT_NOT_CONFIGURED`. A latest receipt with
failed delivery returns `HEARTBEAT_FAILED`. The runner validates HTTPS before
delivery. Unset, empty, whitespace-only, non-HTTPS and malformed values disable
manual-run delivery. A valid HTTPS URL enables it.

Before a scratch or restore-drill run, unset both heartbeat variables in
that PowerShell session.

    Remove-Item Env:CANONICAL_HEARTBEAT_URL,Env:CANONICAL_BACKUP_HEARTBEAT_URL -ErrorAction SilentlyContinue

The heartbeat uses the session URL regardless of the database passed to
`run`. A scratch run that inherits the production URL can reset the production
timer and mask a genuinely missed run, or send a false failure alert. A
dedicated heartbeat rehearsal may set a temporary check URL after clearing
the inherited value.

**2026-11-24 — Database-bound heartbeat guard.** The owner declined a code
guard that restricts the production heartbeat by database path. Scratch and
restore-drill sessions rely on the operator clearing the inherited URL.

The runner sends a success ping once the database commit and run receipt are
complete and the run is `SUCCESS` with no withheld leagues or whole-run block.
Withholding, a whole-run block, or a caught run exception sends a failure ping.
A process that stops before sending leaves the external check waiting for a
ping. This includes a task killed at its 15-minute limit. The production
absence deadline is 27 hours from the last received ping. The JSON body
contains only `run_id`, `state`, and `withheld_leagues`.
Startup errors before a run ID can be allocated send `null` as the run ID.
Unexpected exceptions that escape the CLI still attempt the heartbeat and
propagate with their original traceback. That path does not show a local toast.
The heartbeat uses the current run's outcome independently of `MISSED_FORECASTS`.

Each attempt has a 10-second deadline. A timeout, connection error, HTTP 429,
or 5xx response can trigger one retry. Both attempts carry the same run ID and
payload. A manual rerun gets a new run ID. A retry can produce a duplicate ping
if the service received the first request before its response was lost.

The receipt's `heartbeat` records `SENT`, `FAILED`, or `DISABLED` and kind
`SUCCESS` or `FAILURE`. A separate best-effort write adds this field after the
delivery attempt. If the process stops between delivery and that write, the
original run receipt stays committed without the `heartbeat` field and the
service may already have accepted the ping. If that write raises, the CLI
returns the original run state and exit code with `heartbeat_record_error_type`.
Status reports `HEARTBEAT_FAILED` or
`HEARTBEAT_NOT_CONFIGURED` as yellow warnings when the rest of health is green.
With no recorded delivery yet, status checks the current process environment.
A valid URL then reports `NOT_EVALUATED`. A delivery failure leaves the run
result, forecasts and exit code intact.

For the owner-attended rehearsal, use a temporary check with a five-minute
period, short grace, and a private terminal environment pointing only to that
check. Before a manual run, privately compare the end of the configured URL
against the intended check's URL. A test ping to the wrong check can reset its
timer or send a false alert. The first rehearsal attempt accidentally pinged
the default check. The operator retired that check and the rehearsal passed
against the temporary check. Run the canonical command against a scratch
database. A bad provider key can exercise the startup failure ping. Observe a
success ping, a failure ping and its alert, then another success to reset the
check. Withhold the next ping and observe the absence alert after the period
and grace. The scheduled task environment check follows adoption, a smoke run,
and registration.

For a code freeze, run the complete canonical pytest suite on the laptop
outside the sandbox. Windows sandbox loopback can stall API fixtures at
`socket.accept` while creating an asyncio socket pair. Report API tests
omitted from sandbox runs and use the laptop result as their verification.
Record SHA-256 source and test hashes alongside the exact command and result.
[Runbook checksum](canonical-loop.md.sha256) · [Freeze manifest addendum](canonical-freeze-addendum-2026-11-24.json)

## Data authority and failure behavior

Live authority is data/canonical/football.sqlite3. Historical CSVs and NEXT
artifacts remain evidence, not additional live prediction stores. Provider team
and match IDs carry identity. Only match metadata and full-time goals are kept;
referee/player names, odds, corners and cards are discarded.

The mapping gate applies before inference and when admitting post-seed results
to the E17 feature history. A rejected fixture remains available for result
ingestion and can still settle a forecast already frozen against its official
fixture ID. The current published mapping uses the owner-approved Barcelona
binding, and the completed bridge supplies Barcelona's verified state. The
2024-06-02 cutoff belonged to the prior seed-only binding.

A confirmed TIMED fixture qualifies for its first forecast when kickoff is
strictly after the UTC run start and at or before 48 hours from that instant.
The first forecast is insert-only.
SQLite triggers reject update, delete and replacement. It stores two expected
goals values plus immutable model, market-version and feature provenance.
Specific markets are derived using poisson-dc-raw-v1, with rho=-0.13 and bounded
numerical tail error. There are no confidence caps, H2H overlays or binary
calibrators. Invalid numerical/DC domains fail loudly rather than being clamped.

Results and fixture history append revisions. Cancelled/awarded fixtures are
void; pending/postponed fixtures are not settled. Rescheduling never regenerates
the official forecast. A kickoff corrected to before freeze invalidates its
evaluation eligibility. Result corrections remain auditable; as-of feature
queries cannot use a revision observed later.

Runs are idempotent. The first write transaction commits validated fixtures
and results. A second transaction rechecks the active model and freezes new
forecasts as a batch. If forecasting fails, the batch rolls back, the results
remain settled, and the run reports `SETTLEMENT_ONLY` with a failure exit code.
Requests are paced, bounded, and retried finitely; auth/entitlement errors stop.
No raw response, request
authorization header or secret is retained. Notification delivery is best effort,
not an attested visible alert.

Health turns red after 36 hours without success, on a failed run, missing model
or guardrail halt. Missing results older than 72 hours add a warning.
`missed_forecast_count` counts mappable fixtures with confirmed kickoffs
strictly after the latest model activation and at or before the status check.
A nonzero count adds `MISSED_FORECASTS` and turns otherwise green health yellow.

**2026-11-22, reviewed 2026-11-24 — Local Windows alert availability.** The
owner accepted desktop notifications for the solo setup. Windows pop-ups
require an active interactive session. Healthchecks detected the missed
check-in in the temporary rehearsal and provides external coverage when
the laptop is asleep, powered off, or offline. The production absence deadline
is 27 hours from the last received ping. Revisit the desktop notification
decision if the laptop regularly goes without an active interactive session.
The missed-forecast count uses fixtures already in SQLite and can grow as a
later run discovers missed fixtures.

The live bug guard uses the latest 300 eligible settled forecasts and a frozen same-league
training-frequency 1X2 baseline. If mean natural-log loss is worse by more than
0.03, new forecasts halt while fetching/settlement continue. This is a bug guard,
not a profitability claim or high-precision selection test.

No automatic retraining, promotion, cleanup or structural-model search runs.

## Recovery and backups

    .\venv\Scripts\python.exe -B scripts\run_canonical.py backup "<new snapshot path>"
    .\venv\Scripts\python.exe -B scripts\run_canonical.py rollback --reason "<reviewed reason>"
    .\venv\Scripts\python.exe -B scripts\run_canonical.py resume --reason "<investigation outcome>"

The backup command creates an exclusive SQLite snapshot and checks its
integrity. Back up the full versioned `data/canonical/serving/` bundle,
the published mapping and its receipts, this versioned source and the
evidence packet alongside the database. Retain the serving bundle through
cleanup and verify it against `bundle.sha256` before and after recovery.
The DB artifact reference and the artifact's proof paths are absolute.
Restore the bundle at the original absolute path to preserve its approved
hash and the DB reference. Do not edit the immutable database references
manually.

rollback can select only a prior validated artifact; original forecasts stay
unchanged. resume records the prior halt and reason, and the next run still
rechecks the guardrail. It does not change the threshold.

Keep backups outside the laptop. An evidence copy in a OneDrive folder proves
only a local copy until the owner verifies remote sync independently.

### Daily Drive snapshot and monthly restore drill

`BettingPartnerCanonicalBackup` is a separate Windows task scheduled for
18:30 SAST, after the 18:00 forecast. Its registration uses the current
interactive user, limited privileges, catch-up, a 15-minute limit, battery
operation, and overlap protection. Registration refuses to replace an
existing backup task.

The backup task stages a snapshot with the canonical `backup` command,
checks SQLite integrity, and hashes the logical SQL dump. A busy or timed-out
snapshot gets one retry with a fresh staging name. The task exclusively
creates a UTC-timestamped `canonical-db-*.sqlite3` under the project's Drive
`canonical-snapshots` folder, reopens it, and compares integrity and logical
hash before writing its `.sqlite3.json` sidecar. It retains the latest 14
verified pairs. A missing destination, copy collision, or failed verification
exits nonzero before pruning. Unpaired files and unrelated names are outside
retention.

Use a separate Healthchecks check with a 24-hour period, three hours of
grace, GET allowed, URL-based outcomes, and email alerts. Enter its URL
privately and set `CANONICAL_BACKUP_HEARTBEAT_URL` in the scheduled user's
environment. The backup sends one HTTPS ping with a 10-second deadline. A
failure ping uses `/fail` and carries only the failed step name. The script
reports a delivery failure with a nonzero exit. Keep both heartbeat URLs out
of scratch and restore-drill sessions.

Use these commands from a terminal with the appropriate environment. The
scheduled task uses `run --require-heartbeat`. For a manual copy with the
backup heartbeat unset, use `run`.

    .\venv\Scripts\python.exe -B scripts\run_canonical_backup.py run
    .\venv\Scripts\python.exe -B scripts\run_canonical_backup.py register

Once a month, download a snapshot and its sidecar from the Google Drive web
interface into the same local folder. Clear the two heartbeat variables,
then restore to a new scratch file. Record the result and logical hash with
the recovery evidence.

    .\venv\Scripts\python.exe -B scripts\run_canonical_backup.py restore-check "<downloaded snapshot>" "<new scratch database>"

The daily task verifies the copy on the Drive mount. The web download and
scratch restore exercise the remote copy independently.

