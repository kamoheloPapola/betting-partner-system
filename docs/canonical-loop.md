# Canonical local football loop — staged, not activated

The canonical loop is implemented separately from the legacy CSV, corner/card,
automatic-retraining and promotion routes. It is not yet the active production
system. The current E artifact fails the live identity/history adoption gates.
No scheduler was enabled and no original processed CSV was replaced.

## Current evidence and limits

The fixed E recipe was refitted for the original 22 folds / 8,332 fixtures.
All E prediction hashes matched the accepted run. E minus R log loss was
-0.0170577219, with the frozen bootstrap 95% interval
[-0.0214601715, -0.0125694467], passing the original 0.006 margin.
R predictions were reused only after all 17,241 unsealed input payloads matched
and the original prediction/lineage hashes were verified.

The shared canonical E17 feature code matched frozen E17 on every unsealed
fixture (tolerance 1e-12). This is formula parity, not proof of complete live
history. Historical case/name variants create multiple states per official ID;
other old clubs have no unique official match. Only 33 states were bound,
219 were left unbound. A partial seed is not eligible for activation.

The two sealed windows remain closed. Candidate 2024 deduplication and 2025
relabeling exist inside NEXT-38 only. The four ambiguous PD rows remain excluded,
counted and retained as opaque bytes. Nothing was repaired in place.

## Read-only operator commands

From the canonical workspace, using the existing virtual environment:

    .\venv\Scripts\python.exe -B scripts\run_canonical.py status
    .\venv\Scripts\python.exe -B scripts\run_canonical.py forecasts --league PL
    .\venv\Scripts\python.exe -B scripts\run_canonical.py serve

The dashboard binds 127.0.0.1:8765. It exposes read-only records and health.
It does not generate a forecast on demand. A missing database is shown as
NEEDS_SETUP; reading status does not create one.

The old Phase 19 operator is still historical evidence. The old API and scripts
are not converted or automatically launched by this implementation. Do not use
legacy fetch, CSV-prediction, cleanup or auto-promotion commands as the canonical
loop. Their retirement/cutover must follow the foundation gate, not precede it.

## Activation sequence after the foundation is complete

1. Complete the historical provider-ID crosswalk and continuous verified prior
   history, without opening sealed outcomes prematurely. Rebuild the fixed
   feature inputs and repeat the locked R/E evaluation if their identities change.
   No new model structure or tuning round is authorized by this implementation.
2. Produce a new model artifact and hash-pinned gate evidence. The model must pass
   both provider-identity and history-continuity gates. Editing a status label is
   not evidence. The current candidate deliberately fails check-model and adopt.
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
Registration was not run in this implementation pass.

## Data authority and failure behavior

Live authority is data/canonical/football.sqlite3. Historical CSVs and NEXT
artifacts remain evidence, not additional live prediction stores. Provider team
and match IDs carry identity. Only match metadata and full-time goals are kept;
referee/player names, odds, corners and cards are discarded.

The first forecast saved before a confirmed TIMED kickoff is insert-only.
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

Runs are idempotent. One SQLite write transaction covers validated business
changes; a failed fetch, parse, inference or write rolls it back. Failure records
and safe local notification attempts are separate. Requests are paced, bounded,
and retried finitely; auth/entitlement errors stop. No raw response, request
authorization header or secret is retained. Notification delivery is best effort,
not an attested visible alert.

Health turns red after 36 hours without success, on a failed run, missing model
or guardrail halt. Missing results older than 72 hours add a warning. The live
bug guard uses the latest 300 eligible settled forecasts and a frozen same-league
training-frequency 1X2 baseline. If mean natural-log loss is worse by more than
0.03, new forecasts halt while fetching/settlement continue. This is a bug guard,
not a profitability claim or high-precision selection test.

No automatic retraining, promotion, cleanup or structural-model search runs.

## Recovery and backups

    .\venv\Scripts\python.exe -B scripts\run_canonical.py backup "<new snapshot path>"
    .\venv\Scripts\python.exe -B scripts\run_canonical.py rollback --reason "<reviewed reason>"
    .\venv\Scripts\python.exe -B scripts\run_canonical.py resume --reason "<investigation outcome>"

backup creates an exclusive SQLite snapshot and checks its integrity. It is a
database backup only: retain adopted model JSON, its proof files, this versioned
source and the evidence packet alongside it. Paths in model references/proofs
are currently absolute; restoration to a different root is not automated or
qualified. Do not edit the immutable database references manually.

rollback can select only a prior validated artifact; original forecasts stay
unchanged. resume records the prior halt and reason, and the next run still
rechecks the guardrail. It does not change the threshold.

Keep backups outside the laptop. An evidence copy in a OneDrive folder proves
only a local copy until the owner verifies remote sync independently.

