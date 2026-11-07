# Phase 20 boundary: degraded and terminated historical evidence

This decision refines the **independent** boundary review layer. It does not
modify `phase20_settlement.integrity()`, the frozen Phase 20 authorization,
the Phase 19 pair heads, historical journals, or the Local Operator Workflow.
Settlement remains blocked.

## PENDING proposal v6

The write-once `phase20-boundary-authorization-proposal-v6.json` has file
SHA-256 `06a951b930ae1d9bf88bc83d1bd893bf97153c5b714c69ca425427cef3c3eec6`.
It binds the stable 183-file post-snapshot inventory, exact file identities,
the frozen 19,176-file prefix, Phase 19 heads, the verifier source, and the
new scheduler provenance records. It is `PENDING`, grants **zero** append-only
authorizations, and fixes both coexistence flags to `false`.

| Classification | Files | Meaning |
|---|---:|---|
| `AUTHORIZED_APPEND_ONLY_EVIDENCE` | 0 | Requires explicit approval and existing Phase 12 scientific replay. |
| `QUARANTINED_HISTORICAL_EVIDENCE` | 12 | FL1 journal event 102 and all 11 post-snapshot report 33 files. The report is degraded and its FL1 outcome failed replay. Presence is preserved; scientific clearance remains false. |
| `TERMINATED_INCOMPLETE_EVIDENCE` | 4 | Report 34 intent, SA claim, SA outcome, and their associated SA checkpoint. The scheduled invocation timed out before final report publication. |
| `UNAUTHORIZED` | 167 | Known Phase 12 candidate evidence without Phase 20 scientific append-only approval, including PD source refresh and pair event 118. |
| `UNFINISHED` / `UNKNOWN` | 0 | A currently running or unattributed incomplete package, or a later unlisted file, would still block. |

The direct v6 verifier result was `verified: false`, `protected_existing:
19176`, `changed: []`, `missing: []`, `quarantined_historical_evidence: 12`,
`terminated_incomplete_evidence: 4`, `unauthorized_additions: 167`, and
`unknown_additions: 0`. The reported reason was
`BOUNDARY_APPROVAL_REQUIRED`. An approval hash, even if supplied, cannot make
this PENDING contract pass.

## Terminal provenance and limits

The proposal binds the [captured Task Scheduler events](phase12-report34-scheduler-events.json)
and [exported task definition](phase12-report34-task-definition.xml) by exact
file hash. The verifier parses the original event XML and checks event IDs,
record IDs, task name, one instance ID, engine PID, action, six-hour execution
limit, launch and timeout timestamps, and the currently disabled task state.
It then checks the exact report-34 intent, SA claim/outcome, and checkpoint
hashes and seals; requires their timestamps to fall within the invocation;
requires zero new pairs, settlements, or incidents; requires the SA journal
head to remain unchanged; and requires the report directory to contain exactly
the three declared files and **no final `report.json`**. Any extra or changed
file fails. The scheduler records identify the invocation and timeout, but
they are not a per-file OS write audit.

`TERMINATED_INCOMPLETE_EVIDENCE` can coexist with a passing boundary only if
a **new APPROVED contract** explicitly sets `terminated_may_coexist: true`
and every bound check passes. Coexistence never completes report 34 or clears
SA's historical protocol violation. The older `UNFINISHED` category remains
blocking for incomplete packages without this terminal provenance.

For report 33, the verifier checks FL1 event 102 as a sealed protocol
violation, the completed report as `degraded: true`, the FL1 outcome as
`replay: FAILED` with `SOURCE_BYTES_DRIFT`, and every added file in that
report directory as quarantined. It does not replay or approve the degraded
report as a clean operation. `QUARANTINED_HISTORICAL_EVIDENCE` can coexist
only if a new APPROVED contract explicitly sets
`quarantined_may_coexist: true`; every entry still carries
`scientific_clearance: false`.

## What remains before approval

The 167 candidate files have not been granted scientific authorization.
The earlier PD full-cohort replay attempt was interrupted and remains
**incomplete, not passing**. Any candidate promoted into
`AUTHORIZED_APPEND_ONLY_EVIDENCE` must pass the existing Phase 12 replay
required by the independent verifier. PD pair event 118 can only be approved
as historical Phase 12 evidence; the frozen Phase 19 heads continue to define
Phase 20 settlement eligibility. FL1 event 102 remains unresolved.

A separate scientific approval must issue a new sealed contract and an
independently supplied approval hash. Only after that contract verifies true
could a separate engineering change consider connecting the verifier to the
live Phase 20 integrity gate.

Focused boundary, Phase 20 settlement, and Local Operator tests: **38 passed**.
No historical evidence was edited or deleted, and no settlement was run.
