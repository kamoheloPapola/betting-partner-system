# Phase 20 protected-boundary authorization review

This review package is **pending**. It does not change
`phase20_settlement.integrity()`, the Phase 20 settlement authorization, or
the Local Operator Workflow. It does not grant settlement or append authority.

## Sealed proposal

`phase20-boundary-authorization-proposal-v4.json` is a write-once, self-sealed
review proposal (file SHA-256
`cc8c03f94eab524326bab134ea006dfaea5ef910199d6f1b44496998ac6a4e50`).
It binds the exact Phase 20 authorization and 19,176-file snapshot, Phase 19
pair index/counts/heads, the Phase 12 run specification and operations
configuration, the 181-file investigative inventory, the exact verifier source,
one named PD source-refresh run, starting journal heads, added event sequences
and kinds, permitted schema families, and each added file's path/hash/size.

It classifies the captured additions as follows:

| Class | Files | Meaning |
|---|---:|---|
| `AUTHORIZED_APPEND_ONLY_EVIDENCE` | 0 | No scientific append-only approval has been granted. |
| `QUARANTINED_PRESENT_EVIDENCE` | 2 | FL1 event 102 and report 33's failed FL1 outcome remain present and scientifically unresolved. Quarantine currently blocks boundary verification. |
| `UNFINISHED` | 2 | Report 34 intent and SA claim were unfinished at the inventory snapshot. |
| `UNAUTHORIZED` | 177 | Identified Phase 12 candidate evidence, but its coexistence with Phase 20 has not been scientifically authorized. This includes PD pair event 118. |
| `UNKNOWN` | 0 in the proposal | Any later unlisted file is unknown until separately classified. |

The `UNAUTHORIZED` label records the **current Phase 20 authority state**;
it is not a conclusion that the 177 files are forged or invalid Phase 12
evidence. The proposal fixes `decision_state: PENDING`,
`frozen_prefix_policy: EXACT_BYTES`, `unlisted_additions_policy: REJECT`,
`phase19_pair_policy: FROZEN_HEADS_ONLY`, and
`quarantined_may_coexist: false`. Every entry has
`scientific_clearance: false`, so neither file preservation nor a future
coexistence decision can clear the historical FL1 violation.

## Independent verifier

`src/evaluation/phase20_boundary.py` scans the original protected trees
independently, compares every frozen path's bytes to the snapshot, checks the
proposal's seal and exact references, validates listed file hashes/seals and
source references, checks source request scope, validates journal continuation
from the snapshotted heads, checks event types and declared sequences, and
checks report intent, chain, inventory, and completion. For any *approved*
pair, source, checkpoint, retrieval, or report addition it delegates scientific
replay to the existing Phase 12 cohort verifier; authorized reports also
require complete non-degraded outcomes, claim/head/count agreement, and
checkpoint replay. An approved non-violating incident can authorize the
recording of that incident as evidence, not a new Phase 19 pair or a model
decision.

An authorization can return `verified: true` only if a **new, separately
approved** sealed contract says `APPROVED`, its file SHA-256 matches an
independently supplied approval hash, every current addition is accounted
for, all required replay succeeds, no blocking class is populated, and any
quarantine coexistence is explicitly allowed. The verifier does not mint an
approval hash or modify the live Phase 20 gate. Example read-only inspection:

```powershell
.\venv\Scripts\python.exe scripts\verify_phase20_boundary.py
```

The live inspection after sealing the proposal returned `verified: false` with all
19,176 frozen files unchanged, zero authorized additions, 177 unauthorized
candidates, two quarantined files, and two unfinished files. It also detected
two **new unlisted** files from report 34: an SA checkpoint and an SA outcome.
This is the intended fail-closed result; the proposal is a timestamped
review of 181 additions, not a rule that absorbs future files.

## Scientific evidence still missing

The investigative inventory checked structure and references, not complete
Phase 12 scientific replay. A read-only attempt to replay the PD cohort was
interrupted after prolonged execution and produced no passing certificate;
its result is **incomplete**, not passing. Report 33 is degraded and its FL1
outcome says `SOURCE_BYTES_DRIFT`/failed replay. The cause and append
invocation of FL1 event 102 remain unresolved. Report 34 has gained files
since the captured inventory and must be inventoried under a new exact
package before any completion classification.

A later scientific decision must name each continuation it permits, supply
its independent approval hash, state whether quarantined historical evidence
may coexist with a passing *current-lineage* boundary, and keep Phase 12
protocol violations blocked. Only after that decision and a passing verifier
should a separate change connect this verifier to Phase 20 `integrity()`.

## Tests

`tests/unit/test_phase20_boundary.py` exercises an explicitly approved
append-only incident, approval-hash anchoring, frozen-file mutation/deletion,
arbitrary and unknown additions, invalid seals, journal chain and event-kind
failures, raw-reference and source-scope failures, sequence drift, incomplete
reports, historical quarantine, and unapproved candidates. The Phase 20
settlement and Local Operator tests also passed with the separate verifier
present. No settlement or prediction collection was run by this work.
