# Phase 20 boundary scientific approval readiness

The independent Phase 20 boundary remains separate from the live Phase 20
settlement gate. No settlement, pair collection, scheduling, training, model
activation, or promotion was run in this review.

## Terminal replay

The sealed [read-only replay certificate](phase20-boundary-scientific-replay-20260923.json)
has SHA-256
`a16e8492fa243e344625a7396abdbc916b0c03c15789ae7fd5c35d2575c9b4f9`.
It binds proposal v6, the exact 167 candidate file identities, its replay
script, and the frozen Phase 12/11 verifier sources. It records three
`REPLAY_VERIFIED` packages:

| Package | Candidate files | Verification |
|---|---:|---|
| PD source acquisition | 161 | Frozen PD cohort and 76 raw provider sources replayed; all source metadata, indexes, qualification, current source contract, and assessment companions agree. |
| PD retrieval, journal, checkpoint | 5 | Frozen PD cohort replays 30 pairs, 16 settlements, and 118 ordered events; 52 checkpoints verify. The retrieval reparses, and events 117/118 and the added checkpoint agree. |
| BL1 checkpoint | 1 | Frozen BL1 cohort replays 3 pairs, 2 settlements, and 17 ordered events; 38 checkpoints verify, including the added one. |

The Phase 19 pair index still contains exactly 1,172 pairs, and PD Phase 12
event 118 is absent by both pair and event hash. Phase 20 obtains settlement
pairs from Phase 19 cohorts. Historical FL1 event 102 remains unresolved;
report 33 remains degraded; report 34 remains terminated and incomplete.

## Pending approval candidate

The sealed [v7 candidate](phase20-boundary-approval-candidate-v7.json) has
SHA-256
`b8dddc14f72f8b26aa3a1a9fa9e54f86b8ee3f23705f90ac94f25aba559835a4`.
It preserves `EXACT_BYTES`, `FROZEN_HEADS_ONLY`, and `REJECT` for unlisted
additions. The 183 exact additions are now 167
`AUTHORIZED_APPEND_ONLY_EVIDENCE`, 12
`QUARANTINED_HISTORICAL_EVIDENCE`, and 4
`TERMINATED_INCOMPLETE_EVIDENCE`. Quarantine and terminated coexistence are
explicitly proposed for the exact bound packages. Every entry retains
`scientific_clearance:false`; the two historical categories never gain pair,
settlement, evaluation, or promotion authority.

This is a **PENDING** candidate, not an issued approval. The replay
certificate supports scientific review; it is not an external approval hash.
The independent verifier reports `verified:false`, 19,176 protected existing
files, `changed:[]`, `missing:[]`, zero unauthorized/unknown/unfinished
additions, and `BOUNDARY_NOT_APPROVED`. The verifier itself has not changed
since v6. A reviewer must make the separate approval decision and provide an
independent exact hash before a new APPROVED contract can pass. Connecting a
passing independent verifier to the live Phase 20 gate is a later, separate
engineering change.

Focused boundary, settlement, and Local Operator tests: **40 passed**.
