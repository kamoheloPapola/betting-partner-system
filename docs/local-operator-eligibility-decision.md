# Local operator eligibility decision

Decision date: 2026-09-23. Scope: compose already authorized, local Phase 19
prediction inspection and Phase 20 manual settlement. This decision does not
authorize another prediction acquisition cycle, scheduling, model changes, or
promotion.

## Decision

The existing, independently verified Phase 19 predictions for BL1, SA, FL1,
and PD may be shown and audited in a personal-PC operator workflow. Their
pairs are within the separately authorized Phase 20 manual settlement scope,
but the **current settlement command is blocked** by its protected integrity
check. The older Phase 12 lineages remain visible and blocked where their
journals contain protocol violations.

This separation follows from the frozen lineage bindings, not from clearing
any historical incident. Phase 16 clean cohorts start with an empty pair
prefix and a new v2 protocol hash. Their specifications reference the old
Phase 12 **run specification** for provenance, but do not import its journal or
pairs. Phase 19 binds each prediction to a named clean cohort, frozen model
identities, qualified source evidence, causal feature replay, and the v2 rule.
Phase 20 binds settlement to the exact Phase 19 pair heads. The model and
source artifacts can be shared across lineages, so their present hashes and
replay must still verify whenever the current path is used.

| League | Current cohort | Frozen active model | Phase 19 pairs | Current eligibility | Historical Phase 12 state |
|---|---|---|---:|---|---|
| BL1 | `BL1-clean-001` | `phase9-active-BL1-afb0be84cd1b0717` | 270 | Inspect existing pairs; Phase 20 settlement currently blocked by integrity | Blocked: admitted kickoff revision, event 10 |
| SA | `SA-clean-001` | `phase9-active-SA-cb691070fc0455d7` | 330 | Inspect existing pairs; Phase 20 settlement currently blocked by integrity | Blocked: prediction timestamp mismatch, event 121 |
| FL1 | `FL1-clean-001` | `phase9-active-FL1-e4f44e1a47118c5b` | 261 | Inspect existing pairs; Phase 20 settlement currently blocked by integrity | Blocked and unresolved: `ARTIFACT_DRIFT`, event 102 |
| PD | `PD-clean-001` | `phase9-active-PD-2374d8f5823879ed` | 311 | Inspect existing pairs; Phase 20 settlement currently blocked by integrity | No protocol-violation event in the current journal; historical decision status not inferred |
| PL | `PL-clean-001` | `active-production-pl-goals-phase4-frozen` | 0 | Status and audit only; no prospective prediction to show or settle | No Phase 12 PL cohort |

The Phase 19 run report's five pair counts and journal heads match the live
Phase 19 status inspected for this decision. Its sealed offline verification
reports the four non-PL cohorts replayed. Phase 20 authorization binds the
same pair counts and heads, allows manual result acquisition and settlement,
and forbids pair creation, scheduling, and promotion. It currently reports zero
live settlements in every league. A live Phase 20 integrity run on this date
returned `verified: false`: 167 files were added under older Phase 11/12
baseline trees after the Phase 20 protected snapshot, with no snapshotted file
missing or changed. The additions include FL1 journal event 102 and PD journal
event 117. `phase20_settlement.settle` checks this integrity result before
publishing a settlement and therefore currently refuses settlement for every
league. These additions require a separate boundary decision; they must not
be deleted or silently accepted by the operator workflow.

PL had no qualified frozen causal feature rows for its Phase 19 future
fixtures; it must not be presented as prediction ready. A later PL pair would
also need the separately verified settlement adapter described in the Phase 20
contract.

## Required operator behavior

- `STATUS` and `AUDIT` show the current clean cohort beside historical Phase 12
  incidents. In particular, FL1's historical blocker remains `UNRESOLVED` and
  must never appear as cleared because the current cohort verifies.
- Prediction display reads only sealed Phase 19 pairs after checking its
  authorization, current journal heads, replay certificates, and protected
  integrity. Clearly distinguish a recorded pre-kickoff prediction from an
  unrecorded forecast or a fixture that has already kicked off.
- `SETTLE` exposes the current Phase 20 integrity refusal plainly. Once the
  protected-boundary discrepancy is resolved under a separate decision, it
  calls Phase 20's manual authorization, acquisition, verification, and
  settlement path for previously recorded pairs. It does not settle an
  unrecorded fixture or reuse Phase 12 results. Performance remains
  descriptive; promotion stays unavailable. The operator must inspect the
  integrity command's `verified` field: its current CLI returns exit code 0
  even when that field is `false`.
- A failed current-lineage verification blocks that league's current action
  and gives the operator the failing reason. It does not rewrite either
  lineage. A historical Phase 12 blocker stays visible even while an
  independently verified clean cohort is eligible for its bounded action.

## Boundary before Local Operator Workflow v1

Phase 19 authorized one controlled collection cycle. Its `collect` command
can resume that same archived qualification, not acquire a new one or serve as
a daily prediction command. Phase 20 authorizes only manual settlement of the
existing Phase 19 pairs. Therefore a workflow may implement status, inspection,
audit, and a thin Phase 20 manual-settlement wrapper that reports the current
integrity refusal. Actual settlement requires the protected-boundary issue to
be resolved without changing the frozen Phase 20 authorization in place.
The protected-boundary decision must account for any further append-only Phase
12 operations, or legitimate historical additions will keep tripping the
same global integrity gate.
An `acquire -> predict -> record` command for new days needs a separate,
explicitly frozen repeat-cycle scientific authorization. Local scheduling
needs its own later authorization. The operator interface must call the
scientific components and surface their refusal reasons; it must not copy
their feature, admission, or settlement logic.

Evidence: `docs/phase16-fixture-revision.md`, `docs/phase17-v2-readiness.md`,
`docs/phase18-pl-v2-readiness.md`, `docs/phase19-controlled-collection.md`,
`docs/phase20-controlled-settlement.md`, the Phase 19 and Phase 20 sealed run
reports, and the current sealed Phase 12 journals. FL1's incident assessment
is in `docs/fl1-phase12-artifact-drift-adjudication.md`.
