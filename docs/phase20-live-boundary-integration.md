# Phase 20 approved boundary in the live integrity gate

The live `phase20_settlement.integrity()` now delegates post-snapshot policy to
the independently approved `phase20_boundary.verify()` implementation. It
passes only `docs/phase20-boundary-approved-v1.json` and the fixed, externally
issued whole-file SHA-256
`1ed13b93c37bd18b17a52a256c4d8358abdb0269c8574e7e9f2fe44994b63876`.
It does not discover another approved file or derive its expected hash from
the contract. The approved replay certificate and its source references must
still bind exactly. A verifier error, false result, or broken replay-certificate
reference fails the live gate with the underlying reason.

The live result retains the independent verifier's classification counts and
adds the approved boundary identity. Local Operator `STATUS` and `AUDIT`
display the 12 quarantined historical files and 4 terminated-incomplete files
alongside the active boundary identity. FL1 event 102 remains shown as
`BLOCKED_PROTOCOL_VIOLATION / UNRESOLVED`; the current clean FL1 lineage is
reported separately.

Direct Phase 20 result acquisition now checks integrity before creating an
attempt or requesting provider data. Local Operator retains its integrity
preflight, then calls the same acquisition and settlement components. Phase 20
continues to take settlement pairs exclusively from the frozen Phase 19
cohorts; the approved historical PD Phase 12 event 118 is not a candidate.
Integrity checks themselves do not acquire results or publish settlements.

## Read-only verification on the current PC

- Independent boundary verifier: `verified:true`.
- Live Phase 20 integrity: `verified:true` with the same 19,176 protected
  existing files, 167 authorized append-only additions, 12 quarantined files,
  4 terminated-incomplete files, and zero changed, missing, unauthorized,
  unknown, or unfinished files.
- Local Operator status: BL1, SA, FL1, and PD show `ELIGIBLE`; PL shows
  `NO_RECORDED_PREDICTION`. The FL1 historical blocker remains visible.
- Live gate with no approved hash or an incorrect hash: `verified:false`,
  `BOUNDARY_APPROVAL_HASH_REQUIRED` in both cases.
- Focused boundary, settlement, and Local Operator suite: **51 passed**.
- Phase 19 pair index: **1,172**. Phase 20 acquisition attempts: **0**.
  Settlement journal entries: **0**.

`ELIGIBLE` means only that the protected boundary and current-lineage
preconditions permit a future manual settlement attempt. A genuinely finished
Phase 19-admitted fixture and authoritative post-match source are still needed
before Phase 20 can produce its first settlement. This integration performed
no acquisition or settlement.

The live check performs frozen scientific replay and can take several minutes
on this PC. Acquisition and settlement each recheck integrity before their
respective write steps.
