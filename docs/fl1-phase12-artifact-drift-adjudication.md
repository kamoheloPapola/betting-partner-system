# FL1 Phase 12 `ARTIFACT_DRIFT` adjudication brief

Inspection date: 2026-09-23. This is a read-only classification of the existing
evidence. It does not amend the Phase 12 journal or grant scientific clearance.

## Classification

**Unresolved protocol violation.** FL1 Phase 12 journal event 102 is a valid,
chained incident record, but the available record does not identify the artifact
that failed its hash check or prove which authorized invocation appended it.
Current matching bytes do not establish what bytes were present at 10:01:33 UTC.
The Phase 12 FL1 blocker remains in force. No model promotion or incident
override follows from this review.

## Established facts

1. `baselines/phase12/phase12-prospective-20260908-v1/FL1/journal/00000102.json`
   was created at `2026-09-23T10:01:33.777597+00:00`. Its payload is
   `ARTIFACT_DRIFT`, `protocol_violation: true`, `fixture_id: null`, and
   `source: null`. The Phase 12 journal verifier accepts all 102 event seals,
   sequence numbers, and predecessor links. Event 102 follows event 101's hash.
2. The Phase 12 operations wrapper can append a source-free drift incident in
   `src/evaluation/phase12_operations.py:171` when a league operation raises a
   drift error. The incident format records only the error string, not the
   failing reference, expected hash, observed hash, or operation identity.
   This identifies a plausible code path, not the proven caller of event 102.
3. Operational report 32's FL1 claim began at journal count 99. Its outcome
   records the subsequent source-refresh incident and no error. Report 33 has
   an intent dated `2026-09-23T08:50:25.102193+00:00`; at inspection it has
   no FL1 claim or final report. Its FL1 outcome says `SOURCE_BYTES_DRIFT` and
   `replay: FAILED`. This is nearby evidence of a source-byte check failure,
   but it does not identify the source or bind that failure to event 102.
4. A current hash scan of 1,549 references reachable from FL1 Phase 12 journal
   pairs, their qualifications and source indexes found no mismatch. The
   original FL1 Phase 11 protocol, qualification, index, prefix pair, and
   indexed raw sources also match their recorded hashes. The Phase 19 protected
   integrity command currently reports 17,544 protected files unchanged.
   These are present-state checks, not a reconstruction of the failing bytes.

## Evidence required to close the cause and append-authority questions

- Find contemporaneous diagnostics or an immutable archive naming the exact
  failing path, expected hash, observed hash or missing-file condition, and the
  check that raised `ARTIFACT_DRIFT`. Do not infer that path from a later clean
  scan or from the separate `SOURCE_BYTES_DRIFT` outcome.
- Bind event 102 to a specific authorized invocation and its league lock/claim
  history, or document that this provenance cannot be recovered. A timestamp
  overlap alone is insufficient.
- Independently replay the relevant journal prefix and source evidence, then
  classify the scientific effect under the frozen Phase 12 rules. If the cause
  or append authority remains unprovable, retain the unresolved blocker and
  seek a separate scientific contract decision before any clearance.

The clean Phase 19 FL1 cohort and its independently replayed predictions are
separate evidence. This brief does not alter their records or authorize a new
collection cycle, scheduling, settlement, or promotion.
