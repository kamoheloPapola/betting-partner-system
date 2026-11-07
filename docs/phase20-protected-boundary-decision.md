# Phase 20 protected-boundary inventory and decision record

Inspection window: 2026-09-23 17:32:36–17:33:33 UTC. This is an investigative
classification, not a change to Phase 20 authorization or its integrity gate.
The per-file manifest is [`phase20-protected-boundary-inventory.json`](phase20-protected-boundary-inventory.json);
the read-only reproducer is `scripts/inventory_phase20_additions.py`.

## Present decision

**Keep Phase 20 settlement blocked.** Its sealed authorization freezes an exact
19,176-file snapshot across `baselines`, `data`, and `src/ml/models`, excluding
the Phase 20 run itself. At the captured instant, all existing files matched
their frozen hashes and none was missing, but 181 files had been added. The
frozen `integrity()` rule rejects every addition. No Phase 20 authorization
declares a Phase 12 append-only exception. Structural validity under Phase 12
does not itself grant one. A separate direct call to the genuine Phase 20
`integrity()` gate returned `verified: false`, `protected_files: 19176`,
`changed: 0`, `missing: 0`, and `added: 181` after the inventory.

| Namespace | Added | Observed producer and authority | Boundary state |
|---|---:|---|---|
| `baselines/phase11/phase12-acquisition-8e664a6cd54640d8bc0ad9eaa2a8b4ad` | 161 | PD source refresh under Phase 12; 76 raw responses, 76 metadata records, two indexes, six assessment records, one run specification | Candidate Phase 12 evidence; Phase 20 continuation undeclared |
| Phase 12 journal | 3 | `phase12_prospective.append`; FL1 incident 102, PD incident 117, PD pair 118 | Chains and seals valid; FL1 incident unresolved; pair replay not established by this inventory |
| Phase 12 checkpoints | 2 | `phase12_prospective.write_checkpoint` for BL1 and PD | Seals and journal-head links valid; Phase 20 continuation undeclared |
| Phase 12 result retrieval | 2 | PD `matches.json` plus raw response under `phase12_operations.retrieve` | Metadata/raw reference valid; Phase 20 continuation undeclared |
| Phase 12 operational reports | 13 | Eleven completed `report-000033` files; `report-000034` intent and SA claim | Report 33 is degraded; report 34 is unfinished |

The scanner found no failed local seal, raw-reference, referenced-byte, or
journal-chain check among these additions. The full Phase 12 scientific replay
was **not** performed by this inventory; a valid seal/chain only establishes
preservation and ordering. Its 177 `candidate_phase12_append_only_evidence`
rows are *candidates*, not authorized Phase 20 additions. Two rows record
unresolved FL1 evidence; two belong to the unfinished report 34. No path was
outside an identified Phase 12 namespace at capture time. The manifest gives
each path, file SHA-256, size, producer, contractual basis, local checks, and
incoming evidence references. The scan detected no files arriving or
disappearing during its window; later scans may see a larger count.

## Sequence and lineage facts

- Phase 20 froze at **09:27:43 UTC**. Report 33's intent was created at
  **08:50:25 UTC** and was already in the protected snapshot. Its later
  completion added files after that snapshot. The completed report is
  `degraded: true`.
- FL1 journal event **102** was appended at **10:01:33 UTC** as
  `ARTIFACT_DRIFT`, `protocol_violation: true`, `source: null`. The 102-event
  chain and checkpoint references validate. The exact failing artifact,
  observed bytes, and append invocation remain unproved. Report 33's FL1
  outcome says `SOURCE_BYTES_DRIFT` and `replay: FAILED`; it does not bind
  those missing facts to event 102. Its status stays
  **`BLOCKED_PROTOCOL_VIOLATION / UNRESOLVED`**.
- PD journal event **117** at **10:37:12 UTC** records
  `SOURCE_REFRESH_STARTED` and references the exact new source-refresh run
  specification. Its raw acquisition metadata and indexes resolve by hash.
  PD event **118** at **11:00:22 UTC** is a new Phase 12 pair for
  `PD_football_data_org_564713`, kickoff 2026-10-17 16:30 UTC. Report 33's
  PD claim binds journal head 116, and its outcome records one new admission.
  This pair is outside the frozen Phase 19/20 pair heads and must never be
  treated as a Phase 20 settlement candidate merely because it exists.
- Report **34** has an intent dated **11:50:35 UTC** and an SA claim, but no
  sealed final report in this capture. Its invocation is incomplete; the
  manifest does not classify it as a finished operational package.
- The Phase 12 report key, claims, journal references, and source archives
  identify likely producing subsystems. They do not establish the operating
  system process identity or independent write-authority provenance for
  every append. In particular, they do not resolve FL1 event 102.

## Boundary contract needed before any code change

1. **Frozen prefix:** every originally snapshotted path must remain present
   with exactly its recorded bytes. A changed or missing path always fails.
2. **Enumerated continuation:** a separately sealed boundary decision must
   explicitly classify already-present files and name exact Phase 12 run IDs,
   journal/report namespaces,
   allowed schemas and event kinds, sequence ranges and initial heads,
   source-archive relationship, and verifier. It must verify raw bytes,
   references, source scope, report chain, and publication completeness.
   This investigation cannot retroactively certify an invocation merely
   from a matching path or seal.
3. **Scientific quarantine:** historical protocol-violation records stay
   visible and blocked in their own lineages. Any decision to tolerate an
   unresolved historical append as *present evidence* must say so explicitly
   and must not mark that lineage scientifically cleared. Phase 20 continues
   to bind only the frozen Phase 19 pair heads.
4. **Fail closed:** unknown namespace, unexpected event kind, invalid seal or
   sequence, broken reference, unfinished package where completion is
   required, or unclassified addition keeps `verified: false`. The output
   should separately count protected existing files, candidate/authorized
   append-only additions, unresolved additions, and unauthorized additions.

Do not edit `protected_before.json`, delete added evidence, or relax the
operator `SETTLE` preflight. A future implementation would require a separate
scientific authorization for the new boundary rule and focused tests proving
that all four categories fail or pass as declared. The current Local Operator
Workflow v1 needs no redesign; its settlement path will follow the genuine
Phase 20 result when such a rule is approved and independently verified.
