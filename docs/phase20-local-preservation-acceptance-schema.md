# Phase 20 local preservation acceptance schema v1

Implementation authority: `P20-LOCAL-ACCEPTANCE-SCHEMA-IMPLEMENTATION-20261020-01`.
Schema: `ffps.phase20_local_preservation_acceptance.v1`.

`src/evaluation/phase20_local_acceptance.py` validates supplied objects and bytes.
It contains no publisher, file writer, current-clock lookup, package reader, or
scientific integration. It does not create acceptance, start retention, or close
Phase 20. The approved future destination remains
`C:\Users\papol\my_systems\betting_partner_system\docs\phase20-local-preservation-acceptance-v1.json`.
Creation of that record is not authorized by this implementation authority.

## Exact contract

`_contract(recorded)` is the dedicated executable schema definition. Every object
has an exact field set; missing and unknown fields fail. Native scalar types are
checked before equality, so `16.0`, `true`, and `"16"` cannot substitute for `16`,
and `0`, `null`, and `"false"` cannot substitute for Boolean `false`.

Top-level fields are `schema_version`, `decision`, `recording`, `package`,
`verification`, `limitations`, `retention`, `destruction`, `boundaries`, and
`record_sha256`. The module specifies their exact nested fields and fixed values.

| Group | Contract |
| --- | --- |
| Decision | Identifier matching `P20-FORMAL-ACCEPTANCE-[A-Z0-9][A-Z0-9-]{0,95}`; `ACCEPT_WITH_DOCUMENTED_LIMITATIONS`; ISO owner-stated date; Kamohelo Papola / `FRIDAY\papol`; `OWNER_SELF_ATTESTATION`; independent authority verification `NOT_ESTABLISHED`; exact separate write-authorization reference and digest for completed input. |
| Recording | `recording.record_status` is `DRAFT_NOT_RECORDED` or `RECORDED`; timezone `Africa/Johannesburg`; acceptance-date source `ACTUAL_LOCAL_RECORDING_EVENT`. UTC timestamps require `Z` or `+00:00`, with optional 1–6 fractional digits. |
| Package | Fixed package `phase20-local-20261019T174318Z-fbd145f179a742358bbea7076c65112d`, its exact root and manifest digest `3312cd4f868f7bb9bcef26646ce9d971513d546ea332690c498cf7009818d014`, 16 files / 4,552,385 bytes, ten preserved, unresolved, unrestored discrepancies. These are `PREVIOUSLY_REPORTED`, not reverified by the validator. |
| Verification | Previously reported `CAPTURE_PASS` and `READ_ONLY_VERIFICATION_PASS`; exact `records\single-operator-verification.json` reference under that package; `SINGLE_OPERATOR_VERIFICATION`; independence and independent assurance `NOT_ESTABLISHED`. No receipt is opened. |
| Limitations | Accepted single-volume loss risk; backup and manifest anchoring `NOT_ESTABLISHED; DEFERRED`; encryption `UNVERIFIED; VERIFICATION DEFERRED`; strong immutability, disaster-recovery readiness, independent verification, and acceptance-record independent anchoring `NOT_ESTABLISHED`; privileged local access remains. |
| Retention | Formal P is the recording event's Johannesburg date; start=P; minimum end=P+12 calendar months; nonexistent target day clamps to the last valid target-month day; review date=minimum end; automatic deletion is Boolean `false`. |
| Destruction | Expiry is not authority. Investigation closure, elapsed minimum retention, and a separate explicit decision are all required. This record always says destruction is unauthorized and has no destruction-authorization reference. |
| Boundaries | Package and historical receipts remain unchanged; restoration, evidence deletion, scientific boundary amendment, scientific clearance, and closure remain unauthorized; Phase 20 remains `BLOCKED`. These are declared governance boundaries, not a filesystem audit. |

Drafts require null creation timestamp, P, retention start/end/review date,
write authorization, and seal, with retention `NOT_STARTED`. They reject recording
context. A valid draft returns `self_seal=NOT_APPLICABLE` and never returns
`record_status=RECORDED`. Owner-stated date and P have different sources but may
coincide; P is never inferred from the owner-stated date or historical receipts.

## Calling contract and trust boundary

`validate_record(value, context=...)` returns a `ValidationResult`. `valid` means
that the supplied data meets this contract; it is not proof that recording or
publication occurred. Failures have `valid=false`, no validated record status,
and a coded `P20_ACCEPTANCE_*` error. Missing external facts fail closed.

Completed input requires a separately established `RecordingContext` containing:

- Actual `event_utc` and `observed_at_utc`; the record timestamp must equal the
  event and cannot exceed the observation time. Both come from the trusted caller.
- A `WriteAuthorization` with exact reference, SHA-256, destination, operation
  `EXCLUSIVE_CREATE_ACCEPTANCE_RECORD`, schema identifier, and owner-stated
  decision date. The stated reference/digest/date must match this context. Path
  spelling is exact; no case folding, resolution, or normalization broadens scope.
- A `UniquenessCheck` with a `frozenset` of existing IDs, exact scope
  `PHASE20_LOCAL_PRESERVATION_ACCEPTANCE_DECISIONS`, Boolean `complete=true`, and
  a nonblank snapshot/reference identifier. The candidate ID must be absent.

These context objects are assertions from the caller, not authenticated evidence
minted by the module. Never reconstruct them from the candidate record. The caller
must authenticate authorization, establish the actual event, obtain a complete
authoritative uniqueness snapshot, and serialize the uniqueness check with later
publication. There is no ID reservation, database, or repository-wide uniqueness
claim. Revalidating an existing recorded decision requires an appropriate historical
pre-creation uniqueness context; the module does not invent that history.

The module's `ZoneInfo('Africa/Johannesburg')` lookup must succeed; otherwise
validation returns `P20_ACCEPTANCE_TIMEZONE_UNAVAILABLE`. No fixed-offset fallback
is used. No new dependency is introduced. Actual recording, publication failure
handling, collision checks, and event timing remain a separately authorized
integration responsibility. The existing temporary-file publisher is not invoked.

`destruction_prerequisites_met` tests the conjunction of three native Booleans.
It authenticates no separate decision, checks no live investigation or clock, and
authorizes no deletion, even when the policy conjunction is true.

## Canonicalization and integrity

Canonical bytes match `src/predictions/provenance.py` for native JSON: UTF-8,
sorted keys, `separators=(',', ':')`, `ensure_ascii=False`, `allow_nan=False`, no
BOM or trailing newline. No NumPy, date, path, tuple, set, or key coercions are
accepted. Finite floats can be serialized, but schema integer fields reject them.
The tests compare the exact AST-extracted repository canonical functions on this
native subset, without importing their application dependencies.

`internal_record_sha256` computes SHA-256 of canonical bytes after excluding
**only the top-level `record_sha256`**. It does not assign a seal or modify input.
Completed input requires an exact lower-case 64-hex seal and matching content;
drafts require a null seal. This follows the existing verifier convention; the
Phase 12 constructor must not be handed an already sealed object unchanged.

`parse_record` requires UTF-8 bytes, rejects duplicate keys at every JSON object
level, malformed JSON, nonfinite constants, BOM, and non-object roots. Passing an
already parsed dictionary to `validate_record` cannot recover duplicate-key
information discarded by another parser; use `validate_record_bytes` at a JSON
input boundary.

`validate_record_bytes` optionally compares SHA-256 of the **entire exact byte
sequence including the internal seal** against an externally supplied expected
digest. `require_external_digest=true` rejects omission. Otherwise omission is
reported as `NOT_CHECKED`, never `PASS`. Whitespace can preserve an internal seal
and change the whole-file digest; canonical bytes should be used for any future
authorized publication. Neither digest establishes independent anchoring, and
results always retain `independent_anchoring=NOT_ESTABLISHED`.

## Safe focused test execution

The unit file uses `unittest`, which the repository already uses, and creates only
synthetic in-memory records. Do not use its fixture authorization or dates as
real governance data. The repository pytest autouse fixtures create directories
and import broader application components, so the bounded review runs this file
directly through `unittest`, without pytest configuration or hooks.

Run the existing base Python with `-I -S -B` and an inline launcher that:

1. Installs an audit hook rejecting write-mode file opens, filesystem mutations,
   subprocess creation, and network connections.
2. Adds this repository to `sys.path` for its empty `src` package and dedicated
   validator; no other application modules are imported.
3. Calls `zoneinfo.reset_tzpath` with the existing
   `venv/Lib/site-packages/tzdata/zoneinfo` directory, allowing the standard-library
   loader to read timezone data without importing site startup code. This changes
   only the test process's lookup path, not files or dependencies.
4. Executes `tests/unit/test_phase20_local_acceptance.py` with `runpy.run_path`
   and `run_name='__main__'`. Bytecode, temporary fixtures, logs, reports, caches,
   and acceptance files are not written.

The tests cover every required completed-record field's absence, every defined
object's unknown fields, fixed-value drift, synthetic drafts and events, strict
types, authorization and uniqueness context, digest boundaries, calendar cases,
retention/review agreement, destruction combinations, and forbidden transitions.
No broader scientific or preservation operation is part of these tests.
