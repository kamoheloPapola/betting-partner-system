# Phase 20 publication integration prerequisites

Diagnostic implementation notes, 24 October 2026. This is not an acceptance
record, new wire contract, threat-model approval, qualification package, or witness.
FC_B01 remains BLOCKED. Production integration is NOT_INTEGRATED.

## Sources and authority

The current implementation request is the attachment at
`C:/Users/papol/.codex/attachments/df4d9ce1-7b78-4429-aba9-4bdb5157e1c4/Pasted text.txt`.
Its requirements prohibit changing contracts, budgets, historical evidence, or
completion status to overcome a missing prerequisite.

Source identifiers used below:

- S: exact assistant message, physical line 216, `payload.content[0].text`, in
  `C:/Users/papol/.codex/sessions/2026/10/21/rollout-2026-10-21T19-27-46-01a15f81-3897-7433-823d-058b7f3a70cb.jsonl`.
  Full message: 102812 UTF-8 bytes,
  SHA-256 `7373cad7327bb71a8dd02689bff5c7f6bd11c8a8b65efcfb8765678a04e1d39c`.
  D.6/D.12 define pending namespace assurance; D.13 contains proposed limits.
- T: exact assistant message, physical line 367, in
  `C:/Users/papol/.codex/sessions/2026/10/24/rollout-2026-10-24T10-51-31-01a16d1b-a8ac-7342-9703-1b553360d773.jsonl`.
  55442 UTF-8 bytes,
  SHA-256 `3a22ee5d3f1710657e738e9f70c8b430d31ef45610575102ad36936b84950b82`.
  R4.1 contains the guard-excluded S02 candidate ledger.
- R: exact assistant message, physical line 825, in the same October 24 file.
  52644 UTF-8 bytes,
  SHA-256 `04a362a827f1a0b490617287ae3af4525efc40914d9bcf0e1c58aac3015c1c6f`.
  RD02 leaves guard ownership/acquisition/release undefined; RD03 proposes logical
  loans; RD04 proposes post-stop eligibility; section 7.1 lists all 64 C entries.
- P/H/M: respectively `publisher-completion-design-v1.md`,
  `harness-implementation-plan-v1.md`, and `qualification-manifest-draft-v1.json`
  under `tmp/phase20-windows-qualification-preparation-20261020-01/`.
  Their sizes/digests are 7058/
  `a305c5430a55c8da17c2e95c3c0c5c64094d15da8a5371faaec2fd8bc1960848`,
  11841/`e90221435132d07c4e453f903ff70899998c8c1cfb3977a0bbedbf3c8e72f05d`,
  10271/`9f6ff0eed53035ec4e4b1e0bf63dba38658738900cc4d1d7ef34ff56f9e6c04d`.
- V: `src/evaluation/phase20_local_acceptance.py` and
  `docs/phase20-local-preservation-acceptance-schema.md`. This implemented v1
  schema validates supplied data; it does not authorize creating its destination.

All source identities above were directly rechecked from original files or exact
UTF-8 message strings without trimming or newline normalization. Hash agreement
identifies bytes, not approval, authenticity, or enforcement. S/T/R and P/H/M
remain proposed/unissued material; no adopted complete guard or v2 completion
contract was recovered. The scientific settlement authorization in
`docs/phase20-boundary-approved-v1.json` concerns a different boundary and is not
FC-B01 guard authorization. No source or historical artifact was rewritten.

## Reproduced baseline and implementation changes

Initial branch was `main`, HEAD `79110f9` (pipeline warning fix); preceding commits
were `a93c803` and `acc59de`. Many unrelated tracked modifications/deletions and
untracked scientific/evidence files were already present. The four publisher
source/test files were untracked. The staged diff was empty. Status inspection
used `git --no-optional-locks`; no staging, commit, reset, checkout, or cleanup ran.
No applicable AGENTS.md was found. This was not a full Git metadata audit.

Before changes, the documented isolated unittest method reproduced 198 passes:
166 acceptance, 16 controller, 16 injected Win32-binding tests; zero failures or
skips, process exit 0, 0.141 seconds. The inline launcher installed a write/process/
network-denying audit hook, added the repository to sys.path, selected the existing
venv timezone data, loaded exactly those three files with runpy.run_path, and
collected their unittest.TestCase classes. No pytest hooks ran.

Exact baseline command (before implementation changes):

```powershell
$phase20BaselineCode = @'
import os, sys, runpy, unittest, zoneinfo, json
from pathlib import Path
root = Path.cwd()
def audit(event, args):
    if event == 'open':
        mode, flags = args[1], args[2]
        if (isinstance(mode, str) and any(c in mode for c in 'wax+')) or (isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)):
            raise RuntimeError('test audit rejected file write')
    if event in {'os.remove', 'os.rename', 'os.rmdir', 'os.mkdir', 'os.link', 'os.symlink', 'os.chmod', 'os.truncate', 'subprocess.Popen', 'os.system', 'socket.connect', 'socket.bind'}:
        raise RuntimeError('test audit rejected effect: ' + event)
sys.addaudithook(audit)
sys.path.insert(0, str(root))
zoneinfo.reset_tzpath([str(root / 'venv/Lib/site-packages/tzdata/zoneinfo')])
suite = unittest.TestSuite()
counts = {}
for name in ('test_phase20_local_acceptance.py', 'test_phase20_publication.py', 'test_phase20_windows_io.py'):
    scope = runpy.run_path(str(root / 'tests/unit' / name))
    count = 0
    for obj in scope.values():
        if isinstance(obj, type) and issubclass(obj, unittest.TestCase) and obj is not unittest.TestCase:
            cases = unittest.defaultTestLoader.loadTestsFromTestCase(obj)
            count += cases.countTestCases()
            suite.addTests(cases)
    counts[name] = count
print(json.dumps({'python':sys.version,'platform':sys.platform,'tests':counts}), flush=True)
result = unittest.TextTestRunner(verbosity=1).run(suite)
sys.exit(0 if result.wasSuccessful() else 1)
'@
& 'C:\Users\papol\AppData\Local\Programs\Python\Python314\python.exe' -I -S -B -c $phase20BaselineCode
```

The baseline controller already invoked write_once once, but the adapter allowed
a second native write on the same open handle. The adapter now consumes one write
attempt before calling WriteFile, including short writes, API errors, and
exceptions. Flush requires a fully reported write and also consumes one attempt
before its native call. No repair write or flush retry is allowed. Known owned
handles can still be closed once under the controller's existing release policy.

The controller's failure-prefix tests now compare complete operation sequences,
and new tests cover post-validation payload/context mutation, adapter write/flush
retries, and acquisition attempts after uncertain close. Quarantine is held in the
adapter's process memory; it is not a durable restart ledger. A fresh process or
adapter is not evidence of safe prior release and does not permit replay.

`scripts/test_phase20_publication.py` provides a reproducible isolated runner. It
also denies subsequent DLL loading/symbol lookup, ensuring these tests use injected
API doubles. It writes no report, fixture, acceptance, cache, or evidence file.

Executed command (PowerShell, repository root):

```powershell
& 'C:\Users\papol\AppData\Local\Programs\Python\Python314\python.exe' -I -S -B scripts/test_phase20_publication.py
```

Observed environment: CPython 3.14.2, MSC v.1944, 64-bit AMD64, sys.platform win32.
These facts do not establish Windows edition/build, NTFS identity, effective
rights, filters, device mappings, or qualification applicability. M still records
unresolved filesystem/volume and effective-rights evidence.

Updated run: 204 tests, 166 acceptance + 18 controller + 20 binding tests; zero
failures/errors/skips, exit 0. These are unit tests with injected functions, not
native IO integration, race/security qualification, or end-to-end qualification.
The complete dedicated acceptance suite ran. The broader scientific pytest suite
was not run: it imports unrelated application components and its autouse fixtures
write repository state. No scientific source changed in this task.

## Required protection invariant and lifecycle

Requirement derived from S D.12, R RD02/RD04, P lines 19-44, H preservation/stopping
rules, and the controlling user dispositions: before a dependent namespace effect
is allowed, the intended volume/root, full ancestor chain, effective rights,
naming context, relevant leaves, and included capabilities must be bound to a
reviewed protection argument. Protection must cover every interval where dependent
effects remain possible, including writer close/reopen, partial output, failures,
and the required final handoff. Only intended publication writes are permitted;
seeded/control/partial bytes and entries must be preserved as required. Actual
quiescence and ownership must precede deliberate release. Unexpected holder loss
needs its own prevention/ordering argument. Evidence collection must not depend
circularly on protection already released.

This states the required property; it does not claim a mechanism implements it.
The operational threat artifact, authoritative disposition, exact included actors,
and final protection endpoint remain unresolved. The earlier single-operator
implementation scope does not silently waive this integration prerequisite.

| Boundary | Existing implementation | Missing premise / safe disposition |
|---|---|---|
| Trusted root and protection acquisition | No guard acquisition implementation | Bootstrap/root identity, traversal, rights and scope must be established before production creation. |
| Holder/resource identity | IO tokens are bound to their adapter instance | Tokens are not a process-security boundary or independent holder. RD02 actual guard ownership is undefined. |
| Payload preparation | Frozen canonical v1 bytes, external digest and caller context validated again at entry | Caller context does not authenticate filesystem permission, clock, or namespace protection. |
| CREATE_NEW | One exact target, exclusive create, refusal of existing entry | Ancestors can redirect resolution before the open. An unexpected path may already contain an empty new file before detection. No deletion is inferred. |
| Writer identity / one write / flush | Identity/path/type checks, one native write, exact count, one flush | Observation does not prevent ancestor/security/mapping changes. Short/unknown writes cannot be repaired. |
| Writer close then reader reopen | Ordered and separately gated | A real protection gap remains. No reader handoff or ancestor guard is invented. |
| Readback and close | Volume/file/path equality, length, bytes, EOF, digest; close once | These are observations, not interval-wide preservation or authoritative completion. |
| Stop/error/release | No later ordinary controller stage; separate release permission permits one known-owner close if none attempted | No close retry after uncertainty; capability retained. A Boolean callback is not authenticated external authority. |
| Holder/process loss | No holder, supervisor, job, worker, or restart protocol implemented | No quiescence inferred from cancellation, timeout, process disappearance, or OS eventual cleanup. |
| Final acceptance and protection release | Not connected | Required evidence/acknowledgement, actual-owner release and noncircular reporting remain BLOCKED. |

Supported Windows facilities were considered without selecting a guard:

- CreateFileW documents share compatibility and handle-scoped restrictions, but
  attribute/extended-attribute access is outside those share flags. Directory
  handles alone do not establish protection for every leaf and security change.
  [Parameters and directory handling](https://learn.microsoft.com/en-us/windows/win32/api/fileapi/nf-fileapi-createfilew).
- Object security is evaluated against object descriptors and effective caller
  capabilities; inheritance on creation is not proof of a closed leaf's safety.
  [Security descriptors and access rights](https://learn.microsoft.com/en-us/windows/win32/fileio/file-security-and-access-rights).
- NtCreateFile documents RootDirectory-relative naming. It does not supply the
  missing trusted starting handle or complete lifetime argument for this workflow.
  [ObjectAttributes and Remarks](https://learn.microsoft.com/en-us/windows/win32/api/winternl/nf-winternl-ntcreatefile).
- Jobs offer process association and termination facilities. No evidence here
  establishes that all dependent effects cease before guard loss; process/job
  rights, membership, existing capabilities, and completion ordering need proof.
  [Job management](https://learn.microsoft.com/en-us/windows/win32/procthread/job-objects).

No new guard, ACL, privilege, device mapping, child process, or race test was run.

## Completion transition assessment

These rows are investigation categories, not added protocol states. All local
results still have completion_status COMPLETION_UNRESOLVED. No acknowledgement
interface, receipt, replay rule, persistent quarantine, or recovery format is added.

| Category | Responsible component and entry evidence | Reversal / retry / duplicate / failure behavior |
|---|---|---|
| Payload validated | FrozenAcceptance plus unchanged V validator; exact canonical bytes, seal and independent expected digest | Immutable input; entry revalidation rejects mutation. Validation can be repeated without publication, but grants no write. |
| Publication attempted | Current gate and actual adapter create invocation | Not reversible by cleanup. Existing leaves refused, including identical bytes. Creation error may leave acquisition unresolved. No automatic retry. |
| Publication locally verified | Controller observes ordered write/count/flush/close/reopen/identity/length/bytes/EOF/close and final permission | Local flag only. No transition to authoritative acceptance or completion. |
| Publication outcome uncertain | Exception, insufficient observations, failed close, or unobserved acquisition/result | Preserve output and known capabilities. No repair/reuse/retry or invented observation. Independently authorized safe release only. |
| Authoritative acceptance verified | No established integration component or complete authenticated evidence contract | BLOCKED; local flags and v1 RECORDED data do not establish an actual accepted event. |
| Completion established | P proposes a verified evidence chain plus retained acknowledgement; S/R contain unadopted refinements | BLOCKED; exact final semantics, trust/channel bindings and issuance are unresolved. |
| Acknowledgement received/verified | No approved channel or retrieval/authentication implementation recovered | BLOCKED; missing, delayed, duplicated, replayed, contradictory or lost responses cannot be promoted to success. No guessed retry or idempotency mechanism. |
| Reconciliation required | Diagnostic assessment of retained original evidence | No executable recovery entry point. Restart cannot reconstruct a prior completion or authorize repeating an uncertain operation. |

Timeouts are not currently enforced by these synchronous primitives. Adding a
thread/process watchdog or post-stop cleanup would require the missing holder and
outstanding-IO ordering contract. Passing unit runtime is not evidence of native
deadline compliance or persistence through power/device failure.

## Budget reconciliation

No authoritative adopted complete v2 budget/measurement procedure was recovered.
The exact source S D.13 calls its values proposed; R adds proposed logical loans.
The following comparisons preserve those recorded ceilings and do not approve,
raise, reinterpret, or charge a new native operation to a closed abstract API.
Complete workflow compliance is NOT_VERIFIED.

| ID/source | Boundary / units / recorded ceiling | Existing allocation | Current helper contribution / headroom | Measurement and disposition |
|---|---|---|---|---|
| B1 S D.13 recorded_calls | Admitted active + terminal case / Calls / 128 | T R4.1: 113 abstract Calls, guard-excluded | One helper = 15 injected native calls vs 11 abstract lifecycle Calls; complete native-model mapping absent | Exact injected call list checked in binding test; whole workflow NOT_VERIFIED. |
| B2 S D.13 checks | Case including actions / checks / 128 | T: 116 abstract checks | Current permit callback is invoked at adapter-method boundaries, not before every native identity subquery | Authenticated native-call/check mapping NOT_VERIFIED. |
| B3 S D.13 gates | Case / gate containers / 32 | T: 14 scopes | No serialized gate evidence or guard/action/reporting allocation emitted | Headroom not established; NOT_VERIFIED. |
| B4 R RD03 / T R4.1 | Case / logical loans and groups / 256 each | Guard-excluded abstract calculation: 8*16+2+4*9=166 loans | Native identity has three result buffers; corresponding APIs/loan roles not reconciled | No discount or physical-alias guarantee; NOT_VERIFIED. |
| B5 S D.13 candidate_payload | One output / bytes / 65536 | Existing helper ceiling 65536 | No increase; input/read/write bounds enforced | Unit bounds PASS; complete request memory/serialization NOT_VERIFIED. |
| B6 S D.13 operation/contender | Record / bytes / 65536; example 16 Calls | 16*3072 + 16384 = 65536 | Helper observations are not serialized to these records | Complete representation NOT_VERIFIED; cannot assume remaining space. |
| B7 S D.13 request/context | Request 16777216; trust_context 524288 bytes | Source grouped context bound 502144 | No complete evidence request/context produced | Actual headroom unknown, NOT_VERIFIED. |
| B8 S D.13 durations_us | API 5000000; publication/ordinary case 30000000; two-contender/ack 60000000 | Detection budgets only, not durability or kill permission | Synchronous calls have no safe watchdog or completion deadline proof | Native timing/clock checks NOT_RUN; NOT_VERIFIED. |
| B9 S D.13 stop/clock | Post-stop 10000000; gate-to-call/cancel-observation 100000; sample age 1000000; uncertainty 1000 us; drift 100000 ppb | Requires trusted event/clock bindings | No authenticated clock/cancellation evidence in primitives | NOT_VERIFIED; no expiry or stop bypass inferred. |
| B10 S D.13 other caps | Remaining exact numeric maps and grouped byte ceilings | See source and index below | No complete workflow serialization or closure | NOT_VERIFIED. |

Concrete accounting conflict, conditional on substituting this native trace for
each corresponding T lifecycle: eight publications contribute 8*15=120 calls,
one losing create contributes 1, and four fresh reads contribute 4*8=32
(open + three identity queries + length + body + EOF + close). Total 153 calls,
25 above 128, before guards or additional supervision/reporting. T used
8*11+1+4*6=113. The difference is 40. If every additional native initiation
requires its own check, 116+40=156 checks, 28 above 128. This is an analytical
substitution, not an execution or a new valid wire ledger. The identity attribute
and final-path queries are not silently hidden inside a counted native call.
Removing checks, dropping required reads, changing a cap, or treating DLL/runtime
setup as free is not adopted. An exact supported operation model fitting existing
limits is required before integration. W8 is not adopted.

Exact remaining numeric field/value index from S D.13 (proposal, not an adopted
profile). The original six maps contain 85 fields. These are indexed for scope,
not copied into a runtime configuration:

```text
encoded_bytes:
 candidate_payload=65536 intent=262144 operation=65536 local_confirmation=32768
 observer=524288 contender=65536 terminal=32768 execution_grant=32768
 observer_grant=32768 synthetic_authority=16384 policy=32768 clock_profile=16384
 clock_assertion=4096 acknowledgement=16384 expectation=786432 trust_context=524288
 assessment=32768 core=18000 request=16777216 raw_blob=786432 external=65536
 fixture=65536 ordinary_string_utf8=4096
counts:
 candidates=2 records=512 clock_assertions=256 raw_blobs=64 fixture_inputs=16
 json_depth=24 object_members=64 array_items=512 ordinary_string_scalars=1024
 candidate_base64_characters=87384 gates=32 checks=128 recorded_calls=128
 auxiliary_calls=39 trace_states=11 reasons=64
durations_us:
 ordinary_case=30000000 two_contender_case=60000000 barrier_acquisition=30000000
 api_call=5000000 record_publication=30000000 acknowledgement=60000000
 poststop_observation=10000000 gate_to_call=100000 cancellation_observation=100000
clock:
 sample_age_us=1000000 event_uncertainty_us=1000 drift_ppb=100000
nested_bytes:
 reference=512 event=640 binding=768 producer=512 identity=384 outcome=432 call=3072
 check=1280 gate=23040 candidate_metadata=16384 candidate=104000 closure=230000
 runtime=4096 namespace=48000 scenario=24000 slot_bindings=16000
 runtime_observation=230000 blob_reference=192
group_bytes:
 intent_remainder=24000 expectation_remainder=32768 operation_remainder=16384
 contender_remainder=16384 observer_snapshots=16384 observer_remainder=16384
 local_total=24000 terminal_total=24000 grant_total=24000 clock_profile_total=12000
 clock_assertion_total=3000 acknowledgement_total=12000 context_gates=240000
 context_remainder=262144 diagnostics=8192 terminal_diagnostics=8192
```

## Provenance and witness inventory

Read-only bounded searches covered project docs/tmp/baselines/prospective_evidence/
src/tests text files up to 4 MiB and the three identified S/T/R/guard transcripts.
Three old pytest-temp paths were access denied; they were not bypassed. The search
is not proof of universal absence. Original file bytes were not copied, regenerated,
normalized, or replaced. Transcript reads do not constitute an audit of all file
metadata or independent authentication of the original events.

C06_STATUS remains SOURCE_RECOVERY_REQUIRED. S/T/R contain no C06 family.
Captured user direction at physical line 382 in
`C:/Users/papol/.codex/sessions/2026/10/24/rollout-2026-10-24T13-06-07-01a16d96-e32f-7092-8dab-86d5970a16c6.jsonl`
requires source recovery and does not select obsolescence. No recovery entry point,
failure conditions, retry permission, supersession, or deprecation was established.
The stable witness ID C-06 below is C01 INTENT/READBACK, not the missing C06 family.

Original final C04 CONTROL buffers were not recovered. R section 7.2 and retained
tool outputs at lines 558, 628, and 688 of its transcript contain construction and
identity summaries. All 42 retained RD tool outputs were inspected as source data;
none supplied recovered original buffers. Constructors were not executed. No
historical result was certified, no witness built, and no package captured.

The table is an index of source requirements, not acquired witness records. Every
row's requirement reference is R section 7.1; exact descriptions retain that source.
M = complete baseline/mutated CONTROL data, expected values, bindings, evidence,
and source-defined oracle; N = genuine native-failure observation plus M's relevant
protection/release evidence; P = original C04 serialized CONTROL bytes/provenance.
G = guard/resource baseline unresolved; N = native evidence unavailable; P = original
serialized provenance unresolved. NR means NOT_RECOVERED within the bounded search;
NV means NOT_VERIFIED. No row's diagnostic tests are substituted for a witness.

| ID | Source dimensions | Evidence | Artifact | Acquisition | Provenance | Verification | Blocker |
|---|---|---|---|---|---|---|---|
| C-01 | C01 INTENT x CREATE | M | NR | BLOCKED | NV | BLOCKED | G |
| C-02 | C01 INTENT x WRITE | M | NR | BLOCKED | NV | BLOCKED | G |
| C-03 | C01 INTENT x FLUSH | M | NR | BLOCKED | NV | BLOCKED | G |
| C-04 | C01 INTENT x CLOSE | M | NR | BLOCKED | NV | BLOCKED | G |
| C-05 | C01 INTENT x REOPEN | M | NR | BLOCKED | NV | BLOCKED | G |
| C-06 | C01 INTENT x READBACK | M | NR | BLOCKED | NV | BLOCKED | G |
| C-07 | C01 OPERATION x CREATE | M | NR | BLOCKED | NV | BLOCKED | G |
| C-08 | C01 OPERATION x WRITE | M | NR | BLOCKED | NV | BLOCKED | G |
| C-09 | C01 OPERATION x FLUSH | M | NR | BLOCKED | NV | BLOCKED | G |
| C-10 | C01 OPERATION x CLOSE | M | NR | BLOCKED | NV | BLOCKED | G |
| C-11 | C01 OPERATION x REOPEN | M | NR | BLOCKED | NV | BLOCKED | G |
| C-12 | C01 OPERATION x READBACK | M | NR | BLOCKED | NV | BLOCKED | G |
| C-13 | C01 LOCAL x CREATE | M | NR | BLOCKED | NV | BLOCKED | G |
| C-14 | C01 LOCAL x WRITE | M | NR | BLOCKED | NV | BLOCKED | G |
| C-15 | C01 LOCAL x FLUSH | M | NR | BLOCKED | NV | BLOCKED | G |
| C-16 | C01 LOCAL x CLOSE | M | NR | BLOCKED | NV | BLOCKED | G |
| C-17 | C01 LOCAL x REOPEN | M | NR | BLOCKED | NV | BLOCKED | G |
| C-18 | C01 LOCAL x READBACK | M | NR | BLOCKED | NV | BLOCKED | G |
| C-19 | C01 OBSERVER x CREATE | M | NR | BLOCKED | NV | BLOCKED | G |
| C-20 | C01 OBSERVER x WRITE | M | NR | BLOCKED | NV | BLOCKED | G |
| C-21 | C01 OBSERVER x FLUSH | M | NR | BLOCKED | NV | BLOCKED | G |
| C-22 | C01 OBSERVER x CLOSE | M | NR | BLOCKED | NV | BLOCKED | G |
| C-23 | C01 OBSERVER x REOPEN | M | NR | BLOCKED | NV | BLOCKED | G |
| C-24 | C01 OBSERVER x READBACK | M | NR | BLOCKED | NV | BLOCKED | G |
| C-25 | C01 TERMINAL x CREATE | M | NR | BLOCKED | NV | BLOCKED | G |
| C-26 | C01 TERMINAL x WRITE | M | NR | BLOCKED | NV | BLOCKED | G |
| C-27 | C01 TERMINAL x FLUSH | M | NR | BLOCKED | NV | BLOCKED | G |
| C-28 | C01 TERMINAL x CLOSE | M | NR | BLOCKED | NV | BLOCKED | G |
| C-29 | C01 TERMINAL x REOPEN | M | NR | BLOCKED | NV | BLOCKED | G |
| C-30 | C01 TERMINAL x READBACK | M | NR | BLOCKED | NV | BLOCKED | G |
| C-31 | C01 CONTENDER_1 x CREATE | M | NR | BLOCKED | NV | BLOCKED | G |
| C-32 | C01 CONTENDER_1 x WRITE | M | NR | BLOCKED | NV | BLOCKED | G |
| C-33 | C01 CONTENDER_1 x FLUSH | M | NR | BLOCKED | NV | BLOCKED | G |
| C-34 | C01 CONTENDER_1 x CLOSE | M | NR | BLOCKED | NV | BLOCKED | G |
| C-35 | C01 CONTENDER_1 x REOPEN | M | NR | BLOCKED | NV | BLOCKED | G |
| C-36 | C01 CONTENDER_1 x READBACK | M | NR | BLOCKED | NV | BLOCKED | G |
| C-37 | C01 CONTENDER_2 x CREATE | M | NR | BLOCKED | NV | BLOCKED | G |
| C-38 | C01 CONTENDER_2 x WRITE | M | NR | BLOCKED | NV | BLOCKED | G |
| C-39 | C01 CONTENDER_2 x FLUSH | M | NR | BLOCKED | NV | BLOCKED | G |
| C-40 | C01 CONTENDER_2 x CLOSE | M | NR | BLOCKED | NV | BLOCKED | G |
| C-41 | C01 CONTENDER_2 x REOPEN | M | NR | BLOCKED | NV | BLOCKED | G |
| C-42 | C01 CONTENDER_2 x READBACK | M | NR | BLOCKED | NV | BLOCKED | G |
| C-43 | C02 MISSING | M | NR | BLOCKED | NV | BLOCKED | G |
| C-44 | C02 FORGED | M | NR | BLOCKED | NV | BLOCKED | G |
| C-45 | C02 REPLAYED | M | NR | BLOCKED | NV | BLOCKED | G |
| C-46 | C02 CONFLICTING | M | NR | BLOCKED | NV | BLOCKED | G |
| C-47 | C02 MISMATCHED | M | NR | BLOCKED | NV | BLOCKED | G |
| C-48 | C02 LATE | M | NR | BLOCKED | NV | BLOCKED | G |
| C-49 | C02 UNVERIFIABLE | M | NR | BLOCKED | NV | BLOCKED | G |
| C-50 | C02 SOURCE_UNAVAILABLE | M | NR | BLOCKED | NV | BLOCKED | G |
| C-51 | C03 OBSERVER_FAILED | M | NR | BLOCKED | NV | BLOCKED | G |
| C-52 | C03 OBSERVER_REVOKED | M | NR | BLOCKED | NV | BLOCKED | G |
| C-53 | C03 PINS_LOST | M | NR | BLOCKED | NV | BLOCKED | G |
| C-54 | C03 RUNTIME_DRIFT | M | NR | BLOCKED | NV | BLOCKED | G |
| C-55 | C03 WORKER_UNKNOWN | M | NR | BLOCKED | NV | BLOCKED | G |
| C-56 | C03 IO_IN_FLIGHT | M | NR | BLOCKED | NV | BLOCKED | G |
| C-57 | C04 CROSS_ATTEMPT | P | NR | NOT_RECOVERED | NV | NOT_VERIFIED | P |
| C-58 | C04 REORDERED | P | NR | NOT_RECOVERED | NV | NOT_VERIFIED | P |
| C-59 | C04 DUPLICATE_ID | P | NR | NOT_RECOVERED | NV | NOT_VERIFIED | P |
| C-60 | C04 SELF_CONSISTENT_UNAUTHENTICATED | P | NR | NOT_RECOVERED | NV | NOT_VERIFIED | P |
| C-61 | C05 CLOSE_FAILURE | N | NR | BLOCKED | NV | BLOCKED | G/N |
| C-62 | C05 REOPEN_FAILURE | N | NR | BLOCKED | NV | BLOCKED | G/N |
| C-63 | C05 READ_FAILURE | N | NR | BLOCKED | NV | BLOCKED | G/N |
| C-64 | C05 FLUSH_FAILURE | N | NR | BLOCKED | NV | BLOCKED | G/N |

Reconciliation: 42 C01 + 8 C02 + 6 C03 + 4 C04 + 4 C05 = 64 distinct source rows.
The 60 outstanding rows are C-01..C-56 and C-61..C-64. The other four are historical
C04 construction claims with unavailable original buffers; none is independently
verified here. Models are derived inputs, not 64 independent native executions.
Each requires its own source-defined oracle and provenance. No C06 row is inferred.

## Integration blocker and owner-decision docket

| Blocker | Established requirement / current result | Minimum missing input |
|---|---|---|
| Namespace/leaf protection | Required; not implemented as continuous protection | Exact threat artifact/disposition, environment/capabilities and end-to-end mechanism proof. |
| Guard lifecycle | Required; RD02 undefined | Bootstrap, actual owner, acquisition/release and reporting lifetime. |
| Holder-failure ordering | Required; no holder implementation | Enforced dependent-quiescence ordering or an explicitly approved claim limitation. |
| Workflow budget | Required; NOT_VERIFIED; conditional 153/156 conflicts above | Adopted exact operation/evidence model within unchanged limits, including adverse prefixes. |
| Completion/acknowledgement | Local observation implemented; authoritative transitions BLOCKED | Final contract, trusted channel, receipt/replay/lost-response semantics and post-stop authority. |
| C06 recovery | SOURCE_RECOVERY_REQUIRED; no implementation | Introducing source or authoritative disposition, then any actual recovery contract. |
| Original C04 buffers | ORIGINAL_SERIALIZED_BUFFERS_AND_PROVENANCE_REQUIRED | Original final bytes with independently checkable historical identity and provenance. |
| Remaining 60 witnesses | All inventoried; BLOCKED | Complete guard/resource baselines; C05 additionally genuine native observations. |
| Native qualification | NOT_RUN | Established protection, ordering and completion contracts; supported environment and exact safe execution bindings. |
| Package capture | NOT_CAPTURED | Authoritative qualification-package definition/procedure, actual qualified inputs/results and authorization bindings. The older 16-file preservation package is not this package. |
| Production integration | NOT_INTEGRATED / BLOCKED | Every required gate above plus complete required tests/evidence; local tests cannot waive prerequisites. |

Specific decisions, none adopted here:

1. Does final integration retain the source's complete-chain/leaf/failure property,
   or seek an expressly narrower local observation claim? The implementation
   authorization permits local primitives; S/R and current integration requirements
   do not establish that the narrower claim satisfies FC-B01. The first needs a
   complete protection argument; the second needs an explicit acceptance-scope
   decision and must not be described as the existing guarantee.
2. What is the actual guard owner, protection endpoint, and failure ordering? A
   subject-owned or separate-holder design has different failure/rights costs;
   neither is proven. Provide exact lifecycle, environment, quiescence evidence,
   post-stop authority and noncircular reporting before implementation/integration.
3. Which complete native/abstract operation mapping is authorized? Current identity
   checks expand a modeled operation into three native queries. Counting all
   current contributions exceeds the recorded whole-case ceilings; hiding them
   would invalidate accounting. Provide a reviewed representable workflow within
   unchanged limits. No limit increase or weakened check is requested/adopted.
4. Which exact completion and trusted acknowledgement contract is issued? Same-
   machine local readback and a retained external/reference acknowledgement supply
   different evidence. P proposes the latter and leaves channel identity unresolved.
   No missing/lost/replayed response may become success until the selected contract
   defines authentic evidence, delivery knowledge, finality and permitted recovery.
5. What introduced C06, or authoritatively superseded it? Absence from C01-C05 does
   not establish obsolescence or a recovery algorithm. Recover the source/decision
   and original C04 buffers before constructing their dependent qualification work.

The next safe step is exact threat/scope and lifecycle evidence resolution, together
with the native-operation accounting and completion decisions above. Continue only
independently specified code work. Native security/race tests, package capture and
authoritative publication remain gated; no experiment was needed for the write-
attempt enforcement fix. FC_B01_STATUS remains BLOCKED; completion remains
COMPLETION_UNRESOLVED; NATIVE_QUALIFICATION_STATUS remains NOT_RUN.
