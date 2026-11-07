# FC-B01 — Explicit Owner Authorization and Bounded Execution Instructions

**Authorization record:** `FC-B01-OWNER-AUTHORIZATION-20261024-01`  
**Authorization date:** 2026-10-24  
**Authority:** Explicit owner authorization issued in the current conversation  
**Scope:** FC-B01 only  
**Execution mode:** Bounded, evidence-preserving, implementation-readiness work

## 1. Executive authorization

I explicitly authorize work to proceed on FC-B01 within the boundaries defined in this instruction.

This authorization supersedes the previous owner-decision state of `PENDING` for the purpose of authorizing the bounded FC-B01 work described here.

Record the owner decision as:

`OWNER_DECISION_STATUS: AUTHORIZED`

Record the authorization status separately from technical readiness:

- `FC_B01_STATUS: BLOCKED` until its substantive blockers are resolved through admissible evidence and applicable requirements.
- `IMPLEMENTATION_READINESS: BLOCKED` until the applicable readiness criteria are satisfied.
- `O12_AUTHORIZATION_STATUS: NOT_GRANTED` unless a separate existing authorization establishes otherwise.
- `WIRE_AMENDMENT_STATUS: NOT_AUTHORIZED`
- `BUDGET_CHANGE_STATUS: NONE`
- `NATIVE_QUALIFICATION_STATUS: NOT_RUN`
- `COMPLETION_STATUS: COMPLETION_UNRESOLVED`
- `PACKAGE_CAPTURE_STATUS: NOT_CAPTURED`
- `PRODUCTION_INTEGRATION_STATUS: NOT_INTEGRATED`
- `C06_STATUS: SOURCE_RECOVERY_REQUIRED`

**Owner authorization does not constitute evidence that any technical blocker has been resolved.** Do not mark FC-B01 implementation-ready merely because this authorization has been issued.

## 2. Objective

Proceed with the existing FC-B01 work using the recovered evidence, existing approved requirements, and already-authorized implementation scope.

The objective is to make concrete progress without repeating unproductive broad investigations, weakening existing controls, inventing missing contracts, or silently expanding scope.

Preserve the system's correctness, trustworthiness, evidence provenance, existing budgets, and unresolved completion semantics.

## 3. Mandatory initial inspection

Before making changes, inspect the current repository and the relevant FC-B01 work products.

The previously reported repository baseline is historical evidence, not an assumption that the current repository remains unchanged.

Verify and report:

- Repository root, branch, HEAD and current working-tree state.
- Existing tracked modifications, deletions, staged changes and untracked content.
- Relevant FC-B01 specifications, proposals, tests, preparation artifacts and implementation files.
- Existing authorization records and their actual scope.
- Current contracts, budget definitions, witness requirements and acceptance criteria.
- Which requested work is already authorized and which dependencies remain blocked.

Do not overwrite, discard, clean, reset, or normalize existing user work. Do not assume every untracked artifact is disposable.

If the repository differs from the historical baseline, preserve the current state and determine whether the difference affects the authorized work before proceeding.

## 4. Evidence and authority requirements

Use the existing FC-B01 evidence matrix as the starting point.

Address the outstanding categories individually:

**A. Operational threat model**

Recover the exact applicable operational artifact, its identity and provenance, its issuing authority, approval or disposition, version history, exclusions, and protection endpoint.

Do not substitute a synthetic digest, proposed threat model, or historical task instruction for operational authority.

**B. Environment and protection**

Use admissible, appropriately scoped evidence to evaluate the applicable environment and protection obligations, including runtime binding, filesystem and effective-rights evidence, process capabilities, continuous protection, holder-loss containment, quiescence and release.

Historical observations alone do not establish a current or continuous prevention guarantee.

**C. C06 definition and disposition**

Continue source recovery for the original C06 definition and its required evidence, or obtain an authoritative disposition that explicitly resolves the missing definition.

Do not substitute `C-06` for `C06`, infer missing requirements, or declare the source obsolete without authority.

**D. Original C04 CONTROL buffers**

Recover the four original serialized buffers for C-57 through C-60, together with their storage and creation provenance and historical construction/evaluation references.

Do not reconstruct replacement buffers and present them as the originals. Do not claim that historical hashes verify bytes that have not been recovered.

**E. Accounting and acknowledgement contracts**

Recover applicable issued contract bytes and their approval, applicability and supersession records.

The required accounting and acknowledgement requirements must address the obligations already identified in the evidence matrix, including resource and native-query accounting, final permission checks, supervision, failure tails, unchanged limits, authenticated acknowledgement, replay and conflict handling, timing, restart uncertainty and final release.

Do not promote proposed contracts to authoritative contracts without the necessary evidence.

### Evidence integrity

For every recovered artifact, record its exact location, byte identity where available, digest, provenance, authority, applicability and verification limitations.

A matching digest establishes byte identity only; it does not establish authenticity, approval, completeness or enforcement.

Maintain explicit distinctions between `RECOVERED`, `VERIFIED`, `APPROVED`, `APPLICABLE`, `RESOLVED` and `NOT_VERIFIED`. Use the project's existing vocabulary where defined.

## 5. Bounded execution rules

Proceed with concrete FC-B01 work wherever the requirements and authorization are sufficient.

Prefer the smallest useful, independently verifiable unit of work. Avoid restarting broad historical searches that have already failed to produce authoritative evidence unless a new, specific lead justifies another attempt.

If a dependency is missing, identify the exact missing input, the decision it blocks, and the smallest action needed to obtain it.

If work can proceed independently without violating an unresolved obligation, continue within the existing authorization.

If work cannot safely proceed without an unresolved requirement, stop that dependent activity and document the blocker. Do not invent a contract or weaken an invariant to make progress appear possible.

### Access-denial boundary

The previously reported access-denied locations beneath:

`C:\Users\papol\my_systems\betting_partner_system\tmp\BL1-future-lineage-implementation-20261015-01`

remain an explicit investigation boundary unless access is legitimately changed by the owner or administrator.

Do not bypass permissions, impersonate another identity, or attempt alternative access paths intended to evade the denial.

## 6. Protected contracts and invariants

Unless a separate applicable authorization explicitly permits a change, preserve:

- Existing wire contracts and schemas.
- Existing accounting and resource budgets.
- Existing witness identities and historical evidence.
- Required provenance and evidence-chain integrity.
- Existing permission and protection requirements.
- Completion-state uncertainty and acknowledgement semantics.
- Restart, interruption, timeout and failure-path safety.
- Existing publication and release constraints.
- User modifications and unrelated repository state.

Do not turn `COMPLETION_UNRESOLVED` into a successful completion claim without satisfying its authoritative requirements.

Do not treat a successful test, a proposed contract, a recovered file or a matching hash as a substitute for an outstanding operational guarantee.

## 7. Explicit authorization exclusions

This authorization does not grant permission to:

1. Amend wire contracts or schemas.
2. Change budgets, accounting limits or resource allocations.
3. Run native qualification or qualification experiments.
4. Execute package capture.
5. Integrate the work into production.
6. Bypass access controls.
7. Fabricate, regenerate or relabel missing historical evidence as original evidence.
8. Declare FC-B01 implementation-ready while applicable readiness criteria remain unmet.
9. Treat O12 authorization as granted.
10. Expand the work into unrelated phases or previously blocked work.

Any activity requiring one of these permissions must remain unperformed unless its own existing authorization clearly permits it or a separate explicit authorization is obtained.

## 8. Implementation and verification procedure

For each independently authorized work unit:

1. Establish the relevant current state and requirements.
2. Identify its prerequisites and whether they are satisfied.
3. Make the smallest justified change, if implementation is authorized and sufficiently specified.
4. Add or update focused tests consistent with the authorized scope.
5. Run only tests and verification activities permitted by the existing authorization.
6. Record the exact changes, test commands, results, limitations and unresolved dependencies.
7. Preserve a clear distinction between verified behavior and unverified operational guarantees.

Do not run constructors, evaluators, native experiments, qualification, or package capture merely to gather evidence if those activities are outside the current authorization.

If the repository or evidence reveals a material contradiction, unsafe assumption or incompatible requirement, stop the affected work and report the exact decision requiring resolution.

## 9. Completion report

At the end of this bounded stage, provide:

- Owner authorization status.
- FC-B01 technical status and implementation-readiness status.
- Repository baseline and any changes made.
- Work completed and work deliberately not performed.
- Evidence recovered, with identities and provenance.
- Evidence that remains missing and the exact reason each item matters.
- Tests and permitted verification performed, with actual results.
- Contract, budget, witness and acceptance-record changes, explicitly reporting none where none occurred.
- Remaining blockers and their minimum required inputs.
- Any separate authorization required for subsequent work.

Do not report `PASS`, `RESOLVED`, or `IMPLEMENTATION_READY` unless the corresponding criteria are actually satisfied.

## 10. Transition to other authorized work

Once FC-B01 reaches the status required by its applicable acceptance criteria, report the evidence supporting that transition.

After the FC-B01 status is accurately established, move to the other already-authorized work items in the project's existing priority and dependency order.

Do not require a new owner decision merely to begin unrelated work that is already explicitly authorized and whose prerequisites are satisfied.

Do not use FC-B01's owner authorization to authorize unrelated blocked work.

If FC-B01 remains blocked because authoritative external inputs are unavailable, report the exact outstanding inputs and proceed with independent, already-authorized work where permitted. Do not falsely declare FC-B01 resolved to enable that transition.

## 11. Final instruction

**Proceed under this explicit, bounded owner authorization.**

Make useful progress, preserve all existing controls, and avoid repetitive investigations that have no concrete path to new evidence.

Do not confuse permission to work with proof of correctness, operational protection, or implementation readiness.

The required outcome is demonstrable progress with trustworthy evidence, preserved invariants and precise reporting of anything that remains unresolved.
