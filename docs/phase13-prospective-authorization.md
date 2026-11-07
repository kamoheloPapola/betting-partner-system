# Phase 13 authorization operating guide

Phase 13 installs prerequisites, never an approval. Run `venv\Scripts\python.exe scripts/run_phase13_authorization.py bind` to complete or idempotently verify publication. `status` reconstructs all four bindings; `verify` verifies the sealed implementation report offline. No command activates a model or creates a scientific decision.

Successor specifications live in `baselines/phase9/phase13-authorization-20260915-v2/<league>/deployment_specification.json`. Contracts and installation manifests live in `baselines/phase13/phase13-authorization-20260915-v2/<league>`. Original Phase9 specifications, Phase11 protocols, Phase12 evidence and production pointers remain unchanged. Original Phase12 reports still describe their original specification; use the successor's Phase13 status for the new binding.

The initial September14 v1 binding publication is retained byte-for-byte. Integration review found missing Phase8 activation-record identity fields; v2 binds the corrected adapter. V1 has no sealed completion report and fails implementation-hash reconstruction. It must not be used for deployment.

The contract references the protocol for horizon, bootstrap and scoring rules; it does not define a second scientific configuration. The successor's material hash excludes the contract reference to avoid a circular hash. The installation manifest then binds the actual immutable bytes of both files. Offline verification reconstructs both from the original specification, original protocol and sealed Phase12 report-000001 lineage anchor. Subsequent legitimate Phase12 appends are allowed; anchored records cannot change or disappear.

## Future decision interface

Only a separate future decision phase may publish a final scientific artifact. It must use the contract's `ffps.final_prospective_decision.v1` schema and `FINALIZED_FIXED_HORIZON` status, within the league's `decisions` directory. The normal status location is `decisions/final`. The file is `prospective_result.json`, sealed by canonical `decision_sha256`, plus `artifact_index.json` sealed by `record_sha256` and containing exactly the decision file identity. These are integrity seals, not digital signatures or discretionary approvals.

The required result is exactly the value reconstructed by `decision_material(contract, installation, finalized_at)`: contract and actual successor specification references, protocol, complete ordered current Phase12 events, every admitted pair and settlement, every incident, and the unchanged Phase12 evaluation. Finalization must occur after the frozen calendar horizon and no later than the real clock. All admitted observations must settle; full causal-feature and RNG validation and Phase4 replay must pass. Protocol violations, corrected-result conflicts, mixed active baselines, missing lineage or any altered metric reject authorization. No manual or legacy fallback exists.

Recompute while holding the league's existing `operation.lock`. A correction or later journal append invalidates an older final artifact; never overwrite the old decision. Publish any legitimate successor in a new immutable directory. No final decision is produced in Phase13; synthetic examples exist only in isolated tests.

## Controlled deployment boundary

For a future separately authorized operation, instantiate `src.ml.phase13_bundle_activation.ProspectiveBundleController` with the successor specification and exact league. Its inherited Phase8 `activate(evidence, expected_active=..., operator=...)` retains worker quiescence, expected active generation, model identity verification, smoke replay, atomic transition, activation records, rollback and crash recovery. The adapter additionally holds the league cohort lock through validation, preflight and transition. A candidate-supported scientific result permits preflight only; it does not assert that operational checks passed.

The old Phase8 non-PL CLI constructs the old controller and cannot consume the new evidence schema; it fails closed. Do not bypass that rejection with old evidence or manual authorization. Existing serving, rollback and recovery remain backed by the unchanged Phase8 implementation. Phase13 intentionally provides no activation command.

Phase12's hourly Windows task remains unchanged and continues independently. Its logged-in-user/machine-availability limitation remains deferred. All files existing at the Phase13 integrity snapshot must retain their hashes; newly appended Phase12 source or journal files are reported separately. Models, production pointers and the complete PL Phase7P tree must match the pre-run inventory exactly.
