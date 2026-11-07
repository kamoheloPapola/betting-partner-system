# Controlled PL bundle activation and rollback

Production remains `active-production-pl-goals-phase4-frozen`. The Phase 6 bundle is a shadow challenger. Phase 8 does not establish predictive superiority and has no override flag.

`src/ml/bundle_activation.py` is the controller. `src/ml/registry.py` delegates PL goal loading to it. The existing CLI/API calculation, 30 features, 60/40 ensemble, simulator, and raw home/draw/away semantics remain unchanged. The prediction module is byte-identical to the version present before Phase 8.

The original file manifest is the bootstrap authority and retained rollback metadata. Its four PL pointers and member metadata are protected against generic registry writes. A database mirror cannot replace the file authority. Once created, `src/ml/models/active_pl_bundle.json` is the sole PL goal selector; it holds one bundle ID, aggregate, generation, transition ID, and previous-pointer fingerprint. A missing pointer after successful transition history is an error, never an implicit rollback.

The registry's manifest singleton is not a model cache. The prediction loop retains a suite per league batch. Phase 8 reloads a complete verified bundle on resolution and returns generation-bound model objects. Each inference call checks the current pointer, loaded identity, on-disk bytes, metadata, ordered features and finite values while holding an OS lock. An old suite must be reloaded after a transition. A transition during suite assembly or between its model calls fails the request; it cannot mix accepted old and new members. A calculation whose model calls finished before the transition may finish with its explicit old-bundle provenance.

All operations use the supported repository virtual environment. Activation requires maintenance downtime: stop serving workers using that venv/base interpreter before invoking the administrative command, then restart them after success. Process inspection blocks mutation while another supported-interpreter worker is running, covering processes that imported the registry before Phase 8 was installed. Phase 8 does not kill workers. Newly started workers load the new code and honor its lock/journal. No distributed serving or alternate-runtime deployment is supported.

Run commands from the repository root:

```powershell
venv/Scripts/python.exe scripts/run_phase8_bundle_activation.py dry-run --output tmp/phase8-new-dry-run.json

# Later only: finalized supported Phase 7 evidence, stopped serving workers,
# and a passing technical preflight are mandatory. This command refuses today.
venv/Scripts/python.exe scripts/run_phase8_bundle_activation.py activate --evidence <finalized-phase7-directory> --expected-active active-production-pl-goals-phase4-frozen --operator <operator-identity>

venv/Scripts/python.exe scripts/run_phase8_bundle_activation.py rollback --expected-active phase6-evaluation-only-pl-goals-7fc4d5600bde7e96 --operator <operator-identity> --reason <technical-incident>

venv/Scripts/python.exe scripts/run_phase8_bundle_activation.py recover --operator <operator-identity>
venv/Scripts/python.exe scripts/run_phase8_bundle_activation.py verify <activation-or-rollback-record.json>
venv/Scripts/python.exe scripts/run_phase8_bundle_activation.py verify-infrastructure baselines/phase8/phase8-controlled-pl-bundle-20260905-v1
```

Dry-run writes only the explicitly requested new evidence file and a reusable lock file. It does not write the active pointer, original manifest, candidate files, or activation history. Reusing an evidence filename is rejected.

The promotion gate requires the exact frozen Phase 7 specification fingerprint, candidate aggregate and active control identity. It verifies finalized manifest/index/result relationships and offline status. `CANDIDATE_INCONCLUSIVE` gives `PROMOTION_NOT_ELIGIBLE`. Operator identity cannot alter this classification.

The future Phase 7 finalizer must publish a **new** evidence directory, retaining the existing `ffps.phase7_run_manifest.v1` layout. It must include the unchanged specification and indexed `prospective_predictions.json` and `prospective_settlements.json` arrays containing the entire authoritative Phase 7P cohort. `prospective_result.json.status` must be `FINALIZED_FIXED_HORIZON`; manifest status must be `complete`, with a finalization timestamp after the frozen calendar horizon. Result/manifest/offline classification must agree. Phase 8 independently verifies the prediction provenance, prediction/settlement seals and links, checks export completeness against the immutable store, recomputes the frozen aggregates and decision rule, and requires `CANDIDATE_SUPPORTED_FOR_PROMOTION_REVIEW`. This adapter consumes future authority; it does not create or finalize scientific authority during collection. The current frozen Phase 7 run must never be edited to satisfy it.

`ModelBundleActivationRecord` and `ModelBundleRollbackRecord` use schemas `ffps.model_bundle_activation_record.v1` and `ffps.model_bundle_rollback_record.v1`. Their canonical SHA-256 seals bind:

- Activation ID, UTC timestamp, operator, administrative authorization classification and source revision.
- Previous/requested bundles, all four artifact and lineage identities, preflight and scientific evidence identity (or rollback authorization link).
- Previous/resulting pointer states and generations, pointer byte hashes, unchanged legacy manifest hashes.
- Loaded hashes, deterministic smoke Phase 4 provenance, result status and immutable intent fingerprint.

The hash detects changed evidence; it is not a digital signature or an access-control system. Filesystem ownership remains the administrative trust boundary.

Transitions persist an immutable intent and pending journal, fsync staged pointer bytes, atomically replace one file, reload and verify the entire bundle, run a smoke prediction/replay, and publish immutable result evidence. History publication uses a staged, fsynced hard link with no overwrite. Model artifacts are never moved, rewritten or deleted. The OS lock serializes cooperating readers and writers. The journal blocks production after interruption until explicit recovery.

Failure before replacement preserves the exact old pointer and records `ABORTED_UNCHANGED`. Failure after replacement triggers verified rollback. If recovery itself cannot verify the previous artifacts or smoke, the journal remains and serving stays blocked. Recovery also recognizes a completed transition whose journal cleanup was interrupted. Successful rollback restores the previous four-model reference and adds a separate evidence record. It does not need a broken current candidate to deserialize.

Technical rollback reasons include loading failures, invalid output, feature/provenance/calibration errors, mixed models, or operational regressions. No statistical performance thresholds are introduced.

Historical Phase 4 records reference the registry source that existed when they were produced. `src/ml/bundle_evidence.py` verifies them in a temporary artifact view containing that hash-verified archived source and unchanged model/data bytes; replay uses the unchanged numerical calculation. The original verifier remains strict. Frozen evidence, Phase 7 sources and thresholds are not rewritten. New predictions carry the current registry source and bundle loader hash.

Limitations: this is a local-filesystem, single-serving-runtime mechanism, with deliberate maintenance downtime. Storage must support same-volume atomic replacement and hard links. Files are fsynced; actual power-loss/storage-controller failure has not been physically tested. Incident recovery fails closed if protected artifacts or evidence are lost. The fixed-horizon Phase 7 final export does not exist yet. The current result is therefore `PHASE8_READY_ACTIVATION_BLOCKED_BY_PHASE7`.
