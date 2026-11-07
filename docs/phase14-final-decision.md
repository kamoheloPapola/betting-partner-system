# Phase 14 final scientific decision operations

Use the repository virtual environment:

```
venv\Scripts\python.exe scripts/run_phase14_decision.py status all
venv\Scripts\python.exe scripts/run_phase14_decision.py status SA
venv\Scripts\python.exe scripts/run_phase14_decision.py finalize SA
venv\Scripts\python.exe scripts/run_phase14_decision.py verify <decision-directory>
```

There is no activate command. Status reconstructs the authoritative complete cohort without publishing evidence. Finalize returns exit code 2 when blocked. Each league has its own existing Phase12 operation lock. A busy lock fails closed and can be retried after the current collector finishes; do not stop collection or remove lock files.

All real horizons remain in May 2027: BL1 May23, FL1 May30, SA/PD May31 (UTC). Readiness reports simultaneous horizon, settlement and protocol blockers. Complete lineage, causal features, common RNG, Phase4 replay, historical checkpoint evaluations and the unchanged Phase12/Phase7 scientific rules must all verify. Candidate support is exactly the frozen zero-margin rule, not a comparison of raw log loss. Active-supported and inconclusive terminal decisions are valid scientific artifacts but cannot authorize promotion review.

Only Phase13 v2 bindings at `phase13-authorization-20260915-v2` are accepted. Phase13 v1 remains historical and unusable. This phase does not change either revision, models, pointers, protocols, source qualification, scientific rules or the hourly Windows task.

Normal future publication is `<Phase13-v2>/<league>/decisions/final`, containing exactly `prospective_result.json` and `artifact_index.json`. The result is the exact frozen Phase13 `decision_material`; the index contains exactly its final file identity. Both use the existing canonical JSON and SHA256 integrity seals. These are verifiable evidence identities, not discretionary approvals or new digital-signature infrastructure.

The existing league lock is held across complete reconstruction, evaluation, staging, atomic rename and independent verification. A separate hash snapshot detects writers that ignore the lock. Staging directories sit outside the contract-permitted decisions root; a crash before rename can leave an inadmissible orphan stage. Retry creates a fresh stage. A crash after rename leaves the complete two-file publication, which a retry verifies and returns idempotently. Never manually edit, move into place, or overwrite decision artifacts.

The default finalization timestamp is the real clock. For retries of an existing publication, its original timestamp is reused. `--finalized-at` may specify the same identity explicitly, but cannot be in the future or before the frozen horizon. `--publication-id successor-1` provides a new immutable directory for a legitimate later scientific successor. A later append invalidates an older final; the old bytes remain intact. Scientific conflicts cannot be bypassed by choosing another directory or timestamp.

The standalone verifier imports the frozen Phase13 authorization consumer and Phase12 replay logic; it does not import the publisher. It verifies canonical bytes, directory inventory, both hashes, installation, protocol, complete lineage, every checkpoint, replay, recomputed decision and exact current cohort. It accepts scientifically valid active-supported/inconclusive finals while reporting no promotion-review eligibility.

Initial implementation evidence is under `baselines/phase14/phase14-final-decision-20260915-v1`. Dated real refusal records are independently reconstructed as complete journal prefixes; later valid appends are allowed but removals, reordering or omitted contemporaneous members are rejected. The initial report verifier also proves that no real final artifact exists and that the current clock remains before every frozen horizon. After a horizon legitimately matures, use the final-decision verifier; the initial-report check deliberately ceases to certify a current horizon refusal.

For implementation evidence maintenance, `src.evaluation.phase14_evidence.certify` accepts focused/protected/full JUnit paths, `seal_report` independently verifies each real refusal and seals the report, and `verify_report` independently reconstructs the sealed initial evidence again. No method in Phase14 calls activation. Future deployment still requires the separate explicit Phase13/Phase8 operation and every unchanged operational preflight.
