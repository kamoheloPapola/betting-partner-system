# Phase 15 scientific incident review

```
venv\Scripts\python.exe scripts/run_phase15_review.py status BL1 565839
venv\Scripts\python.exe scripts/run_phase15_review.py evaluate BL1 565839
venv\Scripts\python.exe scripts/run_phase15_review.py verify <review-artifact>
venv\Scripts\python.exe scripts/run_phase15_review.py phase14-status BL1 565839
```

There are no override, force-clear, approval, activation, exclusion or prediction-regeneration options. Every operation takes the existing league operation lock. A busy lock must be retried without removing the lock or stopping collection.

The BL1 review uses stored evidence only. The admitted kickoff was October31 2026 at 00:00 UTC; the later authoritative observation says 14:30 UTC. The original source was retrieved September9 at 18:43:12.819350 UTC. The revision was retrieved September14 at 18:08:19.777412 UTC and recorded as an incident at 18:11:46.024987 UTC. Both predictions share September9 18:47:11.711413 UTC as their prediction timestamp and causal cutoff. The pair was admitted at 18:47:40.648202 UTC. The maximum source event timestamp was September5 13:30 UTC.

These facts do not authorize clearance. Phase12 `same_fixture(admitted=True)` requires identical kickoff instants. Its sealed operating guide explicitly says there is no frozen rule permitting an admitted kickoff change. Phase13 rejects preserved protocol violations and provides no review-clearance hook. Phase15 therefore records `UNRESOLVED_PROTOCOL_CONFLICT`, keeps the observation in the cohort, and grants neither clearance nor fixture exclusion. No experiment restart is ordered: that would require a separate scientific-contract decision.

The deterministic factual classifier can distinguish an actual temporal/identity/order violation, insufficient resolution semantics, and factual identity (for example equivalent timestamp representations of the same instant). `RESOLVED_FACTUAL_IDENTITY` is only a factual finding. With the currently frozen consumers, even that label cannot clear a recorded protocol blocker. There is deliberately no invented production clearance rule. Synthetic tests exercise this distinction and ensure a resolved factual finding cannot waive the horizon.

Reviews are canonical, atomically published JSON under `baselines/phase15/phase15-scientific-review-20260921-v1/reviews/<league>/<fixture>/review-NNNNNN.json`. Each binds its predecessor, original protocol, installation, complete ordered evidence identity, admission, subsequent events, authoritative source versions and raw hashes, causal cutoffs, timing/order findings, frozen rule identities and exact outcome. Duplicate evaluation of unchanged evidence returns the existing artifact. A changed legitimate basis produces an immutable successor. Nothing overwrites an old review.

The independent verifier imports the read-only facts/rule kernel, not the publisher. It reloads the authoritative cohort, source bytes and protocol, replays predictions and causal features, checks the review chain, and rebuilds the exact material/hash. Historical verification may reconstruct the exact old journal prefix; its result explicitly says whether the review applies to current lineage. The Phase14 adapter requires current-lineage verification and rejects stale, malformed, unbound or forged-clearance reviews.

Integration is additive: `phase15_phase14.status` (exposed as `phase14-status`) calls the sealed Phase14 readiness engine under the same lock and attaches strictly verified reviews. It preserves every existing blocker; an invalid review adds a fail-closed review error. The original Phase14 code, its old CLI, the Phase13 consumer and all historical records remain byte-identical. This preserves the prior sealed implementation indexes. Original Phase14 finalization still enforces all horizon, settlement, replay, RNG, metric, lineage and incident restrictions. No alternate finalization or activation path is introduced.

SA, FL1 and PD receive no scientific changes; only isolated generic compatibility tests exercise their identifiers. The hourly collection task and its machine-availability limitation remain untouched. BL1 still must satisfy its May23 2027 horizon, complete settlement and every other scientific condition; the unresolved kickoff conflict independently blocks it.
