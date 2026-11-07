# Phase 16 — versioned fixture revision protocol

Phase 16 freezes `ffps.fixture_revision.v2` for **new** prospective cohorts. The Phase 12–15 BL1 cohort remains scientifically blocked by its admitted kickoff revision. Its pair and Phase 15 `UNRESOLVED_PROTOCOL_CONFLICT` review are untouched. The Phase 16 regression applies v2 to the stored facts as a hypothetical classification only; it never moves that pair into a new cohort.

The rule is league-generic. It sorts immutable authoritative source versions by retrieval instant, checks fixture identity independently of kickoff, and checks each observed kickoff against both the preceding version and the original prediction/feature cutoffs. It requires unchanged kickoff-sorted cohort order and the cohort's explicit fixed horizon. The new selection policy considers every qualifying future fixture independently; there is no earliest-fixture limiter whose selection could be changed by a later schedule revision. Unknown source history, simultaneous conflicting versions, provisional unknown kickoff, material identity or terminal-status changes, result-aware readmission, and post-match discovery of a new kickoff fail closed. A finished result with the already-established kickoff is handled separately from schedule revision and can settle only after that kickoff.

The v2 rule artifact binds the classifier and cohort admission implementation bytes. Each clean cohort has its own sealed `ffps.league_prospective_protocol.v2`, rule reference, league/season, fixed start and end, active/candidate identities, source qualification and an empty pair prefix. The end date is inherited from that league's already-frozen Phase 11 horizon; the start is later than the new protocol freeze. New pair capture must use the v2 protocol hash and occur after its start and before kickoff. The admission path replays the qualification, causal feature state, both Phase 4 predictions and common RNG before atomically appending a sealed pair. It will reject an old Phase 12 pair even if that fixture is still future.

The publisher accepts only archived football-data.org source metadata that the existing qualified adapter can re-normalize. This currently permits clean-cohort initialization for SA, FL1, PD and BL1. PL uses its separate Phase 7P source/capture adapter; the v2 classifier has no PL branch and is tested against PL, but no PL cohort is initialized until a qualified PL adapter is bound. Other subsequently supported competitions can use the same classifier once their source and replay adapter is qualified. No source authority is broadened.

The clean-cohort machinery is ready but Phase 16 does **not** schedule collection or publish a prospective pair. The following operational phase can provide fresh source acquisitions and call `capture_pair`, `admit_pair`, `append_observation` and `settle` under the cohort lock. Each observation retains the raw source identity and normalized event. A conflict remains in the journal and blocks the cohort; it is never repaired by deleting a pair or changing the old protocol. Settlement uses only authoritative post-match results and checks the full version classification again under the lock. The independent verifier replays all stored pair and source bytes, journal sequence, classifications and settlements. No v2 final-decision or activation path exists in Phase 16.

```
venv\Scripts\python.exe scripts/run_phase16_protocol.py freeze-rule
venv\Scripts\python.exe scripts/run_phase16_protocol.py initialize BL1-clean-001 BL1
venv\Scripts\python.exe scripts/run_phase16_protocol.py status BL1-clean-001
venv\Scripts\python.exe scripts/run_phase16_protocol.py historical-regression
venv\Scripts\python.exe scripts/run_phase16_protocol.py verify-regression
```

The CLI has no override, fixture exclusion, historical finalization, model training, calibration, promotion or activation operation. Phase 12's existing scheduler and Phase 13/14 consumers remain bound to their original lineage. The new empty cohort is `PHASE16_PROTOCOL_V2_READY_FOR_CLEAN_COHORT`; that status grants no scientific decision or deployment authority.
