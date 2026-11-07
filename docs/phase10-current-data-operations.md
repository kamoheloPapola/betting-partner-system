# Phase 10 current-data qualification

This is evaluation-only work. PL, Phase 9 candidates, production data and model pointers are protected. No command trains, calibrates or activates a model.

Run commands from the repository root with `venv/Scripts/python.exe scripts/run_phase10_current_data.py`.

1. `prepare --run-dir baselines/phase10/<new-run>` records the protected byte inventory before acquisition.
2. `acquire --run-dir <run> --league BL1 --attempt initial` archives the previous/current season matches, teams and statistics. Repeat independently for FL1, PD and SA. The configured API key is read internally; never add credentials to a command or evidence file. Authorized provider requests may require network access outside the sandbox.
3. If needed, `retry-statistics --run-dir <run> --league BL1 --attempt initial` makes one bounded retry and creates a new `raw_source_index_statistics-retry.json`. Earlier responses remain immutable.
4. `qualify --run-dir <run> --league BL1 --attempt initial` runs the first portability assessment. `qualify-all --run-dir <run> --attempt initial` then processes BL1, FL1, PD and SA without skipping a league after a failure. Use the chosen acquisition attempt explicitly. Assessment attempts are append-only.
5. Only after qualification passes, `freeze --run-dir <run> --league BL1` freezes the current-data contract, candidate/control identities and the remainder of the verified provider season. `collect --run-dir <run> --league BL1` immediately captures every eligible future fixture using that source snapshot. It rejects late/nonfinite/unqualified inputs and reuses already stored pairs. A test pair never counts as live evidence.
6. Later, acquire a **new named attempt** to archive results. `settle --run-dir <run> --league BL1 --source-index <new-source-index.json>` only settles a previously admitted pair from an authoritative FINISHED result. Kickoff or score revisions fail closed. Repeating the identical settlement returns the existing row.
7. `evaluate --run-dir <run> --league BL1` verifies the full stored cohort and applies existing Phase 7 paired metrics/bootstrap/decision. Until the fixed horizon and settlement coverage are complete, the decision remains inconclusive. No Phase 10 decision activates production.

The acquisition request derives the season using the existing league calendar, then verifies it against provider responses. The source season bounds, league codes, teams, fixture IDs, orientation, scores and corners must agree. Provider retrieval timestamps establish when this system possessed bytes; they do not establish publication latency. Any field-level publication/update timestamps remain separate. Current-source gaps never trigger a production-data backfill.

This collector freezes a causal forecast from its qualified acquisition snapshot. It does not automatically replace frozen pairs with later forecasts, advance the protocol horizon, or silently rebase the protocol on another qualification. Acquire and qualify a new run if collection has not begun and the source snapshot needs replacement. Once collection begins, preserve the cohort and use fresh attempts for settlement. A revised forecast protocol needs a separate, prospectively frozen experiment, not a rewritten pair.

Before finalization, run the focused tests to `tmp/phase10-focused.xml`, protected/Phase 7–9 regressions to `tmp/phase10-protected.xml`, and full suite to `tmp/phase10-full.xml`. Run these suites sequentially because protected tests exercise the real PL bundle lock. Run prior Phase 9 offline verification and preserve its JSON as `<run>/prior_phase9_verification.json`.

`finalize --run-dir <run>` creates the matrix, report, test classification, protected after-snapshot and immutable index. This closes the run: do not append collection, settlement or tests afterward. A live cohort should remain open until its reporting horizon; use `evaluate` for interim reports. `verify --run-dir <run>` performs offline verification without network access or writes. The standalone command exits nonzero on any integrity error.

No mass promotion or PL operation exists in this CLI. Phase 8 remains the production transition owner and requires separate league-specific scientific evidence and authorization. The Phase 10 read-only model controller deliberately has no activation or rollback method.
