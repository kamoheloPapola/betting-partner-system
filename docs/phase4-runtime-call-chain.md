# Phase 4 PL 1X2 runtime call chain

This inventory records the repository state inspected before selecting the
Phase 4 integration boundary.

## Shared execution path

API requests enter `src.api.main` or `src.api.routes.cli`, instantiate
`src.predictions.predictor.Predictor`, and delegate to the same
`src.cli.commands.prediction._run_predict_loop` used by CLI
`show-predictions`. The data path is:

`FeaturePipeline.run` -> processed match CSV load -> point-in-time feature
engineering -> date filtering -> `_run_predict_loop`.

For each league, `_run_predict_loop` calls `_load_prediction_models`.
`ModelRegistry.load_model` resolves the pinned manifest metadata, reads the
configured artifact beneath `src/ml/models`, deserializes it, and attaches the
resolved metadata to the model object.

The PL 1X2 calculation path is:

`_run_predict_loop` -> `_calculate_probabilities_v2` -> `_calc_goals` -> four
`_predict_scalar` calls (home/away LightGBM and home/away XGBoost) -> 60/40
goal-lambda ensemble -> `MatchSimulator.simulate` -> ordered home/draw/away
probabilities.

The match-outcome classifier is not used by this path. Goals/BTTS binary
post-processing later in `_calc_goals` does not alter the raw 1X2 values.

## Existing identities and persistence

- The live canonical match identity is `match_id`; the feature pipeline derives
  it from `match_hash` when required.
- Fixture fields retained by the live result are league/competition, home and
  away teams, date, optional `kickoff_utc`, and `match_id`.
- Processed sources are `data/processed/matches/{league}_*.csv`, with optional
  FBref CSV enrichment.
- The externally used resolver persistence is the daily
  `data/predictions/predictions_{league}_{date}.csv`, deduplicated by
  match-hash/market.
- Earlier phases provide compatible canonical JSON/SHA-256, Git state,
  protected-artifact, and evidence-runner patterns in `src/evaluation` and
  `scripts/run_phase*.py`.

## Repository reality that differs from the Phase 4 premise

The production API/CLI loop does not currently derive a fresh fixture-specific
seed. It creates one seed-42 `MatchSimulator` per league batch and advances that
RNG stream in fixture order. Phase 1's evaluation helper separately derives a
fixture-specific seed. Phase 4 does not change either behavior because doing so
would change numerical outputs. The v1 record instead persists the actual
pre-call RNG state and labels the live seed scope `league_batch_shared_rng`,
which is sufficient for exact independent replay.

Legacy active model metadata includes version, type, feature order, training
date, sizes, and a temporal-validation flag, but not immutable training-row,
training-data, and training-code identities. Those artifacts can therefore be
execution-traceable while their training lineage remains explicitly unproven.
