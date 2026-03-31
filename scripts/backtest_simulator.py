"""
Monte Carlo Simulator Backtest.

Tests calibration (not accuracy) of the simulator by:
1. Loading the trained home/away goal models
2. Predicting xG for 300 test-set matches (season >= 2023)
3. Simulating outcomes via MatchSimulator
4. Comparing predicted probabilities vs actual outcomes

Calibration check: If model predicts P(home_win) = 0.70 across a bucket
of matches, ~70% of those should actually be home wins.
"""

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.config import DATA_DIR
from src.simulation.match_simulator import MatchSimulator, clamp_lambda

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# --- Config ---
FEATURES_PATH = DATA_DIR / "features" / "feature_matrix.csv"
MODELS_DIR = Path("models")
N_BACKTEST = 300
N_SIM = 10_000
SEED = 42

EXCLUDE_COLS = [
    "match_hash", "date", "match_date", "league", "home_team", "away_team",
    "home_score", "away_score", "home_goals_ht", "away_goals_ht",
    "home_corners", "away_corners", "home_cards", "away_cards",
    "home_shots", "away_shots", "home_shots_on_target", "away_shots_on_target",
    "home_yellow_cards", "away_yellow_cards", "home_red_cards", "away_red_cards",
    "home_fouls", "away_fouls", "home_total_shots", "away_total_shots", "referee",
    "status", "result"
]


def load_models():
    """Load trained home/away goal models and feature columns."""
    home_model = joblib.load(MODELS_DIR / "home_goals_model.pkl")
    away_model = joblib.load(MODELS_DIR / "away_goals_model.pkl")

    with open(MODELS_DIR / "feature_columns.json") as f:
        features = json.load(f)

    logger.info(f"Loaded models with {len(features)} features")
    return home_model, away_model, features


def prepare_test_data(features: list) -> pd.DataFrame:
    """Load feature matrix and extract test set (season >= 2023)."""
    df = pd.read_csv(FEATURES_PATH)

    # Need actual scores for calibration comparison
    df["season_year"] = df["season"].astype(str).str[:4].astype(int)
    test = df[df["season_year"] >= 2023].copy()

    # Drop rows with missing scores
    test = test.dropna(subset=["home_score", "away_score"])

    # Cast objects to category for LightGBM (must match training data prep)
    for col in test.columns:
        if test[col].dtype == "object":
            test[col] = test[col].astype("category")

    # Sample N matches for backtest
    if len(test) > N_BACKTEST:
        test = test.sample(n=N_BACKTEST, random_state=SEED)

    logger.info(f"Test set: {len(test)} matches (season >= 2023)")
    return test


def run_backtest():
    """Main backtest loop."""
    home_model, away_model, features = load_models()
    test_df = prepare_test_data(features)

    sim = MatchSimulator(n_simulations=N_SIM, seed=SEED)

    results = []
    skipped = 0
    for idx, row in test_df.iterrows():
        try:
            # Build feature vector directly from the dataframe slice
            # This preserves the 'category' dtypes correctly
            X = test_df.loc[[idx], features]

            # Predict xG with lambda clamping
            home_xg = float(clamp_lambda(home_model.predict(X)[0]))
            away_xg = float(clamp_lambda(away_model.predict(X)[0]))

            # Simulate
            result = sim.simulate(home_xg, away_xg)

            # Actual outcome
            actual_home = int(row["home_score"])
            actual_away = int(row["away_score"])
            actual_total = actual_home + actual_away

            if actual_home > actual_away:
                actual_result = "H"
            elif actual_home == actual_away:
                actual_result = "D"
            else:
                actual_result = "A"

            results.append({
                "home_team": row.get("home_team", ""),
                "away_team": row.get("away_team", ""),
                "home_xg": home_xg,
                "away_xg": away_xg,
                "pred_home": result.home_win_prob,
                "pred_draw": result.draw_prob,
                "pred_away": result.away_win_prob,
                "pred_o25": result.over_2_5,
                "pred_btts": result.btts_prob,
                "actual_result": actual_result,
                "actual_home_goals": actual_home,
                "actual_away_goals": actual_away,
                "actual_total": actual_total,
                "actual_o25": int(actual_total > 2),
                "actual_btts": int(actual_home > 0 and actual_away > 0),
                "entropy": result.entropy,
                "match_type": result.match_type,
            })
        except Exception as e:
            skipped += 1
            logger.warning(f"Skipped match {idx}: {e}")

    if skipped > 0:
        logger.warning(f"Total skipped: {skipped}/{len(test_df)}")

    df_results = pd.DataFrame(results)
    if len(df_results) == 0:
        logger.error("No matches processed — cannot generate calibration report.")
        return None

    logger.info(f"Backtest complete: {len(df_results)} matches processed")

    # --- Calibration Analysis ---
    print("\n" + "=" * 70)
    print("MONTE CARLO SIMULATOR BACKTEST — CALIBRATION REPORT")
    print("=" * 70)

    # 1. Overall accuracy (for context, not the goal)
    pred_result = df_results.apply(
        lambda r: "H" if r["pred_home"] > max(r["pred_draw"], r["pred_away"])
        else ("D" if r["pred_draw"] > r["pred_away"] else "A"),
        axis=1
    )
    accuracy = (pred_result == df_results["actual_result"]).mean()
    print(f"\n📊 Prediction accuracy (context only): {accuracy:.1%}")

    # 2. 1X2 Calibration by probability bucket
    print("\n── 1X2 CALIBRATION (Bucket Analysis) ──")
    _calibration_table(df_results, "pred_home", "actual_result", "H", "Home Win")
    _calibration_table(df_results, "pred_draw", "actual_result", "D", "Draw")
    _calibration_table(df_results, "pred_away", "actual_result", "A", "Away Win")

    # 3. Over 2.5 Calibration
    print("\n── OVER 2.5 GOALS CALIBRATION ──")
    _binary_calibration(df_results, "pred_o25", "actual_o25", "O2.5")

    # 4. BTTS Calibration
    print("\n── BTTS CALIBRATION ──")
    _binary_calibration(df_results, "pred_btts", "actual_btts", "BTTS")

    # 5. xG vs Actual Goals
    print("\n── EXPECTED GOALS vs ACTUAL ──")
    home_xg_mae = np.abs(df_results["home_xg"] - df_results["actual_home_goals"]).mean()
    away_xg_mae = np.abs(df_results["away_xg"] - df_results["actual_away_goals"]).mean()
    print(f"  Home xG MAE: {home_xg_mae:.3f}")
    print(f"  Away xG MAE: {away_xg_mae:.3f}")
    print(f"  Avg Home xG: {df_results['home_xg'].mean():.2f} vs Actual: {df_results['actual_home_goals'].mean():.2f}")
    print(f"  Avg Away xG: {df_results['away_xg'].mean():.2f} vs Actual: {df_results['actual_away_goals'].mean():.2f}")

    # 6. Entropy distribution
    print("\n── MATCH TYPE DISTRIBUTION ──")
    for mt in ["predictable", "balanced", "high_uncertainty"]:
        count = (df_results["match_type"] == mt).sum()
        pct = count / len(df_results) * 100
        print(f"  {mt:20s}: {count:4d} ({pct:.1f}%)")

    # 7. Calibration by match type
    print("\n── ACCURACY BY MATCH TYPE ──")
    for mt in ["predictable", "balanced", "high_uncertainty"]:
        subset = df_results[df_results["match_type"] == mt]
        if len(subset) == 0:
            continue
        pred_r = subset.apply(
            lambda r: "H" if r["pred_home"] > max(r["pred_draw"], r["pred_away"])
            else ("D" if r["pred_draw"] > r["pred_away"] else "A"),
            axis=1
        )
        acc = (pred_r == subset["actual_result"]).mean()
        print(f"  {mt:20s}: {acc:.1%} ({len(subset)} matches)")

    print("\n" + "=" * 70)

    # Save results
    out_path = MODELS_DIR / "backtest_results.csv"
    df_results.to_csv(out_path, index=False)
    logger.info(f"Results saved to {out_path}")

    return df_results


def _calibration_table(df, prob_col, actual_col, target_val, label):
    """Print calibration table for multinomial outcome."""
    bins = [(0.0, 0.25), (0.25, 0.35), (0.35, 0.45), (0.45, 0.55), (0.55, 0.70), (0.70, 1.0)]

    print(f"\n  {label}:")
    print(f"  {'Bucket':>12s} │ {'Predicted':>9s} │ {'Actual':>9s} │ {'Count':>5s} │ {'Delta':>7s}")
    print(f"  {'─' * 12}─┼─{'─' * 9}─┼─{'─' * 9}─┼─{'─' * 5}─┼─{'─' * 7}")

    for lo, hi in bins:
        mask = (df[prob_col] >= lo) & (df[prob_col] < hi)
        subset = df[mask]
        if len(subset) == 0:
            continue
        pred_mean = subset[prob_col].mean()
        actual_rate = (subset[actual_col] == target_val).mean()
        delta = actual_rate - pred_mean
        marker = "✓" if abs(delta) < 0.10 else "⚠"
        print(f"  {lo:.0%}–{hi:.0%}     │ {pred_mean:8.1%}  │ {actual_rate:8.1%}  │ {len(subset):5d} │ {delta:+6.1%} {marker}")


def _binary_calibration(df, prob_col, actual_col, label):
    """Print calibration table for binary outcome."""
    bins = [(0.0, 0.30), (0.30, 0.45), (0.45, 0.55), (0.55, 0.65), (0.65, 0.80), (0.80, 1.0)]

    print(f"\n  {label}:")
    print(f"  {'Bucket':>12s} │ {'Predicted':>9s} │ {'Actual':>9s} │ {'Count':>5s} │ {'Delta':>7s}")
    print(f"  {'─' * 12}─┼─{'─' * 9}─┼─{'─' * 9}─┼─{'─' * 5}─┼─{'─' * 7}")

    for lo, hi in bins:
        mask = (df[prob_col] >= lo) & (df[prob_col] < hi)
        subset = df[mask]
        if len(subset) == 0:
            continue
        pred_mean = subset[prob_col].mean()
        actual_rate = subset[actual_col].mean()
        delta = actual_rate - pred_mean
        marker = "✓" if abs(delta) < 0.10 else "⚠"
        print(f"  {lo:.0%}–{hi:.0%}     │ {pred_mean:8.1%}  │ {actual_rate:8.1%}  │ {len(subset):5d} │ {delta:+6.1%} {marker}")


if __name__ == "__main__":
    run_backtest()
