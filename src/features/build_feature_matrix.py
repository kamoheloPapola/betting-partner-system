"""
Feature Matrix Builder.

Transforms canonical match data into an ML-ready feature matrix through
a strict 7-layer pipeline with data leakage protection.

Layer order (never mix):
    1. Raw match stats
    2. Team match tables
    3. Rolling windows
    4. Strength-of-schedule normalisation (delta, not ratio)
    5. Dominance metrics
    6. Matchup deltas (interaction features)
    7. Context signals

Output: data/features/feature_matrix.csv

Usage:
    python -m src.cli.main build-features
"""
import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.config import DATA_DIR, PROCESSED_DATA_DIR
from src.features.engineering import FeatureEngineer, FeatureDefaults

logger = logging.getLogger(__name__)

# Paths
CANONICAL_DIR = DATA_DIR / "canonical"
FEATURES_DIR = DATA_DIR / "features"

# Rolling windows used throughout
WINDOWS = [3, 5, 10, 20]

# Minimum matches before a team's rolling stats are meaningful
MIN_ROLLING_MATCHES = 2


class FeatureMatrixBuilder:
    """
    Builds a canonical feature matrix from processed match data through
    a strict 7‐layer pipeline with row-level leakage protection.
    """

    def __init__(
        self,
        source_path: Optional[Path] = None,
        output_dir: Optional[Path] = None,
    ) -> None:
        self.source_path = source_path or DATA_DIR / "processed" / "matches.csv"
        self.output_dir = output_dir or FEATURES_DIR
        self.engineer = FeatureEngineer()

    # ==================================================================
    # Public API
    # ==================================================================
    def build(self) -> Dict[str, Any]:
        """
        Execute the full 7-layer pipeline.

        Returns:
            Summary dict with feature counts, row counts, output paths.
        """
        logger.info("═══ Feature Matrix Build — START ═══")

        # Layer 1: Raw match stats
        matches = self._layer_1_raw_stats()
        if matches is None or matches.empty:
            return {"status": "error", "message": "No source data"}

        # Layer 2: Team match tables
        team_df = self._layer_2_team_tables(matches)

        # Layer 3: Rolling windows
        team_df = self._layer_3_rolling_windows(team_df)

        # Layer 4: Strength-of-schedule normalisation
        team_df = self._layer_4_sos_normalisation(team_df)

        # Layer 5: Dominance metrics
        team_df = self._layer_5_dominance_metrics(team_df)

        # Layer 6: Matchup deltas (reassemble to match view)
        matrix = self._layer_6_matchup_deltas(matches, team_df)

        # Layer 7: Context signals
        matrix = self._layer_7_context_signals(matrix)

        # STRICT LEAKAGE ASSERTION
        self._assert_temporal_integrity(matrix)

        # Write outputs
        self.output_dir.mkdir(parents=True, exist_ok=True)
        matrix_path = self.output_dir / "feature_matrix.csv"
        matrix.to_csv(matrix_path, index=False)
        logger.info("Wrote %d rows × %d cols to %s", len(matrix), len(matrix.columns), matrix_path)

        # Freeze hash + coverage + manifest
        freeze = self._write_freeze_hash(matrix_path, matrix)
        self._write_coverage_report(matrix)
        self._write_feature_manifest()

        logger.info("═══ Feature Matrix Build — DONE ═══")
        return {
            "status": "success",
            "rows": len(matrix),
            "features": len(matrix.columns),
            "matrix_path": str(matrix_path),
            "sha256": freeze["sha256"],
        }

    # ==================================================================
    # Layer 1 — Raw Match Stats
    # ==================================================================
    def _layer_1_raw_stats(self) -> Optional[pd.DataFrame]:
        """Load canonical matches.csv (or fall back to processed CSVs)."""
        logger.info("[L1] Loading raw match stats")

        if self.source_path.exists():
            df = pd.read_csv(self.source_path)
            logger.info("[L1] Loaded canonical dataset: %d matches", len(df))
        else:
            # Fallback: load individual processed files
            matches_dir = PROCESSED_DATA_DIR / "matches"
            if not matches_dir.exists():
                return None
            frames = []
            for f in sorted(matches_dir.glob("*.csv")):
                if f.name.endswith("_upcoming.csv"):
                    continue
                try:
                    frames.append(pd.read_csv(f))
                except Exception:
                    continue
            if not frames:
                return None
            df = pd.concat(frames, ignore_index=True)
            df = df.drop_duplicates(subset=["match_hash"], keep="last")
            logger.info("[L1] Loaded %d matches from processed CSVs (fallback)", len(df))

        # Validate & coerce from new canonical schema -> expected legacy schema
        if "match_date" in df.columns:
            df["date"] = pd.to_datetime(df["match_date"], errors="coerce")
            df = df.drop(columns=["match_date"])
        else:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])

        # Map goals -> score
        if "home_goals" in df.columns and "home_score" not in df.columns:
            df = df.rename(columns={"home_goals": "home_score", "away_goals": "away_score"})

        for col in ("home_score", "away_score"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["home_score", "away_score"])

        df = df.sort_values("date").reset_index(drop=True)
        return df

    # ==================================================================
    # Layer 2 — Team Match Tables
    # ==================================================================
    def _layer_2_team_tables(self, matches: pd.DataFrame) -> pd.DataFrame:
        """Explode match rows into team-centric rows."""
        logger.info("[L2] Building team match tables")
        team_df = self.engineer.transform_match_to_team_rows(matches)

        # Ensure shot columns propagate to team table
        has_shots = "home_shots" in matches.columns and "away_shots" in matches.columns
        home_mask = team_df["is_home"] == 1
        away_mask = team_df["is_home"] == 0

        if has_shots:
            team_df.loc[home_mask, "shots_for"] = team_df.loc[home_mask, "home_shots"]
            team_df.loc[home_mask, "shots_against"] = team_df.loc[home_mask, "away_shots"]
            team_df.loc[away_mask, "shots_for"] = team_df.loc[away_mask, "away_shots"]
            team_df.loc[away_mask, "shots_against"] = team_df.loc[away_mask, "home_shots"]

            if "home_shots_on_target" in matches.columns:
                team_df.loc[home_mask, "shots_on_target_for"] = team_df.loc[home_mask, "home_shots_on_target"]
                team_df.loc[home_mask, "shots_on_target_against"] = team_df.loc[home_mask, "away_shots_on_target"]
                team_df.loc[away_mask, "shots_on_target_for"] = team_df.loc[away_mask, "away_shots_on_target"]
                team_df.loc[away_mask, "shots_on_target_against"] = team_df.loc[away_mask, "home_shots_on_target"]

        # Ensure cards columns propagate 
        has_cards = "home_cards" in matches.columns and "away_cards" in matches.columns
        if has_cards:
            team_df.loc[home_mask, "cards_for"] = team_df.loc[home_mask, "home_cards"]
            team_df.loc[home_mask, "cards_against"] = team_df.loc[home_mask, "away_cards"]
            team_df.loc[away_mask, "cards_for"] = team_df.loc[away_mask, "away_cards"]
            team_df.loc[away_mask, "cards_against"] = team_df.loc[away_mask, "home_cards"]

        logger.info("[L2] Team table: %d rows (%d matches × 2)", len(team_df), len(matches))
        return team_df

    # ==================================================================
    # Layer 3 — Rolling Windows
    # ==================================================================
    def _layer_3_rolling_windows(self, team_df: pd.DataFrame) -> pd.DataFrame:
        """Compute rolling stats, form, rest days. All use shift(1)."""
        logger.info("[L3] Computing rolling windows")

        # Determine available metrics
        metrics = ["goals"]
        if "corners_scored" in team_df.columns:
            metrics.append("corners")
        if "cards_scored" in team_df.columns:
            metrics.append("cards")

        # Rolling stats for multiple windows
        for w in WINDOWS:
            team_df = self.engineer.calculate_rolling_stats(team_df, window=w, metrics=metrics)
            
        # Season-to-date stats with blended guardrails
        team_df = self.engineer.calculate_std_stats(team_df, metrics=metrics, smoothing=10)

        # Shots & Cards rolling (manual — not in FeatureEngineer's METRIC_COLUMNS)
        manual_cols = []
        if "shots_for" in team_df.columns:
            manual_cols.extend(["shots_for", "shots_against"])
            if "shots_on_target_for" in team_df.columns:
                manual_cols.extend(["shots_on_target_for", "shots_on_target_against"])
        if "cards_for" in team_df.columns:
            manual_cols.extend(["cards_for", "cards_against"])

        if manual_cols:
            date_col = self.engineer._get_date_column(team_df)
            team_df = team_df.sort_values(date_col)
            for w in WINDOWS:
                for col in manual_cols:
                    team_df[f"rolling_{col}_{w}"] = team_df.groupby("team_id")[col].transform(
                        lambda x: x.shift(1).rolling(window=w, min_periods=1).mean()
                    )

        # Form rating
        team_df = self.engineer.calculate_form(team_df)

        # Rest days
        team_df = self.engineer.calculate_rest_days(team_df)

        # Team availability (stub until live injury feed integration)
        team_df = self.engineer.add_team_availability_feature(team_df)

        # Expanding season goal difference (shifted)
        date_col = self.engineer._get_date_column(team_df)
        team_df = team_df.sort_values(date_col)
        team_df["_gd"] = team_df["goals_scored"] - team_df["goals_conceded"]
        team_df["season_goal_diff"] = team_df.groupby(["team_id", "season"])["_gd"].transform(
            lambda x: x.shift(1).expanding(min_periods=1).sum()
        ).fillna(0.0)
        team_df = team_df.drop(columns=["_gd"])

        # Note: Form rating and days_rest are pre-filled safely in FeatureEngineer
        
        # Add feature-specific missing indicators instead of imputing
        for col in team_df.columns:
            if not (col.startswith("rolling_") or col.startswith("std_")):
                continue
            if not team_df[col].isna().any():
                continue
            
            # Explicit indicator
            team_df[f"is_missing_{col}"] = team_df[col].isna().astype(int)

        logger.info("[L3] Rolling features computed — %d columns", len([c for c in team_df.columns if "rolling" in c]))
        return team_df

    # ==================================================================
    # Layer 4 — Strength-of-Schedule Normalisation
    # ==================================================================
    def _layer_4_sos_normalisation(self, team_df: pd.DataFrame) -> pd.DataFrame:
        """
        Delta-based SoS normalisation.

        Self-joins on (match_id, opponent_id → team_id) to attach
        opponent rolling stats, then computes:
            attack_strength  = rolling_scored  − opp_rolling_conceded
            defense_strength = opp_rolling_scored − rolling_conceded
        """
        logger.info("[L4] Computing SoS normalisation")

        # Build opponent lookup: for each (match_hash, team_id) get their rolling stats
        opp_cols = ["match_hash", "team_id"]
        signals = {
            "goal": ("rolling_goals_scored_5", "rolling_goals_conceded_5"),
        }
        if "rolling_corners_scored_5" in team_df.columns:
            signals["corner"] = ("rolling_corners_scored_5", "rolling_corners_conceded_5")
        if "rolling_cards_for_5" in team_df.columns:
            signals["discipline"] = ("rolling_cards_for_5", "rolling_cards_against_5")
        elif "rolling_cards_scored_5" in team_df.columns:
            signals["discipline"] = ("rolling_cards_scored_5", "rolling_cards_conceded_5")
        if "rolling_shots_for_5" in team_df.columns:
            signals["shot"] = ("rolling_shots_for_5", "rolling_shots_against_5")

        # Columns to grab from opponent
        grab_cols = []
        for scored, conceded in signals.values():
            grab_cols.extend([scored, conceded])

        opp_df = team_df[opp_cols + grab_cols].copy()
        opp_df = opp_df.rename(columns={c: f"opp_{c}" for c in grab_cols})
        opp_df = opp_df.rename(columns={"team_id": "opponent_id"})

        # Merge: attach opponent's rolling stats via (match_hash, opponent_id)
        team_df = team_df.merge(opp_df, on=["match_hash", "opponent_id"], how="left")

        # Compute deltas per signal
        for signal_name, (scored_col, conceded_col) in signals.items():
            opp_scored = f"opp_{scored_col}"
            opp_conceded = f"opp_{conceded_col}"

            team_df[f"{signal_name}_attack_strength"] = (
                team_df[scored_col] - team_df[opp_conceded]
            )
            team_df[f"{signal_name}_defense_strength"] = (
                team_df[opp_scored] - team_df[conceded_col]
            )

        # Drop opp_ temporary columns
        opp_temp = [c for c in team_df.columns if c.startswith("opp_")]
        team_df = team_df.drop(columns=opp_temp)

        sos_cols = [c for c in team_df.columns if "_strength" in c]
        logger.info("[L4] SoS features: %s", sos_cols)
        return team_df

    # ==================================================================
    # Layer 5 — Dominance Metrics
    # ==================================================================
    def _layer_5_dominance_metrics(self, team_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute streak-based and ratio-based dominance metrics.
        All shifted to prevent leakage.
        """
        logger.info("[L5] Computing dominance metrics")
        date_col = self.engineer._get_date_column(team_df)
        team_df = team_df.sort_values(date_col)

        # Win / unbeaten streaks (shifted)
        team_df["_win"] = (team_df["goals_scored"] > team_df["goals_conceded"]).astype(int)
        team_df["_not_loss"] = (team_df["goals_scored"] >= team_df["goals_conceded"]).astype(int)
        team_df["_clean_sheet"] = (team_df["goals_conceded"] == 0).astype(int)

        def _streak(series: pd.Series) -> pd.Series:
            """Compute current streak length (shifted to exclude current match)."""
            shifted = series.shift(1)
            groups = (shifted != shifted.shift(1)).cumsum()
            streak = shifted.groupby(groups).cumcount() + 1
            result = streak.where(shifted == 1, 0)
            return result

        team_df["win_streak"] = team_df.groupby("team_id")["_win"].transform(_streak)
        team_df["unbeaten_streak"] = team_df.groupby("team_id")["_not_loss"].transform(_streak)

        # Clean sheet rate (last 5, shifted)
        team_df["clean_sheet_rate_5"] = team_df.groupby("team_id")["_clean_sheet"].transform(
            lambda x: x.shift(1).rolling(window=5, min_periods=1).mean()
        ).fillna(0.0)

        # Goal dominance (rolling window 5, shifted)
        team_df["goal_dominance_5"] = team_df.groupby("team_id").apply(
            lambda g: (
                g["goals_scored"].shift(1).rolling(5, min_periods=1).sum()
                / (
                    g["goals_scored"].shift(1).rolling(5, min_periods=1).sum()
                    + g["goals_conceded"].shift(1).rolling(5, min_periods=1).sum()
                ).replace(0, float("nan"))
            ),
            include_groups=False,
        ).reset_index(level=0, drop=True).fillna(0.5)

        # Shot dominance (if shots available)
        if "shots_for" in team_df.columns:
            team_df["shot_dominance_5"] = team_df.groupby("team_id").apply(
                lambda g: (
                    g["shots_for"].shift(1).rolling(5, min_periods=1).sum()
                    / (
                        g["shots_for"].shift(1).rolling(5, min_periods=1).sum()
                        + g["shots_against"].shift(1).rolling(5, min_periods=1).sum()
                    ).replace(0, float("nan"))
                ),
                include_groups=False,
            ).reset_index(level=0, drop=True).fillna(0.5)

        # Pressure index (shifted, rolling 5)
        if "rolling_shots_for_5" in team_df.columns and "rolling_corners_scored_5" in team_df.columns:
            team_df["pressure_index_5"] = (
                (team_df["rolling_shots_for_5"] * 1.0)
                + (team_df["rolling_corners_scored_5"] * 0.7)
                - (team_df["rolling_shots_against_5"] * 1.0)
                - (team_df["rolling_corners_conceded_5"] * 0.7)
            )
        elif "rolling_corners_scored_5" in team_df.columns:
            team_df["pressure_index_5"] = (
                team_df["rolling_corners_scored_5"]
                - team_df["rolling_corners_conceded_5"]
            )

        # Home Advantage Index (expanding means, conditionally computed to prevent leakage)
        # Using shift(1) to ensure the current match is not included in its own historical average
        def _expanding_away(g: pd.DataFrame, col: str, cond_col: str, cond_val: int) -> pd.Series:
            s = g[col].where(g[cond_col] == cond_val)
            return s.shift(1).expanding().mean().ffill()

        team_df["hist_home_goals"] = team_df.groupby("team_id").apply(
            lambda g: _expanding_away(g, "goals_scored", "is_home", 1), include_groups=False
        ).reset_index(level=0, drop=True).fillna(1.5)
        
        team_df["hist_away_goals"] = team_df.groupby("team_id").apply(
            lambda g: _expanding_away(g, "goals_scored", "is_home", 0), include_groups=False
        ).reset_index(level=0, drop=True).fillna(1.0)

        team_df["home_advantage_index"] = team_df["hist_home_goals"] - team_df["hist_away_goals"]
        team_df = team_df.drop(columns=["hist_home_goals", "hist_away_goals"])

        # Cleanup
        team_df = team_df.drop(columns=["_win", "_not_loss", "_clean_sheet"], errors="ignore")

        dom_cols = [c for c in team_df.columns if any(k in c for k in ("streak", "dominance", "clean_sheet", "pressure_index", "home_advantage_index"))]
        logger.info("[L5] Dominance features: %s", dom_cols)
        return team_df

    # ==================================================================
    # Layer 6 — Matchup Deltas
    # ==================================================================
    def _layer_6_matchup_deltas(
        self,
        matches: pd.DataFrame,
        team_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Reassemble match view from team table and compute interaction features.
        """
        logger.info("[L6] Building matchup features")

        # Select team-level features to carry forward
        id_cols = ["match_hash", "team_id", "is_home"]
        skip_prefixes = ("home_", "away_", "_match")
        skip_exact = {"opponent_id", "goals_scored", "goals_conceded",
                      "corners_scored", "corners_conceded",
                      "cards_scored", "cards_conceded", "cards_for", "cards_against",
                      "shots_for", "shots_against",
                      "shots_on_target_for", "shots_on_target_against",
                      "date", "result", "league", "season", "status", "source",
                      "source_file", "competition", "match_total_cards",
                      "total_corners", "total_shots", "match_hash"}

        feature_cols = [
            c for c in team_df.columns
            if c not in id_cols
            and c not in skip_exact
            and not any(c.startswith(p) for p in skip_prefixes)
            # Allow max_source_date columns to pass through for final leakage assertion, but skip other internals
            and (not c.startswith("_") or "max_source_date" in c)
        ]

        # Split home/away
        carry = id_cols + feature_cols
        # Deduplicate column list
        carry = list(dict.fromkeys(carry))

        home_feats = team_df[team_df["is_home"] == 1][carry].copy()
        away_feats = team_df[team_df["is_home"] == 0][carry].copy()

        home_feats = home_feats.rename(columns={c: f"home_{c}" for c in feature_cols})
        away_feats = away_feats.rename(columns={c: f"away_{c}" for c in feature_cols})

        home_feats = home_feats.rename(columns={"match_hash": "match_hash"})
        away_feats = away_feats.rename(columns={"match_hash": "match_hash"})

        # Drop is_home and team_id from the merge set
        home_feats = home_feats.drop(columns=["is_home", "team_id"], errors="ignore")
        away_feats = away_feats.drop(columns=["is_home", "team_id"], errors="ignore")

        # Merge back to match-level
        # Start from a slim version of matches (identity columns only + targets)
        identity_cols = ["match_hash", "date", "league", "season",
                         "home_team", "away_team", "home_score", "away_score", "result",
                         "home_corners", "away_corners", 
                         "home_cards", "away_cards",
                         "home_yellow_cards", "away_yellow_cards",
                         "home_red_cards", "away_red_cards",
                         "home_total_cards", "away_total_cards", "match_total_cards",
                         "home_shots", "away_shots",
                         "home_shots_on_target", "away_shots_on_target",
                         "referee", "referee_id",
                         "stadium_latitude", "stadium_longitude",
                         "stadium_lat", "stadium_lon", "stadium_lng",
                         "latitude", "longitude", "lat", "lon"]
        identity_cols = [c for c in identity_cols if c in matches.columns]
        matrix = matches[identity_cols].copy()

        matrix = matrix.merge(home_feats, on="match_hash", how="left")
        matrix = matrix.merge(away_feats, on="match_hash", how="left")

        # ── Interaction features ──
        def _safe_delta(col_h: str, col_a: str, name: str) -> None:
            if col_h in matrix.columns and col_a in matrix.columns:
                matrix[name] = matrix[col_h] - matrix[col_a]
                matrix[f"abs_{name}"] = matrix[name].abs()

        # Attack vs defense gaps
        _safe_delta("home_goal_attack_strength", "away_goal_defense_strength", "attack_vs_defense_gap")
        _safe_delta("away_goal_attack_strength", "home_goal_defense_strength", "away_attack_vs_home_defense")
        
        # STD gaps
        if "home_std_goals_scored" in matrix.columns:
            _safe_delta("home_std_goals_scored", "away_std_goals_scored", "std_goals_scored_gap")
            _safe_delta("home_std_goals_conceded", "away_std_goals_conceded", "std_goals_conceded_gap")

        # Dominance gap
        _safe_delta("home_goal_dominance_5", "away_goal_dominance_5", "dominance_gap")

        # Form gap
        _safe_delta("home_form_rating", "away_form_rating", "form_gap")

        # Rest days gap
        _safe_delta("home_days_rest", "away_days_rest", "rest_days_gap")

        # Imbalances
        _safe_delta("home_shot_attack_strength", "away_shot_defense_strength", "tactical_imbalance")
        _safe_delta("home_pressure_index_5", "away_pressure_index_5", "territorial_imbalance")

        # Shot accuracy gap
        if "home_shot_dominance_5" in matrix.columns:
            _safe_delta("home_shot_dominance_5", "away_shot_dominance_5", "shot_dominance_gap")

        # Season goal diff gap
        _safe_delta("home_season_goal_diff", "away_season_goal_diff", "season_gd_gap")

        interaction_cols = [c for c in matrix.columns if "gap" in c]
        logger.info("[L6] Interaction features: %s", interaction_cols)
        return matrix

    # ==================================================================
    # Layer 7 — Context Signals
    # ==================================================================
    def _layer_7_context_signals(self, matrix: pd.DataFrame) -> pd.DataFrame:
        """Add contextual features."""
        logger.info("[L7] Adding context signals")

        matrix["date"] = pd.to_datetime(matrix["date"])

        # Day of week / weekend
        matrix["day_of_week"] = matrix["date"].dt.dayofweek
        matrix["is_weekend"] = matrix["day_of_week"].isin([5, 6]).astype(int)

        # Season progress (0→1, August=0)
        matrix["season_progress"] = (matrix["date"].dt.month - 8) % 12 / 12.0

        # Season phase bucket
        matrix["season_phase"] = pd.cut(
            matrix["season_progress"],
            bins=[-0.01, 0.33, 0.66, 1.01],
            labels=["early", "mid", "late"],
        )

        # Home advantage constant
        matrix["is_home"] = 1  # Always from home perspective in match view

        # Fixture congestion: matches in last 7 days per team
        # Computed from the match dates in the dataset
        matrix = matrix.sort_values("date").reset_index(drop=True)
        matrix["home_fixture_congestion"] = 0
        matrix["away_fixture_congestion"] = 0

        # Build a team→match date lookup for efficient congestion calculation
        team_dates: Dict[str, List[pd.Timestamp]] = {}
        for _, row in matrix.iterrows():
            for team_col in ("home_team", "away_team"):
                team = row[team_col]
                if team not in team_dates:
                    team_dates[team] = []
                team_dates[team].append(row["date"])

        # Re-iterate to count matches in prior 7 days
        import bisect
        for idx, row in matrix.iterrows():
            match_date = row["date"]
            cutoff = match_date - pd.Timedelta(days=7)

            for team_col, cong_col in [("home_team", "home_fixture_congestion"),
                                        ("away_team", "away_fixture_congestion")]:
                team = row[team_col]
                dates = team_dates[team]
                # Count dates in (cutoff, match_date) — exclusive of current match
                lo = bisect.bisect_right(dates, cutoff)
                hi = bisect.bisect_left(dates, match_date)
                matrix.at[idx, cong_col] = hi - lo

        # Standardize naming for regression
        if "home_days_rest" in matrix.columns:
            matrix = matrix.rename(columns={"home_days_rest": "home_days_since_last_match", 
                                            "away_days_rest": "away_days_since_last_match"})

        # Referee rolling card-rate context (last 10 matches, shifted)
        matrix = self.engineer.add_referee_features(matrix, window=10)

        # Pre-match weather context (fails safe to 0.0 on unavailable API/coords)
        matrix = self.engineer.add_weather_features(matrix)

        logger.info("[L7] Context signals added")
        return matrix

    # ==================================================================
    # Post-build: Leakage Check, Freeze Hash & Coverage
    # ==================================================================
    def _assert_temporal_integrity(self, matrix: pd.DataFrame) -> None:
        """Asserts that no feature was derived from data on or after the match date."""
        target_dates = pd.to_datetime(matrix["date"])
        
        max_source_cols = [c for c in matrix.columns if '_max_source_date' in c]
        
        if not max_source_cols:
            logger.warning("[Leakage] No max_source_date columns found to verify.")
            
        for col in max_source_cols:
            source_dates = pd.to_datetime(matrix[col])
            # Strict inequality: source date MUST BE less than match date
            leakage = source_dates >= target_dates
            
            if leakage.any():
                leaked_matches = matrix.loc[leakage, ["match_hash", "date", col]]
                raise ValueError(
                    f"CRITICAL LEAKAGE DETECTED in feature pipeline: \n"
                    f"Column {col} has source dates >= match_date.\n"
                    f"Sample leaked rows:\n{leaked_matches.head()}"
                )
        logger.info(f"[Leakage] Integrity verified against {len(max_source_cols)} timeline trackers.")

    def _write_freeze_hash(self, matrix_path: Path, matrix: pd.DataFrame) -> Dict[str, Any]:
        """Compute SHA256 of the output file and save metadata."""
        sha = hashlib.sha256(matrix_path.read_bytes()).hexdigest()
        
        missing_count = int(matrix.isna().sum().sum())
        
        min_date = matrix["date"].min()
        max_date = matrix["date"].max()
        min_date_str = str(min_date.date()) if not matrix.empty else None
        max_date_str = str(max_date.date()) if not matrix.empty else None

        payload = {
            "sha256": sha,
            "build_timestamp": datetime.now(timezone.utc).isoformat(),
            "feature_generation_cutoff_date": max_date_str,
            "max_feature_lag_days": 1, # since shift(1) is universally enforced
            "max_rolling_window": max(WINDOWS) if WINDOWS else 20,
            "std_smoothing_factor": 10,
            "rows": len(matrix),
            "columns": len(matrix.columns),
            "feature_count": len(matrix.columns),
            "min_date": min_date_str,
            "max_date": max_date_str,
            "date_range": {"start": min_date_str, "end": max_date_str},
            "missing_values_total": missing_count,
        }

        hash_path = self.output_dir / "feature_matrix_hash.json"
        with open(hash_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        logger.info("[Freeze] SHA256: %s", sha[:16] + "…")
        return payload

    def _write_coverage_report(self, matrix: pd.DataFrame) -> None:
        """Generate feature coverage report."""
        missing = {
            col: int(matrix[col].isna().sum())
            for col in matrix.columns
            if matrix[col].isna().any()
        }

        report = {
            "rows": len(matrix),
            "columns": len(matrix.columns),
            "feature_count": len(matrix.columns),
            "missing_values_per_feature": missing,
            "league_distribution": matrix["league"].value_counts().to_dict() if "league" in matrix.columns else {},
            "season_distribution": matrix["season"].value_counts().to_dict() if "season" in matrix.columns else {},
            "features_list": sorted(matrix.columns.tolist()),
        }

        cov_path = self.output_dir / "feature_matrix_coverage.json"
        with open(cov_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str)

        logger.info("[Coverage] %d features, %d missing-value columns", report["feature_count"], len(missing))

    def _write_feature_manifest(self) -> None:
        """Write feature lineage metadata."""
        manifest = {
            "rolling_goals_scored_X": {"source": "goals", "layer": 3, "method": "rolling_mean_shifted"},
            "rolling_shots_for_X": {"source": "shots", "layer": 3, "method": "rolling_mean_shifted"},
            "goal_attack_strength": {"source": "rolling_goals_scored", "layer": 4, "method": "delta_vs_opponent_conceded"},
            "goal_defense_strength": {"source": "rolling_goals_conceded", "layer": 4, "method": "delta_vs_opponent_scored"},
            "pressure_index": {"source": "corners + shots", "layer": 5, "method": "weighted_sum"},
            "home_advantage_index": {"source": "goals_scored", "layer": 5, "method": "expanding_home_away_diff"},
            "tactical_imbalance": {"source": "attack_strength", "layer": 6, "method": "home_away_diff"},
            "fixture_congestion": {"source": "date", "layer": 7, "method": "count_matches_last_7_days"}
        }
        manifest_path = self.output_dir / "feature_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
