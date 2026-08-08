"""
Prediction Reconciler.

Reconciles historical predictions against labeled match results to
generate integrity and performance metrics. Matches predictions to
results using deterministic fingerprints and persists evaluations.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import requests

# Add project root if needed (legacy pattern, kept for safety but cleaned)
sys.path.append(os.getcwd())

from src.config import DATA_DIR, FOOTBALL_DATA_API_KEY
from src.monitoring.performance_engine import PerformanceTracker
from src.utils.naming import generate_match_fingerprint, normalize_team_name

# Define public API
__all__ = ["Reconciler"]

logger = logging.getLogger(__name__)


class Reconciler:
    """
    Reconciles predictions with labeled results.

    Matches stored predictions (by date/hash) against the centralized
    labeled results file. Calculates errors and persists evaluation records.

    Attributes:
        LABELED_PATH: Path to the source-of-truth labeled results CSV.
        EVALS_DIR: Directory where daily evaluation reports are saved.
    """

    LABELED_PATH: Path = DATA_DIR / "results" / "labeled" / "results_labeled.csv"
    EVALS_DIR: Path = DATA_DIR / "evaluations"
    FOOTBALL_DATA_BASE_URL: str = "https://api.football-data.org/v4"
    DEFAULT_LEAGUES: List[str] = ["PL", "PD", "SA", "BL1", "FL1"]
    RETRY_ATTEMPTS: int = 3
    BACKOFF_BASE_SECONDS: float = 1.0
    REQUEST_TIMEOUT_SECONDS: int = 30

    def __init__(self) -> None:
        """Initialize the reconciler and ensure output directories exist."""
        self.EVALS_DIR.mkdir(parents=True, exist_ok=True)

    def reconcile_date(self, prediction_date_str: str) -> None:
        """
        Reconcile all predictions from a specific date.

        Args:
            prediction_date_str: Date string in 'YYYY-MM-DD' format.
        """
        pred_file = DATA_DIR / "predictions" / f"predictions_{prediction_date_str}.csv"

        if not pred_file.exists():
            logger.warning(f"No predictions found for {prediction_date_str}")
            return

        if not self.LABELED_PATH.exists():
            logger.error("Labeled results not found. Ingest results first.")
            return

        preds_df = pd.read_csv(pred_file)

        # 1. Ensure/Map match_hash
        if "match_hash" not in preds_df.columns:
            if "match_id" in preds_df.columns:
                logger.info("Legacy match_id detected. Recalculating fingerprints...")
                preds_df["match_hash"] = preds_df.apply(self._repair_hash, axis=1)
            else:
                logger.error(f"Prediction file {pred_file} missing join keys.")
                return

        labeled_df = pd.read_csv(self.LABELED_PATH)

        # 2. Join on match_hash
        merged = pd.merge(
            preds_df,
            labeled_df,
            on="match_hash",
            how="inner",
            suffixes=("", "_res"),
        )

        if merged.empty:
            logger.info(f"No results found for predictions on {prediction_date_str}")
            return

        # 3. Extract Outcomes
        eval_records = self._extract_outcomes(merged, prediction_date_str, labeled_df.columns)

        if not eval_records:
            logger.info("No matching markets found in labels.")
            return

        eval_df = pd.DataFrame(eval_records)

        # 4. Save (Append-Only per ID)
        self._save_evaluations(prediction_date_str, eval_df)

    def auto_settle(
        self,
        leagues: Optional[List[str]] = None,
        days_back: int = 7,
    ) -> Dict[str, Any]:
        """
        Fetch settled fixtures and feed outcomes into PerformanceTracker.

        Retry policy: 3 attempts with exponential backoff (1s, 2s, 4s).
        Fail-safe: fetch/reconcile/report errors are captured and returned.
        """
        if not FOOTBALL_DATA_API_KEY:
            logger.warning("auto_settle skipped: FOOTBALL_DATA_API_KEY is not configured.")
            return {"status": "skipped", "reason": "missing_api_key"}

        target_leagues = leagues or list(self.DEFAULT_LEAGUES)
        end_date = datetime.now(timezone.utc).date()
        start_date = end_date - pd.Timedelta(days=days_back)
        all_rows: List[Dict[str, Any]] = []
        errors: List[str] = []

        for league in target_leagues:
            fixtures = self._fetch_settled_fixtures_with_retry(
                league=league,
                date_from=str(start_date),
                date_to=str(end_date),
            )
            if fixtures is None:
                errors.append(f"{league}:fetch_failed")
                continue

            for fixture in fixtures:
                row = self._fixture_to_label_row(league, fixture)
                if row is not None:
                    all_rows.append(row)

        upsert = self._upsert_labeled_rows(all_rows)
        affected_dates = sorted(
            {
                str(row.get("kickoff_date_utc", ""))[:10]
                for row in all_rows
                if row.get("kickoff_date_utc")
            }
        )

        reconciled_dates: List[str] = []
        for prediction_date in affected_dates:
            try:
                self.reconcile_date(prediction_date)
                reconciled_dates.append(prediction_date)
            except Exception as exc:
                logger.warning("auto_settle reconciliation failed for %s: %s", prediction_date, exc)
                errors.append(f"{prediction_date}:reconcile_failed")

        report_status = "OK"
        try:
            report = PerformanceTracker().generate_report(last_n_days=days_back)
            report_status = str(report.get("status", "OK")) if isinstance(report, dict) else "OK"
        except Exception as exc:
            logger.warning("auto_settle performance report failed: %s", exc)
            errors.append("performance_report_failed")
            report_status = "ERROR"

        return {
            "status": "ok" if not errors else "partial",
            "leagues": target_leagues,
            "fixtures_settled": len(all_rows),
            "labels_inserted": upsert["inserted"],
            "labels_updated": upsert["updated"],
            "reconciled_dates": reconciled_dates,
            "performance_report_status": report_status,
            "errors": errors,
        }

    def _repair_hash(self, row: pd.Series) -> str:
        """Attempt to regenerate match hash from legacy row data."""
        try:
            match_date = pd.to_datetime(row["date"])
            return generate_match_fingerprint(
                row["league"], match_date, row["home_team"], row["away_team"]
            )
        except Exception:
            # Fallback to existing ID if regeneration fails
            return str(row["match_id"])

    def _extract_outcomes(
        self,
        merged_df: pd.DataFrame,
        prediction_date_str: str,
        labeled_columns: pd.Index,
    ) -> List[Dict[str, Any]]:
        """Extract evaluation records from merged dataframe."""
        eval_records = []
        resolved_date = datetime.now().strftime("%Y-%m-%d")

        for _, row in merged_df.iterrows():
            market = row["market"]
            # The outcome is in the column named after the market in labeled_df
            if market in labeled_columns:
                actual_outcome = row[market]
                prob = row["predicted_probability"]
                error = prob - actual_outcome

                eval_records.append(
                    {
                        "evaluation_id": f"{row['match_hash']}_{market}",
                        "match_hash": row["match_hash"],
                        "market": market,
                        "predicted_probability": prob,
                        "actual_outcome": actual_outcome,
                        "error": error,
                        "abs_error": abs(error),
                        "decision_tier": row.get("decision_tier", "none"),
                        "confidence_tier": row.get("confidence_tier", "unfiltered"),
                        "model_version": row.get("model_version", "unknown"),
                        "prediction_date": prediction_date_str,
                        "resolved_date": resolved_date,
                    }
                )
        return eval_records

    def _save_evaluations(self, pred_date: str, new_evals: pd.DataFrame) -> None:
        """
        Save evaluations to CSV, avoiding duplicates.

        Args:
            pred_date: Date of the predictions.
            new_evals: DataFrame of new evaluation records.
        """
        eval_path = self.EVALS_DIR / f"evaluations_{pred_date}.csv"

        if eval_path.exists():
            existing = pd.read_csv(eval_path)
            # Only append IDs not already in existing
            to_append = new_evals[~new_evals["evaluation_id"].isin(existing["evaluation_id"])]

            if to_append.empty:
                logger.info(f"All predictions for {pred_date} already reconciled.")
                return

            combined = pd.concat([existing, to_append], ignore_index=True)
            combined.to_csv(eval_path, index=False)
            logger.info(f"Appended {len(to_append)} new evaluations to {eval_path}")
        else:
            new_evals.to_csv(eval_path, index=False)
            logger.info(f"Created evaluation file for {pred_date} with {len(new_evals)} records.")

    def _fetch_settled_fixtures_with_retry(
        self,
        *,
        league: str,
        date_from: str,
        date_to: str,
    ) -> Optional[List[Dict[str, Any]]]:
        """Fetch finished fixtures from football-data.org with retries."""
        url = f"{self.FOOTBALL_DATA_BASE_URL}/competitions/{league}/matches"
        headers = {"X-Auth-Token": FOOTBALL_DATA_API_KEY or ""}
        params = {"status": "FINISHED", "dateFrom": date_from, "dateTo": date_to}

        for attempt in range(1, self.RETRY_ATTEMPTS + 1):
            try:
                response = requests.get(
                    url,
                    headers=headers,
                    params=params,
                    timeout=self.REQUEST_TIMEOUT_SECONDS,
                )
                response.raise_for_status()
                payload = response.json()
                matches = payload.get("matches", [])
                if isinstance(matches, list):
                    return matches
                return []
            except Exception as exc:
                if attempt == self.RETRY_ATTEMPTS:
                    logger.error(
                        "auto_settle fetch failed for %s after %d attempts: %s",
                        league,
                        attempt,
                        exc,
                    )
                    return None

                delay = self.BACKOFF_BASE_SECONDS * (2 ** (attempt - 1))
                logger.warning(
                    "auto_settle retry %d/%d for %s in %.1fs (%s)",
                    attempt,
                    self.RETRY_ATTEMPTS,
                    league,
                    delay,
                    exc,
                )
                time.sleep(delay)

        return None

    def _fixture_to_label_row(self, league: str, fixture: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Convert a football-data fixture into partial labeled outcome row."""
        utc_date = fixture.get("utcDate")
        home_name = fixture.get("homeTeam", {}).get("name")
        away_name = fixture.get("awayTeam", {}).get("name")
        full_time = fixture.get("score", {}).get("fullTime", {})
        home_goals = full_time.get("home")
        away_goals = full_time.get("away")

        if not utc_date or not home_name or not away_name:
            return None
        if home_goals is None or away_goals is None:
            return None

        try:
            kickoff = pd.to_datetime(utc_date, utc=True).to_pydatetime()
        except Exception:
            return None

        home = normalize_team_name(str(home_name), league=league)
        away = normalize_team_name(str(away_name), league=league)
        match_hash = generate_match_fingerprint(league, kickoff, home, away)

        total_goals = float(home_goals + away_goals)
        return {
            "match_hash": match_hash,
            "kickoff_date_utc": utc_date,
            "home_win": float(home_goals > away_goals),
            "draw": float(home_goals == away_goals),
            "away_win": float(away_goals > home_goals),
            "over_2_5": float(total_goals > 2.5),
            "under_2_5": float(total_goals <= 2.5),
            "goals_over_2_5": float(total_goals > 2.5),
            "goals_under_2_5": float(total_goals <= 2.5),
            "over_2_5_goals": float(total_goals > 2.5),
            "under_2_5_goals": float(total_goals <= 2.5),
            "home_under_1_5": float(home_goals <= 1),
            "away_under_1_5": float(away_goals <= 1),
            "btts_yes": float(home_goals > 0 and away_goals > 0),
            "btts_no": float(not (home_goals > 0 and away_goals > 0)),
        }

    def _upsert_labeled_rows(self, rows: List[Dict[str, Any]]) -> Dict[str, int]:
        """Upsert rows into labeled results keyed by match_hash."""
        if not rows:
            return {"inserted": 0, "updated": 0}

        self.LABELED_PATH.parent.mkdir(parents=True, exist_ok=True)
        incoming = pd.DataFrame(rows).drop_duplicates(subset=["match_hash"], keep="last")

        if self.LABELED_PATH.exists():
            existing = pd.read_csv(self.LABELED_PATH)
        else:
            existing = pd.DataFrame(columns=incoming.columns)

        existing_count = len(existing)
        existing_hashes = set(existing["match_hash"].astype(str)) if "match_hash" in existing.columns else set()
        incoming_hashes = set(incoming["match_hash"].astype(str))
        updated = len(existing_hashes.intersection(incoming_hashes))

        if existing.empty:
            combined = incoming.copy()
        else:
            combined = pd.concat([existing, incoming], ignore_index=True, sort=False)
        combined = combined.drop_duplicates(subset=["match_hash"], keep="last")
        combined.to_csv(self.LABELED_PATH, index=False)

        inserted = max(len(combined) - existing_count, 0)
        return {"inserted": inserted, "updated": updated}
