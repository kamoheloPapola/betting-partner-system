"""
Live Performance Engine.

Aggregates reconciled evaluations and bet logs into rolling
performance metrics: Brier Score, Win Rate by confidence bucket,
and ROI tracking. Designed to turn the system from a static
predictor into a learning machine.
"""
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.config import DATA_DIR
from src.ml.calibration import (
    ECE_THRESHOLD,
    adjust_dampening_alpha,
    calculate_ece,
    get_dampening_alpha,
)

logger = logging.getLogger(__name__)

__all__ = ["PerformanceTracker"]


class PerformanceTracker:
    """
    Aggregates reconciled prediction evaluations into actionable
    performance metrics for continuous model monitoring.
    """

    EVALS_DIR: Path = DATA_DIR / "evaluations"
    BET_LOG_PATH: Path = DATA_DIR / "betting_log.csv"
    REPORT_DIR: Path = DATA_DIR / "monitoring" / "performance"
    RELIABILITY_DIR: Path = DATA_DIR / "monitoring"
    
    # Confidence buckets for calibration analysis
    CONFIDENCE_BUCKETS = [
        (0.0, 0.40, "low"),
        (0.40, 0.55, "medium"),
        (0.55, 0.70, "high"),
        (0.70, 1.01, "very_high"),
    ]

    def __init__(self) -> None:
        self.REPORT_DIR.mkdir(parents=True, exist_ok=True)
        self.RELIABILITY_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_report(self, last_n_days: Optional[int] = None) -> Dict[str, Any]:
        """
        Generate a comprehensive performance report.

        Args:
            last_n_days: If set, only include evaluations from the last N days.

        Returns:
            Dictionary with brier_score, calibration, roi, and metadata.
        """
        evals_df = self._load_all_evaluations()

        if evals_df.empty:
            logger.warning("PerformanceTracker: No evaluations found.")
            return {"status": "NO_DATA", "message": "No reconciled evaluations available."}

        if last_n_days is not None:
            cutoff = pd.Timestamp.now() - pd.Timedelta(days=last_n_days)
            evals_df["prediction_date_dt"] = pd.to_datetime(evals_df["prediction_date"], errors="coerce")
            evals_df = evals_df[evals_df["prediction_date_dt"] >= cutoff]
            if evals_df.empty:
                return {"status": "NO_DATA", "message": f"No evaluations in last {last_n_days} days."}

        accuracy_by_model = self._compute_accuracy_by_model(evals_df, group_col="model_version")
        accuracy_by_market = self._compute_accuracy_by_model(evals_df, group_col="market")

        report: Dict[str, Any] = {
            "generated_at": datetime.now().isoformat(),
            "evaluation_count": len(evals_df),
            "brier_score": self._compute_brier(evals_df),
            "calibration_by_bucket": self._compute_calibration_buckets(evals_df),
            "ece_by_market": self._compute_ece_by_market(evals_df),
            "accuracy_by_model": accuracy_by_model,
            "accuracy_by_market": accuracy_by_market,
            "roi": self._compute_roi(),
        }
        report["auto_calibration_adjustments"] = self._apply_live_ece_feedback(report["ece_by_market"])
        report["reliability_diagrams"] = self._generate_reliability_diagrams(evals_df)

        # Persist report
        report_path = self.REPORT_DIR / f"performance_{datetime.now().strftime('%Y-%m-%d')}.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str)
        logger.info(f"Performance report saved to {report_path}")

        return report

    def _generate_reliability_diagrams(self, evals_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Generate and persist reliability diagrams per market per league.

        Files are written as:
            data/monitoring/reliability_{market}_{league}_{date}.json
        """
        required = {"market", "predicted_probability", "actual_outcome"}
        if evals_df.empty or not required.issubset(set(evals_df.columns)):
            return []

        valid = evals_df.dropna(subset=["market", "predicted_probability", "actual_outcome"]).copy()
        if valid.empty:
            return []

        if "league" not in valid.columns:
            valid["league"] = "Global"
        else:
            valid["league"] = valid["league"].fillna("Global").astype(str)
            valid.loc[valid["league"].str.strip() == "", "league"] = "Global"

        valid["market"] = valid["market"].astype(str)
        valid["prob"] = pd.to_numeric(valid["predicted_probability"], errors="coerce")
        valid["outcome"] = pd.to_numeric(valid["actual_outcome"], errors="coerce")
        valid = valid.dropna(subset=["prob", "outcome"])
        if valid.empty:
            return []

        valid["prob"] = valid["prob"].clip(0.0, 1.0)
        generated_at = datetime.now()
        date_token = generated_at.strftime("%Y-%m-%d")

        diagrams: List[Dict[str, Any]] = []
        for (market, league), group in valid.groupby(["market", "league"], dropna=False):
            buckets = self._compute_reliability_buckets(group)
            max_deviation = max(
                (
                    float(bucket["abs_gap"])
                    for bucket in buckets
                    if bucket.get("abs_gap") is not None
                ),
                default=0.0,
            )
            systematic_miscalibration = any(
                float(bucket.get("abs_gap", 0.0) or 0.0) > 0.10 for bucket in buckets
            )

            payload: Dict[str, Any] = {
                "generated_at": generated_at.isoformat(),
                "date": date_token,
                "market": str(market),
                "league": str(league),
                "total_samples": int(len(group)),
                "deviation_threshold": 0.10,
                "max_deviation": round(max_deviation, 4),
                "systematic_miscalibration": systematic_miscalibration,
                "buckets": buckets,
            }

            market_token = self._safe_filename_token(str(market))
            league_token = self._safe_filename_token(str(league))
            file_path = self.RELIABILITY_DIR / f"reliability_{market_token}_{league_token}_{date_token}.json"
            with open(file_path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)

            if systematic_miscalibration:
                logger.warning(
                    "Systematic miscalibration detected | market=%s league=%s max_deviation=%.3f",
                    market,
                    league,
                    max_deviation,
                )

            diagrams.append(
                {
                    "market": str(market),
                    "league": str(league),
                    "path": str(file_path),
                    "max_deviation": round(max_deviation, 4),
                    "systematic_miscalibration": systematic_miscalibration,
                }
            )

        return diagrams

    @staticmethod
    def _safe_filename_token(value: str) -> str:
        token = re.sub(r"[^A-Za-z0-9]+", "_", str(value)).strip("_")
        return token or "unknown"

    @staticmethod
    def _compute_reliability_buckets(evals_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Build 10 reliability buckets from 0-10% to 90-100%.
        """
        bins = np.linspace(0.0, 1.0, 11)
        rows: List[Dict[str, Any]] = []

        for idx in range(10):
            lo = float(bins[idx])
            hi = float(bins[idx + 1])
            if idx == 9:
                mask = (evals_df["prob"] >= lo) & (evals_df["prob"] <= hi)
            else:
                mask = (evals_df["prob"] >= lo) & (evals_df["prob"] < hi)

            subset = evals_df[mask]
            count = int(len(subset))
            mean_pred = float(subset["prob"].mean()) if count else None
            actual_rate = float(subset["outcome"].mean()) if count else None
            gap = (mean_pred - actual_rate) if count else None

            rows.append(
                {
                    "bucket": f"{int(lo * 100)}-{int(hi * 100)}%",
                    "range_start": round(lo, 2),
                    "range_end": round(hi, 2),
                    "count": count,
                    "predicted_confidence": round(mean_pred, 4) if mean_pred is not None else None,
                    "actual_win_rate": round(actual_rate, 4) if actual_rate is not None else None,
                    "gap": round(gap, 4) if gap is not None else None,
                    "abs_gap": round(abs(gap), 4) if gap is not None else None,
                }
            )

        return rows

    # ------------------------------------------------------------------
    # ECE By Market + Live Alpha Feedback
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_ece_by_market(evals_df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """Compute per-market ECE and direction of calibration error."""
        required = {"market", "predicted_probability", "actual_outcome"}
        if not required.issubset(set(evals_df.columns)):
            return {}

        valid = evals_df.dropna(subset=["market", "predicted_probability", "actual_outcome"]).copy()
        if valid.empty:
            return {}

        valid["prob"] = valid["predicted_probability"].astype(float)
        valid["outcome"] = valid["actual_outcome"].astype(float)

        ece_by_market: Dict[str, Dict[str, Any]] = {}
        for market, group in valid.groupby("market"):
            probs = group["prob"].to_numpy(dtype=float)
            outcomes = group["outcome"].to_numpy(dtype=float)
            gap = float(probs.mean() - outcomes.mean())
            if gap > 0:
                confidence_state = "overconfident"
            elif gap < 0:
                confidence_state = "underconfident"
            else:
                confidence_state = "balanced"

            ece_by_market[str(market)] = {
                "count": int(len(group)),
                "ece": round(float(calculate_ece(outcomes, probs)), 5),
                "calibration_gap": round(gap, 5),
                "confidence_state": confidence_state,
            }
        return ece_by_market

    @staticmethod
    def _apply_live_ece_feedback(ece_by_market: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Auto-adjust market dampening alpha when ECE breaches threshold.
        """
        adjustments: List[Dict[str, Any]] = []
        for market, metrics in ece_by_market.items():
            ece = float(metrics.get("ece", 0.0))
            if ece <= ECE_THRESHOLD:
                continue

            confidence_state = metrics.get("confidence_state")
            if confidence_state == "overconfident":
                direction = "more_dampening"
            elif confidence_state == "underconfident":
                direction = "less_dampening"
            else:
                continue

            before = get_dampening_alpha(market)
            old_alpha, new_alpha = adjust_dampening_alpha(market, direction=direction, step=0.05)
            if new_alpha == old_alpha:
                continue

            adjustment = {
                "market": market,
                "ece": round(ece, 5),
                "direction": direction,
                "alpha_before": round(float(before), 4),
                "alpha_after": round(float(new_alpha), 4),
                "reason": confidence_state,
            }
            adjustments.append(adjustment)
            logger.info(
                "Auto-adjusted dampening alpha | market=%s ece=%.4f state=%s alpha: %.2f -> %.2f",
                market,
                ece,
                confidence_state,
                old_alpha,
                new_alpha,
            )

        return adjustments

    # ------------------------------------------------------------------
    # Brier Score
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_brier(evals_df: pd.DataFrame) -> Dict[str, Any]:
        """Compute rolling Brier score from reconciled evaluations."""
        valid = evals_df.dropna(subset=["predicted_probability", "actual_outcome"])
        if valid.empty:
            return {"value": None, "sample_size": 0}

        probs = valid["predicted_probability"].astype(float).values
        outcomes = valid["actual_outcome"].astype(float).values
        brier = float(np.mean((probs - outcomes) ** 2))

        return {
            "value": round(brier, 5),
            "sample_size": len(valid),
            "mean_predicted": round(float(np.mean(probs)), 4),
            "mean_actual": round(float(np.mean(outcomes)), 4),
        }

    # ------------------------------------------------------------------
    # Calibration Buckets
    # ------------------------------------------------------------------

    def _compute_calibration_buckets(self, evals_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Split predictions into confidence buckets and compare
        predicted probability vs actual hit rate.
        """
        valid = evals_df.dropna(subset=["predicted_probability", "actual_outcome"]).copy()
        if valid.empty:
            return []

        valid["prob"] = valid["predicted_probability"].astype(float)
        valid["outcome"] = valid["actual_outcome"].astype(float)
        buckets = []

        for lo, hi, label in self.CONFIDENCE_BUCKETS:
            mask = (valid["prob"] >= lo) & (valid["prob"] < hi)
            subset = valid[mask]
            if len(subset) == 0:
                continue

            mean_pred = float(subset["prob"].mean())
            mean_actual = float(subset["outcome"].mean())
            gap = mean_pred - mean_actual

            buckets.append({
                "bucket": label,
                "range": f"{lo:.0%}-{hi:.0%}",
                "count": len(subset),
                "mean_predicted": round(mean_pred, 4),
                "mean_actual": round(mean_actual, 4),
                "calibration_gap": round(gap, 4),
            })

        return buckets

    # ------------------------------------------------------------------
    # Accuracy by Model Version
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_accuracy_by_model(
        evals_df: pd.DataFrame,
        group_col: str = "model_version",
    ) -> Dict[str, Any]:
        """Group accuracy metrics by a selected column (default: model version)."""
        if group_col not in evals_df.columns:
            return {}

        valid = evals_df.dropna(subset=["predicted_probability", "actual_outcome"]).copy()
        if valid.empty:
            return {}

        valid["correct"] = (
            (valid["predicted_probability"].astype(float) >= 0.5)
            == (valid["actual_outcome"].astype(float) >= 0.5)
        )

        result = {}
        for group_value, group in valid.groupby(group_col):
            result[str(group_value)] = {
                "count": len(group),
                "accuracy": round(float(group["correct"].mean()), 4),
                "brier": round(
                    float(np.mean((group["predicted_probability"].astype(float).values
                                   - group["actual_outcome"].astype(float).values) ** 2)),
                    5,
                ),
            }
        return result

    # ------------------------------------------------------------------
    # ROI from Bet Log
    # ------------------------------------------------------------------

    def _compute_roi(self) -> Dict[str, Any]:
        """Compute ROI from the bet logger's CSV."""
        if not self.BET_LOG_PATH.exists():
            return {"status": "NO_BET_LOG"}

        try:
            df = pd.read_csv(self.BET_LOG_PATH)
        except Exception as e:
            logger.error(f"Failed to read bet log: {e}")
            return {"status": "READ_ERROR", "error": str(e)}

        if df.empty:
            return {"status": "EMPTY", "total_bets": 0}

        total_staked = df["stake"].sum() if "stake" in df.columns else 0.0

        settled = df[df["outcome"].notna() & (df["outcome"] != "")]
        if settled.empty:
            return {
                "status": "NO_SETTLEMENTS",
                "total_bets": len(df),
                "total_staked": float(total_staked),
            }

        total_pnl = settled["pnl"].astype(float).sum()
        wins = (settled["outcome"] == "WIN").sum()
        roi = (total_pnl / total_staked * 100) if total_staked > 0 else 0.0

        return {
            "status": "OK",
            "total_bets": len(df),
            "settled_bets": len(settled),
            "wins": int(wins),
            "win_rate": round(float(wins / len(settled)), 4) if len(settled) > 0 else 0.0,
            "total_staked": round(float(total_staked), 2),
            "total_pnl": round(float(total_pnl), 2),
            "roi_pct": round(float(roi), 2),
        }

    # ------------------------------------------------------------------
    # Data Loading
    # ------------------------------------------------------------------

    def _load_all_evaluations(self) -> pd.DataFrame:
        """Load and concatenate all evaluation CSV files."""
        if not self.EVALS_DIR.exists():
            return pd.DataFrame()

        csv_files = sorted(self.EVALS_DIR.glob("evaluations_*.csv"))
        if not csv_files:
            return pd.DataFrame()

        frames = []
        for f in csv_files:
            try:
                frames.append(pd.read_csv(f))
            except Exception as e:
                logger.warning(f"Skipping corrupt evaluation file {f}: {e}")

        if not frames:
            return pd.DataFrame()

        return pd.concat(frames, ignore_index=True)
