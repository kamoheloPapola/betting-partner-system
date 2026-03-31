"""
Corner Calibration Evaluator.

Evaluates calibration of Corner Prediction Models using:
- Expected Calibration Error (ECE)
- Brier Score
- League-level stratification

Generates calibration reports with status classifications
(BOOST, ELIGIBLE, NEUTRAL, EXCLUDE) based on ECE thresholds.
"""
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from tqdm import tqdm

from src.config import DATA_DIR, PROCESSED_DATA_DIR
from src.config.evaluation import CalibrationConfig
from src.core.exceptions import DataValidationError

logger = logging.getLogger(__name__)

@dataclass
class CalibrationResult:
    """Results from calibration evaluation."""
    metrics_df: pd.DataFrame
    total_matches: int
    leagues_evaluated: int
    output_file: Optional[Path]
    timestamp: datetime

    @property
    def is_empty(self) -> bool:
        """Check if evaluation produced any results."""
        return self.metrics_df.empty
    
    @property
    def summary(self) -> str:
        """Human-readable summary."""
        if self.is_empty:
            return "Calibration: No data available"
        return (
            f"Calibration: {self.leagues_evaluated} leagues, "
            f"{self.total_matches} matches evaluated"
        )

class CornerCalibrationEvaluator:
    """
    Evaluates calibration of Corner Prediction Models (NB).
    Metrics: ECE (Expected Calibration Error), Brier Score.
    """
    
    def __init__(
        self, 
        log_dir: Optional[Path] = None,
        matches_dir: Optional[Path] = None,
        output_dir: Optional[Path] = None
    ):
        """
        Initialize Evaluator with configurable paths.
        
        Args:
            log_dir: Directory containing prediction logs.
            matches_dir: Directory containing processed match results.
            output_dir: Directory to save evaluation reports.
        """
        self.log_dir = log_dir or (DATA_DIR / "predictions")
        self.matches_dir = matches_dir or (PROCESSED_DATA_DIR / "matches")
        self.output_dir = output_dir or (DATA_DIR / "evaluation" / "corners")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config = CalibrationConfig()
        
    def _validate_logs(self, logs: pd.DataFrame) -> None:
        """Validate prediction logs have required structure and valid probabilities."""
        required = {'match_id', 'league', 'P_under_11_5'}
        missing = required - set(logs.columns)
        if missing:
            raise DataValidationError(
                f"Prediction logs missing required columns: {missing}",
                context={"available": list(logs.columns)}
            )
        
        # Validate probability ranges
        # Drop check for empty to avoid errors on empty logs which are handled upstream
        if not logs.empty and not logs['P_under_11_5'].between(0, 1).all():
            invalid_count = (~logs['P_under_11_5'].between(0, 1)).sum()
            raise DataValidationError(
                f"Invalid probabilities found: {invalid_count} values outside [0,1]"
            )

    def _load_prediction_logs(self) -> pd.DataFrame:
        """Load and validate prediction logs."""
        log_files = list(self.log_dir.glob("predictions_corners_*.csv"))
        if not log_files:
            logger.warning("No corner prediction logs found.")
            return pd.DataFrame()
            
        dfs = []
        for f in tqdm(log_files, desc="Loading prediction logs", leave=False):
            try:
                df = pd.read_csv(f)
                # Quick pre-validation to avoid appending garbage
                if 'match_id' in df.columns:
                     df['match_id'] = df['match_id'].astype(str)
                else:
                     logger.warning(f"File {f.name} missing match_id, skipping.")
                     continue
                     
                self._validate_logs(df)
                dfs.append(df)
            except Exception as e:
                logger.warning(f"Failed to load {f.name}: {e}")
                
        if not dfs:
             logger.warning("No valid prediction logs loaded.")
             return pd.DataFrame()
             
        logs = pd.concat(dfs, ignore_index=True)
        logs['league'] = logs['league'].astype(str)
        return logs

    def _calculate_target_labels(self, matches: pd.DataFrame) -> pd.DataFrame:
        """Calculate target labels from match results in a NaN-safe way."""
        if 'home_corners' not in matches.columns or 'away_corners' not in matches.columns:
            logger.warning("Match data missing corner columns.")
            return pd.DataFrame()

        # Preserve NaN properly
        # Summing converts NaN to NaN automatically in pandas series addition
        total_corners = matches['home_corners'] + matches['away_corners']
        
        matches['total_corners'] = total_corners
        
        # NaN-safe labeling (fixes silent data corruption)
        # Use np.where to handle NaNs explicitly
        matches['target_u11_5'] = np.where(
            total_corners.isna(),
            np.nan,
            (total_corners <= 11.5).astype(float)
        )
        
        matches['target_o9_5'] = np.where(
            total_corners.isna(),
            np.nan,
            (total_corners > 9.5).astype(float)
        )
        return matches

    def _load_match_results(self) -> pd.DataFrame:
        """Load and validate match results."""
        if not self.matches_dir.exists():
             logger.warning(f"Matches directory not found: {self.matches_dir}")
             return pd.DataFrame()

        match_files = list(self.matches_dir.glob("*.csv"))
        if not match_files:
            logger.warning("No processed match files found.")
            return pd.DataFrame()
            
        dfs = []
        for f in tqdm(match_files, desc="Loading match results", leave=False):
             try:
                 df = pd.read_csv(f)
                 if 'match_id' not in df.columns:
                     logger.warning(f"File {f.name} missing match_id, skipping.")
                     continue
                 df['match_id'] = df['match_id'].astype(str)
                 dfs.append(df)
             except Exception as e:
                 logger.warning(f"Failed to load {f.name}: {e}")
                 
        if not dfs:
            return pd.DataFrame()
            
        matches = pd.concat(dfs, ignore_index=True)
        
        # Deduplicate matches to ensure unique match_id for many-to-one merge
        before_dedup = len(matches)
        matches = matches.drop_duplicates(subset=['match_id'])
        after_dedup = len(matches)
        
        if before_dedup != after_dedup:
            logger.warning(f"Dropped {before_dedup - after_dedup} duplicate matches from result set.")
            
        return self._calculate_target_labels(matches)

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Orchestrate data loading.
        Returns: (prediction_logs, match_results)
        """
        logs = self._load_prediction_logs()
        matches = self._load_match_results()
        return logs, matches
        
    def evaluate(self) -> CalibrationResult:
        """
        Main evaluation loop.
        
        Returns:
            CalibrationResult (may have empty metrics_df if no data)
        """
        logs, matches = self.load_data()
        
        empty_result = CalibrationResult(
            metrics_df=pd.DataFrame(),
            total_matches=0,
            leagues_evaluated=0,
            output_file=None,
            timestamp=datetime.now()
        )

        if logs.empty or matches.empty:
            logger.warning("Data loading failed or returned empty datasets.")
            return empty_result
            
        # Before merge, validate match_id overlap
        logs_ids = set(logs['match_id'].unique())
        match_ids = set(matches['match_id'].unique())
        overlap = logs_ids & match_ids

        if not overlap:
            logger.error(
                "No match_id overlap between logs and matches",
                extra={
                    "sample_log_ids": list(logs_ids)[:5],
                    "sample_match_ids": list(match_ids)[:5],
                    "log_count": len(logs_ids),
                    "match_count": len(match_ids)
                }
            )
            return empty_result

        logger.info(
            f"Match ID overlap: {len(overlap)} / {len(logs_ids)} predictions have results"
        )

        # Merge
        try:
             merged = pd.merge(
                 logs, 
                 matches[['match_id', 'total_corners', 'target_u11_5', 'target_o9_5']], 
                 on='match_id', 
                 how='inner',
                 validate='many_to_one'
             )
        except Exception as e:
             logger.error(f"Failed to merge logs and matches: {e}")
             return empty_result
        
        # CRITICAL: Drop NaNs (Postponement Poisoning Fix)
        # We only calibrate on matches that actually happened.
        before_drop = len(merged)
        merged = merged.dropna(subset=['target_u11_5'])
        after_drop = len(merged)
        
        if before_drop != after_drop:
            logger.info(f"Calibration: Dropped {before_drop - after_drop} incomplete/postponed matches.")
                          
        if merged.empty:
            logger.warning("No matches overlap between logs and actuals (after filtering).")
            return empty_result
            
        results = []
        
        # Group by League
        leagues = merged['league'].unique()
        
        for lg in leagues:
            subset = merged[merged['league'] == lg]
            if len(subset) < self.config.MIN_SAMPLE_SIZE:
                continue
                
            # Metric: Under 11.5
            brier_u11 = self.brier_score(subset['P_under_11_5'], subset['target_u11_5'])
            ece_u11 = self.ece_score(subset['P_under_11_5'], subset['target_u11_5'])
            
            results.append({
                'league': lg,
                'target': 'U11.5',
                'sample_size': len(subset),
                'brier_score': round(brier_u11, 4),
                'ece': round(ece_u11, 4),
                'status': self.get_status(ece_u11),
                'timestamp': pd.Timestamp.now().isoformat(),
                # Additional diagnostics
                'mean_predicted_prob': round(subset['P_under_11_5'].mean(), 3),
                'observed_frequency': round(subset['target_u11_5'].mean(), 3),
                'calibration_gap': round(
                    subset['P_under_11_5'].mean() - subset['target_u11_5'].mean(), 
                    3
                ),
                'pred_std': round(subset['P_under_11_5'].std(), 3)
            })
            
            
        results_df = pd.DataFrame(results)
        
        output_file = None
        if not results_df.empty:
            # Save Report
            output_file = self.output_dir / f"calibration_report_{pd.Timestamp.now().strftime('%Y%m%d')}.csv"
            results_df.to_csv(output_file, index=False)
            
            # Interactive Output
            try:
                from rich.console import Console
                from rich.table import Table
                
                console = Console()
                table = Table(title="[bold cyan]Calibration Results[/bold cyan]")
                table.add_column("League", style="bold")
                table.add_column("Sample", justify="right")
                table.add_column("Brier", justify="right")
                table.add_column("ECE", justify="right")
                table.add_column("Status", justify="center")
                
                for _, row in results_df.iterrows():
                    status_color = {
                        "BOOST": "green",
                        "ELIGIBLE": "cyan",
                        "NEUTRAL": "yellow",
                        "EXCLUDE": "red"
                    }.get(row['status'], "white")
                    
                    table.add_row(
                        str(row['league']),
                        str(row['sample_size']),
                        f"{row['brier_score']:.4f}",
                        f"{row['ece']:.4f}",
                        f"[{status_color}]{row['status']}[/{status_color}]"
                    )
                
                console.print(table)
            except ImportError:
                logger.debug("rich not installed, skipping interactive table.")
            except Exception as e:
                logger.warning(f"Failed to display interactive table: {e}")

        else:
             logger.warning("No leagues met the minimum sample size criteria.")
        
        return CalibrationResult(
            metrics_df=results_df,
            total_matches=len(merged),
            leagues_evaluated=len(results_df),
            output_file=output_file,
            timestamp=datetime.now()
        )
        
    def brier_score(
        self, 
        probs: pd.Series, 
        actuals: pd.Series
    ) -> float:
        """
        Calculate Brier score (mean squared error of probabilities).
        
        Args:
            probs: Predicted probabilities [0, 1]
            actuals: Actual outcomes {0, 1}
            
        Returns:
            Brier score (lower is better, range [0, 1])
            
        References:
            Brier, G. W. (1950). "Verification of forecasts expressed in terms 
            of probability". Monthly Weather Review.
        """
        if len(probs) != len(actuals):
            raise ValueError(
                f"Length mismatch: probs={len(probs)}, actuals={len(actuals)}"
            )
        
        return float(np.mean((probs - actuals) ** 2))
        
    def ece_score(
        self, 
        probs: pd.Series, 
        actuals: pd.Series, 
        n_bins: Optional[int] = None
    ) -> float:
        """
        Calculate Expected Calibration Error.
        
        Measures difference between predicted probability and observed frequency
        across probability bins.
        
        Args:
            probs: Predicted probabilities
            actuals: Actual binary outcomes
            n_bins: Number of bins to divide probability space (default: config.ECE_NUM_BINS)
            
        Returns:
            ECE score (lower is better, range [0, 1])
            
        References:
            Naeini et al. (2015). "Obtaining Well Calibrated Probabilities 
            Using Bayesian Binning"
        """
        if len(probs) != len(actuals):
            raise ValueError(
                f"Length mismatch: probs={len(probs)}, actuals={len(actuals)}"
            )

        n_bins = n_bins or self.config.ECE_NUM_BINS
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        empty_bins = 0
        
        for i in range(n_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i+1]
            
            mask = (probs > bin_lower) & (probs <= bin_upper)
            
            if mask.sum() == 0:
                empty_bins += 1
                continue
            
            # Warn if bin too small
            if mask.sum() < self.config.MIN_BIN_SIZE:
                logger.debug(
                    f"Bin [{bin_lower:.2f}, {bin_upper:.2f}] has only {mask.sum()} samples"
                )
            
            bin_prob = probs[mask].mean()
            bin_actual = actuals[mask].mean()
            
            ece += np.abs(bin_prob - bin_actual) * (mask.sum() / len(probs))
        
        # Diagnostic: Too many empty bins suggests model output is clustered
        if empty_bins > n_bins / 2:
            logger.warning(
                f"ECE calculation: {empty_bins}/{n_bins} bins empty. "
                "Model may have poor probability diversity."
            )
            
        return ece
        
    def get_status(self, ece: float) -> str:
        """
        Classify model calibration quality based on ECE.
        
        Args:
            ece: Expected Calibration Error
            
        Returns:
            Status string: EXCLUDE | NEUTRAL | ELIGIBLE | BOOST
        """
        if ece > self.config.ECE_EXCLUDE_THRESHOLD: return "EXCLUDE"
        if ece < self.config.ECE_BOOST_THRESHOLD: return "BOOST"
        if ece < self.config.ECE_ELIGIBLE_THRESHOLD: return "ELIGIBLE"
        return "NEUTRAL"

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    try:
        evaluator = CornerCalibrationEvaluator()
        res = evaluator.evaluate()
        print(res)
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
