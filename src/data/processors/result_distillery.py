"""
Result Distillery.

Ingests raw CSV match results and distills them into the
Canonical Master Form. Features:
- Football-Data.co.uk format parsing
- Team name normalization
- Prediction-aware filtering
- Append-only master file with atomic writes
"""

import sys
import os
import pandas as pd
import numpy as np
import logging
import tempfile
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List
from functools import partial

from src.config import DATA_DIR
from dataclasses import dataclass
from src.core.exceptions import DataValidationError
from src.utils.naming import normalize_team_name, generate_match_fingerprint, calculate_season

logger = logging.getLogger(__name__)

@dataclass
class IngestionResult:
    """Statistics from ingestion operation."""
    file_path: Path
    rows_read: int
    rows_appended: int
    rows_skipped: int
    master_path: Path

class ResultDistillery:
    """
    Ingests raw CSV data and distills it into the Canonical Master Form.
    Follows Non-Negotiable Append-Only Design Principles.
    """
    
    REQUIRED_COLS = [
        'league', 'season', 'match_date', 'home_team', 'away_team', 
        'home_goals', 'away_goals', 'home_corners', 'away_corners', 
        'home_cards', 'away_cards', 'ingested_at'
    ]

    def __init__(
        self,
        master_path: Optional[Path] = None,
        predictions_dir: Optional[Path] = None
    ):
        """
        Initialize ResultDistillery with configurable paths.
        
        Args:
            master_path: Path to normalized master results file.
            predictions_dir: Path to predictions directory (for filtering).
        """
        self.master_path = master_path or (DATA_DIR / "results" / "normalized" / "results_master.csv")
        self.predictions_dir = predictions_dir or (DATA_DIR / "predictions")

    def _get_prediction_universe(self) -> Dict[str, Any]:
        """
        Loads all match_hashes and their earliest prediction_date from predictions path.
        Returns: {match_hash: min_prediction_date}
        Uses optimized vector loading.
        """
        universe = {}
        if not self.predictions_dir.exists():
            return universe

        # Collect all valid DataFrames
        dfs = []
        for f in self.predictions_dir.glob("predictions_*.csv"):
            try:
                # Read specific columns for speed
                df = pd.read_csv(f, usecols=['match_hash', 'prediction_date'])
                dfs.append(df)
            except ValueError:
                # Missing columns in file, skip
                continue
            except Exception as e:
                logger.warning(f"Error reading prediction file {f}: {e}")
                continue
        
        if not dfs:
            return universe

        # Vectorized aggregation
        try:
            full_df = pd.concat(dfs, ignore_index=True)
            full_df = full_df.dropna(subset=['match_hash', 'prediction_date'])
            
            # Normalize to naive UTC
            if not pd.api.types.is_datetime64_any_dtype(full_df['prediction_date']):
                full_df['prediction_date'] = pd.to_datetime(full_df['prediction_date'], utc=True).dt.tz_localize(None)
            elif full_df['prediction_date'].dt.tz is not None:
                full_df['prediction_date'] = full_df['prediction_date'].dt.tz_convert('UTC').dt.tz_localize(None)
            
            # Group by hash and get earliest date
            universe = full_df.groupby('match_hash')['prediction_date'].min().to_dict()
            
        except Exception as e:
            logger.error(f"Failed to aggregate prediction universe: {e}")
            
        return universe

    def ingest_raw_csv(self, file_path: Path, league: str) -> IngestionResult:
        """
        Ingests a raw CSV (expected Football-Data format or similar).
        Normalizes and appends to Master.
        
        Args:
            file_path: Path to raw CSV.
            league: Internal league identifier.
            
        Returns:
            IngestionResult with processing statistics.
        """
        if isinstance(file_path, str):
            file_path = Path(file_path)
            
        if not file_path.exists():
            raise FileNotFoundError(f"CSV file not found: {file_path}")
            
        # Basic League Validation
        if not league or not isinstance(league, str):
            raise DataValidationError(f"Invalid league provided: {league}")

        try:
            # Use unicode_escape to handle legacy CSV character sets
            df = pd.read_csv(file_path, encoding='unicode_escape')
        except pd.errors.EmptyDataError:
            logger.warning(f"File is empty: {file_path}")
            return IngestionResult(file_path, 0, 0, 0, self.master_path)
        except Exception as e:
            raise DataValidationError(f"Failed to read CSV file: {e}") from e
        
        if df.empty:
            logger.warning(f"File contains no data: {file_path}")
            return IngestionResult(file_path, 0, 0, 0, self.master_path)

        rows_read = len(df)

        # 1. Map to Canonical Columns
        col_map = {
            'Date': 'match_date',
            'HomeTeam': 'home_team',
            'AwayTeam': 'away_team',
            'FTHG': 'home_goals',
            'FTAG': 'away_goals',
            'HC': 'home_corners',
            'AC': 'away_corners',
            'HY': 'home_yellow_cards',
            'AY': 'away_yellow_cards',
            'HR': 'home_red_cards',
            'AR': 'away_red_cards',
        }
        
        # Check available columns
        available = [c for c in col_map.keys() if c in df.columns]
        df = df.rename(columns={c: col_map[c] for c in available})
        
        # Verify numeric safety for goals
        numeric_goal_cols = ['home_goals', 'away_goals']
        df[numeric_goal_cols] = df[numeric_goal_cols].apply(pd.to_numeric, errors='coerce')
        df = df.dropna(subset=numeric_goal_cols)
        
        # 2. Handle Missing Columns (Safety & Integrity)
        # Avoid silent zero-filling for critical missing data
        if 'home_cards' not in df.columns:
            if 'home_yellow_cards' in df.columns and 'home_red_cards' in df.columns:
                df['home_cards'] = df['home_yellow_cards'] + df['home_red_cards']
                df['away_cards'] = df['away_yellow_cards'] + df['away_red_cards']
            else:
                logger.debug(f"Card data missing in {file_path.name}, filling with NaN")
                df['home_cards'] = np.nan
                df['away_cards'] = np.nan
                
        if 'home_corners' not in df.columns:
            logger.debug(f"Corner data missing in {file_path.name}, filling with NaN")
            df['home_corners'] = np.nan
            df['away_corners'] = np.nan

        # 3. Clean and Normalize
        df['match_date'] = pd.to_datetime(df['match_date'], dayfirst=True, errors='coerce', utc=True)
        df = df.dropna(subset=['match_date'])
        
        if df.empty:
            return IngestionResult(file_path, rows_read, 0, rows_read, self.master_path)

        df['league'] = league.upper()
        df['season'] = df['match_date'].apply(partial(calculate_season, league=league))
        
        # 4. Normalize Team Names (Optimized based on season variation)
        if df['season'].nunique() == 1:
            season = df['season'].iloc[0]
            df['home_team'] = df['home_team'].apply(
                partial(normalize_team_name, league=league, season=season)
            )
            df['away_team'] = df['away_team'].apply(
                partial(normalize_team_name, league=league, season=season)
            )
        else:
            def normalize_row_teams(row: pd.Series) -> pd.Series:
                row['home_team'] = normalize_team_name(
                    row['home_team'], 
                    league=league, 
                    season=row['season']
                )
                row['away_team'] = normalize_team_name(
                    row['away_team'], 
                    league=league, 
                    season=row['season']
                )
                return row
            df = df.apply(normalize_row_teams, axis=1)
        
        # 5. Generate Match Hash (Deterministic ID)
        def make_hash(row: pd.Series) -> str:
            return generate_match_fingerprint(
                league, 
                row['match_date'], 
                row['home_team'], 
                row['away_team'], 
                season=row['season']
            )

        df['match_hash'] = df.apply(make_hash, axis=1)
        df['kickoff_date_utc'] = df['match_date'].dt.strftime('%Y-%m-%dT%H:%M:%SZ')
        df['is_final'] = True
        df['ingested_at'] = datetime.now().isoformat()
        
        # Final Structure Validation
        missing_required = set(self.REQUIRED_COLS) - set(df.columns)
        if missing_required:
             raise DataValidationError(f"Missing required columns in processed data: {missing_required}")
        
        # 6. Prediction-Aware Filtering (Phase 2 Safety)
        is_historical = "historical" in str(file_path).lower()
        
        filtered_df = df
        
        if not is_historical:
            universe = self._get_prediction_universe()
            if not universe:
                logger.warning("No prediction universe found. Selective ingestion skipped (0 rows).")
                return IngestionResult(file_path, rows_read, 0, rows_read, self.master_path)
            
            # Filtering
            filtered_df['in_universe'] = filtered_df['match_hash'].isin(universe.keys())
            filtered_df = filtered_df[filtered_df['in_universe']].copy()
            
            if filtered_df.empty:
                logger.info(f"No results in {file_path.name} match awaiting predictions. Skipping.")
                return IngestionResult(file_path, rows_read, 0, rows_read, self.master_path)

            # Temporal Safety Check
            filtered_df['pred_date'] = filtered_df['match_hash'].map(universe)
            violations = filtered_df[filtered_df['match_date'] < filtered_df['pred_date']]
            
            if not violations.empty:
                raise DataValidationError(f"Temporal consistency violations detected (Result Date < Prediction Date): {len(violations)}")
            
            filtered_df = filtered_df.drop(columns=['in_universe', 'pred_date'])
            logger.info(f"Filtered {file_path.name}: {len(filtered_df)} eligible results found.")

        # 7. Append-Only Logic
        appended_count = self._append_to_master(filtered_df)
        
        return IngestionResult(
            file_path=file_path,
            rows_read=rows_read,
            rows_appended=appended_count,
            rows_skipped=rows_read - appended_count,
            master_path=self.master_path
        )

    def _append_to_master(self, new_df: pd.DataFrame) -> int:
        """
        Appends new results to the master file, skipping existing hashes.
        Returns: Number of new rows appended.
        """
        self.master_path.parent.mkdir(parents=True, exist_ok=True)
        
        if self.master_path.exists():
            try:
                master_df = pd.read_csv(self.master_path)
            except Exception as e:
                logger.error(f"Failed to read existing master file: {e}")
                raise DataValidationError(f"Master results file corrupted: {e}") from e
            
            # Find new rows (hash not in master)
            new_rows = new_df[~new_df['match_hash'].isin(master_df['match_hash'])]
            
            if new_rows.empty:
                logger.info("No new results to append.")
                return 0
                
            # Filter intra-batch duplicates
            if new_rows['match_hash'].duplicated().any():
                logger.warning("Duplicates detected in input batch. Filtering.")
                new_rows = new_rows.drop_duplicates(subset=['match_hash'])
            
            combined = pd.concat([master_df, new_rows], ignore_index=True)
        else:
            combined = new_df
            
        # Atomic Write Pattern: prevent corruption during concurrent appends
        with tempfile.NamedTemporaryFile(mode='w', delete=False, dir=self.master_path.parent, suffix=".tmp") as tmp:
            tmp_path = Path(tmp.name)
            
        try:
            # Write to closed temp file to prevent handle leaks/locks
            combined.to_csv(tmp_path, index=False)
            
            # os.replace is atomic and cross-platform
            os.replace(tmp_path, self.master_path)
                
            logger.info(f"Successfully refined master file with {len(combined)} total rows.")
            return len(new_df)
        except Exception as e:
            logger.error(f"Failed to atomically update master file: {e}")
            if tmp_path.exists():
                os.unlink(tmp_path)
            raise DataValidationError(f"Atomic update failed for master results: {e}") from e

if __name__ == "__main__":
    # Test/Bootstrap with PL historical data
    distillery = ResultDistillery()
    hist_dir = DATA_DIR / "historical" / "PL"
    if hist_dir.exists():
        for csv_file in hist_dir.glob("*.csv"):
            distillery.ingest_raw_csv(csv_file, "PL")
