"""
Match Result Labeler.

Generates binary market labels from normalized results:
- Goals markets (Over/Under, BTTS)
- Corner markets (Over/Under)
- Card markets
- Double Chance

Labels are generated deterministically and stored append-only.
"""
import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import logging

# Add project root
sys.path.append(os.getcwd())
from src.config import DATA_DIR
from dataclasses import dataclass
from typing import Optional, List
from tqdm import tqdm
from src.config.markets import MARKET_DEFINITIONS
from src.core.exceptions import DataValidationError

logger = logging.getLogger(__name__)

@dataclass
class LabelingResult:
    """Result of label generation operation."""
    total_processed: int
    new_labels: int
    skipped: int
    labeled_path: Path

class Labeler:
    """
    Consumes Canonical Master results and generates binary market labels.
    Labels are generated ONCE and stored.
    """
    
    def __init__(
        self, 
        master_path: Path = DATA_DIR / "results" / "normalized" / "results_master.csv",
        labeled_path: Path = DATA_DIR / "results" / "labeled" / "results_labeled.csv"
    ):
        """
        Initialize Labeler with configurable paths.
        
        Args:
            master_path: Path to normalized master results
            labeled_path: Path to output labeled results
        """
        self.master_path = master_path
        self.labeled_path = labeled_path

    def generate_labels(self) -> LabelingResult:
        """
        Processes new results from Master and appends to Labeled.
        
        Returns:
            LabelingResult with statistics about the operation.
        """
        if not self.master_path.exists():
            raise DataValidationError(
                f"Master results file not found: {self.master_path}",
                context={"expected_path": str(self.master_path)}
            )

        try:
            master_df = pd.read_csv(self.master_path)
        except pd.errors.ParserError as e:
            raise DataValidationError(
                f"Master results file is corrupted or invalid CSV: {e}",
                context={"path": str(self.master_path)}
            ) from e
        except Exception as e:
            raise DataValidationError(
                f"Failed to read master results: {e}",
                context={"path": str(self.master_path)}
            ) from e

        self._validate_master_data(master_df)
        
        if self.labeled_path.exists():
            try:
                labeled_df = pd.read_csv(self.labeled_path)
            except Exception as e:
                # If labeled file is corrupt, we might want to warn or fail. 
                # For safety, let's treat it as empty or fail. Failing is safer.
                raise DataValidationError(
                    f"Existing labels file is corrupted: {e}",
                    context={"path": str(self.labeled_path)}
                ) from e
                
            # Process only new matches by excluding existing hashes
            new_rows = master_df[~master_df['match_hash'].isin(labeled_df['match_hash'])].copy()
        else:
            labeled_df = pd.DataFrame()
            new_rows = master_df.copy()

        if new_rows.empty:
            logger.info("No new matches to label.")
            return LabelingResult(
                total_processed=len(master_df),
                new_labels=0,
                skipped=len(master_df),
                labeled_path=self.labeled_path
            )

        # Deterministic Label Generation
        logger.info(f"Computing labels for {len(new_rows)} matches...")
        labels = self._compute_labels(new_rows)
        
        # Append and Save
        # Ensure directory exists before saving
        self.labeled_path.parent.mkdir(parents=True, exist_ok=True)
        
        final_df = pd.concat([labeled_df, labels], ignore_index=True)
        final_df.to_csv(self.labeled_path, index=False)
        
        logger.info(f"Generated labels for {len(labels)} new matches.")
        
        return LabelingResult(
            total_processed=len(master_df),
            new_labels=len(labels),
            skipped=len(master_df) - len(labels),
            labeled_path=self.labeled_path
        )

    def _validate_master_data(self, df: pd.DataFrame) -> None:
        """
        Validate master results have required columns and structure.
        
        Checks:
        - DataFrame is not empty
        - All required columns present
        - match_hash values are unique (logs first 5 duplicates if found)
        
        Raises:
            DataValidationError: If validation fails
        """
        if df.empty:
            raise DataValidationError(
                "Master results file is empty",
                context={"path": str(self.master_path)}
            )

        required_cols = [
            'match_hash', 'kickoff_date_utc',
            'home_goals', 'away_goals',
            'home_corners', 'away_corners',
            'home_cards', 'away_cards'
        ]
        
        missing = set(required_cols) - set(df.columns)
        if missing:
            raise DataValidationError(
                f"Master results missing required columns: {missing}",
                context={"available_columns": list(df.columns)}
            )
        
        # Validate match_hash uniqueness
        if df['match_hash'].duplicated().any():
            duplicates = df[df['match_hash'].duplicated()]['match_hash'].tolist()
            raise DataValidationError(
                "Master results contain duplicate match_hash values",
                context={"duplicate_count": len(duplicates), "examples": duplicates[:5]}
            )

    def _compute_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Pure deterministic logic for market labels using configuration.
        NaN-Safe: If result columns are NaN, the label remains NaN.
        """
        results = pd.DataFrame({
            'match_hash': df['match_hash'],
            'kickoff_date_utc': df['kickoff_date_utc']
        })
        
        # Count total markets for granular progress bar
        total_markets = sum(
            len([k for k in m.keys() if k != 'dependencies'])
            for m in MARKET_DEFINITIONS.values()
        )
        
        # Use granular progress bar
        pbar = tqdm(total=total_markets, desc="Generating labels")
        
        for market_group, markets in MARKET_DEFINITIONS.items():
            deps = markets.get('dependencies', [])
            
            for market_name, condition_fn in markets.items():
                if market_name == 'dependencies':
                    continue
                
                # Apply efficient vectorised lambda
                bool_series = condition_fn(df)
                results[market_name] = self._apply_label_with_deps(bool_series, df, deps)
                pbar.update(1)
                
        pbar.close()    
        return results

    def _apply_label_with_deps(
        self, 
        bool_series: pd.Series, 
        df: pd.DataFrame, 
        deps: List[str]
    ) -> pd.Series:
        """
        Convert boolean series to 0.0/1.0, preserving NaNs if dependencies are missing.
        
        Args:
            bool_series: Boolean result of label condition
            df: Source DataFrame containing dependency columns
            deps: List of column names required for this label
            
        Returns:
            Float series (0.0, 1.0, or NaN)
        """
        res = bool_series.astype(float)
        
        # If ANY dependency column is NaN for a row, the result must be NaN
        if deps:
            mask = df[deps].isna().any(axis=1)
            res[mask] = np.nan
            
        return res

if __name__ == "__main__":
    try:
        labeler = Labeler()
        result = labeler.generate_labels()
        if result.new_labels > 0:
            print(f"Success: {result.new_labels} labeled, saved to {result.labeled_path}")
        else:
            print("No new labels generated.")
    except Exception as e:
        logger.error(f"Label generation failed: {e}", exc_info=True)
        sys.exit(1)
