"""
Training Data Validation.

Provides utilities for loading, filtering, and validating datasets
specifically for model training and backtesting. Ensures data integrity
before it enters the training pipeline.
"""
import logging
import os
from pathlib import Path
from typing import Union

import pandas as pd

from src.config.thresholds import Thresholds
from src.core.constants import STATUS_FINISHED
from src.core.exceptions import DataValidationError, InsufficientDataError

# Define public API
__all__ = ["validate_training_data", "filter_historical_matches", "load_feature_file"]

logger = logging.getLogger(__name__)


def validate_training_data(df: pd.DataFrame, context: str) -> None:
    """
    Validate training data meets minimum requirements.
    
    Args:
        df: The DataFrame to validate.
        context: Description of the dataset (for error messages).
        
    Raises:
        InsufficientDataError: If data is missing or below sample threshold.
    """
    if df is None or df.empty:
        raise InsufficientDataError(
            f"No training data available for {context}",
            context={"reason": "empty_dataframe"}
        )
    
    if len(df) < Thresholds.MIN_TRAINING_SAMPLES:
        raise InsufficientDataError(
            f"Insufficient training samples for {context}",
            context={
                "available": len(df), 
                "required": Thresholds.MIN_TRAINING_SAMPLES
            }
        )
    
    logger.info(f"Data validation successful for {context} ({len(df)} samples)")


def filter_historical_matches(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter for completed historical matches only (non-mutating).
    
    Args:
        df: Input DataFrame containing 'status' and 'date' columns.
        
    Returns:
        New DataFrame containing only finished matches from the past.
    """
    df = df.copy()
    
    # Ensure date column is datetime
    if not pd.api.types.is_datetime64_any_dtype(df['date']):
        df['date'] = pd.to_datetime(df['date'])
    
    # Filter for finished matches before today
    today = pd.Timestamp.now(tz='UTC').normalize()  # Start of today (UTC)
    
    # Normalize status for filtering
    # Handle 'FT', 'FINISHED', 'finished', 'Full Time'
    valid_statuses = {STATUS_FINISHED, 'FT', 'FINISHED', 'FULL TIME', 'finished'}
    
    mask = (
        (df['status'].isin(valid_statuses)) &
        (df['date'] < today)
    )
    
    return df[mask]


def load_feature_file(path: Union[str, Path]) -> pd.DataFrame:
    """
    Load and validate a feature CSV file.
    
    Args:
        path: Path to the CSV file.
        
    Returns:
        DataFrame with parsed dates.
        
    Raises:
        DataValidationError: If file is missing, empty, or malformed.
    """
    path_str = str(path)
    if not os.path.exists(path_str):
        raise DataValidationError(
            f"Feature file not found: {path_str}", 
            context={"path": path_str}
        )
        
    logger.info(f"Loading features from {path_str}")
    try:
        df = pd.read_csv(path_str)
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            
        if df.empty:
            raise DataValidationError(
                f"Feature file is empty: {path_str}", 
                context={"file": path_str}
            )
            
        return df
        
    except pd.errors.EmptyDataError:
        raise DataValidationError(f"Feature file is empty: {path_str}")
    except pd.errors.ParserError as e:
        raise DataValidationError(
            f"Feature file is not valid CSV: {path_str}", 
            context={"error": str(e)}
        )
    except Exception as e:
        raise DataValidationError(
            f"Unexpected error loading {path_str}: {str(e)}"
        )

