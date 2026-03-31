"""
DataFrame Validators.

Provides fail-fast validation functions for match DataFrames to ensure
data quality before processing in pipelines and predictions.

Usage:
    from src.core.validators import validate_match_dataframe
    
    validate_match_dataframe(df, require_settled=True, context="training")
"""
import logging
from typing import FrozenSet, Optional

import pandas as pd

from src.core.exceptions import DataValidationError

# Define public API
__all__ = ["validate_match_dataframe", "REQUIRED_MATCH_COLUMNS"]

logger = logging.getLogger(__name__)

# Centralized required columns
REQUIRED_MATCH_COLUMNS: FrozenSet[str] = frozenset({
    'match_id', 'date', 'home_team', 'away_team', 'league'
})


def validate_match_dataframe(
    df: pd.DataFrame, 
    require_settled: bool = False, 
    context: str = ""
) -> None:
    """
    Fail-fast validation for match DataFrames.
    
    Args:
        df: The DataFrame to validate.
        require_settled: If True, checks for 'home_score' and validates it's not null.
        context: Additional context string for error messages.
        
    Raises:
        DataValidationError: If validation fails.
    """
    # Empty DataFrame is valid (no matches found for filter)
    if df.empty:
        logger.debug("Empty DataFrame passed validation", extra={"context": context})
        return
    
    logger.debug(
        "Validating match DataFrame",
        extra={"rows": len(df), "columns": len(df.columns), "context": context}
    )
    
    # Required columns check
    missing = REQUIRED_MATCH_COLUMNS - set(df.columns)
    if missing:
        raise DataValidationError(
            f"Missing required columns: {missing}",
            context={"missing_columns": list(missing), "context": context}
        )
    
    # Settled matches require scores
    if require_settled:
        if 'home_score' not in df.columns:
            raise DataValidationError(
                "Settled matches require 'home_score' column",
                context={"columns": df.columns.tolist(), "context": context}
            )
        
        # Check for null scores in settled data
        null_count = df['home_score'].isna().sum()
        if null_count > 0:
            raise DataValidationError(
                f"{null_count} matches missing scores in settled data",
                context={"invalid_count": int(null_count), "context": context}
            )

    # Type validation for 'date'
    if 'date' in df.columns:
        if not pd.api.types.is_datetime64_any_dtype(df['date']):
            raise DataValidationError(
                "'date' column must be datetime64 type",
                context={"actual_type": str(df['date'].dtype), "context": context}
            )

    # Sanity check for empty strings in key fields
    for col in ['home_team', 'away_team', 'league']:
        if col in df.columns:
            if df[col].astype(str).str.strip().eq("").any():
                raise DataValidationError(
                    f"Found empty strings in required column '{col}'",
                    context={"column": col, "context": context}
                )
    
    logger.debug("DataFrame validation passed", extra={"context": context})

