"""
ML Feature Selection.

Selects appropriate features for model training based on
feature set configuration (BASE vs ENHANCED_XG). Handles
dynamic feature discovery and validation.
"""
import json
import logging
from pathlib import Path
from typing import List

import pandas as pd

from src.config import MODELS_DIR
from src.ml.models.base_models import BASE_FEATURES
from src.ml.training.model_configs import FeatureSet
from src.core.exceptions import DataValidationError

logger = logging.getLogger(__name__)

# Path to the feature list produced by train_probability_models.py
ENHANCED_XG_COLUMNS_FILE: Path = MODELS_DIR / "feature_columns.json"


def select_features(
    df: pd.DataFrame,
    target_col: str,
    feature_set: FeatureSet = FeatureSet.BASE,
) -> List[str]:
    """
    Select feature subset for model training.

    Raises:
        DataValidationError: If no valid features found
    """
    if feature_set == FeatureSet.BASE:
        features = _select_base_features(df, target_col)
    elif feature_set == FeatureSet.ENHANCED_XG:
        features = _select_enhanced_xg_features(df, target_col)
    else:
        raise ValueError(f"Unknown feature set: {feature_set}")

    if not features:
        raise DataValidationError(
            f"No valid features found for {target_col}",
            context={"feature_set": feature_set.value, "available_columns": len(df.columns)},
        )

    logger.info(f"Selected {len(features)} features for {target_col} ({feature_set.value})")
    return features


def _select_base_features(df: pd.DataFrame, target_col: str) -> List[str]:
    """Select features from BASE_FEATURES whitelist."""
    available = [c for c in BASE_FEATURES if c in df.columns and c != target_col]

    missing = set(BASE_FEATURES) - set(df.columns)
    if missing:
        logger.warning(f"Some base features missing: {missing}")

    return available


def _select_enhanced_xg_features(df: pd.DataFrame, target_col: str) -> List[str]:
    """
    Load Phase-5 LightGBM feature list from models/feature_columns.json.

    This is the bridge between the Phase-5 trainer and the live pipeline.
    The JSON is produced by train_probability_models.py — run that script first.

    Raises:
        DataValidationError: If feature_columns.json does not exist.
        RuntimeError: If more than 5% of features are missing.
    """
    if not ENHANCED_XG_COLUMNS_FILE.exists():
        raise DataValidationError(
            "ENHANCED_XG feature set requires models/feature_columns.json — "
            "run train_probability_models.py first to generate it.",
            context={"expected_path": str(ENHANCED_XG_COLUMNS_FILE)},
        )

    with open(ENHANCED_XG_COLUMNS_FILE, "r", encoding="utf-8") as f:
        cols: List[str] = json.load(f)

    # Enforce order: Keep columns that exist, and strictly measure missing ratio
    expected_cols = [c for c in cols if c != target_col]
    missing = [c for c in expected_cols if c not in df.columns]
    
    missing_ratio = len(missing) / len(expected_cols) if expected_cols else 0.0

    if missing_ratio > 0.05:
        raise RuntimeError(
            f"Too many missing features ({missing_ratio:.1%} > 5%) — abort prediction. "
            f"Missing features: {missing}"
        )
    elif missing_ratio > 0:
        logger.warning(
            f"ENHANCED_XG: {len(missing)} feature column(s) absent from DataFrame "
            f"(ratio: {missing_ratio:.1%}). They will be silently dropped. Missing: {sorted(missing)}"
        )

    available = [c for c in expected_cols if c in df.columns]
    return available
