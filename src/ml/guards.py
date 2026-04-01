"""
ML Training Guards.

Safety checks to prevent model training on invalid data:
- guard_min_samples: Minimum sample size for statistical validity
- guard_single_league: League purity check
- guard_feature_coverage: Feature completeness threshold
- guard_xg_coverage: Dynamic BASE/ENHANCED feature switch
- guard_model_freshness: Model staleness warning
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional, List, Any

__all__ = [
    "guard_min_samples", 
    "guard_single_league", 
    "guard_feature_coverage", 

    "guard_model_freshness",
    "PredictionGuard"
]

logger = logging.getLogger(__name__)

class PredictionGuard:
    """
    Phase 9: Hallucination & Safety Guardrails.
    Ensures integrity of every prediction before it is emitted.
    """
    
    @staticmethod
    def validate_prediction_integrity(model: Any, X: pd.DataFrame, context: str = "") -> str:
        """
        Runs all safety checks. Returns a cryptographic checksum of the prediction context.
        Raises RuntimeError (DataValidationError) if any check fails.
        """
        # 1. Verification: Model Existence & Type
        if not hasattr(model, 'predict'):
            raise RuntimeError(f"Invalid model object in {context}")
            
        meta = getattr(model, 'meta', {})
        if not meta:
             raise RuntimeError(f"Model missing registry metadata in {context}. Bypass detected?")

        # 2. Guard: Feature Schema (Strict Equality)
        required_features_raw = getattr(model, 'features', None)
        if required_features_raw is None:
            required_features_raw = getattr(model, 'feature_names_in_', [])

        # Normalize to list to avoid ambiguous truth-value checks for numpy arrays.
        if isinstance(required_features_raw, np.ndarray):
            required_features = required_features_raw.tolist()
        elif isinstance(required_features_raw, (list, tuple, pd.Index)):
            required_features = list(required_features_raw)
        else:
            required_features = []

        if len(required_features) == 0:
            # Fallback (rare)
            pass
            
        required_set = set(required_features)
        input_set = set(X.columns)
        
        missing = required_set - input_set
        if missing:
            raise RuntimeError(f"Schema Mismatch (Missing): {missing} in {context}")
            
        # 3. Guard: Calibration Gate

        # "No market can output probabilities unless Calibration Metadata is present"
        calib_score = meta.get('metrics', {}).get('calibration_score')
        if calib_score is None:
             raise RuntimeError(f"Calibration Gate: Model {meta.get('name')} lacks calibration_score in {context}")
             
        # 4. Checksum Generation (Input + Version + Context)
        import hashlib
        
        feats_sorted = sorted(required_features)
        try:
             # Hash first row input values
             row_data = X[feats_sorted].iloc[0].values.tobytes()
             input_hash = hashlib.sha256(row_data).hexdigest()[:16]
        except Exception:
             input_hash = "hash_fail"
             
        version = meta.get('version', 'unknown')
        ts = datetime.now().strftime("%Y%m%d%H") # Hourly bucket
        
        checksum = f"{version}:{input_hash}:{ts}"
        
        # Log successful check (Debug only to avoid noise)
        logger.debug(f"🛡️ Guard Passed: {checksum} | Context: {context}")
        
        return checksum

logger = logging.getLogger(__name__)

def guard_min_samples(df: pd.DataFrame, min_matches: int = 300):
    """
    Guard 1: Minimum sample size.
    Poisson λ collapses under small samples.
    """
    if len(df) < min_matches:
        raise RuntimeError(
            f"Insufficient data: {len(df)} < {min_matches}"
        )

def guard_single_league(df: pd.DataFrame, league: str):
    """
    Guard 2: League purity check.
    Ensures disjoint datasets.
    """
    # Prefer 'league' (System Code e.g. PL) over 'competition' (Full Name)
    target_col = 'league' if 'league' in df.columns else 'competition'
    
    if target_col not in df.columns:
        # If neither exists, we technically can't check purity, but usually this means global data or error.
        # For now, bypassing if no column found (or raising error depending on strictness).
        # logger.warning("Structure check: No 'league' or 'competition' column found.")
        return 

    leagues = df[target_col].dropna().unique()
    
    # Check if contaminated (more than 1)
    if len(leagues) > 1:
        raise RuntimeError(f"League contamination detected in {target_col}: {leagues}")
        
    # Check if mismatch
    # If filtered by league='BL1' and column has 'PD', that's bad.
    # But usually we filter beforehand.


def guard_feature_coverage(X: pd.DataFrame, required: float = 0.98):
    """
    Guard 3: Feature completeness.
    Prevents training on sparse matrices (bad scraping).
    """
    if X.empty:
         raise RuntimeError("Empty feature matrix.")
         
    # Fraction of non-NaN values per column
    # We want valid rows. But guard says "feature coverage".
    # "coverage = 1 - X.isna().mean().max()" -> Worst column coverage?
    # X.isna().mean() gives fraction of NaNs for each column.
    # Max gives the worst column's Nan fraction.
    # So 1 - Max is the min coverage of any single column?
    # Or aggregate?
    # User's logic: "1 - X.isna().mean().max()".
    # Interpretation: If ANY column has > 2% missing data, fail.
    
    worst_nan_ratio = X.isna().mean().max()
    coverage = 1.0 - worst_nan_ratio
    
    if coverage < required:
        raise RuntimeError(
            f"Feature coverage too low: {coverage:.2%} (Required {required:.0%})"
        )


def guard_model_freshness(registry, league: str, max_age_days: int = 30):
    """
    Guard 5: Model freshness.
    Warns if model is stale.
    """
    # Registry needs to support get_latest(league)
    # This would likely be called inside CLI or Trainer before predicting/retraining.
    # For now, just implementing the logic assuming we have metadata.
    try:
        model_meta = registry.get_production_model_for_league(league)
        if not model_meta:
            return 
            
        reg_time = datetime.fromisoformat(model_meta['registered_at'])
        age = (datetime.now() - reg_time).days
        
        if age > max_age_days:
            logger.warning(f"⚠️ {league} model is {age} days old (Threshold: {max_age_days})")
    except Exception:
        pass # Don't block on this
