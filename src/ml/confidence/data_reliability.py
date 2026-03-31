"""
Data Reliability Score (DRS).

Measures confidence in input data quality:
- H2H sample size
- Recency of data  
- Feature completeness

Includes kill switch for catastrophic data scenarios.
"""
import math
from typing import List


def clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp value between min and max."""
    return max(min_val, min(value, max_val))


def calculate_drs(
    h2h_count: int,
    days_since_h2h: float = 0.0,
    missing_features: List[str] = None
) -> float:
    """
    Calculate Data Reliability Score.
    
    Args:
        h2h_count: Number of H2H matches in history
        days_since_h2h: Days since most recent H2H match
        missing_features: List of missing feature names
        
    Returns:
        DRS ∈ [0.3, 1.0]
    """
    missing_features = missing_features or []
    
    # === KILL SWITCH: Catastrophic data ===
    if h2h_count == 0 and len(missing_features) >= 3:
        return 0.3
    
    # === Sample Size Score ===
    # Full score at 10+ H2H, capped at 0.6 if < 5
    if h2h_count >= 5:
        sample_score = min(h2h_count / 10, 1.0)
    else:
        sample_score = 0.6 * (h2h_count / 5) if h2h_count > 0 else 0.0
    
    # === Recency Score ===
    # Exponential decay over 1 year
    recency_score = math.exp(-days_since_h2h / 365) if days_since_h2h >= 0 else 1.0
    
    # === Completeness Score ===
    # Exponential penalty: missing 3 features hurts badly
    completeness_score = math.exp(-0.35 * len(missing_features))
    
    # === Weighted Combination ===
    drs = (
        0.4 * sample_score +
        0.3 * recency_score +
        0.3 * completeness_score
    )
    
    return clamp(drs, 0.3, 1.0)


def get_missing_features(row: dict, required: List[str]) -> List[str]:
    """Check which required features are missing."""
    import pandas as pd
    missing = []
    for feat in required:
        val = row.get(feat)
        if val is None or (isinstance(val, float) and pd.isna(val)):
            missing.append(feat)
    return missing
