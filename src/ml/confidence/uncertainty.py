"""
Uncertainty Estimation.

Uses soft saturation curve for tail risk penalty.
High prob + high variance = LOW confidence.
"""
import math
import numpy as np
from typing import Optional


def estimate_uncertainty(
    mu: float, 
    variance: float, 
    n_bootstrap: int = 0
) -> float:
    """
    Estimate prediction uncertainty using soft saturation.
    
    Args:
        mu: Mean prediction (lambda for Poisson/NegBin)
        variance: Model variance
        n_bootstrap: Optional bootstrap samples (0 = use variance only)
        
    Returns:
        uncertainty ∈ [0, 1] where higher = more uncertain
    """
    if mu <= 0:
        return 0.5  # Default uncertainty for edge cases
    
    # Coefficient of variation
    cv = math.sqrt(max(variance, 0)) / max(mu, 0.01)
    
    # Soft saturation: small variance → small penalty, exploding → collapse
    uncertainty = 1 - math.exp(-cv)
    
    return min(max(uncertainty, 0.0), 1.0)


def bootstrap_uncertainty(
    predictions: np.ndarray,
    n_samples: int = 100
) -> float:
    """
    Estimate uncertainty via bootstrap resampling.
    
    Args:
        predictions: Array of predictions from bootstrap samples
        n_samples: Number of bootstrap iterations
        
    Returns:
        uncertainty based on prediction variance
    """
    if len(predictions) < 2:
        return 0.5
    
    std = np.std(predictions)
    mean = np.mean(predictions)
    
    if mean <= 0:
        return 0.5
    
    cv = std / max(mean, 0.01)
    return 1 - math.exp(-cv)
