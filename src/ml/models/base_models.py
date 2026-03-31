"""
Base Machine Learning Models.

Defines the standard interface (`BaseModel`) and wrapper implementations
for Poisson and Negative Binomial models used throughout the prediction system.

Includes specialized logic for:
- Model serialization (save/load)
- Feature sanitization (NaN/Inf handling)
- Baseline-anchored scaling for temporal stability
"""
import abc
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# Heavy ML imports moved to lazy loading for faster CLI startup
# from sklearn.linear_model import PoissonRegressor  # Lazy in PoissonWrapper
# from sklearn.preprocessing import StandardScaler   # Lazy in PoissonWrapper

# Define public API
__all__ = ["BaseModel", "PoissonWrapper", "BASE_FEATURES"]

logger = logging.getLogger(__name__)

# Core features used across multiple models
# TODO: Move to a dedicated feature registry in future refactors
BASE_FEATURES: List[str] = [
    "home_rolling_goals_scored_5",
    "home_rolling_goals_conceded_5",
    "away_rolling_goals_scored_5",
    "away_rolling_goals_conceded_5",
    "home_form_rating",
    "away_form_rating",
    "home_days_rest",
    "away_days_rest",
    # New Phase Features (Optional depending on data availability)
    "home_rolling_corners_scored_5",
    "home_rolling_corners_conceded_5",
    "away_rolling_corners_scored_5",
    "away_rolling_corners_conceded_5",
    "home_rolling_cards_scored_5",
    "away_rolling_cards_scored_5"
]

# Baseline-Anchored Scaling Constants
ANCHOR_SAMPLE_SIZE = 600
ANCHOR_WINDOW_SIZE = 400

# Prediction bounds (Lambda floor/ceiling)
LAMBDA_MIN = 0.05
LAMBDA_MAX = 15.0


class BaseModel(abc.ABC):
    """
    Standard interface for all models in the system.
    
    Enforces a consistent API for training, prediction, and persistence.
    """
    
    @abc.abstractmethod
    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train the model on the provided data."""
        pass

    @abc.abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions for the input data."""
        pass
    
    @abc.abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Returns probabilities or expected values.
        
        For regression models (like Poisson goals), this might need interpretation (e.g. prob > 2.5).
        For classifiers, it returns class probabilities.
        """
        pass

    def save(self, path: Path) -> None:
        """Serialize and save the model to disk."""
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        logger.info(f"Model saved to {path}")

    @staticmethod
    def load(path: Path) -> 'BaseModel':
        """Load a serialized model from disk."""
        with open(path, 'rb') as f:
            return pickle.load(f)





class PoissonWrapper(BaseModel):
    """
    Wrapper for Poisson Regression (predicting goal/corner counts).
    
    Includes StandardScaler for GLM stability and implements
    Baseline-Anchored Scaling to handle drift.
    """
    def __init__(self, **kwargs: Any) -> None:
        # Lazy import for faster CLI startup
        from sklearn.linear_model import PoissonRegressor
        from sklearn.preprocessing import StandardScaler
        
        self.model = PoissonRegressor(**kwargs)
        self.scaler = StandardScaler()
        self.features: Optional[List[str]] = None

    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.features = X.columns.tolist()
        
        # Drop missing targets
        mask = y.notna()
        X = X[mask]
        y = y[mask]
        
        if len(y) == 0:
            logger.warning("No valid samples for training (all targets NaN).")
            return
            
        # Baseline-Anchored Scaling (Bug 3.2 Fix)
        # To avoid damping modern intensity by historical shifts,
        # we fit the scaler on the 'Modern Baseline' (last N matches)
        # if the dataset is sufficiently large.
        if len(X) > ANCHOR_SAMPLE_SIZE:
            modern_anchor = X.iloc[-ANCHOR_WINDOW_SIZE:]
            self.scaler.fit(modern_anchor)
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = self.scaler.fit_transform(X)
            
        self.model.fit(X_scaled, y)

    def predict(
        self, 
        X: pd.DataFrame, 
        offset: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Predict expected value (lambda).
        
        Args:
            X: Input features.
            offset: Optional log-space offset (mu = exp(Xb + offset)).
        """
        if self.features is None:
            raise RuntimeError("Model must be trained before prediction")

        # Hardened Contract Check
        missing = set(self.features) - set(X.columns)
        if missing:
            raise RuntimeError(
                f"Feature mismatch. Missing at inference: {missing}"
            )
        
        # Sanitize
        X_clean = X[self.features].replace([np.inf, -np.inf], np.nan).fillna(0)
            
        X_scaled = self.scaler.transform(X_clean)
        preds = self.model.predict(X_scaled)
        
        # Apply offset if provided (mu = exp(Xb + offset))
        if offset is not None:
             preds = preds * np.exp(offset)
        
        # 4.2 Add λ floor/ceiling during training/prediction
        # Prevent log(0) and model hallucinations
        return np.clip(preds, LAMBDA_MIN, LAMBDA_MAX) 

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Returns the Expected Value (Lambda).
        Calling code must interpret this as a regression output.
        """
        return self.predict(X)

