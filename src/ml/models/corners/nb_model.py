"""
Negative Binomial Corner Model.

Statsmodels-based Negative Binomial regression for predicting
corner counts with over-dispersion (variance > mean). Provides:
- scikit-learn compatible fit/predict interface
- Probability calculations (PMF, CDF)
- Model persistence with backward compatibility
"""

import logging
import numpy as np
import pandas as pd
import statsmodels.api as sm
import pickle
from pathlib import Path

from typing import Tuple, Dict, Any, List, Optional, Union
# sklearn imports moved to lazy loading for faster CLI startup
# from sklearn.preprocessing import StandardScaler  # Lazy in __init__
from scipy.stats import nbinom, poisson

logger = logging.getLogger(__name__)

class NegativeBinomialWrapper:
    """
    Wrapper for Statsmodels Negative Binomial (NB2) regression.
    Learns coefficients and overdispersion parameter (alpha).
    """
    
    CURRENT_VERSION = "1.1.2"
    
    def __init__(self, alpha: float = None):
        """
        :param alpha: Initial or fixed dispersion. If None, it is estimated during fit (MLE).
        """
        self.results = None # Kept transiently for fitting interpretation if needed, but not saved.
        self.alpha_ = alpha
        
        # Sklearn-style attributes
        self.coef_: Optional[np.ndarray] = None
        self.intercept_: Optional[float] = None
        self.feature_names_in_: Optional[List[str]] = None
        
        # Statistics
        self.bse_: Optional[np.ndarray] = None # Standard errors for coef_
        self.intercept_bse_: Optional[float] = None
        self.alpha_bse_: Optional[float] = None
        
        self.pvalues_: Optional[np.ndarray] = None # P-values for coef_
        self.intercept_pvalue_: Optional[float] = None
        self.alpha_pvalue_: Optional[float] = None
        
        # Lazy import for faster CLI startup
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()
        self.version = self.CURRENT_VERSION
        
    def fit(self, X: pd.DataFrame, y: pd.Series, strict_convergence: bool = True):
        """
        Fits the Negative Binomial model using Maximum Likelihood.
        Automatically estimates alpha.
        
        Args:
            X: Feature matrix (Must be DataFrame to capture feature names)
            y: Target vector
            strict_convergence: If True, raises RuntimeError if MLE doesn't converge.
        """
        # Enforce DataFrame for training to ensure we have feature names
        if not isinstance(X, pd.DataFrame):
             raise TypeError("Input X must be a pandas DataFrame for training to capture feature names.")

        self.feature_names_in_ = X.columns.tolist()
        
        # Drop missing targets
        mask = y.notna()
        # Reset index to ensure alignment (robustness)
        X = X[mask].reset_index(drop=True)
        y = y[mask].reset_index(drop=True)
        
        if len(y) == 0:
            raise ValueError("Cannot train: all target values are NaN or input is empty.")

        # VALIDATION: Non-negative targets
        if (y < 0).any():
            raise ValueError("Target values must be non-negative for Negative Binomial model.")

        # VALIDATION: Sufficient samples (heuristic: N > k + 2)
        if len(y) < len(self.feature_names_in_) + 2:
            raise ValueError(f"Insufficient samples: {len(y)} < {len(self.feature_names_in_) + 2} (Features + 2)")
        
        # Scale (returns numpy array)
        X_scaled = self.scaler.fit_transform(X)
        
        # Add intercept directly to numpy array
        X_const = sm.add_constant(X_scaled, has_constant='add')
        
        # Using discrete_model.NegativeBinomial for MLE optimization of params + alpha
        try:
            model = sm.NegativeBinomial(y.values, X_const, loglike_method='nb2')
            self.results = model.fit(disp=0) # disp=0 silences convergence logs
            
            # Check Convergence
            if hasattr(self.results, 'mle_retvals') and 'converged' in self.results.mle_retvals:
                converged = self.results.mle_retvals['converged']
                if not converged:
                     msg = "NegativeBinomial training did not converge (mle_retvals['converged'] == False)."
                     if strict_convergence:
                         raise RuntimeError(msg)
                     logger.warning(msg)

            # Extract Parameters
            expected_params = X_const.shape[1] + 1 # features + const + alpha
            if len(self.results.params) != expected_params:
                logger.error(f"Unexpected params length: {len(self.results.params)} vs expected {expected_params}")
                
            # Alpha extraction (Last parameter in MLE NB2)
            self.alpha_ = self.results.params[-1]
            
            # Coefficients and Intercept
            self.intercept_ = self.results.params[0]
            self.coef_ = self.results.params[1:-1]
            
            # Extract statistics for persistence
            # Splitting stats to align with attributes
            # Stats array structure: [Intercept, Coef1, Coef2, ..., Alpha]
            
            # Standard Errors
            bse_all = self.results.bse
            self.intercept_bse_ = bse_all[0]
            self.bse_ = bse_all[1:-1]
            self.alpha_bse_ = bse_all[-1]
            
            # P-Values
            pvalues_all = self.results.pvalues
            self.intercept_pvalue_ = pvalues_all[0]
            self.pvalues_ = pvalues_all[1:-1]
            self.alpha_pvalue_ = pvalues_all[-1]
            
            # Persist summary as text for reports
            try:
                xnames = ['const'] + self.feature_names_in_ + ['alpha']
                self.summary_text_ = self.results.summary(xname=xnames).as_text()
            except Exception as e:
                logger.warning(f"Could not generate summary text: {e}")
                self.summary_text_ = "Summary generation failed."
            
            logger.info(f"NB Model fitted successfully. Alpha={self.alpha_:.4f}, Intercept={self.intercept_:.4f}, Samples={len(y)}")
            logger.debug(f"Features: {self.feature_names_in_}")
            logger.debug(f"Scaler Mean: {self.scaler.mean_}, Var: {self.scaler.var_}")
            
            if self.alpha_ < 1e-4:
                logger.warning(f"Alpha is very small ({self.alpha_:.6f}). Model is effectively Poisson.")

        except Exception as e:
            logger.error(f"Error fitting NB model: {e}")
            raise

    # Alias for system compatibility
    train = fit

    def summary(self):
        """
        Returns statsmodels summary as text.
        Works even after loading from disk (returns cached summary).
        """
        if hasattr(self, 'summary_text_') and self.summary_text_:
            return self.summary_text_
            
        if self.results is not None:
             xnames = ['const'] + self.feature_names_in_ + ['alpha']
             return self.results.summary(xname=xnames).as_text()
             
        raise ValueError("No fitted results or cached summary available.")

    def _validate_data(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        if self.coef_ is None:
            raise ValueError("Model not fitted.")
        
        # Handle Numpy Arrays by wrapping them (assuming order is correct)
        if isinstance(X, np.ndarray):
             if X.shape[1] != len(self.feature_names_in_):
                  raise ValueError(f"Input numpy array has {X.shape[1]} features, expected {len(self.feature_names_in_)}.")
             X = pd.DataFrame(X, columns=self.feature_names_in_)
        
        # Check column uniqueness
        if not X.columns.is_unique:
             raise ValueError("Input X contains duplicate columns.")

        # Check features existence
        missing = set(self.feature_names_in_) - set(X.columns)
        if missing:
            raise RuntimeError(f"Feature mismatch. Missing: {missing}")
            
        # Reorder and Select (Robust to order changes)
        X_selected = X[self.feature_names_in_]
        
        return self.scaler.transform(X_selected)

    def predict(self, X: Union[pd.DataFrame, np.ndarray], offset: np.ndarray = None) -> np.ndarray:
        """
        Returns mu (predicted means).
        
        Args:
            X: Features DataFrame or Numpy Array.
            offset: Optional LOG-SCALE offset (exposure), e.g., log(time). Must match X length.
        """
        X_scaled = self._validate_data(X)

        
        if offset is not None and len(offset) != len(X):
            raise ValueError(f"Offset length {len(offset)} doesn't match X length {len(X)}")
            
        # Linear Predictor: Intercept + X @ Coef + Offset
        # shape: (n_samples, n_features) @ (n_features,) -> (n_samples,)
        linear_pred = np.dot(X_scaled, self.coef_) + self.intercept_
        
        if offset is not None:
            linear_pred += offset
            
        # Stability: Clip before exp
        linear_pred = np.clip(linear_pred, -20, 20)
            
        return np.exp(linear_pred)
        
    def predict_with_alpha(self, X: pd.DataFrame, offset: np.ndarray = None) -> Tuple[np.ndarray, float]:
        """
        Returns (mu, alpha).
        """
        return self.predict(X, offset=offset), self.alpha_

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Alias for predict, returns mu only.
        """
        return self.predict(X)

    def pmf(self, k: int, X: pd.DataFrame, offset: np.ndarray = None) -> np.ndarray:
        """
        Probability Mass Function P(Y=k). Vectorized.
        """
        mu = self.predict(X, offset=offset)
        alpha = self.alpha_
        
        if alpha <= 1e-9:
             return poisson.pmf(k, mu)
        
        # Vectorized nbinom (n is scalar, p is vector)   
        n = 1.0 / alpha
        p = n / (n + mu)
        return nbinom.pmf(k, n, p)
    
    def cdf(self, k: int, X: pd.DataFrame, offset: np.ndarray = None) -> np.ndarray:
        """
        Cumulative Distribution Function P(Y<=k). Vectorized.
        """
        mu = self.predict(X, offset=offset)
        alpha = self.alpha_
        
        if alpha <= 1e-9:
             return poisson.cdf(k, mu)
        
        # Vectorized nbinom
        n = 1.0 / alpha
        p = n / (n + mu)
        return nbinom.cdf(k, n, p)

    def predict_proba_range(self, X: pd.DataFrame, lower: int, upper: int, offset: np.ndarray = None) -> np.ndarray:
        """
        Calculates Probability P(lower <= Y <= upper).
        """
        if lower < 0 or upper < 0:
            raise ValueError("lower and upper must be non-negative")
        if lower > upper:
            raise ValueError(f"lower ({lower}) cannot exceed upper ({upper})")
            
        # Use CDF method
        # P(L <= Y <= U) = CDF(U) - CDF(L-1)
        prob_upper = self.cdf(upper, X, offset)
        prob_lower_minus_1 = self.cdf(lower - 1, X, offset) if lower > 0 else 0.0
        
        return prob_upper - prob_lower_minus_1

    def __setstate__(self, state):
        """
        Handle unpickling state, including migration for legacy models (pre v1.1.0).
        """
        # 1. Naive update (useful for version, scaler, etc)
        self.__dict__.update(state)
        
        # 2. Explicit mapping for core parameters (Key -> Attribute)
        # Because __getstate__ uses 'coef' but attribute is 'coef_'
        if 'coef' in state: self.coef_ = state['coef']
        if 'intercept' in state: self.intercept_ = state['intercept']
        if 'alpha' in state: self.alpha_ = state['alpha']
        
        # 3. Stats mapping
        if 'bse' in state: self.bse_ = state['bse']
        if 'intercept_bse' in state: self.intercept_bse_ = state['intercept_bse']
        if 'alpha_bse' in state: self.alpha_bse_ = state['alpha_bse']
        
        if 'pvalues' in state: self.pvalues_ = state['pvalues']
        if 'intercept_pvalue' in state: self.intercept_pvalue_ = state['intercept_pvalue']
        if 'alpha_pvalue' in state: self.alpha_pvalue_ = state['alpha_pvalue']
        
        if 'feature_names' in state: self.feature_names_in_ = state['feature_names']

        # Backward compatibility for legacy models
        if not hasattr(self, 'coef_') or self.coef_ is None:
            # Try both keys
            params = state.get('params', state.get('params_'))
            
            if params is not None and len(params) >= 2:
                # Handle Pandas Series vs Numpy Array
                if hasattr(params, 'iloc'):
                    self.intercept_ = params.iloc[0]
                    self.coef_ = params.iloc[1:-1].values if hasattr(params.iloc[1:-1], 'values') else params.iloc[1:-1]
                    # Alpha
                    if not hasattr(self, 'alpha_') or self.alpha_ is None:
                        # Only grab alpha if it looks like it's included (NB2 usually has it at end)
                        # Check labels if possible? 
                        # If const is 0, alpha is -1?
                        self.alpha_ = params.iloc[-1]
                else:
                    self.intercept_ = params[0]
                    self.coef_ = params[1:-1]
                    if not hasattr(self, 'alpha_') or self.alpha_ is None:
                        self.alpha_ = params[-1]
            else:
                 self.coef_ = None
                 self.intercept_ = None
                 
        if not hasattr(self, 'alpha_'):
             self.alpha_ = state.get('alpha')

        if not hasattr(self, 'feature_names_in_'):
             self.feature_names_in_ = state.get('feature_names', state.get('features'))

        self.results = None

    def __getstate__(self):
        """
        Custom pickling state to avoid saving fragile statsmodels results.
        Returns the same dictionary structure as the old save() method.
        """
        if self.coef_ is None:
             # Even if not fitted, we should return a valid state (though less useful)
             # But usually specific to fitted models.
             pass

        state = {
            'coef': self.coef_,
            'intercept': self.intercept_,
            'alpha': self.alpha_,
            'feature_names': self.feature_names_in_,
            'scaler': self.scaler,
            'summary_text': getattr(self, 'summary_text_', None),
            
            # Check for new separated stats (v1.1.1)
            'bse': getattr(self, 'bse_', None),
            'intercept_bse': getattr(self, 'intercept_bse_', None),
            'alpha_bse': getattr(self, 'alpha_bse_', None),
            
            'pvalues': getattr(self, 'pvalues_', None),
            'intercept_pvalue': getattr(self, 'intercept_pvalue_', None),
            'alpha_pvalue': getattr(self, 'alpha_pvalue_', None),
            
            'version': self.CURRENT_VERSION
        }
        return state

    def save(self, filepath: Path):
        """
        Saves model object to pickle file.
        Uses __getstate__ to control serialization.
        """
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
            
    @classmethod
    def load(cls, filepath: Path) -> 'NegativeBinomialWrapper':
        with open(filepath, 'rb') as f:
            obj = pickle.load(f)
        return obj
            
        # Version Check
        file_ver = state.get('version', 'unknown')
        
        obj = cls()
        obj.coef_ = state.get('coef')
        obj.intercept_ = state.get('intercept')
        obj.alpha_ = state.get('alpha')

        # Backward compatibility for legacy models (pre v1.1.0) using 'params'
        if obj.coef_ is None and 'params' in state:
            params = state['params']
            # Assuming params structure: [Intercept, ...coefs..., alpha]
            if len(params) >= 2:
                obj.intercept_ = params[0]
                obj.coef_ = params[1:-1] 
                # Note: alpha_ is already loaded from 'alpha' key which was saved separately,
                # but if it was None for some reason, we could grab params[-1].
                # We assume 'alpha' key is reliable from old save().

        
        # Backwards compatibility for 'features' vs 'feature_names'
        obj.feature_names_in_ = state.get('feature_names', state.get('features'))
        
        obj.summary_text_ = state.get('summary_text')
        
        # Recover separated stats (v1.1.1)
        obj.bse_ = state.get('bse')
        obj.intercept_bse_ = state.get('intercept_bse')
        obj.alpha_bse_ = state.get('alpha_bse')
        
        obj.pvalues_ = state.get('pvalues')
        obj.intercept_pvalue_ = state.get('intercept_pvalue')
        obj.alpha_pvalue_ = state.get('alpha_pvalue')
        
        # Legacy/transition support for v1.1.0/1.0.3 pvalues/bse if they were full arrays
        # If we loaded an older model, bse might include all params (intercept...alpha)
        # We can implement smart migration if needed, but for now assuming clean break or re-train.
        if obj.bse_ is not None and len(obj.bse_) > len(obj.feature_names_in_) and obj.intercept_bse_ is None:
             # Likely old format v1.1.0 logic where bse_ was everything
             # Attempt to slice
             bse_full = obj.bse_
             obj.intercept_bse_ = bse_full[0]
             obj.bse_ = bse_full[1:-1]
             obj.alpha_bse_ = bse_full[-1]
             
             pvalues_full = obj.pvalues_
             if pvalues_full is not None:
                 obj.intercept_pvalue_ = pvalues_full[0]
                 obj.pvalues_ = pvalues_full[1:-1]
                 obj.alpha_pvalue_ = pvalues_full[-1]

        # Explicitly set results to None
        obj.results = None

        obj.scaler = state['scaler']
        obj.version = file_ver
        
        return obj
