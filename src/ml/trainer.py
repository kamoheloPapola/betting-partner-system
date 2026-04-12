"""
Model Trainer Module.

Orchestrates the training lifecycle for ML models, including:
- Data splitting (time-series aware)
- Feature selection and sanitization
- Model training and evaluation
- Artifact persistence and registration

Enforces production safeguards to prevent data leakage and ensure model integrity.
"""
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# sklearn imports moved to lazy loading for faster CLI startup
# from sklearn.metrics import mean_absolute_error  # Lazy imported in _evaluate_model

from src.config import DATA_DIR, MODELS_DIR
from src.ml.calibration import (
    apply_binary_calibrator,
    fit_best_binary_calibrator,
    save_binary_calibrator_artifact,
)
from src.ml.guards import (
    guard_feature_coverage,
    guard_min_samples,
    guard_single_league,

)
from src.ml.models.base_models import BaseModel, PoissonWrapper
from src.ml.models.corners.nb_model import NegativeBinomialWrapper
from src.ml.registry import ModelRegistry

# Define public API
__all__ = ["ModelTrainer"]

logger = logging.getLogger(__name__)

# Constants
DRIFT_BASELINE_FILE = DATA_DIR / "models" / "drift_baselines.json"
ODDS_KEYWORDS = [
    'odds', 'coef', 'bet365', 'bwin', 'pinnacle', 'williamhill', 'marathon'
]
LEAKAGE_KEYWORDS = [
    'score', 'corners', 'cards', 'yellow', 'red', 'shots', 'xg', 'goals', 'target'
]
EXCLUDE_COLS = [
    'match_id', 'date', 'home_team', 'away_team', 'source', 'status', 
    'competition', 'season', 'created_at', 'updated_at'
]


class ModelTrainer:
    """
    Handles data splitting, model training, evaluation, and registration.
    
    Attributes:
        registry: The model registry service for versioning and metadata.
    """
    
    def __init__(self, registry: ModelRegistry) -> None:
        self.registry = registry

    def time_series_split(
        self, 
        df: pd.DataFrame, 
        test_size: float = 0.2
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data chronologically to prevent temporal leakage.
        
        Args:
            df: Input DataFrame.
            test_size: Fraction of data to use for testing (0.0 to 1.0).
            
        Returns:
            Tuple of (train_df, test_df).
        """
        df = df.sort_values('date')
        cut_idx = int(len(df) * (1 - test_size))
        return df.iloc[:cut_idx], df.iloc[cut_idx:]

    def time_series_split_train_val_test(
        self,
        df: pd.DataFrame,
        val_size: float = 0.2,
        test_size: float = 0.2,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Chronological train/validation/test split (no leakage)."""
        ordered = df.sort_values("date")
        n = len(ordered)
        if n < 3:
            raise ValueError("Need at least 3 samples for train/validation/test split.")

        test_n = max(1, int(round(n * test_size)))
        val_n = max(1, int(round(n * val_size)))
        if test_n + val_n >= n:
            overflow = (test_n + val_n) - (n - 1)
            if val_n > 1:
                reduce_val = min(val_n - 1, overflow)
                val_n -= reduce_val
                overflow -= reduce_val
            if overflow > 0 and test_n > 1:
                test_n -= min(test_n - 1, overflow)

        train_end = n - test_n - val_n
        val_end = n - test_n
        train_df = ordered.iloc[:train_end]
        val_df = ordered.iloc[train_end:val_end]
        test_df = ordered.iloc[val_end:]
        return train_df, val_df, test_df

    def train_model(
        self, 
        df: pd.DataFrame, 
        target_col: str,
        league: Optional[str] = None,
        model_type: str = 'poisson', 
        model_name: str = 'model_v1', 
        features: Optional[List[str]] = None,
        params: Optional[Dict[str, Any]] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
        mode: str = 'debug'
    ) -> Tuple[BaseModel, Dict[str, Any]]:
        """
        Train, evaluate, and register a model.
        
        Args:
            df: Training data.
            target_col: Target column name.
            league: Optional league code for specific models.
            model_type: Architecture type ('poisson', 'nb').
            model_name: Base name for the model artifact.
            features: explicit list of features (auto-detected if None).
            params: Hyperparameters for the model.
            extra_metadata: Additional metadata to store in registry.
            mode: 'debug' (lenient) or 'production' (strict guards).
            
        Returns:
            Tuple of (trained_model, metadata_dict).
        """
        # 1. Guards
        if league:
             guard_single_league(df, league)
        
        min_prod = 300 if mode == 'production' else 10
        guard_min_samples(df, min_matches=min_prod)
        
        # 2. Feature Selection
        if features is None:
            features = self._auto_select_features(df, target_col)
            
        # Log Training Start
        n_samples = len(df)
        feats_desc = extra_metadata.get('feature_set', 'UNKNOWN') if extra_metadata else f"{len(features)} params"
        logger.info(
            f"Training {model_name} ({model_type}) on {n_samples} samples. "
            f"League: {league or 'Global'}. Features: {feats_desc}. Mode: {mode}"
        )
        
        # 3. Split Data (chronological): train -> validation -> test
        train_df, val_df, test_df = self.time_series_split_train_val_test(df)
        X_train, y_train = train_df[features], train_df[target_col]
        X_val, y_val = val_df[features], val_df[target_col]
        X_test, y_test = test_df[features], test_df[target_col]

        # Guard: Feature Coverage
        if mode == 'production':
            guard_feature_coverage(X_train, required=0.98)

        # 4. Init Model
        model = self._init_model(model_type, params)
        
        # 5. Sanitization
        X_train = self._sanitize_features(X_train, mode)
        X_val_clean = self._sanitize_features(X_val, mode='debug')
        X_test_clean = self._sanitize_features(X_test, mode='debug') # Always lenient on test set

        # 6. Train
        model.train(X_train, y_train)

        # 6.5 Log Feature Importance Hook
        try:
            importances = None
            if hasattr(model, "model"):
                if hasattr(model.model, "feature_importances_"):
                    importances = model.model.feature_importances_
                elif hasattr(model.model, "coef_"):
                    importances = np.abs(model.model.coef_)

            if importances is not None and len(importances) == len(features):
                imp_df = pd.DataFrame({
                    "feature_name": features,
                    "importance": importances
                }).sort_values("importance", ascending=False)
                imp_path = MODELS_DIR / "feature_importance.csv"
                imp_df.to_csv(imp_path, index=False)
                logger.info(f"Exported feature importance to {imp_path}")
        except Exception as e:
            logger.warning(f"Failed to log feature importance: {e}")

        # 7. Evaluate
        preds = model.predict(X_test_clean)
        metrics = self._calculate_metrics(y_test, preds, model_type)
        
        # 8. Save Model
        version = self.registry.get_next_version(model_name, league, increment="minor")
        filename = f"{model_name}_v{version}.pkl"
        
        if league:
             sub_dir = MODELS_DIR / model_type / league
             sub_dir.mkdir(parents=True, exist_ok=True)
             save_path = sub_dir / filename
             rel_path = f"{model_type}/{league}/{filename}"
             logger.info(f"Saving league-specific model to: {save_path}")
        else:
             save_path = MODELS_DIR / filename
             rel_path = filename
             logger.info(f"Saving global model to: {save_path}")

        model.save(save_path)

        # 8.5 Fit post-hoc calibrator on validation split when available (never test).
        calibrator_meta = self._fit_posthoc_calibrator(
            model=model,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val_clean,
            y_val=y_val,
            model_type=model_type,
            save_path=save_path,
        )

        # 9. Register
        metadata = {
            "type": model_type,
            "target": target_col,
            "features": features,
            "params": params,
            "filename": str(rel_path),
            "train_size": len(train_df),
            "val_size": len(val_df),
            "test_size": len(test_df),
            "mode": mode,
            "status": "productive" if mode == 'production' else "provisional",
            "league": league, 
            "metrics": metrics
        }

        if calibrator_meta is not None:
            metadata.update(calibrator_meta)
        
        if extra_metadata:
            metadata.update(extra_metadata)

        baseline_export = self._export_drift_baselines(
            extra_metadata=extra_metadata,
            model_name=model_name,
            model_type=model_type,
            league=league,
            version=version,
            train_df=train_df,
            test_df=test_df,
        )
        if baseline_export is not None:
            metadata["drift_baselines_file"] = str(baseline_export)
            
        self.registry.register_model(model_name, version, metadata)
        
        return model, metadata

    def train_model_in_memory(
        self, 
        df: pd.DataFrame, 
        target_col: str, 
        model_type: str = 'poisson', 
        params: Optional[Dict[str, Any]] = None,
        features: Optional[List[str]] = None
    ) -> BaseModel:
        """
        Train a model without saving to disk (used for backtesting).
        
        Hardened for stability:
        1. Filters out NaN targets.
        2. Enforces minimum sample size (50).
        3. Uses consistent feature selection logic.
        """
        # 1. Prepare Data
        valid = df[df[target_col].notna()].copy()
        
        # Guard: Insufficient Data (ISSUE #10)
        if len(valid) < 50:
            raise ValueError(
                f"Insufficient training data for {target_col}: {len(valid)} samples (min 50 required)"
            )

        # 2. Feature Selection
        if features is None:
            features = self._auto_select_features(valid, target_col)

        X = valid[features].replace([np.inf, -np.inf], np.nan).fillna(0)
        y = valid[target_col]
        
        # 3. Init & Train
        model = self._init_model(model_type, params)
        model.train(X, y)
        
        return model

    def _auto_select_features(self, df: pd.DataFrame, target_col: str) -> List[str]:
        """Detect valid feature columns, excluding odds and leakage."""
        exclude = set(EXCLUDE_COLS + [target_col])
        
        candidates = [
            c for c in df.columns 
            if c not in exclude and df[c].dtype in [np.float64, np.int64]
        ]
        
        features = []
        for c in candidates:
            # 1. Reject Odds
            if any(k in c.lower() for k in ODDS_KEYWORDS):
                # logger.warning(f"Excluding potential odds column: {c}")
                continue
                
            # 2. Reject IDs and Artifacts
            if c.endswith('_id') or '_id_' in c or c.endswith('_x') or c.endswith('_y'):
                continue
                
            # 3. LEAKAGE PREVENTION & CLEANUP
            # "Future" match stats are not valid unless they are known priors (Rolling/Form)
            is_safe = 'rolling' in c or 'form' in c or 'days_rest' in c
            is_leakage = any(x in c for x in LEAKAGE_KEYWORDS)
            
            if is_leakage and not is_safe:
                continue
            
            # 4. MULTICOLLINEARITY PREVENTION (Linear Models)
            if c.endswith('_diff'):
                continue
            
            features.append(c)
            
        return features

    def _init_model(self, model_type: str, params: Optional[Dict[str, Any]]) -> BaseModel:
        """Initialize the model wrapper based on type."""
        if model_type == 'poisson':
            return PoissonWrapper(**(params or {}))
        elif model_type == 'nb':
            return NegativeBinomialWrapper(alpha=params.get('alpha') if params else None)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def _sanitize_features(self, X: pd.DataFrame, mode: str) -> pd.DataFrame:
        """Check for and fix NaNs/Infs in feature matrix."""
        nan_count = X.isna().sum().sum()
        inf_count = np.isinf(X.values).sum()
        total_cells = X.size
        
        bad_ratio = (nan_count + inf_count) / total_cells if total_cells > 0 else 0
        
        msg = f"Feature Sanitization: Found {nan_count} NaNs and {inf_count} Infs ({bad_ratio:.2%})"
        
        if bad_ratio > 0.01:
            if mode == 'production':
                logger.error(f"{msg}. ABORTING in Production mode (>1%).")
                raise ValueError(f"Data corruption too high ({bad_ratio:.2%}) for PRODUCTION training.")
            else:
                logger.warning(f"{msg}. Proceeding in DEBUG mode (fixing blindly).")
        else:
            logger.info(msg)
            
        return X.replace([np.inf, -np.inf], np.nan).fillna(0)

    def _calculate_metrics(
        self, 
        y_true: pd.Series, 
        y_pred: np.ndarray, 
        model_type: str
    ) -> Dict[str, float]:
        """Calculate evaluation metrics (MAE, Calibration Score)."""
        try:
            # Mask out NaNs in ground truth (Audit Integrity)
            eval_mask = ~np.isnan(y_true)
            y_eval_clean = y_true[eval_mask]
            preds_eval_clean = y_pred[eval_mask]

            if len(y_eval_clean) == 0:
                return {"calibration_score": 999.0, "mae": 999.0}

            # Lazy import for faster CLI startup
            from sklearn.metrics import mean_absolute_error
            mae = float(mean_absolute_error(y_eval_clean, preds_eval_clean))
            
            # Phase 23: Real ECE for Poisson (Binary Projection: p > 0)
            if model_type in ['poisson', 'nb']:
                lambdas = preds_eval_clean
                p_at_least_one = 1 - np.exp(-lambdas)
                y_binary = (y_eval_clean > 0).astype(int)
                col_score = self._calculate_binary_ece(p_at_least_one, y_binary)
            else:
                y_binary = (y_eval_clean > 0).astype(int)
                col_score = self._calculate_binary_ece(preds_eval_clean, y_binary)
                
            return {
                "calibration_score": float(col_score),
                "mae": float(mae)
            }
            
        except Exception as e:
            logger.warning(f"Could not calculate metrics: {e}")
            return {"calibration_score": 999.0, "mae": 999.0}

    def _calculate_binary_ece(
        self, 
        probs: np.ndarray, 
        actuals: np.ndarray, 
        n_bins: int = 10
    ) -> float:
        """Calculate Expected Calibration Error (ECE) for binary outcomes."""
        if len(probs) == 0: 
            return 999.0
        
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        
        for i in range(n_bins):
            # Mask for predictions in this bin
            mask = (probs > bin_boundaries[i]) & (probs <= bin_boundaries[i+1])
            if np.any(mask):
                bin_prob = probs[mask].mean()
                bin_actual = actuals[mask].mean()
                bin_weight = mask.sum() / len(probs)
                ece += bin_weight * np.abs(bin_prob - bin_actual)
                
        return float(ece)

    @staticmethod
    def _to_binary_probability(preds: np.ndarray, model_type: str) -> np.ndarray:
        """Map model outputs to binary-event probabilities for post-hoc calibration."""
        values = np.asarray(preds, dtype=float)
        if model_type in {"poisson", "nb"}:
            lambdas = np.clip(values, 0.0, None)
            return 1.0 - np.exp(-lambdas)
        return np.clip(values, 0.0, 1.0)

    def _fit_posthoc_calibrator(
        self,
        *,
        model: BaseModel,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        model_type: str,
        save_path: Path,
    ) -> Optional[Dict[str, Any]]:
        """
        Fit and persist the best post-hoc calibrator using validation data only.

        When the validation split is empty, use the earliest 20% of the training
        rows as a calibration-only fallback. This never changes model fitting or
        the held-out test set.
        """
        try:
            if len(X_val) == 0:
                if len(X_train) == 0 or len(y_train) == 0:
                    return None

                fallback_n = max(1, int(np.ceil(len(X_train) * 0.2)))
                X_val = X_train.iloc[:fallback_n]
                y_val = y_train.iloc[:fallback_n]
                logger.warning(
                    "Val split empty for calibration — using training holdout fallback (n=%s)",
                    fallback_n,
                )

            if len(y_val) == 0:
                return None

            val_preds = model.predict(X_val)
            p_raw = self._to_binary_probability(val_preds, model_type=model_type)
            y_binary = (pd.Series(y_val).fillna(0).astype(float) > 0).astype(int).to_numpy()

            best = fit_best_binary_calibrator(p_raw=p_raw, y_true=y_binary)
            if best is None:
                return None

            p_cal = apply_binary_calibrator(best["calibrator"], str(best["type"]), p_raw)
            ece_raw = self._calculate_binary_ece(p_raw, y_binary)
            ece_cal = self._calculate_binary_ece(p_cal, y_binary)

            calibrator_file = save_path.with_name(f"{save_path.stem}_calibrator.pkl")
            artifact_payload: Dict[str, Any] = {
                "calibrator": best["calibrator"],
                "type": best["type"],
                "fitted_at": best.get("fitted_at"),
                "validation": {
                    "ece_raw": float(ece_raw),
                    "ece_calibrated": float(ece_cal),
                    "n_samples": int(len(y_binary)),
                },
            }
            save_binary_calibrator_artifact(calibrator_file, artifact_payload)

            rel_file = calibrator_file.relative_to(MODELS_DIR).as_posix()
            return {
                "calibrator_filename": rel_file,
                "calibrator_type": str(best["type"]),
                "calibration_split": "validation_only",
                "calibrator_validation_ece": float(ece_cal),
                "calibrator_validation_raw_ece": float(ece_raw),
            }
        except Exception as e:
            logger.warning("Post-hoc calibration fit skipped: %s", e)
            return None

    def _export_drift_baselines(
        self,
        *,
        extra_metadata: Optional[Dict[str, Any]],
        model_name: str,
        model_type: str,
        league: Optional[str],
        version: str,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> Optional[Path]:
        """Persist drift baselines when a training run provides measured baseline metrics."""
        if not extra_metadata:
            return None

        drift_baselines = extra_metadata.get("drift_baselines")
        if not drift_baselines:
            return None

        required_metrics = ("hit_rate", "ece", "mean_conf")
        missing_metrics = [
            metric for metric in required_metrics if metric not in drift_baselines
        ]
        if missing_metrics:
            missing = ", ".join(sorted(missing_metrics))
            raise ValueError(
                f"drift_baselines missing required metrics: {missing}"
            )

        training_date = (
            drift_baselines.get("training_date")
            or extra_metadata.get("training_date")
            or datetime.now(timezone.utc).isoformat()
        )
        training_window_days = (
            drift_baselines.get("training_window_days")
            or extra_metadata.get("training_window_days")
        )
        selection_rate = drift_baselines.get("selection_rate", 0.03)

        payload = {
            "baseline_hit_rate": float(drift_baselines["hit_rate"]),
            "baseline_ece": float(drift_baselines["ece"]),
            "baseline_mean_conf": float(drift_baselines["mean_conf"]),
            "baseline_selection_rate": float(selection_rate),
            "training_date": training_date,
            "training_window_days": training_window_days,
            "baseline_metrics": {
                "hit_rate": float(drift_baselines["hit_rate"]),
                "ece": float(drift_baselines["ece"]),
                "mean_confidence": float(drift_baselines["mean_conf"]),
                "selection_rate": float(selection_rate),
            },
            "source": "training_artifact",
            "model_name": model_name,
            "model_type": model_type,
            "league": league or "Global",
            "version": version,
            "train_size": int(len(train_df)),
            "test_size": int(len(test_df)),
        }

        DRIFT_BASELINE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(DRIFT_BASELINE_FILE, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

        logger.info("Exported drift baselines to %s", DRIFT_BASELINE_FILE)
        return DRIFT_BASELINE_FILE
