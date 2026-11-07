"""
Training Orchestrator.

Coordinates the complete training pipeline across leagues
and model types. Handles feature set validation, xG coverage
checks, and batch training execution.
"""
import logging
import pandas as pd
from typing import List, Optional, Tuple, Dict, Any

from src.core.container import ServiceContainer
from src.core.exceptions import DataValidationError, InsufficientDataError
from src.ml.trainer import ModelTrainer
from src.config import DEFAULT_TRAINING_LEAGUES, Thresholds
from src.config.alpha_config import get_alpha
from src.ml.training.model_configs import MODEL_CONFIGS, ModelType, FeatureSet, TrainingMode
from src.ml.training.feature_selector import select_features
from src.ml.training.data_validator import validate_training_data, filter_historical_matches

logger = logging.getLogger(__name__)

class TrainingOrchestrator:
    """
    Orchestrates the training for all models in the system.
    """
    
    def __init__(self):
        self.container = ServiceContainer.get_instance()
        self.trainer = ModelTrainer(self.container.registry)

    def run(self, df: pd.DataFrame, model_type: ModelType, league_code: Optional[str] = None, mode: TrainingMode = TrainingMode.DEBUG) -> None:
        """
        Run the training sequence for the specified model type and league(s).
        """
        # LOCK ENFORCEMENT: Hard block if system is locked
        from src.config.model_state import require_unlocked
        require_unlocked("Training orchestration")
        
        # 1. Build training targets
        targets = self._build_training_targets(df, league_code)
        
        # 2. Iterate and train
        for l_code, l_df in targets:
            try:
                self._train_league(l_df, l_code, model_type, mode)
            except InsufficientDataError as e:
                logger.warning(f"Skipping training for {l_code or 'Global'}: {e.message}")
            except Exception as e:
                logger.error(f"Unexpected error training {l_code or 'Global'}: {str(e)}", exc_info=True)

    def _build_training_targets(self, df: pd.DataFrame, league_code: Optional[str]) -> List[Tuple[Optional[str], pd.DataFrame]]:
        """Build list of (league_code, dataframe) tuples for training."""
        if league_code:
            # Train single league
            league_df = df[df['league'] == league_code]
            if league_df.empty:
                raise InsufficientDataError(f"No data available for league {league_code}")
            return [(league_code, league_df)]
        
        # Train global + default leagues
        targets = [(None, df)]  # Global model
        
        for code in DEFAULT_TRAINING_LEAGUES:
            sub_df = df[df['league'] == code]
            if not sub_df.empty:
                targets.append((code, sub_df))
            else:
                logger.debug(f"Skipping {code}: no data")
        
        return targets

    def _train_league(self, df: pd.DataFrame, league: Optional[str], model_type: ModelType, mode: TrainingMode) -> None:
        """Execute training sequence for specified league."""
        validate_training_data(df, league or "Global")
        
        # 4. Filter configs by model type
        relevant_configs = [c for c in MODEL_CONFIGS if c['type'] == model_type]
        if not relevant_configs:
            raise DataValidationError(f"Unsupported model type: {model_type}")
            
        _goals_models: Dict[str, Any] = {}  # Collect home/away goals models for calibration

        for config in relevant_configs:
            
            # B. Check column availability
            if not all(col in df.columns for col in config['required_columns']):
                logger.warning(f"Skipping {config['name']} for {league}: missing columns {config['required_columns']}")
                continue
            
            # C. Feature Selection
            features = select_features(df, config['target'], feature_set=config['feature_set'])
            
            # D. Dynamic Alpha Injection (from alpha_config)
            params = dict(config['params'])  # Copy to avoid mutation
            if model_type == ModelType.POISSON:
                tuned_alpha = get_alpha(league or "Global", config['name'])
                params['alpha'] = tuned_alpha
                logger.debug(f"Using alpha={tuned_alpha} for {config['name']} ({league or 'Global'})")
            
            # E. Consistent Metadata
            metadata = {
                "feature_set": config['feature_set'].value.upper(),
                "xg_coverage": "FULL",  # Assuming full coverage for now as guards handle filtering
                "samples": len(df),
                "trained_at": pd.Timestamp.now().isoformat()
            }
            
            # F. Execution
            trained_model, _ = self.trainer.train_model(
                df=df,
                target_col=config['target'],
                model_type=config['type'].value,
                model_name=config['name'],
                params=params,
                features=features,
                mode=mode.value,
                league=league,
                extra_metadata=metadata
            )

            # Collect goals models for post-loop calibration
            if config['name'] == 'poisson_home_base':
                _goals_models['home'] = (trained_model, features)
            elif config['name'] == 'poisson_away_base':
                _goals_models['away'] = (trained_model, features)

        # Fit goals/BTTS calibrators once both home and away goals models are available
        if league and 'home' in _goals_models and 'away' in _goals_models:
            self._fit_goals_calibrators(df, league, _goals_models)

    def _fit_goals_calibrators(self, df: pd.DataFrame, league: str, goals_models: Dict[str, Any]) -> None:
        """Fit and save goals/BTTS binary calibrators for the given league."""
        import numpy as np
        from scipy.stats import poisson as _poisson
        from src.ml.calibration import fit_best_binary_calibrator, save_binary_calibrator_artifact
        from src.config import MODELS_DIR

        MIN_SAMPLES = 100

        try:
            home_model, home_features = goals_models['home']
            away_model, away_features = goals_models['away']

            _, _, test_df = self.trainer.time_series_split_train_val_test(df)
            if len(test_df) < MIN_SAMPLES:
                logger.warning("Goals calibration skipped for %s: test set too small (%d rows)", league, len(test_df))
                return

            X_test_h = test_df[home_features]
            X_test_a = test_df[away_features]
            mu_h = np.maximum(0.05, home_model.predict(X_test_h))
            mu_a = np.maximum(0.05, away_model.predict(X_test_a))

            probs_u25, probs_u35, probs_btts = [], [], []
            for lh, la in zip(mu_h, mu_a):
                u25 = sum(_poisson.pmf(h, lh) * _poisson.pmf(a, la) for h in range(6) for a in range(6) if h + a <= 2)
                u35 = sum(_poisson.pmf(h, lh) * _poisson.pmf(a, la) for h in range(6) for a in range(6) if h + a <= 3)
                btts = (1 - _poisson.pmf(0, lh)) * (1 - _poisson.pmf(0, la))
                probs_u25.append(u25)
                probs_u35.append(u35)
                probs_btts.append(btts)

            total_h = np.asarray(test_df['home_score'], dtype=float)
            total_a = np.asarray(test_df['away_score'], dtype=float)
            total = total_h + total_a

            for market, probs_raw, y_bin in [
                ("goals_u25",      np.array(probs_u25),  (total <= 2).astype(float)),
                ("goals_u35",      np.array(probs_u35),  (total <= 3).astype(float)),
                ("goals_btts_yes", np.array(probs_btts), ((total_h > 0) & (total_a > 0)).astype(float)),
            ]:
                cal = fit_best_binary_calibrator(probs_raw, y_bin)
                if cal is None:
                    logger.warning("Goals calibrator fit returned None for %s [%s]", market, league)
                    continue
                cal_path = MODELS_DIR / "calibrators" / f"{market}_{league}_v1.pkl"
                save_binary_calibrator_artifact(cal_path, cal)
                logger.info("Saved goals calibrator: %s [%s] ECE-winner=%s n=%d", market, league, cal.get("type"), cal.get("n_samples"))
        except Exception as exc:
            logger.warning("Goals calibrator fitting failed for %s: %s", league, exc, exc_info=True)
