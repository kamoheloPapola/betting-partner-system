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
            self.trainer.train_model(
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

