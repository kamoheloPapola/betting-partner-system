"""
Walk-Forward Backtesting Engine.

Evaluates model performance using walk-forward validation
without data leakage. Features:
- Expanding training window
- Configurable retrain frequency
- Brier score and accuracy metrics
- Wilson confidence intervals
"""
import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Any, Tuple
from datetime import timedelta
import copy
from scipy import stats
from rich.progress import track

from src.features.pipeline import FeaturePipeline
from src.ml.registry import ModelRegistry
from src.ml.trainer import ModelTrainer
from src.ml.distributions import PoissonEngine
from src.cli.utils import LeagueCode, MATCH_SEPARATOR
from src.ml.training.feature_selector import select_features
from src.ml.training.model_configs import FeatureSet

logger = logging.getLogger(__name__)

class BacktestConfig:
    """Backtesting hyperparameters"""
    RETRAIN_FREQUENCY = 10  # Matches between retrains (balance speed/realism)
    MIN_TRAIN_SAMPLES = 50  # Using 50 to match ModelTrainer.train_model_in_memory guard
    
    # Realism profiles
    PROFILES = {
        'fast': 20,      # Retrain every 20 matches (~2 weeks)
        'balanced': 10,  # Retrain every 10 matches (~1 week)
        'realistic': 1   # Retrain daily (slow but most accurate)
    }

class Backtester:
    def __init__(self):
        self.registry = ModelRegistry()
        self.trainer = ModelTrainer(self.registry)
        self.pipeline = FeaturePipeline()
        self.engine = PoissonEngine()

    def _parse_season(self, season: Any) -> int:
        """
        Parse season to integer (start year).
        
        Examples:
            2024 -> 2024
            "2024" -> 2024
            "2024/25" -> 2024
            "invalid" -> raises ValueError
        
        Raises:
            ValueError: If season cannot be parsed
        """
        if isinstance(season, int):
            return season
        
        if isinstance(season, str):
            # Handle "YYYY/YY" format
            if '/' in season:
                try:
                    return int(season.split('/')[0])
                except (ValueError, IndexError):
                    raise ValueError(f"Invalid season format: {season}")
            
            # Handle "YYYY" format
            try:
                return int(season)
            except ValueError:
                raise ValueError(f"Invalid season format: {season}")
        
        raise ValueError(f"Season must be int or str, got {type(season)}")
    def _calculate_brier_score(self, probs: Dict[str, float], actual: str) -> float:
        """
        Calculate multi-class Brier score (normalized 0-1).
        
        Args:
            probs: Dict with keys 'home_win', 'draw', 'away_win'
            actual: One of 'HOME_WIN', 'DRAW', 'AWAY_WIN'
        """
        actual_vector = {
            'home_win': 1.0 if actual == 'HOME_WIN' else 0.0,
            'draw': 1.0 if actual == 'DRAW' else 0.0,
            'away_win': 1.0 if actual == 'AWAY_WIN' else 0.0
        }
        
        # Sum of squared errors across all 3 outcomes
        sse = sum(
            (probs.get(m, 0) - actual_vector[m]) ** 2
            for m in ['home_win', 'draw', 'away_win']
        )
        
        # Normalize by number of classes (3) to keep in [0, 1] range
        return sse / 3.0

    def _calculate_confidence_interval(
        self, 
        accuracy: float, 
        n_samples: int, 
        confidence: float = 0.95
    ) -> Tuple[float, float]:
        """
        Calculate Wilson score confidence interval for accuracy.
        
        Args:
            accuracy: Observed accuracy (0-1)
            n_samples: Number of predictions
            confidence: Confidence level (default 95%)
        
        Returns:
            (lower_bound, upper_bound)
        """
        if n_samples == 0:
            return 0.0, 0.0
            
        z = stats.norm.ppf((1 + confidence) / 2)
        
        denominator = 1 + z**2 / n_samples
        center = (accuracy + z**2 / (2 * n_samples)) / denominator
        margin = z * np.sqrt(accuracy * (1 - accuracy) / n_samples + z**2 / (4 * n_samples**2)) / denominator
        
        return (center - margin, center + margin)

    def _train_models(self, df: pd.DataFrame) -> Tuple[Any, Any, List[str], List[str]]:
        """
        Train home and away models.
        Returns: (model_home, model_away, feats_home, feats_away)
        """
        feats_home = select_features(df, 'home_score', feature_set=FeatureSet.BASE)
        model_home = self.trainer.train_model_in_memory(
            df=df, target_col='home_score', 
            model_type='poisson', params={'alpha': 0.01}, features=feats_home
        )
        
        feats_away = select_features(df, 'away_score', feature_set=FeatureSet.BASE)
        model_away = self.trainer.train_model_in_memory(
            df=df, target_col='away_score', 
            model_type='poisson', params={'alpha': 0.01}, features=feats_away
        )
        return model_home, model_away, feats_home, feats_away

    def run(self, train_seasons: List[int], test_season: int, league: str = "PL") -> pd.DataFrame:
        """
        Runs Walk-Forward Validation without data leakage.
        """
        logger.info(f"🚀 Starting Leak-Free Backtest. Train: {train_seasons}, Test: {test_season}")
        
        # 1. Load RAW data
        raw_matches = self.pipeline.load_stored_matches(league=league)
        if raw_matches.empty:
            logger.error(f"No match data found for league {league}")
            return pd.DataFrame()
            
        resolved_league = LeagueCode(league) if league in LeagueCode.__members__ else LeagueCode.PL
        target_comp = resolved_league.full_name
        raw_matches = raw_matches[raw_matches['competition'].isin([target_comp, league])].copy()
        raw_matches['season_start'] = raw_matches['season'].apply(self._parse_season)
        
        # 2. Split BEFORE feature engineering
        train_raw = raw_matches[raw_matches['season_start'].isin(train_seasons)].copy()
        test_raw = raw_matches[raw_matches['season_start'] == test_season].sort_values('date').copy()
        
        if train_raw.empty or test_raw.empty:
            logger.error("Training or Test set empty")
            return pd.DataFrame()
            
        results = []
        brier_scores = []
        correct_outcomes = 0
        skipped_matches = 0
        
        # 3. Walk Forward Loop
        block_size = BacktestConfig.RETRAIN_FREQUENCY
        test_matches_list = test_raw.to_dict('records')
        
        # Initial training pool
        train_df = self.pipeline.run(match_data=train_raw)
        current_expanding_features = train_df.copy()
        
        for i in track(range(0, len(test_matches_list), block_size), description=f"Simulating {league} {test_season}..."):
            batch_raw = pd.DataFrame(test_matches_list[i:i+block_size])
            match_count = i + len(batch_raw)
            # logger.info(f"  Batch {i//block_size + 1}: Processing {i+1}-{match_count}/{len(test_raw)}")
            
            # A. RE-CALCULATE FEATURES (Walk-forward)
            prior_test_raw = pd.DataFrame(test_matches_list[:i+block_size])
            combined_history_raw = pd.concat([train_raw, prior_test_raw])
            full_window_features = self.pipeline.run(match_data=combined_history_raw)
            
            batch_ids = batch_raw['match_id'].tolist() if 'match_id' in batch_raw.columns else batch_raw['match_hash'].tolist()
            batch_features = full_window_features[full_window_features['match_id'].isin(batch_ids)].copy()
            batch_features = batch_features.reset_index(drop=True)
            
            # B. TRAIN MODELS
            model_home, model_away, feats_home, feats_away = self._train_models(current_expanding_features)
            
            # C. PREDICT BATCH
            try:
                X_home = batch_features[feats_home].fillna(0)
                X_away = batch_features[feats_away].fillna(0)
                
                lambdas_home = model_home.predict(X_home)
                lambdas_away = model_away.predict(X_away)
                
                for idx, row in batch_features.iterrows():
                    lh, la = lambdas_home[idx], lambdas_away[idx]
                    probs = self.engine.calculate_probabilities(lh, la)
                    
                    home_score, away_score = row['home_score'], row['away_score']
                    if pd.isna(home_score) or pd.isna(away_score):
                        logger.warning(f"  Skipping postponed: {row['home_team']}{MATCH_SEPARATOR}{row['away_team']}")
                        skipped_matches += 1
                        continue
                        
                    outcome = "DRAW"
                    if home_score > away_score: outcome = "HOME_WIN"
                    elif away_score > home_score: outcome = "AWAY_WIN"
                    
                    brier = self._calculate_brier_score(probs, outcome)
                    brier_scores.append(brier)
                    
                    pred_outcome = "DRAW"
                    if probs['home_win'] > probs['away_win'] and probs['home_win'] > probs['draw']:
                        pred_outcome = "HOME_WIN"
                    elif probs['away_win'] > probs['home_win'] and probs['away_win'] > probs['draw']:
                        pred_outcome = "AWAY_WIN"
                    
                    if pred_outcome == outcome: correct_outcomes += 1
                        
                    results.append({
                        'date': row['date'], 'home': row['home_team'], 'away': row['away_team'],
                        'actual': outcome, 'pred': pred_outcome, 'prob_home': probs['home_win'],
                        'brier_home': brier, 'xg_home': lh, 'xg_away': la
                    })
            except Exception as e:
                logger.error(f"Batch prediction failure: {e}")
            
            # D. EXPAND TRAINING WINDOW
            current_expanding_features = pd.concat([current_expanding_features, batch_features])
            
        final_df = pd.DataFrame(results)
        
        # 4. Validate Results (ISSUE #16)
        if final_df.empty:
            logger.warning("Backtest produced no results")
            return final_df
            
        required_columns = ['date', 'home', 'away', 'actual', 'pred', 'prob_home', 'brier_home']
        missing = set(required_columns) - set(final_df.columns)
        if missing:
            raise ValueError(f"Backtest results missing required columns: {missing}")
            
        # Summary stats
        accuracy = (final_df['pred'] == final_df['actual']).mean()
        avg_brier = final_df['brier_home'].mean()
        ci_lower, ci_upper = self._calculate_confidence_interval(accuracy, len(final_df))
        
        logger.info("Backtest Summary:")
        logger.info(f"  Total predictions: {len(final_df)}")
        logger.info(f"  Skipped matches:   {skipped_matches}")
        logger.info(f"  Accuracy:          {accuracy:.2%} (95% CI: [{ci_lower:.2%}, {ci_upper:.2%}])")
        logger.info(f"  Mean Brier:        {avg_brier:.4f}")
            
        return final_df
