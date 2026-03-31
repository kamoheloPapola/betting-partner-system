import pandas as pd
import logging
from src.features.pipeline import FeaturePipeline
from src.ml.trainer import ModelTrainer
from src.ml.registry import ModelRegistry

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_all_cards_models():
    pipeline = FeaturePipeline()
    registry = ModelRegistry()
    trainer = ModelTrainer(registry)
    
    # 1. Load data
    logger.info("Loading master features...")
    df = pipeline.run()
    
    leagues = ['PL', 'PD', 'SA', 'BL1', 'FL1']
    
    # Features (Minimal & Safe)
    cards_features = [
        'home_rolling_cards_total_5', # Actually, in re-assembled match-view, it's home_rolling_cards_scored_5?
        'away_rolling_cards_total_5', # Let's check the column name from a sample
        'home_days_rest',
        'away_days_rest'
    ]
    
    # Wait, let's verify column names first. 
    # Usually it's home_rolling_cards_scored_5 and away_rolling_cards_scored_5
    
    available_cols = df.columns.tolist()
    home_feat = 'home_rolling_cards_scored_5' if 'home_rolling_cards_scored_5' in available_cols else 'home_rolling_cards_total_5'
    away_feat = 'away_rolling_cards_scored_5' if 'away_rolling_cards_scored_5' in available_cols else 'away_rolling_cards_total_5'
    
    final_features = [home_feat, away_feat, 'home_days_rest', 'away_days_rest']
    logger.info(f"Using features: {final_features}")

    for lg in leagues:
        logger.info(f">>> STARTING LEAGUE: {lg}")
        lg_df = df[df['league'] == lg]
        
        if lg_df.empty:
            logger.warning(f"No data for {lg}")
            continue
            
        # Training Home Cards Model
        trainer.train_model(
            df=lg_df,
            target_col='home_total_cards',
            league=lg,
            model_type='nb', # Negative Binomial as requested
            model_name='nb_cards_base',
            features=final_features,
            params={'alpha': 0.1},
            mode='production'
        )
        logger.info(f"   [OK] Home Cards for {lg}")
        
        # Training Away Cards Model
        trainer.train_model(
            df=lg_df,
            target_col='away_total_cards',
            league=lg,
            model_type='nb',
            model_name='nb_cards_base_away',
            features=final_features,
            params={'alpha': 0.1},
            mode='production'
        )
        logger.info(f"   [OK] Away Cards for {lg}")
        logger.info(f"<<< FINISHED LEAGUE: {lg}")

if __name__ == "__main__":
    train_all_cards_models()
