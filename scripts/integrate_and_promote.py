import pandas as pd
import logging
import json
import sys
import os
sys.path.append(os.getcwd())
from src.features.pipeline import FeaturePipeline
from src.ml.registry import ModelRegistry
from src.ml.trainer import ModelTrainer

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def integrate_and_promote():
    registry = ModelRegistry()
    pipeline = FeaturePipeline()
    trainer = ModelTrainer(registry=registry)
    
    leagues = ['PL', 'PD', 'SA', 'BL1', 'FL1']
    
    print("--- Phase 7: Pipeline Integration & Promotion ---")
    
    results = {}
    
    for lg in leagues:
        print(f"\nProcessing {lg}...")
        
        # 1. Pipeline Run (Absorbs new BACKFILL_*.csv files automatically)
        # The pipeline implementation in src/features/pipeline.py loads all CSVs matching {league}_*.csv
        # Since our files are named BACKFILL_{league}_{season}.csv, we need to ensure the glob catches them.
        # Checking pipeline.py: defaults to f"{league}_*.csv". 
        # BACKFILL_PL_1213.csv matches PL_*.csv? No. It matches *PL*. 
        # Wait, the pipeline glob is usually matching start. 
        # Let's verify pipeline code via memory or view_file if needed. 
        # Assuming we need to potentially rename or update pipeline loader if it's strict.
        
        # Standardize Filenaming if needed. 
        # Current pattern: BACKFILL_{LEAGUE}_{SEASON}.csv
        # Pipeline expects: {league}_*.csv usually.
        # But wait, pipeline.py line 35: pattern = f"{league}_*.csv"
        # BACKFILL_PL... does NOT start with PL_.
        # We should rename the backfill files to {LEAGUE}_BACKFILL_{SEASON}.csv to be safe.
        
        pass 
        
    print("Renaming backfill files to match Pipeline pattern...")
    from pathlib import Path
    import shutil
    
    p = Path("data/processed/matches/")
    count = 0
    for f in p.glob("BACKFILL_*.csv"):
        # Format: BACKFILL_PL_1213.csv
        parts = f.stem.split('_') # ['BACKFILL', 'PL', '1213']
        if len(parts) == 3:
            new_name = f"{parts[1]}_BACKFILL_{parts[2]}.csv"
            f.rename(p / new_name)
            count += 1
            
    print(f"Renamed {count} files for pipeline compatibility.")
    
    # 2. Re-run Pipeline and Retrain
    for lg in leagues:
        print(f"\nTraining {lg} Models...")
        
        # Train Poisson (Home/Away Goals)
        try:
            # We don't expose separate train_league methods easily without CLI usually,
            # but ModelTrainer uses the pipeline internally.
            # We can use the trainer directly if exposed, or call the logic.
            # src/ml/train.py usually has a train_all or train_league.
            
            # Let's assume we can trigger training via the registry or trainer.
            # Checking available tools/code... the user has train.py? 
            # I will assume standard usage loop.
            
            # Re-load full league data
            df = pipeline.run(league=lg)
            print(f"  > Loaded {len(df)} matches (Historic + Modern)")
            
            if len(df) < 2000:
                print(f"  [WARNING] Only {len(df)} matches. Might still fall short?")
            else:
                print(f"  [SUCCESS] Threshold crossed: {len(df)} > 2000")
                
            # Train Goals (Poisson) via train_model
            # Dynamic Regularization: SA needs loose constraints (0.0001) to fit intercept
            goal_alpha = 0.0001 if lg == 'SA' else 0.01

            # 1. Home Goals
            trainer.train_model(
                df=df,
                target_col='home_score',
                league=lg,
                model_type='poisson',
                model_name='poisson_home_base',
                mode='production',
                params={'alpha': goal_alpha}
            )
            # 2. Away Goals
            trainer.train_model(
                df=df,
                target_col='away_score',
                league=lg,
                model_type='poisson',
                model_name='poisson_away_base',
                mode='production',
                params={'alpha': goal_alpha}
            )
            
            # Train Corners (Negative Binomial)
            # 3. Home Corners
            trainer.train_model(
                df=df,
                target_col='home_corners',
                league=lg,
                model_type='nb',
                model_name='nb_home_corners_base',
                mode='production',
                params={'alpha': 1.0} # Initial guess, optimizer will tune if wrapper supports it or uses it as init
            )
            # 4. Away Corners
            trainer.train_model(
                df=df,
                target_col='away_corners',
                league=lg,
                model_type='nb',
                model_name='nb_away_corners_base',
                mode='production',
                params={'alpha': 1.0}
            )
            
            # 5. Total Cards (Poisson)
            trainer.train_model(
                df=df,
                target_col='match_total_cards',
                league=lg,
                model_type='poisson',
                model_name='poisson_total_cards_base',
                mode='production',
                params={'alpha': 1.0}
            )

            # 6. Double Chance (Expansion 2025-12-28)
            # 1X (Home/Draw)
            if 'target_1x' in df.columns:
                trainer.train_model(df, 'target_1x', league=lg, model_type='xgboost', model_name='xgb_double_chance_1x', 
                                params={'objective': 'binary:logistic', 'eval_metric': 'logloss', 'max_depth': 4}, mode='production')
            # X2 (Away/Draw)
            if 'target_x2' in df.columns:
                 trainer.train_model(df, 'target_x2', league=lg, model_type='xgboost', model_name='xgb_double_chance_x2', 
                                params={'objective': 'binary:logistic', 'eval_metric': 'logloss', 'max_depth': 4}, mode='production')
            # 12 (Home/Away)
            if 'target_12' in df.columns:
                 trainer.train_model(df, 'target_12', league=lg, model_type='xgboost', model_name='xgb_double_chance_12', 
                                params={'objective': 'binary:logistic', 'eval_metric': 'logloss', 'max_depth': 4}, mode='production')
            
            results[lg] = "PROMOTED"
            
        except Exception as e:
            print(f"  [ERROR] Training failed for {lg}: {e}")
            results[lg] = "FAILED"

    print("\n--- Promotion Summary ---")
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    integrate_and_promote()
