import pandas as pd
import numpy as np
from src.ml.registry import ModelRegistry
from src.ml.trainer import ModelTrainer
from src.config import MODELS_DIR

def test_training_flow():
    # 1. create dummy data
    df = pd.DataFrame({
        'feature_1': np.random.rand(100),
        'feature_2': np.random.rand(100),
        'target': np.random.randint(0, 2, 100),
        'date': pd.date_range(start='2023-01-01', periods=100),
        'match_id': range(100),
        'home_team': 'A',
        'away_team': 'B',
        'source': 'test',
        'status': 'FINISHED',
        'competition': 'TEST',
        'season': 2023
    })
    
    # 2. Init components
    registry = ModelRegistry()
    trainer = ModelTrainer(registry)
    
    # 3. Train Model
    print("Training dummy XGBoost...")
    model, metadata = trainer.train_model(
        df=df,
        target_col='target',
        model_type='xgboost',
        model_name='test_xgb_model',
        params={'n_estimators': 10}
    )
    
    # 4. Verify presence
    manifest_path = MODELS_DIR / "manifest.json"
    assert manifest_path.exists(), "Manifest not created"
    
    model_path = MODELS_DIR / metadata['filename']
    assert model_path.exists(), f"Model file missing: {model_path}"
    
    print("MetaData:", metadata)
    print("✅ Training flow verified!")

if __name__ == "__main__":
    test_training_flow()
