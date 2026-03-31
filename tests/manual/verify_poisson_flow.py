import pandas as pd
import numpy as np
from src.ml.registry import ModelRegistry
from src.ml.trainer import ModelTrainer
from src.ml.distributions import ProbabilityEngine

def test_poisson_flow():
    print("--- Testing Poisson Flow ---")
    
    # 1. Create Dummy Data (Simulating a scoring trend)
    # Features: Rolling goals (correlation with future goals)
    n_samples = 200
    df = pd.DataFrame({
        'home_rolling_goals': np.random.uniform(0.5, 3.0, n_samples),
        'away_rolling_conceded': np.random.uniform(0.5, 3.0, n_samples),
        'match_id': range(n_samples), 'date': pd.date_range('2023-01-01', periods=n_samples),
        'home_team': 'H', 'away_team': 'A', 'source': 'test', 'status': 'FINISHED', 'competition': 'X', 'season': 2023
    })
    
    # Target: Poisson distributed goals based on features
    # Lambda = (Home Attack + Away Defense) / 2 approx
    true_lambda = (df['home_rolling_goals'] + df['away_rolling_conceded']) / 2
    df['home_score'] = np.random.poisson(true_lambda)
    
    # 2. Train Home Goals Model
    registry = ModelRegistry()
    trainer = ModelTrainer(registry)
    
    print("Training Home Poisson Model...")
    model_home, meta_home = trainer.train_model(
        df=df, target_col='home_score', model_type='poisson', 
        model_name='home_goals_v1', params={'alpha': 0.1}
    )
    
    # 3. Predict Lambda for a test case
    test_case = df.iloc[-1:].copy()
    pred_lambda_home = model_home.predict(test_case)[0]
    
    # Assume we did the same for Away and got 1.2
    pred_lambda_away = 1.2 
    
    print(f"\nPredicted Lambdas -> Home: {pred_lambda_home:.2f}, Away: {pred_lambda_away:.2f}")
    
    # 4. Derive Probabilities
    engine = ProbabilityEngine()
    probs = engine.derive_markets(pred_lambda_home, pred_lambda_away)
    
    print("\nDerived Market Probabilities:")
    for k, v in probs.items():
        if "goals" in k:
            print(f"  {k}: {v:.2f}")
        else:
            print(f"  {k}: {v:.1%}")
        
    # Validation checks
    assert 0.99 < (probs['home_win'] + probs['draw'] + probs['away_win']) < 1.01, "1X2 probabilities sum != 1"
    assert 0.99 < (probs['btts_yes'] + probs['btts_no']) < 1.01, "BTTS probabilities sum != 1"
    
    print("\n✅ Poisson Flow Verified: consistent derived probabilities.")

if __name__ == "__main__":
    test_poisson_flow()
