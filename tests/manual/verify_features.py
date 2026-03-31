import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.features.pipeline import FeaturePipeline

def test_pipeline_logic():
    # 1. Create Dummy Data
    base_date = datetime(2023, 8, 1)
    
    # Matches with clear stats to Verify logic
    matches = [
        # Match 1: TeamA vs TeamB
        {
            "match_id": "m1", "date": base_date, "home_team": "TeamA", "away_team": "TeamB",
            "home_score": 1, "away_score": 0, "status": "FINISHED", "competition": "PL", "season": 2023, "source": "dummy"
        },
        # Match 2: TeamA vs TeamC
        {
            "match_id": "m2", "date": base_date + timedelta(days=7), "home_team": "TeamA", "away_team": "TeamC",
            "home_score": 3, "away_score": 1, "status": "FINISHED", "competition": "PL", "season": 2023, "source": "dummy"
        },
        # Match 3: TeamB vs TeamA
        {
            "match_id": "m3", "date": base_date + timedelta(days=14), "home_team": "TeamB", "away_team": "TeamA",
            "home_score": 2, "away_score": 0, "status": "FINISHED", "competition": "PL", "season": 2023, "source": "dummy"
        }
    ]
    
    pipeline = FeaturePipeline()
    df = pipeline.run(matches)
    
    # Debug info
    print("Columns:", df.columns.tolist())
    
    # --- CRITICAL CHECKS ---
    
    # 1. Check for Away Features
    assert "away_rolling_goals_scored_5" in df.columns, "Missing away_rolling_goals_scored_5"
    assert "away_rolling_goals_conceded_3" in df.columns, "Missing away_rolling_goals_conceded_3"
    
    # 2. Check for Differentials
    assert "rolling_goals_scored_5_diff" in df.columns, "Missing rolling_goals_scored_5_diff"
    
    # 3. Check NaN Handling
    # M1 should have filled values, NOT NaNs (due to imputation)
    # With min_periods=1, shift(1) is NaN for first match. Pipeline imputes it.
    m1 = df[df['match_id'] == 'm1'].iloc[0]
    print(f"M1 Home Rolling Goals (Imputed): {m1['home_rolling_goals_scored_5']}")
    assert not pd.isna(m1['home_rolling_goals_scored_5']), "M1 NaN check failed (should be imputed)"
    
    # Sanity Check Script portion
    nan_count = df["home_rolling_goals_scored_5"].isna().sum() 
    assert nan_count < len(df) * 0.1, f"Too many NaNs: {nan_count}"
    
    diff_dtype = df["rolling_goals_scored_5_diff"].dtype
    assert np.issubdtype(diff_dtype, np.number), f"Diff column not numeric: {diff_dtype}"
    
    print("\n✅ All assertions passed!")

if __name__ == "__main__":
    test_pipeline_logic()
