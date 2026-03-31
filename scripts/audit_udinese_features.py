import pandas as pd
from src.core.container import ServiceContainer
from src.features.pipeline import FeaturePipeline

def audit_udinese_features():
    print("=== Udinese Feature Audit (Reconstruction) ===")
    
    # 1. Load data via pipeline (simulating show-predictions)
    pipeline = ServiceContainer.get_instance().pipeline
    df = pipeline.run(league='SA', force_refresh=True)
    
    # 2. Find Udinese vs Pisa
    match = df[df['home_team'].str.upper() == 'UDINESE'].iloc[-1:]
    if match.empty:
        print("Udinese match not found.")
        return
        
    m = match.iloc[0]
    print(f"Match: {m['home_team']} vs {m['away_team']} ({m['date']})")
    
    # 3. Check for Default Priors
    # StaticPriors.ROLLING_GOALS = 1.35
    # StaticPriors.ROLLING_CORNERS = 4.5
    
    goals_feat = 'home_rolling_goals_scored_5'
    corners_feat = 'home_rolling_corners_scored_5'
    
    g_val = m.get(goals_feat)
    c_val = m.get(corners_feat)
    
    print(f"\nFeature: {goals_feat} = {g_val}")
    print(f"Feature: {corners_feat} = {c_val}")
    
    is_default_g = abs(g_val - 1.35) < 1e-5
    is_default_c = abs(c_val - 4.5) < 1e-5
    
    if is_default_g and is_default_c:
        print("\n❌ CRITICAL: Features are using DEFAULT PRIORS.")
        print("This means the naming bug caused HISTORY LOSS for this match.")
    else:
        print("\n✅ Features are using HISTORICAL DATA.")
        print("The naming bug did not block history for this specific match.")

if __name__ == "__main__":
    audit_udinese_features()
