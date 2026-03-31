"""Quick check of column names in the data."""
import sys
sys.path.insert(0, '.')

from src.features.pipeline import FeaturePipeline

p = FeaturePipeline()
df = p.run(league='PL')

# Find goal-related columns
goal_cols = [c for c in df.columns if 'goal' in c.lower() or 'fthg' in c.lower() or 'ftag' in c.lower() or 'score' in c.lower()]
print("Goal columns:", goal_cols)

# Find BTTS-related columns  
btts_cols = [c for c in df.columns if 'bt' in c.lower()]
print("BTTS columns:", btts_cols)

# Print first few rows of relevant columns
print("\nSample data:")
print(df[['home_team', 'away_team', 'home_score', 'away_score']].head(3) if 'home_score' in df.columns else "home_score not found")
