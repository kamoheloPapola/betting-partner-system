
import sys
import os
sys.path.insert(0, os.getcwd())

import pandas as pd
from src.features.pipeline import FeaturePipeline
from src.cli.commands.prediction import _load_prediction_models, _calculate_probabilities

league = "BL1"
print(f"Investigating {league}...")

# 1. Load Data
p = FeaturePipeline()
df = p.run(league=league)

# 2. Find Mainz vs Heidenheim (01-13)
match = df[
    (df['home_team'].str.contains("Mainz", case=False)) & 
    (df['away_team'].str.contains("Heidenheim", case=False))
]

if match.empty:
    print("Match NOT found!")
    sys.exit(1)

match = match.iloc[-1]
print(f"\nMatch: {match['home_team']} vs {match['away_team']} ({match['date']})")
print(f"H2H Match Count: {match.get('h2h_match_count', 0)}")
print(f"H2H Goals O2.5 Rate: {match.get('h2h_goals_o25_rate', 'N/A')}")
print(f"H2H BTTS Rate: {match.get('h2h_btts_rate', 'N/A')}")

# 3. Predict
suite = _load_prediction_models(league)
probs, attr = _calculate_probabilities(match, suite, league)

print("\n--- Probabilities ---")
print(f"Under 2.5: {probs['u25']*100:.1f}%")
print(f"Over 2.5:  {probs['o25']*100:.1f}%")
print(f"Under 3.5: {probs['u35']*100:.1f}%")

if probs['u35'] < probs['u25']:
    print("\n[!] LOGIC ERROR DETECTED: Under 3.5 < Under 2.5")
else:
    print("\n[OK] Math is coherent.")
