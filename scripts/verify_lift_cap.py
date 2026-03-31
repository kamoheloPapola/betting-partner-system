"""
Verify H2H lift cap is applied correctly in actual predictions.
"""
import sys
import os
sys.path.insert(0, os.getcwd())

import pandas as pd
from src.features.pipeline import FeaturePipeline
from src.cli.commands.prediction import _load_prediction_models, _calculate_probabilities

# Check Hoffenheim vs Leverkusen and Stuttgart vs Bayern
matches_to_check = [
    ("HOFFENHEIM", "LEVERKUSEN", "BL1"),
    ("STUTTGART", "BAYERN", "BL1"),
    ("MAINZ", "HEIDENHEIM", "BL1"),
]

for home_part, away_part, league in matches_to_check:
    print(f"\n{'='*60}")
    print(f"Checking: {home_part} vs {away_part} ({league})")
    print('='*60)
    
    p = FeaturePipeline()
    df = p.run(league=league)
    
    match = df[
        (df['home_team'].str.contains(home_part, case=False)) & 
        (df['away_team'].str.contains(away_part, case=False))
    ]
    
    if match.empty:
        print("Match NOT found!")
        continue
    
    match = match.iloc[-1]
    h2h_count = match.get('h2h_match_count', 0)
    h2h_o25_rate = match.get('h2h_goals_o25_rate', 0)
    
    print(f"H2H Match Count: {h2h_count}")
    print(f"H2H Goals O2.5 Rate: {h2h_o25_rate*100:.1f}%" if h2h_o25_rate else "N/A")
    
    suite = _load_prediction_models(league)
    probs, attr = _calculate_probabilities(match, suite, league)
    
    print(f"\nFinal Probabilities:")
    print(f"  O2.5: {probs['o25']*100:.1f}%")
    print(f"  U2.5: {probs['u25']*100:.1f}%")
    print(f"  U3.5: {probs['u35']*100:.1f}%")
