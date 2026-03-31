"""
Investigate Cards U5.5 calibration warning (-5.8% gap)
"""
import sys
import os
sys.path.insert(0, os.getcwd())

import pandas as pd
import numpy as np
from pathlib import Path

# Load historical data using the pipeline
from src.features.pipeline import FeaturePipeline

print("Loading data via pipeline...")
dfs = []
for league in ['PL', 'PD', 'SA', 'BL1', 'FL1']:
    try:
        p = FeaturePipeline()
        df = p.run(league=league)
        dfs.append(df)
    except Exception as e:
        print(f"Failed to load {league}: {e}")

df = pd.concat(dfs, ignore_index=True)

# Calculate total cards per match
if 'home_yellow_cards' in df.columns and 'home_red_cards' in df.columns:
    df['home_cards'] = df['home_yellow_cards'].fillna(0) + df['home_red_cards'].fillna(0)
if 'away_yellow_cards' in df.columns and 'away_red_cards' in df.columns:
    df['away_cards'] = df['away_yellow_cards'].fillna(0) + df['away_red_cards'].fillna(0)
    
if 'match_total_cards' in df.columns:
    df['total_cards'] = df['match_total_cards']
elif 'home_cards' in df.columns and 'away_cards' in df.columns:
    df['total_cards'] = df['home_cards'] + df['away_cards']

if 'total_cards' in df.columns:
    df = df.dropna(subset=['total_cards'])
    
    print("=== CARDS U5.5 INVESTIGATION ===")
    print(f"Total matches with card data: {len(df)}")
    print()
    
    # Base rate for U5.5
    u55_rate = (df['total_cards'] < 5.5).mean()
    print(f"Cards U5.5 Base Rate: {u55_rate*100:.1f}%")
    
    # Distribution of cards
    print()
    print("Card Distribution:")
    for i in range(12):
        count = (df['total_cards'] == i).sum()
        pct = count / len(df) * 100
        cumulative = (df['total_cards'] <= i).mean() * 100
        marker = " <-- U5.5 threshold" if i == 5 else ""
        print(f"  {i} cards: {count:5d} ({pct:5.1f}%) | Cumulative: {cumulative:5.1f}%{marker}")
    
    # Average cards
    avg_cards = df['total_cards'].mean()
    print(f"\nAverage total cards: {avg_cards:.2f}")
    
    # What the ZeroInflatedEngine predicts
    from src.ml.distributions import ZeroInflatedEngine
    engine = ZeroInflatedEngine()
    
    print()
    print("=== MODEL SIMULATION ===")
    print("What does ZeroInflatedEngine predict for different mu values?")
    print()
    
    for mu in [3.5, 4.0, 4.25, 4.5, 5.0]:
        res = engine.calculate_probabilities(mu, pi_zero=0.03)
        u55 = res.get('cards_under_5_5', 0)
        print(f"  mu={mu:.2f}: U5.5={u55*100:.1f}% | Gap from base: {(u55 - u55_rate)*100:+.1f}pp")
    
    print()
    print("=== DIAGNOSIS ===")
    print(f"Base Rate U5.5: {u55_rate*100:.1f}%")
    print(f"Model Avg U5.5: {68.2:.1f}% (from audit)")
    print(f"Gap: {68.2 - u55_rate*100:.1f}pp")
    print()
    
    # Check if the issue is pi_zero
    print("=== SENSITIVITY: pi_zero ===")
    for pi in [0.01, 0.02, 0.03, 0.05, 0.08]:
        res = engine.calculate_probabilities(avg_cards, pi_zero=pi)
        u55 = res.get('cards_under_5_5', 0)
        print(f"  pi_zero={pi:.2f}: U5.5={u55*100:.1f}% (gap: {(u55 - u55_rate)*100:+.1f}pp)")
else:
    print("Card columns not found in data")
