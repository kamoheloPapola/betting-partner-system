"""Investigate corner prediction for JUVENTUS vs CREMONESE"""
import sys
import os
sys.path.insert(0, os.getcwd())

import pandas as pd
from src.features.pipeline import FeaturePipeline

# Get SA data
p = FeaturePipeline()
df = p.run(league='SA')

# Find JUVENTUS vs CREMONESE H2H
juve_crem = df[(df['home_team'].str.contains('JUVENTUS', na=False) & df['away_team'].str.contains('CREMONESE', na=False)) |
               (df['home_team'].str.contains('CREMONESE', na=False) & df['away_team'].str.contains('JUVENTUS', na=False))]

print('=== JUVENTUS vs CREMONESE Corner History ===')
if 'home_corners' in df.columns:
    for _, row in juve_crem.iterrows():
        home = row['home_team']
        away = row['away_team']
        hc = row.get('home_corners', 'N/A')
        ac = row.get('away_corners', 'N/A')
        total = (hc + ac) if pd.notna(hc) and pd.notna(ac) else 'N/A'
        date = row['date']
        print(f"  {date}: {home} vs {away} - H:{hc} A:{ac} Total:{total}")
    
    if len(juve_crem) > 0:
        avg_total = (juve_crem['home_corners'] + juve_crem['away_corners']).mean()
        u115_rate = ((juve_crem['home_corners'] + juve_crem['away_corners']) < 11.5).mean()
        print(f"\n  H2H Avg Total Corners: {avg_total:.1f}")
        print(f"  H2H U11.5 Rate: {u115_rate*100:.1f}%")
else:
    print('No corner columns found')

# Check JUVENTUS overall corner stats (home)
print("\n=== JUVENTUS Overall Corner Stats (Home) ===")
juve_home = df[df['home_team'].str.contains('JUVENTUS', na=False)]
if 'home_corners' in juve_home.columns:
    juve_corners = juve_home['home_corners'].dropna()
    print(f"  Avg home corners for JUVE: {juve_corners.mean():.2f}")
    print(f"  Matches: {len(juve_corners)}")

# Check CREMONESE overall stats (away)
print("\n=== CREMONESE Overall Corner Stats (Away) ===")
crem_away = df[df['away_team'].str.contains('CREMONESE', na=False)]
if 'away_corners' in crem_away.columns:
    crem_corners = crem_away['away_corners'].dropna()
    print(f"  Avg away corners for CREM: {crem_corners.mean():.2f}")
    print(f"  Matches: {len(crem_corners)}")

# Total for both teams
print("\n=== COMBINED EXPECTATION ===")
if 'home_corners' in juve_home.columns and 'away_corners' in crem_away.columns:
    juve_avg = juve_home['home_corners'].mean()
    crem_avg = crem_away['away_corners'].mean()
    expected_total = juve_avg + crem_avg
    print(f"  JUVE home avg: {juve_avg:.2f}")
    print(f"  CREM away avg: {crem_avg:.2f}")
    print(f"  Expected total: {expected_total:.2f}")
    
    # What does this mean for U11.5?
    from src.ml.distributions import NegativeBinomialEngine
    engine = NegativeBinomialEngine()
    
    # Assume typical variance
    v_h = juve_avg * 1.3
    v_a = crem_avg * 1.3
    
    res = engine.calculate_probabilities(juve_avg, v_h, crem_avg, v_a)
    print(f"\n  Model U11.5 probability: {res['corners_under_11_5']*100:.1f}%")
