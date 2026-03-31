"""Test H2H corner adjustment for JUVENTUS vs CREMONESE"""
import sys
import os
sys.path.insert(0, os.getcwd())

import pandas as pd
from src.features.pipeline import FeaturePipeline
from src.ml.distributions import NegativeBinomialEngine

# Get data with H2H features
p = FeaturePipeline()
df = p.run(league='SA', force_refresh=True)

# Get JUVENTUS vs CREMONESE Jan 12
juve_crem = df[(df['home_team'].str.contains('JUVENTUS', na=False)) & 
               (df['away_team'].str.contains('CREMONESE', na=False))]

print("=== JUVENTUS vs CREMONESE H2H Corner Features ===")
for _, row in juve_crem.iterrows():
    print(f"\nDate: {row['date']}")
    print(f"  h2h_match_count: {row.get('h2h_match_count', 'N/A')}")
    print(f"  h2h_avg_corners: {row.get('h2h_avg_corners', 'N/A')}")
    print(f"  h2h_corners_u115_rate: {row.get('h2h_corners_u115_rate', 'N/A')}")

# Get the Jan 12 match
jan12 = juve_crem[juve_crem['date'].astype(str).str.contains('2026-01-12')]
if len(jan12) > 0:
    row = jan12.iloc[0]
    h2h_count = row.get('h2h_match_count', 0)
    h2h_avg_corners = row.get('h2h_avg_corners')
    
    print(f"\n=== SIMULATION: H2H CORNER ADJUSTMENT ===")
    
    # Assume model predicts mu_h=7.8, mu_a=5.2 (giving total=13)
    mu_h_model = 7.8
    mu_a_model = 5.2
    mu_total_model = mu_h_model + mu_a_model
    
    print(f"Model expects: mu_h={mu_h_model}, mu_a={mu_a_model}, total={mu_total_model}")
    print(f"H2H avg corners: {h2h_avg_corners}")
    print(f"H2H match count: {h2h_count}")
    
    if h2h_count >= 2 and h2h_avg_corners is not None:
        h2h_weight = min(h2h_count / 5, 0.4)
        mu_total_blended = (1 - h2h_weight) * mu_total_model + h2h_weight * h2h_avg_corners
        scale = mu_total_blended / mu_total_model
        
        mu_h_new = mu_h_model * scale
        mu_a_new = mu_a_model * scale
        
        print(f"\nH2H weight: {h2h_weight:.2%}")
        print(f"Blended total: {mu_total_blended:.1f}")
        
        engine = NegativeBinomialEngine()
        
        # Before
        v_h = mu_h_model * 1.3
        v_a = mu_a_model * 1.3
        res_before = engine.calculate_probabilities(mu_h_model, v_h, mu_a_model, v_a)
        
        # After
        v_h_new = mu_h_new * 1.3
        v_a_new = mu_a_new * 1.3
        res_after = engine.calculate_probabilities(mu_h_new, v_h_new, mu_a_new, v_a_new)
        
        print(f"\nBEFORE H2H fix: Corners U11.5 = {res_before['corners_under_11_5']*100:.1f}%")
        print(f"AFTER H2H fix:  Corners U11.5 = {res_after['corners_under_11_5']*100:.1f}%")
        print(f"\nImprovement: +{(res_after['corners_under_11_5'] - res_before['corners_under_11_5'])*100:.1f} pp for U11.5")
