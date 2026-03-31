"""
Simulation: H2H Adjustment for JUVENTUS vs CREMONESE
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from src.ml.distributions import ZeroInflatedEngine

engine = ZeroInflatedEngine()

# BEFORE FIX: Model predicted mu ~4.1 (giving 72%)
mu_model = 4.1

# H2H data for JUVENTUS vs CREMONESE (2026-01-12)
h2h_avg_cards = 3.67  # (8+2+1)/3 = 3.67 avg from prior 3 meetings  
h2h_match_count = 3

# H2H weight: min(3/5, 0.4) = 0.4 (40% weight)
h2h_weight = min(h2h_match_count / 5, 0.4)

# Blended mu
mu_blended = (1 - h2h_weight) * mu_model + h2h_weight * h2h_avg_cards

print("=== H2H ADJUSTMENT SIMULATION ===")
print(f"Model predicted mu: {mu_model:.2f}")
print(f"H2H average cards: {h2h_avg_cards:.2f}")
print(f"H2H weight: {h2h_weight:.2%}")
print(f"Blended mu: {mu_blended:.2f}")
print()

# Calculate probabilities
res_before = engine.calculate_probabilities(mu_model, pi_zero=0.03)
res_after = engine.calculate_probabilities(mu_blended, pi_zero=0.03)

print(f"BEFORE FIX: Cards O2.5 = {res_before['cards_over_2_5']*100:.1f}%")
print(f"AFTER FIX:  Cards O2.5 = {res_after['cards_over_2_5']*100:.1f}%")
print()
print(f"Improvement: {(res_before['cards_over_2_5'] - res_after['cards_over_2_5'])*100:.1f} percentage points lower probability")
print()
print("Actual outcome: 1 card (O2.5 was WRONG)")
print("The lower prediction is more aligned with the H2H history (25% O2.5 rate)")
