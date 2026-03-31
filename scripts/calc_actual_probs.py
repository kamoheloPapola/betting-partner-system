"""Calculate probability for actual mu values"""
import sys
import os
sys.path.insert(0, os.getcwd())

from src.ml.distributions import ZeroInflatedEngine
engine = ZeroInflatedEngine()

# Actual model values from debug output
mu_before = 4.55  # Model raw prediction
mu_after = 4.20   # After H2H adjustment

res_before = engine.calculate_probabilities(mu_before, pi_zero=0.03)
res_after = engine.calculate_probabilities(mu_after, pi_zero=0.03)

print("=== ACTUAL PROBABILITIES ===")
print(f"mu BEFORE H2H: {mu_before} => Cards O2.5 = {res_before['cards_over_2_5']*100:.1f}%")
print(f"mu AFTER H2H:  {mu_after} => Cards O2.5 = {res_after['cards_over_2_5']*100:.1f}%")
print()
print(f"Improvement: {(res_before['cards_over_2_5'] - res_after['cards_over_2_5'])*100:.1f} percentage points")
