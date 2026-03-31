"""Check what corner mu gives 37% U11.5"""
import sys
import os
sys.path.insert(0, os.getcwd())

from src.ml.distributions import NegativeBinomialEngine
engine = NegativeBinomialEngine()

print("What total expected corners gives 37% U11.5?")
print("=" * 50)

for mu_total in range(8, 18):
    mu_h = mu_total * 0.6  # Typical home/away split
    mu_a = mu_total * 0.4
    v_h = mu_h * 1.3
    v_a = mu_a * 1.3
    res = engine.calculate_probabilities(mu_h, v_h, mu_a, v_a)
    marker = " <-- 37%" if abs(res["corners_under_11_5"] - 0.37) < 0.03 else ""
    print(f"Total mu={mu_total}: U11.5={res['corners_under_11_5']*100:.1f}%{marker}")

print()
print("If model shows 37% U11.5, it expects ~15 total corners!")
print("But H2H average is only 7.7 corners (75% U11.5 rate)")
