
import sys
import os
sys.path.insert(0, os.getcwd())
from src.ml.distributions import ZeroInflatedEngine
import numpy as np
from scipy.stats import poisson

engine = ZeroInflatedEngine()
mu = 4.25
pi = 0.03

x = np.arange(16)
poi_pmf = poisson.pmf(x, mu)
probs = (1 - pi) * poi_pmf
probs[0] += pi
raw_u55 = np.sum(probs[:6])
res = engine.calculate_probabilities(mu, pi)

print(f"mu={mu}, pi={pi}")
print(f"Raw U5.5: {raw_u55:.2%}")
print(f"Calculated U5.5 (after damping & clamp): {res['cards_under_5_5']:.2%}")

# Test with various mu
print("\nSensitivity to mu:")
for test_mu in [3.8, 4.0, 4.2, 4.4, 4.6]:
    r = engine.calculate_probabilities(test_mu, pi)
    print(f"  mu={test_mu:.1f}: U5.5={r['cards_under_5_5']:.2%}")

# Test with various damping alpha
print("\nSensitivity to damping alpha for U5.5:")
raw_p = raw_u55
for alpha in [0.70, 0.72, 0.75, 0.80, 0.85, 0.90]:
    dampened = 0.5 + alpha * (raw_p - 0.5)
    clamped = min(dampened, 0.82)
    print(f"  alpha={alpha:.2f}: {clamped:.2%}")
