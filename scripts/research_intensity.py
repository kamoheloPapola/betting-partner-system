import numpy as np
import pandas as pd
from src.ml.distributions import NegativeBinomialEngine, ZeroInflatedEngine

def simulate_intensity_impact():
    print("--- Phase 3: Match Intensity Research ---")
    
    # Base Case: Freiburg vs Hamburg (Corners)
    # Reconstructed raw mu's
    mu_h_raw = 6.278
    mu_a_raw = 3.632
    alpha_h = 0.04 # Estimating from earlier logs
    alpha_a = 0.05
    
    v_h_base = mu_h_raw + alpha_h * mu_h_raw**2
    v_a_base = mu_a_raw + alpha_a * mu_a_raw**2
    
    nb = NegativeBinomialEngine()
    res_base = nb.calculate_probabilities(mu_h_raw, v_h_base, mu_a_raw, v_a_base)
    
    print(f"\nBaseline (No Intensity Damping):")
    print(f"  Corner Home Win Prob: {res_base['corners_home_win']:.1%}")
    
    # Simulation: Cup Game Intensity (is_cup_game=True)
    # Rule: Multiply variance, not mean. Cap confidence.
    INTENSITY_VARIANCE_MULT = 1.5 # 50% increase in volatility
    
    v_h_high = v_h_base * INTENSITY_VARIANCE_MULT
    v_a_high = v_a_base * INTENSITY_VARIANCE_MULT
    
    res_high = nb.calculate_probabilities(mu_h_raw, v_h_high, mu_a_raw, v_a_high)
    
    print(f"\nSimulated High Intensity (Variance x{INTENSITY_VARIANCE_MULT}):")
    print(f"  Corner Home Win Prob: {res_high['corners_home_win']:.1%}")
    print(f"  Delta: {res_high['corners_home_win'] - res_base['corners_home_win']:.1%}")

    # Case 2: Cards U5.5
    mu_cards = 3.1 # Freiburg vs Hamburg reconstructed
    cards_eng = ZeroInflatedEngine()
    res_cards_base = cards_eng.calculate_probabilities(mu_cards, pi_zero=0.1)
    
    print(f"\nCards U5.5 Baseline: {res_cards_base['cards_under_5_5']:.1%}")
    
    # For Cards (Poisson/ZIP), "multiplying variance" means shifting the distribution?
    # In a Poisson, mean=variance. If we want more variance, we need a different dist (like NB)
    # or we can just shift the mu slightly for simulation purpose, but the user said "multiply variance".
    # ZIP variance is mu(1+pi_zero*mu).
    # If we increase volatility in cards, we expect shorter tails to get longer, hitting U5.5.
    
    print("\nConclusion: Increasing variance effectively 'flattens' the probability distribution,")
    print("pulling confidence away from the center (peaks) and toward the extremes/draws.")
    print("This naturally dampens high-confidence winner predictions without changing the 'expected' mu.")

if __name__ == "__main__":
    simulate_intensity_impact()
