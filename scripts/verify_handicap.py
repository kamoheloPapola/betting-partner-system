"""
Verification Script for European Handicap Logic

Checks:
1. Sanity: EH+2 prob > Double Chance (X2) prob
2. Monotonicity: EH+2 prob > EH+1 prob
3. Guardrails: Rejects close matches (diff < 0.6)
4. Ceiling: Rejects probs > 0.85
"""
import sys
import os
sys.path.append(os.getcwd())

from src.ml.handicap import EuropeanHandicap
from src.ml.distributions import PoissonEngine

def verify_sanity():
    print("--- Sanity Checks ---")
    
    # Scenario: Strong Home Favorite implies Away is Underdog
    # Home: 2.2 goals, Away: 0.8 goals (Diff = 1.4, Valid)
    lam_h = 2.2
    lam_a = 0.8
    
    eh = EuropeanHandicap()
    pe = PoissonEngine()
    
    # 1. Calculate EH +2 (Away)
    p_eh2 = eh.win_probability(lam_h, lam_a, 2, "away")
    
    # 2. Calculate Double Chance (X2)
    # DC = Draw + Away Win
    probs = pe.calculate_probabilities(lam_h, lam_a)
    p_dc = probs['draw'] + probs['away_win']
    
    print(f"Scenario: Home(2.2) vs Away(0.8)")
    print(f"EH (+2) Away:  {p_eh2:.1%}")
    print(f"Double Chance: {p_dc:.1%}")
    
    if p_eh2 > p_dc:
        print("[PASS] EH(+2) > Double Chance")
    else:
        print("[FAIL] EH(+2) is NOT greater than Double Chance!")
        
    # 3. Monotonicity
    # Calculate EH +1 (Manually bypassing guardrail if needed, but 1.4 diff is fine)
    # Actually wait, EH+1 is not implemented in the class, the class takes 'handicap' arg.
    p_eh1 = eh.win_probability(lam_h, lam_a, 1, "away")
    
    print(f"EH (+1) Away:  {p_eh1:.1%}")
    
    if p_eh2 > p_eh1:
        print("[PASS] EH(+2) > EH(+1)")
    else:
        print("[FAIL] Monotonicity broken")

def verify_guardrails():
    print("\n--- Guardrail Checks ---")
    eh = EuropeanHandicap()
    
    # 1. Close Match (Diff < 0.6)
    # Home 1.5, Away 1.2 (Diff 0.3)
    p_close = eh.win_probability(1.5, 1.2, 2, "away")
    print(f"Close Match (1.5 vs 1.2): {p_close} (Expected 0.0)")
    if p_close == 0.0:
        print("[PASS] Close match rejected")
    else:
        print(f"[FAIL] Close match accepted: {p_close}")
        
    # 2. Ceiling check
    # Extreme underdog: Home 3.5, Away 0.1? No, we want EH to be GUARANTEED.
    # If Away is Favorite: Away 2.5, Home 0.5. 
    # Bet on Away + 2? No, bet on Underdog.
    # Let's say Home is underdog (0.5) vs Away (0.8). Diff < 0.6 check covers this.
    # To hit ceiling, we need a valid diff (>=0.6) but result > 0.85
    # Home 2.0 vs Away 1.2 (Diff 0.8). Away + 2 should be very likely.
    p_ceil = eh.win_probability(2.0, 1.2, 2, "away")
    print(f"Ceiling Test (2.0 vs 1.2) EH+2: {p_ceil}")
    
    # Is 2.0 vs 1.2 actually > 85% for Away+2?
    # Mean score 2-1. Away+2 = 2-3. Away wins.
    # It should be high.
    if p_ceil == 0.0:
        print("[PASS] Ceiling Hit (Rejected > 0.85)")
    elif p_ceil > 0.85:
        print(f"[FAIL] Ceiling Breached: {p_ceil}")
    else:
        print(f"[INFO] Probability {p_ceil:.1%} within limits")

if __name__ == "__main__":
    verify_sanity()
    verify_guardrails()
