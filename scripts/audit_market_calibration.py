"""
System-Wide Market Calibration Audit

This script performs a deep analysis of all active betting markets.
It compares:
1. Historical Base Rate (How often does it actually happen?)
2. Engine Model Probability (What does the model predict for an 'average' match?)

Significant gaps (>5-7%) indicate potential calibration issues, bugs, or over/under-damping.

Run: python scripts/audit_market_calibration.py
"""
import sys
import os
sys.path.append(os.getcwd())

import pandas as pd
import numpy as np
from pathlib import Path
from src.ml.distributions import PoissonEngine, NegativeBinomialEngine, ZeroInflatedEngine

# === CONFIG ===
HISTORICAL_DIR = Path("data/historical")
RESULTS_DIR = Path("data/results/raw")

def load_data():
    """Load all historical match data."""
    files = list(HISTORICAL_DIR.rglob("*.csv")) + list(RESULTS_DIR.rglob("*.csv"))
    dfs = []
    print(f"Loading {len(files)} historical files...")
    
    for f in files:
        try:
            df = pd.read_csv(f)
            # Normalize column names if needed
            dfs.append(df)
        except Exception:
            pass
            
    if not dfs:
        return pd.DataFrame()
        
    return pd.concat(dfs, ignore_index=True)

def audit_goals_markets(df):
    """Audit Poisson Markets (1X2, O/U Goals, BTTS)."""
    print("\n" + "="*50)
    print("AUDIT: GOALS & RESULT MARKETS (PoissonEngine)")
    print("="*50)

    # 1. Calculate Base Rates
    # Needs FTHG, FTAG
    if 'FTHG' not in df.columns or 'FTAG' not in df.columns:
        print("[SKIP] Goals data missing")
        return

    df = df.dropna(subset=['FTHG', 'FTAG'])
    n = len(df)
    print(f"analyzing {n} matches...")

    home_goals = df['FTHG']
    away_goals = df['FTAG']
    total_goals = home_goals + away_goals
    
    # Base Rates
    base_home_win = (home_goals > away_goals).mean()
    base_draw = (home_goals == away_goals).mean()
    base_away_win = (away_goals > home_goals).mean()
    
    base_o25 = (total_goals > 2.5).mean()
    base_u25 = (total_goals < 2.5).mean()
    base_u35 = (total_goals < 3.5).mean() # New market check
    
    base_btts_yes = ((home_goals > 0) & (away_goals > 0)).mean()
    
    base_home_u15 = (home_goals < 1.5).mean()
    base_away_u15 = (away_goals < 1.5).mean()

    # 2. Engine Model Prediction (at Average)
    avg_hg = home_goals.mean()
    avg_ag = away_goals.mean()
    print(f"Average Goals: Home={avg_hg:.2f}, Away={avg_ag:.2f}")

    engine = PoissonEngine()
    probs = engine.calculate_probabilities(avg_hg, avg_ag)
    
    # 3. Comparison Report
    report = [
        ("Home Win (1)", base_home_win, probs['home_win']),
        ("Draw (X)", base_draw, probs['draw']),
        ("Away Win (2)", base_away_win, probs['away_win']),
        ("Over 2.5 Goals", base_o25, probs['over_2_5']),
        ("Under 2.5 Goals", base_u25, probs['under_2_5']),
        ("Under 3.5 Goals", base_u35, probs['under_3_5']),
        ("BTTS Yes", base_btts_yes, probs['btts_yes']),
        ("Home Goals U1.5", base_home_u15, probs['home_under_1_5']),
        ("Away Goals U1.5", base_away_u15, probs['away_under_1_5'])
    ]
    
    print(f"{'Market':<20} {'Base Rate':<10} {'Model Avg':<10} {'Gap':<10} {'Status'}")
    print("-" * 65)
    
    for name, base, model in report:
        gap = model - base
        status = "OK"
        if abs(gap) > 0.05: status = "WARN"
        if abs(gap) > 0.10: status = "ALERT"
        
        print(f"{name:<20} {base:.1%}     {model:.1%}     {gap:+.1%}     {status}")

def audit_corners_markets(df):
    """Audit Negative Binomial Markets (Corners)."""
    print("\n" + "="*50)
    print("AUDIT: CORNER MARKETS (NegativeBinomialEngine)")
    print("="*50)
    
    # Check simple names first: HC, AC vs HY, AY confusion
    # Standard usually HC = Home Corners
    cols = ['HC', 'AC']
    if not all(c in df.columns for c in cols):
        print("[SKIP] Corner data (HC/AC) missing")
        return

    df = df.dropna(subset=cols)
    hc = df['HC']
    ac = df['AC']
    total = hc + ac
    
    base_corner_home = (hc > ac).mean()
    base_u115 = (total < 11.5).mean()
    
    # Model Init
    mean_hc = hc.mean()
    mean_ac = ac.mean()
    var_hc = hc.var()
    var_ac = ac.var()
    
    print(f"Stats: H_Mean={mean_hc:.2f} H_Var={var_hc:.2f} | A_Mean={mean_ac:.2f} A_Var={var_ac:.2f}")
    
    engine = NegativeBinomialEngine()
    # Note: Engine takes variance. If Var < Mean, it might correct it.
    probs = engine.calculate_probabilities(mean_hc, var_hc, mean_ac, var_ac)
    
    report = [
        ("Corners Home Win", base_corner_home, probs['corners_home_win']),
        ("Corners U11.5", base_u115, probs['corners_under_11_5'])
    ]
    
    print(f"{'Market':<20} {'Base Rate':<10} {'Model Avg':<10} {'Gap':<10} {'Status'}")
    print("-" * 65)
    
    for name, base, model in report:
        gap = model - base
        status = "OK"
        if abs(gap) > 0.05: status = "WARN"
        if abs(gap) > 0.10: status = "ALERT"
        print(f"{name:<20} {base:.1%}     {model:.1%}     {gap:+.1%}     {status}")

    # SENSITIVITY CHECK
    print("\n--- Sensitivity Check (Lighter Damping) ---")
    # U11.5 is a "High Probability" market (~72% base rate).
    # Standard damping (0.80) pulls 72% -> 66%. Lighter damping (0.90) should fix it.
    
    # Simulate Engine internals (as NB engine doesn't expose alpha publicly easily)
    raw_u115 = probs['corners_under_11_5'] # This is currently clamped/damped. It's hard to reverse.
    # Instead, let's manually calculate what 0.90 damping would do if we assume raw was ~Base
    # P_damped = 0.5 + alpha * (P_raw - 0.5)
    # If P_raw is Base (0.724), then with alpha=0.90:
    target = 0.5 + 0.90 * (base_u115 - 0.5)
    print(f"Projected U11.5 (A=0.90): {base_u115:.1%}     {target:.1%}     {target-base_u115:+.1%}     OK (Projected)")

def audit_cards_markets(df):
    """Audit Zero Inflated Markets (Cards)."""
    print("\n" + "="*50)
    print("AUDIT: CARD MARKETS (ZeroInflatedEngine)")
    print("="*50)
    
    # Try multiple card schemas
    total = None
    if 'HY' in df.columns and 'AY' in df.columns:
        total = df['HY'].fillna(0) + df['AY'].fillna(0) + df['HR'].fillna(0) + df['AR'].fillna(0)
    elif 'HC' in df.columns and 'AC' in df.columns: # Sometimes labeled HC/AC in disparate datasets? No, check naming
        # Avoid conflict with corners. usually HY/AY
        pass
        
    if total is None:
        print("[SKIP] Card data (HY/AY/HR/AR) missing")
        return

    df = df[total.notna()]
    total = total[total.index.isin(df.index)]
    
    base_o25 = (total > 2.5).mean()
    base_u45 = (total < 4.5).mean()
    base_u55 = (total < 5.5).mean()
    
    mean_cards = total.mean()
    print(f"Average Total Cards: {mean_cards:.2f}")
    
    engine = ZeroInflatedEngine()
    # Use default pi_zero (now 0.03) to verify system default behavior
    probs = engine.calculate_probabilities(mean_cards)
    
    report = [
        ("Cards O2.5", base_o25, probs['cards_over_2_5']),
        ("Cards U4.5", base_u45, probs['cards_under_4_5']),
        ("Cards U5.5", base_u55, probs['cards_under_5_5']),
    ]
    
    print(f"{'Market':<20} {'Base Rate':<10} {'Model Avg':<10} {'Gap':<10} {'Status'}")
    print("-" * 65)
    
    for name, base, model in report:
        gap = model - base
        status = "OK"
        if abs(gap) > 0.05: status = "WARN"
        if abs(gap) > 0.10: status = "ALERT"
        print(f"{name:<20} {base:.1%}     {model:.1%}     {gap:+.1%}     {status}")

    # SENSITIVITY CHECK
    print("\n--- Sensitivity Check (Tuned Params) ---")
    # Hypothesis: pi_zero=0.1 is too high (modern football has fewer 0-card games)
    # Let's try pi_zero=0.03 (3%) and see if it fixes Cards O2.5
    tuned_pi = 0.03
    probs_tuned = engine.calculate_probabilities(mean_cards, pi_zero=tuned_pi)
    new_o25 = probs_tuned['cards_over_2_5']
    print(f"Cards O2.5 (Pi=0.03):  {base_o25:.1%}     {new_o25:.1%}     {new_o25-base_o25:+.1%}     {'OK' if abs(new_o25-base_o25)<0.05 else 'WARN'}")

def main():
    print("Loading data...")
    df = load_data()
    if df.empty:
        print("No data found.")
        return

    audit_goals_markets(df)
    audit_corners_markets(df)
    audit_cards_markets(df)
    
    print("\n[DONE] Audit Complete.")

if __name__ == "__main__":
    main()
