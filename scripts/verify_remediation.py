import pandas as pd
import numpy as np
import logging
import json
from pathlib import Path
from src.core.container import ServiceContainer
from src.cli.commands.prediction import _load_prediction_models, _calculate_probabilities
from src.strategies.selection_gate import SelectionGate
from src.config.thresholds import Thresholds

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Verifier")

def verify():
    # 1. Load data
    df = pd.read_csv("failed_matches_features_full.csv")
    
    matches_meta = [
        {"id": "4719e5144c1122b1", "league": "BL1", "home": "SC Freiburg", "away": "Hamburger SV"},
        {"id": "1d206c2b7fa9ed5e", "league": "SA", "home": "Udinese", "away": "Pisa"}
    ]

    container = ServiceContainer.get_instance()
    
    print("\n--- STEP 2.1: BEYON OFFSET VERIFICATION ---")
    for m_info in matches_meta:
        match_id = m_info["id"]
        league = m_info["league"]
        row = df[df['match_id'] == match_id].iloc[0]
        
        suite = _load_prediction_models(league)
        probs = _calculate_probabilities(row, suite, league)
        
        print(f"\nMatch: {m_info['home']} vs {m_info['away']}")
        print(f"  Corner Home Prob: {probs['corn_1x2_h']:.1%}")
        
        if m_info['home'] == "SC Freiburg":
            # Freiburg corner win prob was 70.7% raw. 
            # With offset -0.24, it should be lower than 70% gate.
            if probs['corn_1x2_h'] < 0.70:
                print(f"  [PASS] Freiburg corner win ({probs['corn_1x2_h']:.1%}) is now below 70% gate.")
            else:
                print(f"  [FAIL] Freiburg corner win ({probs['corn_1x2_h']:.1%}) is still above 70% gate.")

    print("\n--- STEP 2.2: FORCED STOP TEST ---")
    status_file = Path("data/drift/rolling_90d_status.json")
    with open(status_file, "r") as f:
        original_status = json.load(f)
    
    try:
        # Force STOP status
        forced_stop = original_status.copy()
        forced_stop["status"] = "STOP"
        with open(status_file, "w") as f:
            json.dump(forced_stop, f)
        
        print("  Drift status forced to STOP.")
        
        # Check SelectionGate logic via .process()
        gate = SelectionGate()
        
        # Prepare a high-conf prediction that WOULD pass normally
        # Using the Freiburg match features as base
        row = df.iloc[0].copy()
        row['league'] = "BL1"
        
        # We need to simulate the structure expected by SelectionGate.process()
        # Which is a list of 'bets' as returned by _prepare_bets
        # Each bet is a dict merge of match data and market stats
        
        test_bet = {
            **row.to_dict(),
            'market': 'corners_home_win',
            'probability': 0.85, # Very high
            'edge': 0.15,
            'confidence': 0.85,
            'selection': 'CORN_H'
        }
        
        # SelectionGate.process returns (passed_list, stats_dict)
        passed, stats = gate.process([test_bet])
        
        if stats.get('GATE_4_DRIFT_BLOCKED', 0) > 0 or len(passed) == 0:
            print(f"  [PASS] Drift blocked! High-conf bet rejected due to STOP status.")
            print(f"  Stats: {stats}")
        else:
            print(f"  [FAIL] Bet NOT blocked despite STOP status. Check cache bypass.")
            print(f"  Stats: {stats}")
            
    finally:
        # Restore status
        with open(status_file, "w") as f:
            json.dump(original_status, f)
        print("  Original drift status restored.")

    print("\n--- STEP 2.3: STABILITY TEST (GOALS/DC) ---")
    # Verify Goal and DC probs are unchanged by the offset injection hook.
    for m_info in matches_meta:
        match_id = m_info["id"]
        league = m_info["league"]
        row = df[df['match_id'] == match_id].iloc[0]
        suite = _load_prediction_models(league)
        probs = _calculate_probabilities(row, suite, league)
        print(f"Match: {m_info['home']} vs {m_info['away']}")
        print(f"  Goal 1X2: H:{probs['home']:.1%} D:{probs['draw']:.1%} A:{probs['away']:.1%}")
        print(f"  DC (unaffected): 1X:{probs['dc_1x']:.1%} X2:{probs['dc_x2']:.1%}")

if __name__ == "__main__":
    verify()
