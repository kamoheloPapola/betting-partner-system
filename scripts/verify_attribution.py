import pandas as pd
import json
from datetime import datetime
from pathlib import Path
from src.core.container import ServiceContainer
from src.cli.commands.prediction import _run_predict_loop
from src.monitoring.events import load_events

def verify_attribution():
    print("--- Phase 5: Attribution Stamp Verification ---")
    
    # 1. Load data for Jan 10 (Freiburg match)
    df = pd.read_csv("failed_matches_features_full.csv")
    
    # 2. Run prediction loop for these matches
    # This will trigger logging to today's events file
    print("Running predictions...")
    results = _run_predict_loop(df)
    
    # 3. Load today's events and check for the stamp
    today_str = datetime.now().strftime('%Y-%m-%d')
    print(f"Loading events for {today_str}...")
    events = load_events(today_str)
    
    prediction_events = [e for e in events if e.event_type == "prediction_generated"]
    
    if not prediction_events:
        print(" [FAIL] No prediction events found in logs!")
        return

    # Check the last event
    last_event = prediction_events[-1]
    metadata = last_event.metadata
    stamp = metadata.get('attribution_stamp')
    
    if stamp:
        print(" [PASS] Attribution stamp found in event metadata!")
        print(f"  Sample Stamp (Freiburg Corner): {json.dumps(stamp.get('corn_1x2_h_attribution'), indent=2)}")
        
        # Verify fields
        if all(k in stamp.get('corn_1x2_h_attribution', {}) for k in ['raw_prob', 'adj_prob', 'drift_state', 'variance_mult']):
             print(" [PASS] All mandatory attribution fields present.")
        else:
             print(" [FAIL] Missing mandatory fields in stamp.")
    else:
        print(" [FAIL] No attribution stamp found in event metadata.")

if __name__ == "__main__":
    # Ensure the failed_matches_features_full.csv exists or recreate if needed
    # (In previous steps it was deleted, but I should have it in memory or recreate it)
    # Actually, I deleted it in the cleanup. I need to recreate it or just use a dummy row.
    
    # Let's recreate a dummy row for Freiburg
    try:
        df = pd.read_csv("failed_matches_features_full.csv")
    except FileNotFoundError:
        print("Recreating dummy feature row...")
        df = pd.DataFrame([{
            'match_id': '4719e5144c1122b1',
            'home_team': 'SC Freiburg',
            'away_team': 'Hamburger SV',
            'league': 'BL1',
            'date': '2026-01-10',
            'kickoff_utc': '2026-01-10T15:30:00Z',
            # Add minimal features required by models
            'Rolling_Home_Corners': 6.0,
            'Rolling_Away_Corners': 4.0,
            'Rolling_Home_Cards': 1.0,
            'Rolling_Away_Cards': 2.0
        }])
        df.to_csv("failed_matches_features_full.csv", index=False)
    
    verify_attribution()
