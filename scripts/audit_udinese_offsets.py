from src.ml.models.corners.team_offsets import TeamOffsetManager
import pandas as pd

def check_udinese_offsets():
    print("=== Udinese Offset Audit ===")
    manager = TeamOffsetManager()
    
    # Check what's in the manager's keys
    sample_keys = list(manager.offsets.keys())[:5]
    print(f"Sample Keys in Manager: {sample_keys}")
    
    # Try lookup with uppercase (simulating new policy)
    test_name = "UDINESE"
    league = "SA"
    
    offset = manager.get_offset(test_name, league)
    if offset:
        print(f"✅ Offset FOUND for '{test_name}': {offset.home_corner_bias}")
    else:
        print(f"❌ Offset MISSING for '{test_name}'")
        
        # Try with mixed case (historical)
        mixed_name = "Udinese"
        offset_m = manager.get_offset(mixed_name, league)
        if offset_m:
            print(f"  (However, it EXISTS as '{mixed_name}')")
            print("  This confirms the CASE-SENSITIVITY BUG is causing offset bypass.")
        else:
            print(f"  (Also missing as '{mixed_name}'. Maybe match count < 200?)")

if __name__ == "__main__":
    check_udinese_offsets()
