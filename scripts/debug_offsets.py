from src.ml.models.corners.team_offsets import TeamOffsetManager
import pandas as pd
import logging

# Set up logging to see the guardrail hits
logging.basicConfig(level=logging.DEBUG)

def debug():
    m = TeamOffsetManager()
    
    teams_to_check = [
        ("HAMBURGER SV", "BL1"),
        ("RAYO VALLECANO", "PD"),
        ("REAL OVIEDO", "PD"),
        ("Hamburger SV", "BL1"),  # Check un-normalized case for guardrail
        ("Rayo Vallecano", "PD")  # Check un-normalized case for guardrail
    ]
    
    print("\n--- Team Offset Verification ---")
    for team, league in teams_to_check:
        print(f"\nChecking: '{team}' ({league})")
        offset = m.get_offset(team, league)
        if offset:
            print(f"  Result: FOUND")
            print(f"  Canonical Name: {offset.team_name}")
            print(f"  Match Count: {offset.match_count}")
            print(f"  Corner Bias (H/A): {offset.home_corner_bias:.3f} / {offset.away_corner_bias:.3f}")
        else:
            print(f"  Result: NOT FOUND")

if __name__ == "__main__":
    debug()
